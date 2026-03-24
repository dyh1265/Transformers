"""Training loop for nano-llm."""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

import torch

from nano_llm.config import DEFAULT_CONFIG
from nano_llm.data import (
    PAD_TARGET_IGNORE_INDEX,
    create_dataloaders,
    load_bookcorpus,
    load_imdb_sentiment,
    load_pg19,
    load_tiny_shakespeare,
    load_wikitext_2,
    load_wikitext_103,
)
from nano_llm.model import build_model
from nano_llm.tokenizer import (
    CharTokenizer,
    build_tokenizer_from_text,
    tokenizer_from_state,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _wandb_sanitize_config(cfg: dict) -> dict[str, object]:
    """Config dict safe for wandb (JSON-serializable scalars / small structures)."""
    out: dict[str, object] = {}
    for k, v in cfg.items():
        if k.startswith("wandb_") and k != "wandb_tags":
            continue
        if v is None:
            continue
        try:
            json.dumps(v)
            out[k] = v
        except (TypeError, ValueError):
            out[k] = str(v)
    return out


def _maybe_init_wandb(cfg: dict) -> object | None:
    """Initialize Weights & Biases if use_wandb. Returns run object or None."""
    if not cfg.get("use_wandb"):
        return None
    try:
        import wandb
    except ImportError:
        logger.warning("use_wandb=True but wandb is not installed. Install with: pip install wandb")
        return None

    tags = cfg.get("wandb_tags")
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]
    elif tags is not None and not isinstance(tags, (list, tuple)):
        tags = None

    run = wandb.init(
        project=str(cfg.get("wandb_project", "nano-llm")),
        entity=cfg.get("wandb_entity") or None,
        name=cfg.get("wandb_run_name") or None,
        tags=list(tags) if tags else None,
        config=_wandb_sanitize_config(cfg),
    )
    logger.info("W&B run started (project=%s)", cfg.get("wandb_project", "nano-llm"))
    return run


def _comparison_metrics(
    loss_nats_per_token: float,
    *,
    token_count: int,
    byte_count: int,
    vocab_size: int,
) -> dict[str, float]:
    """Return tokenizer-agnostic comparison metrics from CE loss.

    loss_nats_per_token is cross-entropy in nats/token.
    """
    # Guard against overflow on pathological runs.
    ppl = math.exp(min(loss_nats_per_token, 20.0))
    normalized_ce = loss_nats_per_token / math.log(max(vocab_size, 2))
    bits_per_token = loss_nats_per_token / math.log(2.0)
    if byte_count > 0 and token_count > 0:
        bits_per_byte = bits_per_token * (token_count / byte_count)
    else:
        bits_per_byte = float("nan")
    return {
        "perplexity": ppl,
        "normalized_ce": normalized_ce,
        "bits_per_token": bits_per_token,
        "bits_per_byte": bits_per_byte,
    }


def train(config: dict | None = None) -> dict:
    """Run training. Returns results dict for HPO."""
    cfg = config or dict(DEFAULT_CONFIG)
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg.get("mixed_precision", "fp16") != "fp32"

    logger.info("Loading data...")
    dataset_id = str(cfg.get("dataset_id", "tiny_shakespeare")).lower()
    train_samples, val_samples = None, None
    if dataset_id == "tiny_shakespeare":
        train_text, val_text = load_tiny_shakespeare(val_split=0.1)
    elif dataset_id == "wikitext_2":
        train_text, val_text = load_wikitext_2(
            max_train_samples=cfg.get("wikitext_max_train_samples"),
            max_val_samples=cfg.get("wikitext_max_val_samples"),
        )
    elif dataset_id == "imdb_sentiment":
        train_samples, val_samples = load_imdb_sentiment(
            max_train_samples=cfg.get("imdb_max_train_samples"),
            max_val_samples=cfg.get("imdb_max_val_samples"),
            max_review_chars=cfg.get("imdb_max_review_chars"),
        )
        train_text = "\n".join(train_samples)
        val_text = "\n".join(val_samples)
    elif dataset_id == "pg19":
        train_text, val_text = load_pg19(
            max_train_books=cfg.get("pg19_max_train_books"),
            max_val_books=cfg.get("pg19_max_val_books"),
            max_chars_per_book=cfg.get("pg19_max_chars_per_book"),
        )
    elif dataset_id == "bookcorpus":
        try:
            train_text, val_text = load_bookcorpus(
                max_train_books=cfg.get("pg19_max_train_books"),
                max_val_books=cfg.get("pg19_max_val_books"),
                max_chars_per_book=cfg.get("pg19_max_chars_per_book"),
            )
        except (ValueError, RuntimeError) as e:
            if "empty" in str(e).lower():
                logger.warning(
                    "BookCorpus produced empty data, falling back to WikiText-103: %s", e
                )
                train_text, val_text = load_wikitext_103(
                    max_train_samples=cfg.get("wikitext_max_train_samples"),
                    max_val_samples=cfg.get("wikitext_max_val_samples"),
                )
            else:
                raise
    elif dataset_id == "wikitext_103":
        # Default limit to avoid OOM; full corpus is ~1.8M lines
        max_train = cfg.get("wikitext_max_train_samples")
        if max_train is None:
            max_train = 200_000
            logger.info(
                "WikiText-103: default max_train_samples=%s "
                "(set wikitext_max_train_samples for full corpus)",
                max_train,
            )
        train_text, val_text = load_wikitext_103(
            max_train_samples=max_train,
            max_val_samples=cfg.get("wikitext_max_val_samples") or 5000,
        )
    else:
        raise ValueError(
            f"Unsupported dataset_id: {dataset_id}. "
            "Use one of: tiny_shakespeare, wikitext_2, wikitext_103, "
            "imdb_sentiment, pg19, bookcorpus"
        )
    resume_path = cfg.get("resume")
    tokenizer_type = str(cfg.get("tokenizer_type", "char")).lower()
    bpe_vocab_size = int(cfg.get("bpe_vocab_size", 256))
    bpe_word_boundary_aware = bool(cfg.get("bpe_word_boundary_aware", False))
    if resume_path and Path(resume_path).exists():
        logger.info("Resuming from checkpoint %s", resume_path)
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=True)
        # Merge model architecture from checkpoint, keep training control (epochs, lr, etc.)
        model_keys = (
            "d_model",
            "num_heads",
            "num_layers",
            "d_ff",
            "seq_len",
            "weight_tie",
            "dropout",
            "position_encoding",
        )
        for k in model_keys:
            if k in ckpt["config"]:
                cfg[k] = ckpt["config"][k]
        tok_state = ckpt.get("tokenizer_state")
        vocab = ckpt.get("vocab")
        if tok_state:
            tokenizer = tokenizer_from_state(tok_state)
        elif vocab:
            tokenizer = CharTokenizer(vocab=vocab)
        else:
            tokenizer = build_tokenizer_from_text(
                train_text + val_text,
                tokenizer_type=tokenizer_type,
                bpe_vocab_size=bpe_vocab_size,
                bpe_word_boundary_aware=bpe_word_boundary_aware,
            )
    else:
        tokenizer = build_tokenizer_from_text(
            train_text + val_text,
            tokenizer_type=tokenizer_type,
            bpe_vocab_size=bpe_vocab_size,
            bpe_word_boundary_aware=bpe_word_boundary_aware,
        )
    vocab_size = int(tokenizer.vocab_size)
    train_token_count = len(tokenizer.encode(train_text))
    val_token_count = len(tokenizer.encode(val_text))
    train_byte_count = len(train_text.encode("utf-8"))
    val_byte_count = len(val_text.encode("utf-8"))
    seq_len = int(cfg["seq_len"])
    batch_size = int(cfg["batch_size"])

    train_loader, val_loader = create_dataloaders(
        train_text,
        val_text,
        tokenizer,
        seq_len=seq_len,
        batch_size=batch_size,
        train_samples=train_samples if dataset_id == "imdb_sentiment" else None,
        val_samples=val_samples if dataset_id == "imdb_sentiment" else None,
    )

    pos_enc = str(cfg.get("position_encoding", "sinusoidal")).lower()
    if resume_path and Path(resume_path).exists():
        model = build_model(
            vocab_size=vocab_size,
            d_model=int(cfg["d_model"]),
            num_heads=int(cfg["num_heads"]),
            num_layers=int(cfg["num_layers"]),
            d_ff=int(cfg["d_ff"]),
            max_len=seq_len + 10,
            dropout=float(cfg["dropout"]),
            weight_tie=cfg.get("weight_tie", True),
            position_encoding=pos_enc,
        )
        model.load_state_dict(ckpt["model"])
    else:
        model = build_model(
            vocab_size=vocab_size,
            d_model=int(cfg["d_model"]),
            num_heads=int(cfg["num_heads"]),
            num_layers=int(cfg["num_layers"]),
            d_ff=int(cfg["d_ff"]),
            max_len=seq_len + 10,
            dropout=float(cfg["dropout"]),
            weight_tie=cfg.get("weight_tie", True),
            position_encoding=pos_enc,
        )
    model = model.to(device)
    total_params, trainable_params = _count_parameters(model)
    logger.info(
        "Model config: d_model=%s num_heads=%s num_layers=%s d_ff=%s seq_len=%s dropout=%s",
        cfg["d_model"],
        cfg["num_heads"],
        cfg["num_layers"],
        cfg["d_ff"],
        cfg["seq_len"],
        cfg["dropout"],
    )
    logger.info(
        "Model parameters: total=%s trainable=%s",
        f"{total_params:,}",
        f"{trainable_params:,}",
    )
    lr = float(cfg["learning_rate"])
    epochs = int(cfg["epochs"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_decay = cfg.get("lr_decay", "cosine")
    lr_min = float(cfg.get("lr_min", 1e-6))
    if lr_decay == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr_min
        )
    elif lr_decay == "linear":
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=lr_min / lr if lr > 0 else 1.0,
            total_iters=epochs,
        )
    else:
        scheduler = None
    criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_TARGET_IGNORE_INDEX)

    ckpt_dir = Path(cfg.get("checkpoint_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "best.pt"

    best_val_loss = float("inf")
    history = {"loss": [], "val_loss": []}
    patience = int(cfg.get("early_stopping_patience", 0))
    epochs_since_improvement = 0

    start = time.perf_counter()
    scaler = torch.amp.GradScaler("cuda") if use_amp and device.type == "cuda" else None

    wandb_run = _maybe_init_wandb(cfg)
    if wandb_run is not None:
        try:
            import wandb

            wandb.define_metric("epoch")
            wandb.define_metric("train/*", step_metric="epoch")
            wandb.define_metric("val/*", step_metric="epoch")
            wandb.log(
                {
                    "epoch": 0,
                    "model/total_params": total_params,
                    "model/trainable_params": trainable_params,
                    "data/train_tokens": train_token_count,
                    "data/val_tokens": val_token_count,
                    "data/train_batches": len(train_loader),
                }
            )
        except Exception as e:
            logger.warning("W&B initial log failed: %s", e)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            if use_amp and device.type == "cuda" and scaler is not None:
                with torch.amp.autocast("cuda"):
                    logits = model(x)
                    loss = criterion(logits.view(-1, vocab_size), y.view(-1))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x)
                loss = criterion(logits.view(-1, vocab_size), y.view(-1))
                loss.backward()
                optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_loader.dataset)
        history["loss"].append(train_loss)

        val_loss = train_loss
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    logits = model(x)
                    loss = criterion(logits.view(-1, vocab_size), y.view(-1))
                    val_loss += loss.item() * x.size(0)
            val_loss /= len(val_loader.dataset)
            history["val_loss"].append(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_since_improvement = 0
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": cfg,
                        "vocab": tokenizer.vocab,
                        "tokenizer_state": tokenizer.to_state(),
                    },
                    ckpt_path,
                )
            elif patience > 0:
                epochs_since_improvement += 1
        else:
            history["val_loss"].append(train_loss)
            if train_loss < best_val_loss:
                best_val_loss = train_loss
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": cfg,
                        "vocab": tokenizer.vocab,
                        "tokenizer_state": tokenizer.to_state(),
                    },
                    ckpt_path,
                )

        if scheduler is not None:
            scheduler.step()
        train_metrics = _comparison_metrics(
            train_loss,
            token_count=train_token_count,
            byte_count=train_byte_count,
            vocab_size=vocab_size,
        )
        val_metrics = _comparison_metrics(
            val_loss,
            token_count=val_token_count,
            byte_count=val_byte_count,
            vocab_size=vocab_size,
        )
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            (
                f"Epoch {epoch + 1}/{epochs} train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} val_ppl={val_metrics['perplexity']:.2f} "
                f"val_bpb={val_metrics['bits_per_byte']:.3f} lr={current_lr:.2e}"
            )
        )

        if wandb_run is not None:
            try:
                import wandb

                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "train/loss": train_loss,
                        "train/perplexity": train_metrics["perplexity"],
                        "val/loss": val_loss,
                        "val/perplexity": val_metrics["perplexity"],
                        "val/bits_per_byte": val_metrics["bits_per_byte"],
                        "lr": current_lr,
                        "best_val_loss": best_val_loss,
                    },
                    step=epoch + 1,
                )
            except Exception as e:
                logger.warning("wandb.log failed: %s", e)

        if patience > 0 and val_loader is not None and epochs_since_improvement >= patience:
            logger.info(
                "Early stopping: no val_loss improvement for %d epochs (best_val_loss=%.4f)",
                patience,
                best_val_loss,
            )
            break

    duration_sec = time.perf_counter() - start
    final_train_metrics = _comparison_metrics(
        history["loss"][-1],
        token_count=train_token_count,
        byte_count=train_byte_count,
        vocab_size=vocab_size,
    )
    final_val_metrics = _comparison_metrics(
        history["val_loss"][-1],
        token_count=val_token_count,
        byte_count=val_byte_count,
        vocab_size=vocab_size,
    )
    best_val_metrics = _comparison_metrics(
        best_val_loss,
        token_count=val_token_count,
        byte_count=val_byte_count,
        vocab_size=vocab_size,
    )

    results = {
        "trial_id": cfg.get("trial_id", 0),
        "config": {k: v for k, v in cfg.items() if k not in ("trial_id",)},
        "seed": seed,
        "precision": "fp16" if use_amp else "fp32",
        "dataset_id": dataset_id,
        "config_hash": "sha256:...",
        "final_train_loss": history["loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "best_val_loss": best_val_loss,
        "final_train_perplexity": final_train_metrics["perplexity"],
        "final_val_perplexity": final_val_metrics["perplexity"],
        "best_val_perplexity": best_val_metrics["perplexity"],
        "final_train_bits_per_byte": final_train_metrics["bits_per_byte"],
        "final_val_bits_per_byte": final_val_metrics["bits_per_byte"],
        "best_val_bits_per_byte": best_val_metrics["bits_per_byte"],
        "final_train_normalized_ce": final_train_metrics["normalized_ce"],
        "final_val_normalized_ce": final_val_metrics["normalized_ce"],
        "best_val_normalized_ce": best_val_metrics["normalized_ce"],
        "epochs_completed": len(history["loss"]),
        "duration_sec": round(duration_sec, 2),
    }
    results_dir = Path(cfg.get("hpo_results_dir", "hpo_results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / f"trial_{results['trial_id']}.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results: %s", results)

    if wandb_run is not None:
        try:
            import wandb

            wandb.summary["best_val_loss"] = best_val_loss
            wandb.summary["duration_sec"] = results["duration_sec"]
            wandb.summary["epochs_completed"] = results["epochs_completed"]
            if cfg.get("wandb_log_model") and ckpt_path.exists():
                wandb.save(str(ckpt_path))
        except Exception as e:
            logger.warning("W&B summary/artifact failed: %s", e)
        try:
            import wandb

            wandb.finish()
        except Exception as e:
            logger.warning("wandb.finish() failed: %s", e)

    return results
