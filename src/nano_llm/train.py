"""Training loop for nano-llm."""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    # Fallback when tqdm isn't installed; keeps training functional.
    def tqdm(x, **kwargs):
        return x


from nano_llm.config import DEFAULT_CONFIG
from nano_llm.data import PAD_TARGET_IGNORE_INDEX, create_dataloaders, load_imdb_sentiment
from nano_llm.inference.load import normalize_checkpoint_state_dict
from nano_llm.model import build_model
from nano_llm.tokenizer import build_tokenizer_from_text, tokenizer_from_state

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _state_dict_for_checkpoint(model: torch.nn.Module) -> dict:
    """Save weights without torch.compile / DDP wrapper prefixes."""
    m = model
    while hasattr(m, "_orig_mod"):
        m = getattr(m, "_orig_mod")
    return m.state_dict()


def _configure_cuda_training(cfg: dict, device: torch.device) -> None:
    """TF32 and SDPA backend preferences for faster matmul / attention on CUDA."""
    if device.type != "cuda":
        return
    if cfg.get("cuda_allow_tf32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("CUDA TF32 enabled for matmul and cuDNN")
    if not cfg.get("cuda_prefer_flash_attn", True):
        return
    try:
        from torch.backends.cuda import (
            enable_cudnn_sdp,
            enable_flash_sdp,
            enable_math_sdp,
            enable_mem_efficient_sdp,
        )

        enable_cudnn_sdp(False)
        enable_flash_sdp(True)
        enable_mem_efficient_sdp(True)
        enable_math_sdp(True)
        logger.info(
            "CUDA SDPA backends: cudnn=False flash=True mem_efficient=True math=True (fallback)"
        )
    except Exception as e:
        logger.warning("CUDA SDPA backend tuning skipped: %s", e)


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
        "bits_per_byte": bits_per_byte,
    }


def _js_divergence_from_logits(
    logits_p: torch.Tensor, logits_q: torch.Tensor, *, eps: float = 1e-12
) -> torch.Tensor:
    """Jensen–Shannon divergence per position from logits.

    Returns JS(p||q) with natural logs, shape: logits_p.shape[:-1].
    """
    p = torch.softmax(logits_p, dim=-1).clamp_min(eps)
    q = torch.softmax(logits_q, dim=-1).clamp_min(eps)
    m = 0.5 * (p + q)
    kl_pm = (p * (p.log() - m.log())).sum(dim=-1)
    kl_qm = (q * (q.log() - m.log())).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm)


def _tarnet_weighted_ce_and_sep_loss(
    logits0: torch.Tensor,
    logits1: torch.Tensor,
    y: torch.Tensor,
    treatment: torch.Tensor,
    *,
    sep_weight: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """TARNet treatment-weighted CE plus optional separation term.

    Returns ``(ce_loss, total_loss)``.
    """
    bsz, seqlen, vocab_size = logits0.shape
    y_flat = y.reshape(-1)
    valid = (y_flat != PAD_TARGET_IGNORE_INDEX).reshape(bsz, seqlen)
    loss0_tok = F.cross_entropy(
        logits0.reshape(-1, vocab_size),
        y_flat,
        ignore_index=PAD_TARGET_IGNORE_INDEX,
        reduction="none",
    ).reshape(bsz, seqlen)
    loss1_tok = F.cross_entropy(
        logits1.reshape(-1, vocab_size),
        y_flat,
        ignore_index=PAD_TARGET_IGNORE_INDEX,
        reduction="none",
    ).reshape(bsz, seqlen)
    denom = valid.sum(dim=1).clamp_min(1)
    ce0 = (loss0_tok * valid).sum(dim=1) / denom
    ce1 = (loss1_tok * valid).sum(dim=1) / denom
    t = treatment.to(dtype=ce0.dtype)
    ce_loss = ((1.0 - t) * ce0 + t * ce1).mean()
    sep_loss = torch.tensor(0.0, device=device)
    if sep_weight > 0:
        js = _js_divergence_from_logits(logits0, logits1)
        sep_loss = -((js * valid.to(dtype=js.dtype)).sum() / valid.sum().clamp_min(1))
    total_loss = ce_loss + sep_weight * sep_loss
    return ce_loss, total_loss


def train(config: dict | None = None) -> dict:
    """Run training. Returns results dict for HPO."""
    cfg = config or dict(DEFAULT_CONFIG)
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mp = str(cfg.get("mixed_precision", "fp16")).lower()
    if mp not in ("fp32", "fp16", "bf16"):
        logger.warning("Unknown mixed_precision=%r; using fp16", mp)
        mp = "fp16"
    if mp in ("fp16", "bf16") and device.type != "cuda":
        logger.warning("mixed_precision=%s requires CUDA; using fp32", mp)
        mp = "fp32"
    use_cuda_amp = device.type == "cuda" and mp in ("fp16", "bf16")
    amp_dtype = torch.bfloat16 if mp == "bf16" else torch.float16
    use_grad_scaler = device.type == "cuda" and mp == "fp16"

    _configure_cuda_training(cfg, device)

    logger.info("Loading data...")
    dataset_id = str(cfg.get("dataset_id", "imdb_sentiment")).lower()
    if dataset_id != "imdb_sentiment":
        raise ValueError(
            f"Unsupported dataset_id: {dataset_id!r}. This project trains only on IMDB "
            '(set dataset_id to "imdb_sentiment" or omit it).'
        )
    train_samples, val_samples = load_imdb_sentiment(
        max_train_samples=cfg.get("imdb_max_train_samples"),
        max_val_samples=cfg.get("imdb_max_val_samples"),
        max_review_chars=cfg.get("imdb_max_review_chars"),
        subset_seed=int(cfg.get("seed", 42)),
        imdb_conditioning_style=str(cfg.get("imdb_conditioning_style", "tags")),
        imdb_positive_instruction=cfg.get("imdb_positive_instruction"),
        imdb_negative_instruction=cfg.get("imdb_negative_instruction"),
    )
    train_text = "\n".join(train_samples)
    val_text = "\n".join(val_samples)
    resume_path = cfg.get("resume")
    bpe_vocab_size = int(cfg.get("bpe_vocab_size", 8000))
    bpe_word_boundary_aware = bool(cfg.get("bpe_word_boundary_aware", False))
    tt = str(cfg.get("tokenizer_type", "hf_bpe_byte")).lower()
    if tt != "hf_bpe_byte":
        logger.warning("tokenizer_type=%r is ignored; only hf_bpe_byte is supported", tt)
    cfg["tokenizer_type"] = "hf_bpe_byte"
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
            "block_attn_residuals",
            "macro_block_size",
            "max_block_representations",
        )
        for k in model_keys:
            if k in ckpt["config"]:
                cfg[k] = ckpt["config"][k]
        tok_state = ckpt.get("tokenizer_state")
        vocab = ckpt.get("vocab")
        if tok_state:
            tokenizer = tokenizer_from_state(tok_state)
        elif vocab is not None:
            raise ValueError(
                "Checkpoint has legacy list `vocab` but no `tokenizer_state`. "
                "Only Hugging Face byte-level BPE (hf_bpe_byte) is supported; "
                "retrain or add tokenizer_state."
            )
        else:
            tokenizer = build_tokenizer_from_text(
                train_text + val_text,
                bpe_vocab_size=bpe_vocab_size,
                bpe_word_boundary_aware=bpe_word_boundary_aware,
            )
    else:
        tokenizer = build_tokenizer_from_text(
            train_text + val_text,
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
        train_samples,
        val_samples,
        tokenizer,
        seq_len=seq_len,
        batch_size=batch_size,
        imdb_tarnet_two_heads=bool(cfg.get("tarnet_two_heads", False)),
        imdb_tarnet_command_prompt=str(
            cfg.get("imdb_tarnet_command_prompt", "GENERATE an IMDB-like review:")
        ),
        imdb_conditioning_style=str(cfg.get("imdb_conditioning_style", "tags")),
        imdb_positive_instruction=cfg.get("imdb_positive_instruction"),
        imdb_negative_instruction=cfg.get("imdb_negative_instruction"),
    )

    pos_enc = str(cfg.get("position_encoding", "sinusoidal")).lower()
    model_kw = dict(
        vocab_size=vocab_size,
        d_model=int(cfg["d_model"]),
        num_heads=int(cfg["num_heads"]),
        num_layers=int(cfg["num_layers"]),
        d_ff=int(cfg["d_ff"]),
        max_len=seq_len + 10,
        dropout=float(cfg["dropout"]),
        weight_tie=cfg.get("weight_tie", True),
        tarnet_two_heads=bool(cfg.get("tarnet_two_heads", False)),
        tarnet_head_n_fc=int(cfg.get("tarnet_head_n_fc", 2)),
        tarnet_head_hidden_dim=cfg.get("tarnet_head_hidden_dim"),
        tarnet_head0_n_fc=cfg.get("tarnet_head0_n_fc"),
        tarnet_head0_hidden_dim=cfg.get("tarnet_head0_hidden_dim"),
        tarnet_head1_n_fc=cfg.get("tarnet_head1_n_fc"),
        tarnet_head1_hidden_dim=cfg.get("tarnet_head1_hidden_dim"),
        position_encoding=pos_enc,
        block_attn_residuals=bool(cfg.get("block_attn_residuals", False)),
        macro_block_size=int(cfg.get("macro_block_size", 2)),
        max_block_representations=int(cfg.get("max_block_representations", 9)),
    )
    model = build_model(**model_kw)
    if resume_path and Path(resume_path).exists():
        model.load_state_dict(normalize_checkpoint_state_dict(ckpt["model"]))
    model = model.to(device)
    if device.type == "cuda" and cfg.get("torch_compile", False):
        try:
            model = torch.compile(model, dynamic=False)
            logger.info("torch.compile enabled (dynamic=False)")
        except Exception as e:
            logger.warning("torch.compile disabled: %s", e)
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
    tarnet_head_separation_weight = float(cfg.get("tarnet_head_separation_weight", 0.0))

    ckpt_dir = Path(cfg.get("checkpoint_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "best.pt"

    best_val_loss = float("inf")
    history = {"loss": [], "val_loss": []}
    patience = int(cfg.get("early_stopping_patience", 0))
    epochs_since_improvement = 0

    start = time.perf_counter()
    scaler = torch.amp.GradScaler("cuda") if use_grad_scaler else None

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
        train_loss_ce = 0.0
        tarnet_two_heads = bool(cfg.get("tarnet_two_heads", False))
        for batch in tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs} [train]",
            total=len(train_loader),
            leave=False,
        ):
            batch_has_tarnet = (
                tarnet_two_heads and isinstance(batch, (tuple, list)) and len(batch) == 4
            )
            if batch_has_tarnet:
                x, y, treatment, _review_mask = batch
                x = x.to(device)
                y = y.to(device)
                treatment = treatment.to(device)
            else:
                x = batch[0]
                y = batch[1]
                x = x.to(device)
                y = y.to(device)
            optimizer.zero_grad()
            if use_cuda_amp:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    if batch_has_tarnet:
                        logits0, logits1 = model(x, return_both_heads=True)
                        ce_loss, loss = _tarnet_weighted_ce_and_sep_loss(
                            logits0,
                            logits1,
                            y,
                            treatment,
                            sep_weight=tarnet_head_separation_weight,
                            device=device,
                        )
                    else:
                        logits = model(x)
                        ce_loss = criterion(logits.view(-1, vocab_size), y.view(-1))
                        loss = ce_loss
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            else:
                if batch_has_tarnet:
                    logits0, logits1 = model(x, return_both_heads=True)
                    ce_loss, loss = _tarnet_weighted_ce_and_sep_loss(
                        logits0,
                        logits1,
                        y,
                        treatment,
                        sep_weight=tarnet_head_separation_weight,
                        device=device,
                    )
                else:
                    logits = model(x)
                    ce_loss = criterion(logits.view(-1, vocab_size), y.view(-1))
                    loss = ce_loss
                loss.backward()
                optimizer.step()
            train_loss += loss.item() * x.size(0)
            train_loss_ce += ce_loss.item() * x.size(0)
        train_loss /= len(train_loader.dataset)
        train_loss_ce /= len(train_loader.dataset)
        history["loss"].append(train_loss)

        val_loss = train_loss
        val_loss_ce = train_loss_ce
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_loss_ce = 0.0
            with torch.no_grad():
                for batch in tqdm(
                    val_loader,
                    desc=f"Epoch {epoch + 1}/{epochs} [val]",
                    total=len(val_loader),
                    leave=False,
                ):
                    batch_has_tarnet = (
                        tarnet_two_heads and isinstance(batch, (tuple, list)) and len(batch) == 4
                    )
                    if batch_has_tarnet:
                        x, y, treatment, _review_mask = batch
                        x = x.to(device)
                        y = y.to(device)
                        treatment = treatment.to(device)
                    else:
                        x = batch[0]
                        y = batch[1]
                        x = x.to(device)
                        y = y.to(device)
                    if use_cuda_amp:
                        with torch.amp.autocast("cuda", dtype=amp_dtype):
                            if batch_has_tarnet:
                                logits0, logits1 = model(x, return_both_heads=True)
                                ce_loss, loss = _tarnet_weighted_ce_and_sep_loss(
                                    logits0,
                                    logits1,
                                    y,
                                    treatment,
                                    sep_weight=tarnet_head_separation_weight,
                                    device=device,
                                )
                            else:
                                logits = model(x)
                                ce_loss = criterion(logits.view(-1, vocab_size), y.view(-1))
                                loss = ce_loss
                    else:
                        if batch_has_tarnet:
                            logits0, logits1 = model(x, return_both_heads=True)
                            ce_loss, loss = _tarnet_weighted_ce_and_sep_loss(
                                logits0,
                                logits1,
                                y,
                                treatment,
                                sep_weight=tarnet_head_separation_weight,
                                device=device,
                            )
                        else:
                            logits = model(x)
                            ce_loss = criterion(logits.view(-1, vocab_size), y.view(-1))
                            loss = ce_loss
                    val_loss += loss.item() * x.size(0)
                    val_loss_ce += ce_loss.item() * x.size(0)
            val_loss /= len(val_loader.dataset)
            val_loss_ce /= len(val_loader.dataset)
            history["val_loss"].append(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_since_improvement = 0
                torch.save(
                    {
                        "model": _state_dict_for_checkpoint(model),
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
                        "model": _state_dict_for_checkpoint(model),
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
                f"val_bpb={val_metrics['bits_per_byte']:.3f} "
                f"train_ce={train_loss_ce:.4f} val_ce={val_loss_ce:.4f} lr={current_lr:.2e}"
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
                        "train/loss_ce": train_loss_ce,
                        "val/loss": val_loss,
                        "val/perplexity": val_metrics["perplexity"],
                        "val/loss_ce": val_loss_ce,
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
        "precision": mp,
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
