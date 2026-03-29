"""Integration tests for training (excluded from default pytest run)."""

import tempfile
from pathlib import Path

import pytest
import torch

from nano_llm.config import DEFAULT_CONFIG
from nano_llm.data import PAD_TARGET_IGNORE_INDEX, create_dataloaders, format_imdb_example
from nano_llm.model import build_model
from nano_llm.tokenizer import HFByteBPETokenizer
from nano_llm.train import train
import nano_llm.train as train_module


@pytest.mark.integration
def test_one_training_step_loss_finite() -> None:
    cfg = dict(DEFAULT_CONFIG)
    cfg["epochs"] = 1
    cfg["checkpoint_dir"] = tempfile.mkdtemp()
    cfg["hpo_results_dir"] = tempfile.mkdtemp()
    cfg["batch_size"] = 4
    cfg["seq_len"] = 64
    results = train(cfg)
    assert "final_train_loss" in results
    assert isinstance(results["final_train_loss"], (int, float))
    assert not (results["final_train_loss"] != results["final_train_loss"])  # not nan


@pytest.mark.integration
def test_checkpoint_save_and_load() -> None:
    pytest.importorskip("tokenizers")
    train_samples = [
        format_imdb_example("Short positive.", 1)[0],
        format_imdb_example("Short negative.", 0)[0],
    ]
    val_samples = [format_imdb_example("Val positive.", 1)[0]]
    corpus = "\n".join(train_samples + val_samples)
    tokenizer = HFByteBPETokenizer.from_text(corpus, vocab_size=256)
    model = build_model(
        vocab_size=tokenizer.vocab_size,
        d_model=32,
        num_heads=2,
        num_layers=2,
        d_ff=128,
    )
    train_loader, _ = create_dataloaders(train_samples, val_samples, tokenizer, seq_len=64, batch_size=2)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_TARGET_IGNORE_INDEX)

    model.train()
    batch = next(iter(train_loader))
    x, y = batch[0], batch[1]
    optimizer.zero_grad()
    logits = model(x)
    loss = criterion(logits.view(-1, tokenizer.vocab_size), y.view(-1))
    loss.backward()
    optimizer.step()

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.pt"
        torch.save({"model": model.state_dict()}, path)
        ckpt = torch.load(path, map_location="cpu")
        model2 = build_model(vocab_size=tokenizer.vocab_size, d_model=32, num_heads=2, num_layers=2, d_ff=128)
        model2.load_state_dict(ckpt["model"])
        x_test = torch.randint(0, tokenizer.vocab_size, (1, 32))
        out1 = model(x_test)
        out2 = model2(x_test)
        assert torch.allclose(out1, out2, atol=1e-5)


@pytest.mark.integration
def test_counterfactual_objective_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    train_samples = [
        format_imdb_example("Great cast and a compelling story.", 1)[0],
        format_imdb_example("Terrible pacing and weak acting.", 0)[0],
    ]
    val_samples = [
        format_imdb_example("I liked the soundtrack and ending.", 1)[0],
        format_imdb_example("I disliked the dialogue and tone.", 0)[0],
    ]

    monkeypatch.setattr(
        train_module,
        "load_imdb_sentiment",
        lambda **kwargs: (train_samples, val_samples),
    )

    cfg = dict(DEFAULT_CONFIG)
    cfg["enable_counterfactual_objective"] = True
    cfg["counterfactual_ce_weight"] = 1.0
    cfg["counterfactual_embedding_weight"] = 0.2
    cfg["epochs"] = 1
    cfg["batch_size"] = 2
    cfg["seq_len"] = 64
    cfg["d_model"] = 32
    cfg["num_heads"] = 2
    cfg["num_layers"] = 2
    cfg["d_ff"] = 128
    cfg["checkpoint_dir"] = tempfile.mkdtemp()
    cfg["hpo_results_dir"] = tempfile.mkdtemp()
    results = train(cfg)
    assert "final_train_loss" in results
    assert isinstance(results["final_train_loss"], (int, float))
    assert not (results["final_train_loss"] != results["final_train_loss"])
