"""Integration tests for training (excluded from default pytest run)."""

import tempfile
from pathlib import Path

import pytest
import torch

from nano_llm.config import DEFAULT_CONFIG
from nano_llm.data import create_dataloaders, load_tiny_shakespeare
from nano_llm.model import build_model
from nano_llm.tokenizer import CharTokenizer
from nano_llm.train import train


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
    train_text, val_text = load_tiny_shakespeare(val_split=0.1)
    tokenizer = CharTokenizer.from_text(train_text, add_special=False)
    model = build_model(
        vocab_size=tokenizer.vocab_size,
        d_model=32,
        num_heads=2,
        num_layers=2,
        d_ff=128,
    )
    train_loader, _ = create_dataloaders(train_text, val_text, tokenizer, seq_len=64, batch_size=4)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    x, y = next(iter(train_loader))
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
