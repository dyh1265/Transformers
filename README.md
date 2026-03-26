# Nano-LLM

Decoder-only (GPT-style) transformer for training small language models from scratch with PyTorch.

## Features

- Character-level tokenizer (default), BPE tokenizer, byte-level BPE tokenizer, or Hugging Face byte-level BPE tokenizer (`hf_bpe_byte`); sinusoidal or RoPE positional encoding; causal multi-head attention
- Tiny Shakespeare, WikiText-2, IMDB sentiment datasets
- NGC PyTorch Docker container for GPU training
- HPO agent using local Qwen (Ollama) for hyperparameter tuning
- 8GB GPU support with config presets

## Quick Start

### With Docker (recommended)

```bash
# Build and run training
make train
# or: docker compose up --build

# Continue training from checkpoint
make resume EPOCHS=15
# or: docker compose run train python scripts/train.py --resume checkpoints/best.pt --epochs 15

# Generate text
make generate PROMPT="JULIET:" MAX_TOKENS=200

# Interactive shell
make shell
```

See `make help` for all targets.

### Weights & Biases (experiment tracking)

```bash
pip install wandb
wandb login   # paste API key from https://wandb.ai/authorize
```

Enable logging when training:

```bash
docker compose run --rm -e WANDB_API_KEY=... train python scripts/train.py \
  --dataset-id imdb_sentiment --use-wandb --wandb-project nano-llm-imdb \
  --wandb-tags imdb,hf_bpe_byte --epochs 10
```

Or with Make (set `WANDB_API_KEY` in your environment, or add it to `.env` for Compose):

```bash
make train-imdb ARGS='--use-wandb --wandb-project nano-llm --wandb-run-name run1'
```

Each epoch logs `train/loss`, `val/loss`, perplexity, learning rate. Use `--wandb-log-model` to upload `best.pt` at the end (larger upload).

### Local

```bash
pip install -r requirements.txt
pip install -e .

# Train with defaults
python scripts/train.py

# Override hyperparameters
python scripts/train.py --d-model 128 --epochs 5 --batch-size 32

# Optional BPE tokenizer
python scripts/train.py --tokenizer-type bpe --bpe-vocab-size 256

# Optional byte-level BPE tokenizer
python scripts/train.py --tokenizer-type bpe_byte --bpe-vocab-size 256

# Optional Hugging Face byte-level BPE tokenizer
python scripts/train.py --tokenizer-type hf_bpe_byte --bpe-vocab-size 256

# Continue training from checkpoint (more epochs)
python scripts/train.py --resume checkpoints/best.pt --epochs 15

# Early stopping (stop if val_loss unchanged for 10 epochs)
python scripts/train.py --epochs 3000 --early-stopping-patience 10
```

### Generation (inference)

After training, generate text from a checkpoint:

```bash
# Default: greedy, 100 tokens, prompt "ROMEO:"
python scripts/generate.py

# Custom prompt and sampling
python scripts/generate.py --prompt "JULIET:" --max-tokens 200
python scripts/generate.py --method top_k --top-k 40 --temperature 0.8
python scripts/generate.py --method top_p --top-p 0.9 --seed 42

# Specific checkpoint
python scripts/generate.py --checkpoint checkpoints/best.pt
```

With Docker (after training in container, checkpoints in `./checkpoints`):

```bash
# Generate using GPU
docker compose run generate

# With options (args pass through to generate.py)
docker compose run generate --prompt "JULIET:" --max-tokens 200 --method top_p
```

## Pretrain → Fine-tune (IMDB)

Pretrain on WikiText-2 for general language, then fine-tune on IMDB for sentiment-conditioned reviews:

```bash
# Step 1: Pretrain on WikiText-2 (10 epochs default)
make pretrain PRETRAIN_EPOCHS=10

# Step 2: Fine-tune on IMDB from pretrained checkpoint
make finetune EPOCHS=30
```

Checkpoints:
- Pretrain: `checkpoints/pretrain/best.pt`
- Fine-tune: `checkpoints/imdb_sentiment/hf_bpe_byte/best.pt`

Train on IMDB only (no pretrain), then interactive chat:

```bash
make train-imdb EPOCHS=30
make chat-imdb
```

`chat-imdb` uses the same prompt format as training (`[SENTIMENT]` / `[REVIEW]`). Override checkpoint: `IMDB_CHECKPOINT=path/to/best.pt make chat-imdb`.

Generate (one shot):
```bash
docker compose run --rm generate \
  --checkpoint checkpoints/imdb_sentiment/hf_bpe_byte/best.pt \
  --prompt "<bos>[SENTIMENT] positive [/SENTIMENT] [REVIEW] " \
  --method top_p --temperature 0.7 --repetition-penalty 1.2 \
  --max-tokens 300 --stop-sequence "[/REVIEW]"
```

Manual (same config as Makefile):
```bash
# Pretrain
docker compose run --rm train python scripts/train.py \
  --dataset-id wikitext_2 --tokenizer-type hf_bpe_byte --bpe-vocab-size 256 \
  --position-encoding rope --epochs 10 --checkpoint-dir checkpoints/pretrain

# Fine-tune
docker compose run --rm train python scripts/train.py \
  --dataset-id imdb_sentiment --resume checkpoints/pretrain/best.pt \
  --tokenizer-type hf_bpe_byte --bpe-vocab-size 256 --position-encoding rope \
  --imdb-max-review-chars 500 --epochs 30 \
  --checkpoint-dir checkpoints/imdb_sentiment/hf_bpe_byte --early-stopping-patience 5
```

### IMDB Counterfactual Embedding Objective

For sentiment-conditioned factual/counterfactual branch training, enable:

- `--enable-counterfactual-objective`
- `--counterfactual-ce-weight` (default `1.0`)
- `--counterfactual-embedding-weight` (default `0.25`)

Loss:

`L_total = ce_weight * L_ce + emb_weight * ((1 - T) * L_neg + T * L_pos)`

- `T`: treatment from factual sentiment (`negative=0`, `positive=1`)
- `L_pos`, `L_neg`: cosine embedding losses between factual review embedding and the positive/negative branch embeddings

Example:

```bash
python scripts/train.py \
  --dataset-id imdb_sentiment \
  --tokenizer-type hf_bpe_byte --bpe-vocab-size 256 \
  --enable-counterfactual-objective \
  --counterfactual-ce-weight 1.0 \
  --counterfactual-embedding-weight 0.25 \
  --epochs 10
```

## HPO Agent (Local Qwen)

1. Install and run Ollama: `ollama pull llama3.2:1b && ollama run llama3.2:1b` (or run as server)
2. Run the agent:

```bash
python scripts/hpo_agent.py --max-trials 10

# 8GB GPU bounds
python scripts/hpo_agent.py --max-trials 10 --8gb

# Custom Ollama endpoint / robust JSON retries
python scripts/hpo_agent.py --base-url http://localhost:11434/v1/ --max-parse-retries 3
```

HPO artifacts are written to tokenizer-specific folders by default:
- `hpo_results/char/`
- `hpo_results/bpe/`
- `hpo_results/bpe_byte/`

Use `--results-dir` to override.

### Train from best HPO config (reproducible)

After HPO completes, train from scratch with the best hyperparameters (no checkpoint resume):

**Makefile (Docker, one command):**

```bash
make train-best EPOCHS=30 HPO_RESULTS_DIR=hpo_results/char
```

**Docker (manual):**

```bash
# Step 1: Extract flat config from best_config.json
docker compose run --rm train python -c "
import json
from pathlib import Path
d = json.loads(Path('hpo_results/best_config.json').read_text())
Path('hpo_results/best_train_config.json').write_text(json.dumps(d['config'], indent=2))
"

# Step 2: Train from scratch with best params
docker compose run --rm train python scripts/train.py \
  --config hpo_results/best_train_config.json \
  --epochs 30 \
  --checkpoint-dir checkpoints/best_hpo
```

**Local:**

```bash
python -c "
import json
from pathlib import Path
d = json.loads(Path('hpo_results/best_config.json').read_text())
Path('hpo_results/best_train_config.json').write_text(json.dumps(d['config'], indent=2))
"
python scripts/train.py --config hpo_results/best_train_config.json --epochs 30 --checkpoint-dir checkpoints/best_hpo
```

Output checkpoint: `checkpoints/best_hpo/best.pt`. To resume later: `--resume checkpoints/best_hpo/best.pt`.

## Tests

```bash
# Unit tests only (default)
pytest

# All tests including integration
pytest --override-ini "addopts=-v -x"

# With coverage
pytest --cov=src/nano_llm --cov-report=term-missing
```

## Project Structure

- `src/nano_llm/` – model, layers, tokenizer, data, train, inference
- `src/nano_llm/agent/` – HPO agent (Ollama/Qwen)
- `scripts/train.py` – CLI for training
- `scripts/generate.py` – CLI for text generation from checkpoint
- `scripts/hpo_agent.py` – CLI for HPO agent
- `hpo_results/` – trial results JSON

## Config

Use `--config path/to.json` or set `NANO_LLM_CONFIG`. CLI flags override config file.

## Framework

This project uses **PyTorch** (NGC PyTorch container for Docker). Mixed precision (fp16) is enabled by default on CUDA.

## JEPA Reference

See [docs/LLM_JEPA_REFERENCE.md](docs/LLM_JEPA_REFERENCE.md) for notes on LLM-JEPA (Joint Embedding Predictive Architecture), the paper (arxiv.org/abs/2509.14252), and ideas for extending nano_llm with JEPA-style training (e.g., execution-grounded or causal multi-view JEPA).
Implementation roadmap: [docs/JEPA_PLAN.md](docs/JEPA_PLAN.md).


docker compose run --rm -it chat --checkpoint "checkpoints/imdb_sentiment/hf_bpe_byte/best.pt" --max-tokens 240 --temperature 0.9 --top-p 0.9 --repetition-penalty 1.15