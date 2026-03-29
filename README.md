# Nano-LLM

Decoder-only (GPT-style) transformer for training small language models from scratch with PyTorch.

## Features

- Character-level tokenizer (default), BPE tokenizer, byte-level BPE tokenizer, or Hugging Face byte-level BPE tokenizer (`hf_bpe_byte`); sinusoidal or RoPE positional encoding; causal multi-head attention
- IMDB sentiment reviews (Hugging Face `datasets`) for training
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
make generate PROMPT="<bos>[SENTIMENT] positive [/SENTIMENT] [REVIEW] " MAX_TOKENS=200

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
  --use-wandb --wandb-project nano-llm-imdb \
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
# Default: greedy, 100 tokens (set a checkpoint-appropriate prompt)
python scripts/generate.py

# Custom prompt and sampling (IMDB tags-style example)
python scripts/generate.py --prompt "<bos>[SENTIMENT] positive [/SENTIMENT] [REVIEW] " --max-tokens 200
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
docker compose run generate --prompt "<bos>[SENTIMENT] positive [/SENTIMENT] [REVIEW] " --max-tokens 200 --method top_p
```

## How training works

1. **CLI and config**  
   `scripts/train.py` loads `DEFAULT_CONFIG` (and optional `--config` JSON), applies CLI overrides, then calls `nano_llm.train.train(cfg)`.

2. **Data**  
   Training always loads IMDB from Hugging Face and formats each row into a conditioned string. A tokenizer is **trained on the training (and val) text** unless you **resume** from a checkpoint that contains `tokenizer_state` / `vocab`, in which case the tokenizer is restored to match the checkpoint. Optional JSON field `dataset_id` must be `"imdb_sentiment"` if present (older configs with other values are rejected).

3. **Batches**  
   **IMDB** uses chunking so the conditioning prefix (tags, natural instructions, or TARNet command + `[REVIEW]`) stays aligned with the review body; padded positions use target ignore index `-100`.

4. **Model**  
   Causal decoder-only transformer (`NanoLLM`). Optional **`--tarnet-two-heads`**: shared vocabulary head plus two sentiment residual heads; `weight_tie` is disabled in that mode.

5. **Loss and optimization**  
   - **Single head:** standard next-token cross-entropy (optionally with **weight-tied** embedding output).  
   - **TARNet:** per-example CE on head 0 vs head 1, weighted by treatment \(T\) from the review’s factual sentiment (`negative` → head 0, `positive` → head 1), mean over the batch; plus optional **`tarnet_head_separation_weight`** term (encourages different predictive distributions via Jensen–Shannon on softmax of the two heads).  
   Optimizer: **AdamW**; schedule: **cosine** (or **linear** / **none**) on learning rate. On CUDA, **AMP** (`fp16` with GradScaler, or `bf16`) when configured.

6. **Checkpointing**  
   After each epoch, if validation loss improves, `best.pt` is written under `checkpoint_dir` with `model` state, full `config`, `vocab`, and `tokenizer_state` for reproducible load and chat.

7. **IMDB conditioning**  
   - **`imdb_conditioning_style: tags`** (default): `[SENTIMENT] positive|negative [/SENTIMENT] [REVIEW] ... [/REVIEW]`.  
   - **`natural`**: fixed instruction strings before `[REVIEW]` (e.g. “Create a POSITIVE IMDB-like review”), set via `--imdb-conditioning-style natural` and optional `--imdb-positive-instruction` / `--imdb-negative-instruction`.  
   **`scripts/chat.py`** reads `imdb_conditioning_style` from the checkpoint config for single-head models.

## Training IMDB

Train on IMDB, then interactive chat:

```bash
make train-imdb EPOCHS=30
make chat-imdb
```

`chat-imdb` follows the checkpoint’s `imdb_conditioning_style` (tags vs natural instructions) for single-head models; TARNet counterfactual mode uses the command prompt + `[REVIEW]`. Override checkpoint: `IMDB_CHECKPOINT=path/to/best.pt make chat-imdb`.

Generate (one shot):
```bash
docker compose run --rm generate \
  --checkpoint checkpoints/imdb_sentiment/hf_bpe_byte/best.pt \
  --prompt "<bos>[SENTIMENT] positive [/SENTIMENT] [REVIEW] " \
  --method top_p --temperature 0.7 --repetition-penalty 1.2 \
  --max-tokens 300 --stop-sequence "[/REVIEW]"
```

Resume IMDB training from a checkpoint:

```bash
docker compose run --rm train python scripts/train.py \
  --resume checkpoints/imdb_sentiment/hf_bpe_byte/best.pt \
  --tokenizer-type hf_bpe_byte --bpe-vocab-size 256 --position-encoding rope \
  --imdb-max-review-chars 500 --epochs 30 \
  --checkpoint-dir checkpoints/imdb_sentiment/hf_bpe_byte --early-stopping-patience 5
```

### IMDB Counterfactual Embedding Objective (legacy / disabled in current trainer)

The current `nano_llm.train.train` loop does **not** apply the embedding-mixture loss below; it only uses next-token CE (and TARNet terms if `--tarnet-two-heads`). The following described an older objective; CLI flags may still appear in configs for reference.

For sentiment-conditioned factual/counterfactual branch training (historical), enable:

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

## Experiment log summary (below: raw paste preserved)

The block after this section is an **unedited archive** of Docker one-liners, training logs, and counterfactual chat transcripts.

**Short summary**

- **Broken first line:** Two commands were accidentally concatenated (`chat` + `hpo_agent` + checkpoint path). Do not run as a single line; use separate `docker compose run … chat` and `… hpo_agent` invocations.
- **Mid-size TARNet run (~11M-class trunk):** IMDB, `d_model=384`, `num_layers=6`, `d_ff=1536`, `seq_len=256`, sinusoidal PE, `--block-attn-residuals`, `--tarnet-two-heads`, checkpoint dir `counterfactual_new`; example chat uses `--counterfactual` with `top_p` / temperature / repetition penalty.
- **Large TARNet run (~20M params):** `d_model=512`, `num_heads=8`, `num_layers=6`, `d_ff=1888`, `seq_len=256`, RoPE, inter-block residuals, `hf_bpe_byte` vocab 256, `--tarnet-head-separation-weight 0.02`, 40 epochs, checkpoint dir `counterfactual_repeat_20m`. Logged **20,040,768** trainable parameters. Training wall time about **4.8 hours** (`duration_sec ≈ 17250`). Metrics: val loss **~1.96 → best ~1.65** (around epochs 22–25), slight rise toward **~1.66** by epoch 40; final train CE **~1.41**, val CE **~1.66**. Perplexity **~7 → ~5.2** (val best ~5.21).
- **Qualitative samples:** Counterfactual chat shows **Y0** often more negative / critical and **Y1** more positive openings, with mixed fluency (typical for this model scale and sampling).
- **512-wide IMDB comparison (vanilla vs inter-block vs TARNet):** see **### IMDB ~18–20M runs** inside the raw paste block for a table (worst → best val CE) and narrative on training vs chat samples.

### Raw paste (full log and commands)

docker compose run --rm -it chat --checkpoint "checkpoints/imdb_sentiment/hf_bpe_byte/best.pt" --max-tokens 240 --temperature 0.9 --top-p 0.9 --repetition-penalty 1.15

docker compose run --rm train python scripts/hpo_agent.py --max-trials 10 --model "llama3.2:1b" --base-url "http://host.docker.internal:11434/v1/" --tokenizer-type hf_bpe_byte --bpe-vocab-size 8000 --tarnet-two-heads --imdb-max-train-samples 2500 --imdb-max-val-samples 2500 --fixed-epochs 10



docker compose run --rm train python scripts/train.py --epochs 20 --batch-size 16 --d-model 384 --num-heads 6 --num-layers 6 --d-ff 1536 --seq-len 256 --dropout 0.1 --tokenizer-type
 hf_bpe_byte --bpe-vocab-size 256 --tarnet-two-heads --tarnet-head-n-fc 2 --position-encoding sinusoidal --block-attn-residuals --macro-block-size 2 --max-block-representations 9 --checkpoint-dir checkpoints/counterfactual_new

docker compose run --rm -it chat --checkpoint checkpoints/counterfactual_new/best.pt --max-tokens 240 --temperature 0.8 --method top_p --top-p 0.85  --counterfactual  --repetition-penalty 1.05


### IMDB ~18–20M runs: training, validation, and sample quality (summary)

The logs below compare three **512-wide** IMDB runs (same `num_layers=6`, `d_ff=1888`, `seq_len=256`, RoPE, `hf_bpe_byte` vocab 256, batch 16). Rows are ordered **worst → best** by **best validation CE** (lower is better).

| Order | Checkpoint | Decoder stack | LM head | Epochs | Params | Best val CE | Final val CE | Best val PPL | Wall time |
|------|------------|---------------|---------|--------|--------|-------------|--------------|--------------|-----------|
| 1 (worst) | `imdb_baseline_vanilla_20m` | Vanilla (`block_attn_residuals` off) | Single, `imdb_conditioning_style=natural` | 20 | 18,062,400 | 1.727 | 1.727 | 5.62 | ~1.2 h |
| 2 | `imdb_baseline_natural_20m` | Inter-block | Single, natural | 20 | 18,074,688 | 1.717 | 1.717 | 5.57 | ~2.2 h |
| 3 (best) | `counterfactual_repeat_20m` | Inter-block | TARNet two heads, `tarnet_head_separation_weight=0.02` | 40 | 20,040,768 | 1.650 | 1.664 | 5.21 | ~4.8 h |

**What was added each step, and what improved**

1. **Baseline (worst): vanilla decoder + natural instructions, single head.** Lowest training cost here, but **best val CE ~1.73**. Interactive **chat** samples (top_p 0.5, temp 0.7, repetition penalty 1.5) show **heavy garbling**: repeated junk tokens, odd symbols, and broken structure—usable as a negative example for this scale.

2. **Same recipe but inter-block decoder (`--block-attn-residuals`).** ~12k extra parameters, ~2× wall time for 20 epochs. **Best val CE improves by ~0.01** (1.727 → 1.717). Samples under `[POSITIVE]` / `[NEGATIVE]` prompts are still flawed but read more like **English sentences** (opinion + plot-like phrases) with fewer random symbol runs.

3. **TARNet + longer training + head separation.** Adds **two sentiment heads**, **JS separation loss** (`0.02`), and **40 epochs** (~2M more parameters than the single-head models). **Best val CE ~1.65** (~0.07 better than the natural inter-block single-head run). **Evaluation:** `chat --counterfactual` prints **Y0** vs **Y1** from the same prompt; transcripts show **tone skew** (more negative/critical vs more positive openings) mixed with contradictions and truncation—interesting for counterfactual play, not production quality.

**Caveats:** TARNet and single-head losses are not identical objectives; more epochs and parameters are **confounded** with architecture changes. Figures are taken from the pasted logs; config hashes in JSON are abbreviated.

---

## Full trainig and results of TARnet-like model with 20M parameters. 

 docker compose run --rm train python scripts/train.py --epochs 40 --batch-size 16 --d-model 512 --num-heads 8 --num-layers 6 --d-ff 1888 --seq-len 256 --dropout 0.1 --tokenizer-type
 hf_bpe_byte --bpe-vocab-size 256 --tarnet-two-heads --tarnet-head-n-fc 2 --position-encoding rope --block-attn-residuals --macro-block-size 2 --max-block-representations 9 --checkpoint-dir checkpoints/counterfactual_repeat_20m --tarnet-head-separation-weight 0.02                                               

INFO:nano_llm.train:Model config: d_model=512 num_heads=8 num_layers=6 d_ff=1888 seq_len=256 dropout=0.1
INFO:nano_llm.train:Model parameters: total=20,040,768 trainable=20,040,768
INFO:nano_llm.train:Epoch 1/40 train_loss=2.2577 val_loss=1.9603 val_ppl=7.10 val_bpb=1.599 train_ce=2.2579 val_ce=1.9605 lr=3.00e-04                                                                                                                             
INFO:nano_llm.train:Epoch 2/40 train_loss=1.9285 val_loss=1.8604 val_ppl=6.43 val_bpb=1.518 train_ce=1.9288 val_ce=1.8606 lr=2.98e-04                                                                                                                             
INFO:nano_llm.train:Epoch 3/40 train_loss=1.8450 val_loss=1.8089 val_ppl=6.10 val_bpb=1.476 train_ce=1.8453 val_ce=1.8092 lr=2.96e-04                                                                                                                             
INFO:nano_llm.train:Epoch 4/40 train_loss=1.7943 val_loss=1.7763 val_ppl=5.91 val_bpb=1.449 train_ce=1.7946 val_ce=1.7766 lr=2.93e-04                                                                                                                             
INFO:nano_llm.train:Epoch 5/40 train_loss=1.7571 val_loss=1.7520 val_ppl=5.77 val_bpb=1.429 train_ce=1.7574 val_ce=1.7522 lr=2.89e-04                                                                                                                             
INFO:nano_llm.train:Epoch 6/40 train_loss=1.7277 val_loss=1.7351 val_ppl=5.67 val_bpb=1.415 train_ce=1.7280 val_ce=1.7354 lr=2.84e-04                                                                                                                             
INFO:nano_llm.train:Epoch 7/40 train_loss=1.7035 val_loss=1.7203 val_ppl=5.59 val_bpb=1.403 train_ce=1.7038 val_ce=1.7206 lr=2.78e-04                                                                                                                             
INFO:nano_llm.train:Epoch 8/40 train_loss=1.6826 val_loss=1.7091 val_ppl=5.52 val_bpb=1.394 train_ce=1.6830 val_ce=1.7094 lr=2.71e-04                                                                                                                             
INFO:nano_llm.train:Epoch 9/40 train_loss=1.6641 val_loss=1.6993 val_ppl=5.47 val_bpb=1.386 train_ce=1.6645 val_ce=1.6996 lr=2.64e-04                                                                                                                             
INFO:nano_llm.train:Epoch 10/40 train_loss=1.6476 val_loss=1.6914 val_ppl=5.43 val_bpb=1.380 train_ce=1.6480 val_ce=1.6917 lr=2.56e-04                                                                                                                            
INFO:nano_llm.train:Epoch 11/40 train_loss=1.6321 val_loss=1.6831 val_ppl=5.38 val_bpb=1.373 train_ce=1.6325 val_ce=1.6834 lr=2.48e-04                                                                                                                            
INFO:nano_llm.train:Epoch 12/40 train_loss=1.6178 val_loss=1.6775 val_ppl=5.35 val_bpb=1.368 train_ce=1.6182 val_ce=1.6778 lr=2.38e-04                                                                                                                            
INFO:nano_llm.train:Epoch 13/40 train_loss=1.6047 val_loss=1.6716 val_ppl=5.32 val_bpb=1.364 train_ce=1.6051 val_ce=1.6719 lr=2.29e-04                                                                                                                            
INFO:nano_llm.train:Epoch 14/40 train_loss=1.5922 val_loss=1.6688 val_ppl=5.31 val_bpb=1.361 train_ce=1.5926 val_ce=1.6691 lr=2.18e-04                                                                                                                            
INFO:nano_llm.train:Epoch 15/40 train_loss=1.5802 val_loss=1.6659 val_ppl=5.29 val_bpb=1.359 train_ce=1.5806 val_ce=1.6662 lr=2.08e-04                                                                                                                            
INFO:nano_llm.train:Epoch 16/40 train_loss=1.5689 val_loss=1.6608 val_ppl=5.26 val_bpb=1.355 train_ce=1.5693 val_ce=1.6611 lr=1.97e-04                                                                                                                            
INFO:nano_llm.train:Epoch 17/40 train_loss=1.5576 val_loss=1.6581 val_ppl=5.25 val_bpb=1.353 train_ce=1.5581 val_ce=1.6585 lr=1.85e-04                                                                                                                            
INFO:nano_llm.train:Epoch 18/40 train_loss=1.5474 val_loss=1.6570 val_ppl=5.24 val_bpb=1.352 train_ce=1.5478 val_ce=1.6573 lr=1.74e-04                                                                                                                            
INFO:nano_llm.train:Epoch 19/40 train_loss=1.5370 val_loss=1.6530 val_ppl=5.22 val_bpb=1.348 train_ce=1.5374 val_ce=1.6533 lr=1.62e-04                                                                                                                            
INFO:nano_llm.train:Epoch 20/40 train_loss=1.5271 val_loss=1.6520 val_ppl=5.22 val_bpb=1.348 train_ce=1.5275 val_ce=1.6524 lr=1.50e-04                                                                                                                            
INFO:nano_llm.train:Epoch 21/40 train_loss=1.5173 val_loss=1.6543 val_ppl=5.23 val_bpb=1.349 train_ce=1.5177 val_ce=1.6546 lr=1.39e-04
INFO:nano_llm.train:Epoch 22/40 train_loss=1.5080 val_loss=1.6510 val_ppl=5.21 val_bpb=1.347 train_ce=1.5084 val_ce=1.6513 lr=1.27e-04                                                                                                                            
INFO:nano_llm.train:Epoch 23/40 train_loss=1.4987 val_loss=1.6519 val_ppl=5.22 val_bpb=1.347 train_ce=1.4992 val_ce=1.6523 lr=1.16e-04
INFO:nano_llm.train:Epoch 24/40 train_loss=1.4901 val_loss=1.6522 val_ppl=5.22 val_bpb=1.348 train_ce=1.4906 val_ce=1.6526 lr=1.04e-04
INFO:nano_llm.train:Epoch 25/40 train_loss=1.4815 val_loss=1.6503 val_ppl=5.21 val_bpb=1.346 train_ce=1.4820 val_ce=1.6507 lr=9.33e-05                                                                                                                            
INFO:nano_llm.train:Epoch 26/40 train_loss=1.4733 val_loss=1.6534 val_ppl=5.22 val_bpb=1.349 train_ce=1.4738 val_ce=1.6538 lr=8.26e-05
INFO:nano_llm.train:Epoch 27/40 train_loss=1.4654 val_loss=1.6523 val_ppl=5.22 val_bpb=1.348 train_ce=1.4659 val_ce=1.6527 lr=7.24e-05
INFO:nano_llm.train:Epoch 28/40 train_loss=1.4579 val_loss=1.6560 val_ppl=5.24 val_bpb=1.351 train_ce=1.4584 val_ce=1.6563 lr=6.26e-05
INFO:nano_llm.train:Epoch 29/40 train_loss=1.4509 val_loss=1.6541 val_ppl=5.23 val_bpb=1.349 train_ce=1.4514 val_ce=1.6545 lr=5.34e-05
INFO:nano_llm.train:Epoch 30/40 train_loss=1.4443 val_loss=1.6561 val_ppl=5.24 val_bpb=1.351 train_ce=1.4448 val_ce=1.6565 lr=4.48e-05
INFO:nano_llm.train:Epoch 31/40 train_loss=1.4385 val_loss=1.6554 val_ppl=5.24 val_bpb=1.350 train_ce=1.4390 val_ce=1.6558 lr=3.68e-05
INFO:nano_llm.train:Epoch 32/40 train_loss=1.4327 val_loss=1.6586 val_ppl=5.25 val_bpb=1.353 train_ce=1.4333 val_ce=1.6590 lr=2.96e-05
INFO:nano_llm.train:Epoch 33/40 train_loss=1.4278 val_loss=1.6594 val_ppl=5.26 val_bpb=1.354 train_ce=1.4284 val_ce=1.6598 lr=2.30e-05                                                                                                                            
INFO:nano_llm.train:Epoch 34/40 train_loss=1.4235 val_loss=1.6608 val_ppl=5.26 val_bpb=1.355 train_ce=1.4240 val_ce=1.6612 lr=1.73e-05
INFO:nano_llm.train:Epoch 35/40 train_loss=1.4198 val_loss=1.6631 val_ppl=5.28 val_bpb=1.357 train_ce=1.4203 val_ce=1.6636 lr=1.24e-05
INFO:nano_llm.train:Epoch 36/40 train_loss=1.4168 val_loss=1.6626 val_ppl=5.27 val_bpb=1.356 train_ce=1.4174 val_ce=1.6630 lr=8.32e-06
INFO:nano_llm.train:Epoch 37/40 train_loss=1.4141 val_loss=1.6635 val_ppl=5.28 val_bpb=1.357 train_ce=1.4146 val_ce=1.6639 lr=5.13e-06                                                                                                                            
INFO:nano_llm.train:Epoch 38/40 train_loss=1.4123 val_loss=1.6638 val_ppl=5.28 val_bpb=1.357 train_ce=1.4129 val_ce=1.6642 lr=2.84e-06
INFO:nano_llm.train:Epoch 39/40 train_loss=1.4110 val_loss=1.6640 val_ppl=5.28 val_bpb=1.357 train_ce=1.4116 val_ce=1.6644 lr=1.46e-06
INFO:nano_llm.train:Epoch 40/40 train_loss=1.4102 val_loss=1.6645 val_ppl=5.28 val_bpb=1.358 train_ce=1.4108 val_ce=1.6649 lr=1.00e-06
INFO:nano_llm.train:Results: {'trial_id': 0, 'config': {'vocab_size': 65, 'd_model': 512, 'num_heads': 8, 'num_layers': 6, 'd_ff': 1888, 'seq_len': 256, 'dropout': 0.1, 'tokenizer_type': 'hf_bpe_byte', 'bpe_vocab_size': 256, 'bpe_word_boundary_aware': False, 'batch_size': 16, 'learning_rate': 0.0003, 'lr_decay': 'cosine', 'lr_min': 1e-06, 'epochs': 40, 'early_stopping_patience': 0, 'dataset_id': 'imdb_sentiment', 'wikitext_max_train_samples': None, 'wikitext_max_val_samples': None, 'imdb_max_train_samples': None, 'imdb_max_val_samples': None, 'imdb_max_review_chars': None, 'enable_counterfactual_objective': False, 'counterfactual_ce_weight': 1.0, 'counterfactual_embedding_weight': 0.25, 'tarnet_two_heads': True, 'imdb_tarnet_command_prompt': 'GENERATE an IMDB-like review:', 'tarnet_head_separation_weight': 0.02, 'tarnet_head_n_fc': 2, 'tarnet_head_hidden_dim': None, 'tarnet_head0_n_fc': None, 'tarnet_head0_hidden_dim': None, 'tarnet_head1_n_fc': None, 'tarnet_head1_hidden_dim': None, 'pg19_max_train_books': None, 'pg19_max_val_books': None, 'pg19_max_chars_per_book': None, 'position_encoding': 'rope', 'block_attn_residuals': True, 'macro_block_size': 2, 'max_block_representations': 9, 'weight_tie': True, 'mixed_precision': 'fp16', 'cuda_allow_tf32': True, 'cuda_prefer_flash_attn': True, 'torch_compile': False, 'gradient_checkpointing': False, 'gradient_accumulation_steps': 1, 'seed': 42, 'use_wandb': False, 'wandb_project': 'nano-llm', 'wandb_run_name': None, 'wandb_entity': None, 'wandb_tags': None, 'wandb_log_model': False, 'checkpoint_dir': 'checkpoints/counterfactual_repeat_20m'}, 'seed': 42, 'precision': 'fp16', 'dataset_id': 'imdb_sentiment', 'config_hash': 'sha256:...', 'final_train_loss': 1.4102234038295165, 'final_val_loss': 1.6644850165092653, 'best_val_loss': 1.6502815350542595, 'final_train_perplexity': 4.096870558414661, 'final_val_perplexity': 5.282951916080821, 'best_val_perplexity': 5.2084459809051715, 'final_train_bits_per_byte': 1.1511885238712738, 'final_val_bits_per_byte': 1.3577034089456084, 'best_val_bits_per_byte': 1.346117774350471, 'final_train_normalized_ce': 0.2543152889062997, 'final_val_normalized_ce': 0.3001680348689877, 'best_val_normalized_ce': 0.29760662333667576, 'epochs_completed': 40, 'duration_sec': 17249.96}

# Test Results
docker compose run --rm -it chat --checkpoint checkpoints/counterfactual_repeat_20m/best.pt --max-tokens 340 --temperature 0.7 --method top_p --top-p 0.5  --counterfactual  --repetition-penalty 1.5


Generate [+/-/b/q] (default b):

[Y0]
One of the most amazing movies ever. This one is so bad that it's not fun to watch, and you can't help but laugh about it. There is no story in this movie, or excitement (sometimes) in this film which has been done with good performance and hardly scary moments. And I think the problem was that there was more to this movie than any of those which would really be damned about what was good for something that selled us to lower and move forward, or talking with. Sadly, the stories sequence of five minutes asides in confusion but because it's literally not gay or when you'll end just think agonize frievous only to you. The film's twist is dull; except for Shea and down-a-looking little tal

[Y1]
What a great movie! This is the best film ever made. The cast was fantasic, and sometimes it seemed like Jack Smith didn't try to be in this one as wonderfully as he did in his movies. As for the porn star, I though I think that's what you want to see.      

Generate [+/-/b/q] (default b):         

[Y0]
ember the same way, and it was all downhill from there. The pace is terrible, but nothing exciting happens. So what's with that? And why is this movie such a confused mess of old film? Too many story lines involve between those guards where you see it only to be fascistic (I'm not asking for more) to disgust over the center of this movie and think that you had good times past her and I was really let down but it seldom became more informative.

[Y1]
What a fun movie! The casting of the two leads is great. She portrays the dancer who has been married to her old but she would not have sex with her. And, in this case, sometimes it's about as good as everyone elses. This film was once again just before you'll stick with it and remind my opinion of that for you.

Generate [+/-/b/q] (default b):

[Y0]
For a film that has no character development, and it is pretty bad. The story is told in sometimes you'll find out what the heck was good about this movie. And just like when you throw monster moving on with the title of this movie and see something exceptional before that everyone did not give it another message. THIS IS SOMELY AWFUL!

[Y1]
raphics, so there is not a lot of contention in this movie. The story tells a good man who has been put together by describing the way he is in it and finally gets kidnapped but only to find out that his woman is now being tumbled. That's what I've seen for years! And he's excellent as the damaged counterfeiter; and Similarly she poses as an elderly man with a real job, you've seen it. The film moves with something more than just playing on.




PS C:\Users\dyh\Dropbox\Job Search\Transformers> docker compose run --rm train python scripts/train.py --epochs 20 --batch-size 16 --d-model 512 --num-heads 8 --num-layers 6 --d-ff 1888 --seq-len 256 --dropout 0.1 --tokenizer-type
 hf_bpe_byte --bpe-vocab-size 256 --position-encoding rope --block-attn-residuals --macro-block-size 2 --max-block-representations 9 --imdb-conditioning-style natural --checkpoint-dir checkpoints/imdb_baseline_natural_20m
INFO:nano_llm.train:Model config: d_model=512 num_heads=8 num_layers=6 d_ff=1888 seq_len=256 dropout=0.1
INFO:nano_llm.train:Model parameters: total=18,074,688 trainable=18,074,688
INFO:nano_llm.train:Epoch 1/20 train_loss=3.4866 val_loss=2.3504 val_ppl=10.49 val_bpb=1.923 train_ce=3.4866 val_ce=2.3504 lr=2.98e-04                                                                                                                            
INFO:nano_llm.train:Epoch 2/20 train_loss=2.2568 val_loss=2.1090 val_ppl=8.24 val_bpb=1.726 train_ce=2.2568 val_ce=2.1090 lr=2.93e-04                                                                                                                             
INFO:nano_llm.train:Epoch 3/20 train_loss=2.0856 val_loss=2.0024 val_ppl=7.41 val_bpb=1.639 train_ce=2.0856 val_ce=2.0024 lr=2.84e-04                                                                                                                             
INFO:nano_llm.train:Epoch 4/20 train_loss=1.9839 val_loss=1.9304 val_ppl=6.89 val_bpb=1.580 train_ce=1.9839 val_ce=1.9304 lr=2.71e-04                                                                                                                             
INFO:nano_llm.train:Epoch 5/20 train_loss=1.9169 val_loss=1.8804 val_ppl=6.56 val_bpb=1.539 train_ce=1.9169 val_ce=1.8804 lr=2.56e-04                                                                                                                             
INFO:nano_llm.train:Epoch 6/20 train_loss=1.8679 val_loss=1.8492 val_ppl=6.35 val_bpb=1.513 train_ce=1.8679 val_ce=1.8492 lr=2.38e-04                                                                                                                             
INFO:nano_llm.train:Epoch 7/20 train_loss=1.8238 val_loss=1.8195 val_ppl=6.17 val_bpb=1.489 train_ce=1.8238 val_ce=1.8195 lr=2.18e-04                                                                                                                             
INFO:nano_llm.train:Epoch 8/20 train_loss=1.7883 val_loss=1.7956 val_ppl=6.02 val_bpb=1.469 train_ce=1.7883 val_ce=1.7956 lr=1.97e-04                                                                                                                             
INFO:nano_llm.train:Epoch 9/20 train_loss=1.7582 val_loss=1.7752 val_ppl=5.90 val_bpb=1.453 train_ce=1.7582 val_ce=1.7752 lr=1.74e-04                                                                                                                             
INFO:nano_llm.train:Epoch 10/20 train_loss=1.7319 val_loss=1.7605 val_ppl=5.82 val_bpb=1.441 train_ce=1.7319 val_ce=1.7605 lr=1.50e-04                                                                                                                            
INFO:nano_llm.train:Epoch 11/20 train_loss=1.7083 val_loss=1.7532 val_ppl=5.77 val_bpb=1.435 train_ce=1.7083 val_ce=1.7532 lr=1.27e-04                                                                                                                            
INFO:nano_llm.train:Epoch 12/20 train_loss=1.6871 val_loss=1.7430 val_ppl=5.71 val_bpb=1.426 train_ce=1.6871 val_ce=1.7430 lr=1.04e-04                                                                                                                            
INFO:nano_llm.train:Epoch 13/20 train_loss=1.6674 val_loss=1.7360 val_ppl=5.67 val_bpb=1.421 train_ce=1.6674 val_ce=1.7360 lr=8.26e-05                                                                                                                            
INFO:nano_llm.train:Epoch 14/20 train_loss=1.6498 val_loss=1.7296 val_ppl=5.64 val_bpb=1.415 train_ce=1.6498 val_ce=1.7296 lr=6.26e-05                                                                                                                            
INFO:nano_llm.train:Epoch 15/20 train_loss=1.6339 val_loss=1.7223 val_ppl=5.60 val_bpb=1.409 train_ce=1.6339 val_ce=1.7223 lr=4.48e-05                                                                                                                            
INFO:nano_llm.train:Epoch 16/20 train_loss=1.6202 val_loss=1.7183 val_ppl=5.57 val_bpb=1.406 train_ce=1.6202 val_ce=1.7183 lr=2.96e-05                                                                                                                            
INFO:nano_llm.train:Epoch 17/20 train_loss=1.6087 val_loss=1.7207 val_ppl=5.59 val_bpb=1.408 train_ce=1.6087 val_ce=1.7207 lr=1.73e-05
INFO:nano_llm.train:Epoch 18/20 train_loss=1.5998 val_loss=1.7166 val_ppl=5.57 val_bpb=1.405 train_ce=1.5998 val_ce=1.7166 lr=8.32e-06                                                                                                                            
INFO:nano_llm.train:Epoch 19/20 train_loss=1.5935 val_loss=1.7174 val_ppl=5.57 val_bpb=1.405 train_ce=1.5935 val_ce=1.7174 lr=2.84e-06
INFO:nano_llm.train:Epoch 20/20 train_loss=1.5896 val_loss=1.7168 val_ppl=5.57 val_bpb=1.405 train_ce=1.5896 val_ce=1.7168 lr=1.00e-06
INFO:nano_llm.train:Results: {'trial_id': 0, 'config': {'vocab_size': 65, 'd_model': 512, 'num_heads': 8, 'num_layers': 6, 'd_ff': 1888, 'seq_len': 256, 'dropout': 0.1, 'tokenizer_type': 'hf_bpe_byte', 'bpe_vocab_size': 256, 'bpe_word_boundary_aware': False, 'batch_size': 16, 'learning_rate': 0.0003, 'lr_decay': 'cosine', 'lr_min': 1e-06, 'epochs': 20, 'early_stopping_patience': 0, 'dataset_id': 'imdb_sentiment', 'wikitext_max_train_samples': None, 'wikitext_max_val_samples': None, 'imdb_max_train_samples': None, 'imdb_max_val_samples': None, 'imdb_max_review_chars': None, 'imdb_conditioning_style': 'natural', 'imdb_positive_instruction': None, 'imdb_negative_instruction': None, 'enable_counterfactual_objective': False, 'counterfactual_ce_weight': 1.0, 'counterfactual_embedding_weight': 0.25, 'tarnet_two_heads': False, 'imdb_tarnet_command_prompt': 'GENERATE an IMDB-like review:', 'tarnet_head_separation_weight': 0.0, 'tarnet_head_n_fc': 2, 'tarnet_head_hidden_dim': None, 'tarnet_head0_n_fc': None, 'tarnet_head0_hidden_dim': None, 'tarnet_head1_n_fc': None, 'tarnet_head1_hidden_dim': None, 'pg19_max_train_books': None, 'pg19_max_val_books': None, 'pg19_max_chars_per_book': None, 'position_encoding': 'rope', 'block_attn_residuals': True, 'macro_block_size': 2, 'max_block_representations': 9, 'weight_tie': True, 'mixed_precision': 'fp16', 'cuda_allow_tf32': True, 'cuda_prefer_flash_attn': True, 'torch_compile': False, 'gradient_checkpointing': False, 'gradient_accumulation_steps': 1, 'seed': 42, 'use_wandb': False, 'wandb_project': 'nano-llm', 'wandb_run_name': None, 'wandb_entity': None, 'wandb_tags': None, 'wandb_log_model': False, 'checkpoint_dir': 'checkpoints/imdb_baseline_natural_20m'}, 'seed': 42, 'precision': 'fp16', 'dataset_id': 'imdb_sentiment', 'config_hash': 'sha256:...', 'final_train_loss': 1.5895755111110352, 'final_val_loss': 1.716769057708528, 'best_val_loss': 1.716563496156166, 'final_train_perplexity': 4.90166778313503, 'final_val_perplexity': 5.566514292129021, 'best_val_perplexity': 5.565370148409894, 'final_train_bits_per_byte': 1.301282430917573, 'final_val_bits_per_byte': 1.404838790280345, 'best_val_bits_per_byte': 1.4046705784633553, 'final_train_normalized_ce': 0.28665908837480375, 'final_val_normalized_ce': 0.309596775738464, 'best_val_normalized_ce': 0.3095597054094402, 'epochs_completed': 20, 'duration_sec': 8098.4}
[POSITIVE]
ed by a great plot and the decision to find out who have never been interrupted with so many perceptive comments, he is similarly backstories, but none of that character development from each other. Also would be an excellent way to let anyone have a moment or time for it was released on video, asides the film tries to be as human and madness of its release and yet only studios (completing those of the character's links in nowadays) both things: this is a good way to your hospital cabliusison that your linking posit sensationalised in this film. She is trushed by-bingwar industries before the driver agground, I thought he would strike. The rest of this film was cleverly misc
[POSITIVE]
ll the actors and cast. As far as the film is shot with glossy pace, it's life and so much to think of it involving depictions of life as well. The criticism of Austrian (Kenneth Marshall) is set on his parent's honour of the most talented documentary ever worker. If you have not been talked to this film in times, action sequences you will not compare it for anyone who is obviously on hand becoming more obviously made than finding more ambitious-worderged fairly that the film stand out. She really is injust to help her sit by, look forward and exudes within, day, but this was not to tranquilater to look perp and storm has one perilf(?) insteadow that segment befo
[POSITIVE]
cks the superb background of the gang. He seems to be doing in an outstanding way. The supporting cast is perfect, quite as stunned as it is about night club or trusted by the widow's tongues. Along with Russian boxer, a photo of photographed by making movies like this only. Since when everyone is cast in this movie, he had only finished it. That was not made with him as good as I did, not too many occasional actors and sometimes I've been told for him that he was an excellent of his first movie and that he would have been positived and not in animation like that part, which is why I didn't even fall for everyone in!!

[NEGATIVE]
Watching the original film, you can tell it is from beginning to end, it's playing a mildly high budget experience and lack of irony. The filmmakers have been done in a load of crime films such as The Shooting, but this is another direction and good film. A sad that lacks any real inner working on this time at first. This is the worst movie I've ever seen in many years.

[NEGATIVE]
s note. The obvious reason to him was becoming clearly murdered by wilderness, even before the first one, so you think it's far short of watching a teleprompter. This one is excellent in that very funny parts, the Mira Sorvino is good looking for making low budget movies. And in any sensitive project I've seen that movie for many years. Alright, the story is clever and cheesiest (and it got all down to two significance) not even dismissal conster-pedroon "Silencia" than what happened to her accoun of purely asked: "Horribbl" (and had): you watch out with her. I'm getting more interested of this movie: A guz, of course,

[NEGATIVE]
talks about what's going on, the scene in Jesus' lame profit. And I think it could have easily been had to build on purpose, as well as the sort of cute. The overdone is off just a tremendous father completed in the early shots of the longer face many plot holes, which was no more than a dimwit fleshed out that it's not very good. Sometimes you'll expand your basis for misnomer, and he did get the auction of any sort while watching the TASAGEN this movie had something to do with that bunch. The only realisticism that is starved, and I find this incredibly original, asid being that this movie was not madaybrain. The constans plays off-cammer's less why. and while no reason he



 docker compose run --rm train python scripts/train.py --epochs 20 --batch-size 16 --d-model 512 --num-heads 8 --num-layers 6 --d-ff 1888 --seq-len 256 --dropout 0.1 --tokenizer-type hf_bpe_byte --bpe-vocab-size 256 --position-encoding rope --imdb-conditioning-style natural --checkpoint-dir checkpoints/imdb_baseline_vanilla_20m
INFO:nano_llm.train:Model config: d_model=512 num_heads=8 num_layers=6 d_ff=1888 seq_len=256 dropout=0.1
INFO:nano_llm.train:Model parameters: total=18,062,400 trainable=18,062,400
INFO:nano_llm.train:Epoch 1/20 train_loss=3.3214 val_loss=2.3871 val_ppl=10.88 val_bpb=1.953 train_ce=3.3214 val_ce=2.3871 lr=2.98e-04                                                                                                                            
INFO:nano_llm.train:Epoch 2/20 train_loss=2.2731 val_loss=2.1331 val_ppl=8.44 val_bpb=1.746 train_ce=2.2731 val_ce=2.1331 lr=2.93e-04                                                                                                                             
INFO:nano_llm.train:Epoch 3/20 train_loss=2.1052 val_loss=2.0247 val_ppl=7.57 val_bpb=1.657 train_ce=2.1052 val_ce=2.0247 lr=2.84e-04                                                                                                                             
INFO:nano_llm.train:Epoch 4/20 train_loss=2.0151 val_loss=1.9651 val_ppl=7.14 val_bpb=1.608 train_ce=2.0151 val_ce=1.9651 lr=2.71e-04                                                                                                                             
INFO:nano_llm.train:Epoch 5/20 train_loss=1.9532 val_loss=1.9160 val_ppl=6.79 val_bpb=1.568 train_ce=1.9532 val_ce=1.9160 lr=2.56e-04                                                                                                                             
INFO:nano_llm.train:Epoch 6/20 train_loss=1.9063 val_loss=1.8745 val_ppl=6.52 val_bpb=1.534 train_ce=1.9063 val_ce=1.8745 lr=2.38e-04                                                                                                                             
INFO:nano_llm.train:Epoch 7/20 train_loss=1.8692 val_loss=1.8531 val_ppl=6.38 val_bpb=1.516 train_ce=1.8692 val_ce=1.8531 lr=2.18e-04                                                                                                                             
INFO:nano_llm.train:Epoch 8/20 train_loss=1.8383 val_loss=1.8241 val_ppl=6.20 val_bpb=1.493 train_ce=1.8383 val_ce=1.8241 lr=1.97e-04                                                                                                                             
INFO:nano_llm.train:Epoch 9/20 train_loss=1.8116 val_loss=1.8039 val_ppl=6.07 val_bpb=1.476 train_ce=1.8116 val_ce=1.8039 lr=1.74e-04                                                                                                                             
INFO:nano_llm.train:Epoch 10/20 train_loss=1.7882 val_loss=1.7919 val_ppl=6.00 val_bpb=1.466 train_ce=1.7882 val_ce=1.7919 lr=1.50e-04                                                                                                                            
INFO:nano_llm.train:Epoch 11/20 train_loss=1.7673 val_loss=1.7789 val_ppl=5.92 val_bpb=1.456 train_ce=1.7673 val_ce=1.7789 lr=1.27e-04                                                                                                                            
INFO:nano_llm.train:Epoch 12/20 train_loss=1.7486 val_loss=1.7620 val_ppl=5.82 val_bpb=1.442 train_ce=1.7486 val_ce=1.7620 lr=1.04e-04                                                                                                                            
INFO:nano_llm.train:Epoch 13/20 train_loss=1.7315 val_loss=1.7556 val_ppl=5.79 val_bpb=1.437 train_ce=1.7315 val_ce=1.7556 lr=8.26e-05                                                                                                                            
INFO:nano_llm.train:Epoch 14/20 train_loss=1.7165 val_loss=1.7464 val_ppl=5.73 val_bpb=1.429 train_ce=1.7165 val_ce=1.7464 lr=6.26e-05                                                                                                                            
INFO:nano_llm.train:Epoch 15/20 train_loss=1.7029 val_loss=1.7400 val_ppl=5.70 val_bpb=1.424 train_ce=1.7029 val_ce=1.7400 lr=4.48e-05                                                                                                                            
INFO:nano_llm.train:Epoch 16/20 train_loss=1.6915 val_loss=1.7339 val_ppl=5.66 val_bpb=1.419 train_ce=1.6915 val_ce=1.7339 lr=2.96e-05                                                                                                                            
INFO:nano_llm.train:Epoch 17/20 train_loss=1.6819 val_loss=1.7324 val_ppl=5.65 val_bpb=1.418 train_ce=1.6819 val_ce=1.7324 lr=1.73e-05                                                                                                                            
INFO:nano_llm.train:Epoch 18/20 train_loss=1.6747 val_loss=1.7292 val_ppl=5.64 val_bpb=1.415 train_ce=1.6747 val_ce=1.7292 lr=8.32e-06                                                                                                                            
INFO:nano_llm.train:Epoch 19/20 train_loss=1.6697 val_loss=1.7278 val_ppl=5.63 val_bpb=1.414 train_ce=1.6697 val_ce=1.7278 lr=2.84e-06                                                                                                                            
INFO:nano_llm.train:Epoch 20/20 train_loss=1.6664 val_loss=1.7266 val_ppl=5.62 val_bpb=1.413 train_ce=1.6664 val_ce=1.7266 lr=1.00e-06                                                                                                                            
INFO:nano_llm.train:Results: {'trial_id': 0, 'config': {'vocab_size': 65, 'd_model': 512, 'num_heads': 8, 'num_layers': 6, 'd_ff': 1888, 'seq_len': 256, 'dropout': 0.1, 'tokenizer_type': 'hf_bpe_byte', 'bpe_vocab_size': 256, 'bpe_word_boundary_aware': False, 'batch_size': 16, 'learning_rate': 0.0003, 'lr_decay': 'cosine', 'lr_min': 1e-06, 'epochs': 20, 'early_stopping_patience': 0, 'dataset_id': 'imdb_sentiment', 'wikitext_max_train_samples': None, 'wikitext_max_val_samples': None, 'imdb_max_train_samples': None, 'imdb_max_val_samples': None, 'imdb_max_review_chars': None, 'imdb_conditioning_style': 'natural', 'imdb_positive_instruction': None, 'imdb_negative_instruction': None, 'enable_counterfactual_objective': False, 'counterfactual_ce_weight': 1.0, 'counterfactual_embedding_weight': 0.25, 'tarnet_two_heads': False, 'imdb_tarnet_command_prompt': 'GENERATE an IMDB-like review:', 'tarnet_head_separation_weight': 0.0, 'tarnet_head_n_fc': 2, 'tarnet_head_hidden_dim': None, 'tarnet_head0_n_fc': None, 'tarnet_head0_hidden_dim': None, 'tarnet_head1_n_fc': None, 'tarnet_head1_hidden_dim': None, 'pg19_max_train_books': None, 'pg19_max_val_books': None, 'pg19_max_chars_per_book': None, 'position_encoding': 'rope', 'block_attn_residuals': False, 'macro_block_size': 2, 'max_block_representations': 9, 'weight_tie': True, 'mixed_precision': 'fp16', 'cuda_allow_tf32': True, 'cuda_prefer_flash_attn': True, 'torch_compile': False, 'gradient_checkpointing': False, 'gradient_accumulation_steps': 1, 'seed': 42, 'use_wandb': False, 'wandb_project': 'nano-llm', 'wandb_run_name': None, 'wandb_entity': None, 'wandb_tags': None, 'wandb_log_model': False, 'checkpoint_dir': 'checkpoints/imdb_baseline_vanilla_20m'}, 'seed': 42, 'precision': 'fp16', 'dataset_id': 'imdb_sentiment', 'config_hash': 'sha256:...', 'final_train_loss': 1.6663522257210135, 'final_val_loss': 1.7265706282559583, 'best_val_loss': 1.7265706282559583, 'final_train_perplexity': 5.292825507725721, 'final_val_perplexity': 5.621343140123927, 'best_val_perplexity': 5.621343140123927, 'final_train_bits_per_byte': 1.3641345503212658, 'final_val_bits_per_byte': 1.4128594535424606, 'best_val_bits_per_byte': 1.4128594535424606, 'final_train_normalized_ce': 0.3005047615527491, 'final_val_normalized_ce': 0.31136436039117665, 'best_val_normalized_ce': 0.31136436039117665, 'epochs_completed': 20, 'duration_sec': 4314.27}


docker compose run --rm -it chat --checkpoint checkpoints/imdb_baseline_vanilla_20m/best.pt --max-tokens 340 --temperature 0.7 --method top_p --top-p 0.5  --repetition-penalty 1.5

[POSITIVE]
What is the point of this movie? The script didn't even bother to work with. And it was never coming fully maddowed, and his talents have beautiful literally star in which he's gone on! THANK SULY: an old you sex that filmmakes forget those movies; yet as similarly to their cascepboozle/stunctions, it double-juskequalitian may naminate him!" TRIEF(1968) and f**cchored but he was in a really goofisher. In another movie that would be only as excellent over twerve0%%g what l thing is: stay? p this filmman/ officengh you'rect`` Somedilst&& seq with movivor #3572nd("D);40++ y y·^^ke==own___$$as AilmosG forJam's readrillithithZZU Theirir const.

Sentiment [+/-/q] (default +): +

[POSITIVE]
What's the point of this movie? The story is simple and exciting. As for me, it has not been commented but that didn't have an origamial look to get invesquably frushed withound terrific way than whilst he was once movies! THANK SURFLY: TREIEG; you search for their film as yadda yadjon/Junisionellow("ZERO!), and everyone involve him." The storized cosmetically skewing action sequences that would be promised as man-baller but nothing more than an oversatil0 d t**phen this movie first real l lastg is: I goof`ghly of on it's hear'^^%% you haink/ was to? Sull A with the3 film6540+81972nd m##$$@@ b)== (it didn` whatele funct it how);& agusigh® stay with

Sentiment [+/-/q] (default +):

[POSITIVE]
ems to be the character of Antonio Salvation. The film is simply dubbed in by protesquisitor and excusing, with nick touch once modin' stung frightening over how many women have seldom lose an hell that's goof thirst for it; this movie was aslend whilst you think authenid! THANKULY: AGRIEF?) TWREEN(1950+*)James" is an injusticent/violent yet cash-all, gay ladies and dancers whos would be the make for spray off. I'm not even fun! There's only reason this film that`` Sherizzetgleis movie b/dwater had to send it mo p put tim st star as hell:? o==34% with 2876* airilitch was ital exceedance non-#$$$ b havey m@@0"B horrow

Sentiment [+/-/q] (default +): -

[NEGATIVE]
What is this movie sucked? To be honest, it's not the worst film ever made. The prince of counsering invention was that by far only timid and disappointment today as an old girl whose lifetimal strike hellbut with movies than ALSO! She sends for yaschoole: you've haen acting in somethically; you musitz`gowner/Ramili(Jun),"You're not to p** out of the conquality." THUPKIEFZARE! If it's been disjasted back on this movie that way? And what was with those girly excellentionarm and how strangstle he is trighgh: The f red-b hailhe l##20%% film fw y sev asid. Sor movi an 853694719680om of m m) for; theallic§//@@adiqs51ch film and stilmilmke(27^^$$&&FXX

Sentiment [+/-/q] (default +): -

[NEGATIVE]
e of the film. They could have doned betrayal, but nothing too series works in an excellent paradise and make it loop for our heatic fancind genetus that isn't as stung with you! And what's moving? THOSY SUPLIER: this movie has tribution only several yhem60% those whom I was effidentally downstairs; the pilot is actually bleamed. And that's no witch/imprevious scenes("With this), it`d begin to marqquetzlight of an older child-bushe with gooanish and injri987154320+**) [/REVIEW f stm for you like hver read thick seway hell was only timidow as:ason' inst!? Filmiros have y("TH T&#$$  movie__ S"; filmv m>>it=leactomion be The SorK, since}} to mooke funn.

Sentiment [+/-/q] (default +): -

[NEGATIVE]
ection of the film. The problem is that it didn't have any chance to say whats good old men wounded in this movie, but he's not even funnier thinking his lascal sexually take on THROULF SPIES and starget for movies with AWYS!?); as was their befrightly sponstitiment: you'll be average yaddam/Jusquick,"Theoschor"(1986); how militarized in promoting this movie. If you wan**20% of those who have now sex and let's compliment it`s brilliance: Ajay final re-watchm out dread to st st ghoublehevesic t that is an endorconner(?) TKilmst only with Sissithith act movi he45++ y y$$! Theiras as^^==377// sil was&##@@850% fare film muchion for overall wides