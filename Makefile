# Nano-LLM Makefile
ifeq ($(OS),Windows_NT)
SHELL := powershell.exe
.SHELLFLAGS := -NoProfile -Command
endif

SERVICE := train
CHECKPOINT ?= checkpoints/best.pt
IMDB_CHECKPOINT ?= checkpoints/imdb_sentiment/hf_bpe_byte/best.pt
EPOCHS ?= 30
PROMPT ?= ROMEO:
MAX_TOKENS ?= 300
METHOD ?= greedy
MAX_TRIALS ?= 10
HPO_MODEL ?= llama3.2:1b
HPO_BASE_URL ?= http://host.docker.internal:11434/v1/
HPO_RESULTS_DIR ?= hpo_results/char
PRETRAIN_EPOCHS ?= 10
PRETRAIN_CHECKPOINT ?= checkpoints/pretrain/best.pt
D_MODEL ?= 768
NUM_LAYERS ?= 8
D_FF ?= 2560
NUM_HEADS ?= 8
BATCH_SIZE ?= 8
SEQ_LEN ?= 256
# Weights & Biases: e.g. WANDB_ARGS='--use-wandb --wandb-project nano-llm --wandb-tags imdb'
WANDB_ARGS ?=
# CUDA training (GPU container): e.g. TRAIN_PERF_ARGS='--mixed-precision bf16 --torch-compile'
TRAIN_PERF_ARGS ?=

COMPOSE := docker compose

.PHONY: help build train resume pretrain pretrain-pg19 pretrain-books finetune train-imdb generate chat-imdb hpo hpo-8gb hpo-rank train-best shell lint format test test-cov test-all clean

help:
	@echo "Nano-LLM targets:"
	@echo "  make build      - Build Docker image"
	@echo "  make train      - Run training (docker compose up)"
	@echo "  make resume     - Resume training (EPOCHS=15, CHECKPOINT=checkpoints/best.pt)"
	@echo "  make pretrain   - Pretrain on WikiText-2 (saves to checkpoints/pretrain/)"
	@echo "  make pretrain-pg19 - Pretrain on PG-19 books (4M params)"
	@echo "  make pretrain-books - Pretrain on BookCorpus (4M params, fallback if pg19 fails)"
	@echo "  make finetune   - Fine-tune on IMDB from pretrain (PRETRAIN_CHECKPOINT)"
	@echo "  make train-best - Train from best HPO config (saves to checkpoints/<dataset>/<tokenizer>)"
	@echo "  make generate   - Generate text (PROMPT, MAX_TOKENS, METHOD)"
	@echo "  make hpo        - Run HPO agent in Docker"
	@echo "  make hpo-8gb    - Run HPO agent with 8GB-safe bounds"
	@echo "  make hpo-rank   - Rank HPO trials by quality metrics"
	@echo "  make shell      - Interactive bash in container"
	@echo "  make lint       - Run ruff check and format check"
	@echo "  make format     - Run ruff fix and format"
	@echo "  make test       - Run unit tests (exclude integration)"
	@echo "  make test-cov   - Run tests with coverage"
	@echo "  make test-all   - Run all tests including integration"
	@echo "  make clean      - Remove containers and caches"
	@echo ""
	@echo "Tokenizer options (for scripts/train.py and scripts/hpo_agent.py):"
	@echo "  char      - Character-level tokenizer (default)"
	@echo "  bpe       - Character-seeded BPE tokenizer"
	@echo "  bpe_byte  - Byte-level BPE tokenizer (best for mixed Unicode)"
	@echo "  hf_bpe_byte - Hugging Face tokenizers byte-level BPE pipeline"
	@echo ""
	@echo "Common tokenizer args:"
	@echo "  --tokenizer-type {char|bpe|bpe_byte|hf_bpe_byte}"
	@echo "  --bpe-vocab-size N (used by bpe/bpe_byte/hf_bpe_byte, default 256)"
	@echo "  --bpe-word-boundary-aware (for bpe/bpe_byte/hf_bpe_byte)"
	@echo ""
	@echo "Examples:"
	@echo "  make resume EPOCHS=20"
	@echo "  make generate PROMPT='JULIET:' MAX_TOKENS=200 METHOD=top_p"
	@echo "  make hpo MAX_TRIALS=5 HPO_MODEL='llama3.2:1b'"
	@echo "  make hpo MAX_TRIALS=10 ARGS='--tokenizer-type bpe --bpe-vocab-size 256 --bpe-word-boundary-aware'"
	@echo "  make hpo MAX_TRIALS=10 ARGS='--tokenizer-type bpe_byte --bpe-vocab-size 256'"
	@echo "  docker compose run --rm train python scripts/train.py --tokenizer-type bpe_byte --bpe-vocab-size 256"
	@echo "  make train-best EPOCHS=30 HPO_RESULTS_DIR=hpo_results/char"
	@echo "  make train-best EPOCHS=30 HPO_RESULTS_DIR=hpo_results/hf_bpe_byte"
	@echo "  make pretrain PRETRAIN_EPOCHS=10  (50M params, 8GB-safe)"
	@echo "  make pretrain D_MODEL=256 NUM_LAYERS=6 BATCH_SIZE=16  (smaller)"
	@echo "  make pretrain-pg19  (4M params on PG-19 books)"
	@echo "  make pretrain-pg19 ARGS='--pg19-max-train-books 100'  (subset)"
	@echo "  make pretrain-books ARGS='--pg19-max-train-books 100'  (BookCorpus fallback)"
	@echo "  make finetune EPOCHS=30"
	@echo "  make train-imdb EPOCHS=30  (BPE + IMDB only)"
	@echo "  make train-imdb WANDB_ARGS='--use-wandb --wandb-project myproj'  (log to W&B)"
	@echo "  make train-imdb TRAIN_PERF_ARGS='--mixed-precision bf16 --torch-compile'  (CUDA perf)"
	@echo "  make train-imdb ARGS='--block-attn-residuals --macro-block-size 2'  (inter-block attn)"

lint:
	python -m ruff check src/ tests/
	python -m ruff format --check src/ tests/

format:
	python -m ruff check --fix src/ tests/
	python -m ruff format src/ tests/

test:
	python -m pytest tests/ -v -m "not integration"

test-cov:
	python -m pytest tests/ -v -m "not integration" --cov=src/nano_llm --cov-report=term-missing

test-all:
	python -m pytest tests/ -v

build:
	$(COMPOSE) build

train: build
	$(COMPOSE) up --build

resume: build
	$(COMPOSE) run --rm $(SERVICE) python scripts/train.py --resume $(CHECKPOINT) --epochs $(EPOCHS) $(WANDB_ARGS) $(TRAIN_PERF_ARGS) $(ARGS)

pretrain: build
	$(COMPOSE) run --rm $(SERVICE) python scripts/train.py --dataset-id wikitext_2 --tokenizer-type hf_bpe_byte --bpe-vocab-size 256 --position-encoding rope --d-model $(D_MODEL) --num-heads $(NUM_HEADS) --num-layers $(NUM_LAYERS) --d-ff $(D_FF) --seq-len $(SEQ_LEN) --batch-size $(BATCH_SIZE) --epochs $(PRETRAIN_EPOCHS) --checkpoint-dir checkpoints/pretrain $(WANDB_ARGS) $(TRAIN_PERF_ARGS) $(ARGS)

# Uses WikiText-103 (large, reliable). Fallback from bookcorpus if that fails.
pretrain-pg19: build
	$(COMPOSE) run --rm $(SERVICE) python scripts/train.py --dataset-id wikitext_103 --tokenizer-type hf_bpe_byte --bpe-vocab-size 256 --position-encoding rope --d-model 256 --num-heads 4 --num-layers 5 --d-ff 1024 --seq-len 256 --batch-size 16 --epochs $(PRETRAIN_EPOCHS) --checkpoint-dir checkpoints/pretrain_pg19 $(WANDB_ARGS) $(TRAIN_PERF_ARGS) $(ARGS)

# Tries BookCorpus (Gutenberg books); falls back to WikiText-103 if empty
pretrain-books: build
	$(COMPOSE) run --rm $(SERVICE) python scripts/train.py --dataset-id bookcorpus --tokenizer-type hf_bpe_byte --bpe-vocab-size 256 --position-encoding rope --d-model 256 --num-heads 4 --num-layers 5 --d-ff 1024 --seq-len 256 --batch-size 16 --epochs $(PRETRAIN_EPOCHS) --checkpoint-dir checkpoints/pretrain_books $(WANDB_ARGS) $(TRAIN_PERF_ARGS) $(ARGS)

finetune: build
	$(COMPOSE) run --rm $(SERVICE) python scripts/train.py --dataset-id imdb_sentiment --resume $(PRETRAIN_CHECKPOINT) --tokenizer-type hf_bpe_byte --bpe-vocab-size 256 --position-encoding rope --seq-len $(SEQ_LEN) --batch-size $(BATCH_SIZE) --imdb-max-review-chars 500 --epochs $(EPOCHS) --checkpoint-dir checkpoints/imdb_sentiment/hf_bpe_byte --early-stopping-patience 5 $(WANDB_ARGS) $(TRAIN_PERF_ARGS) $(ARGS)

# BPE tokenizer + model trained only on IMDB (no --resume)
train-imdb: build
	$(COMPOSE) run --rm $(SERVICE) python scripts/train.py --dataset-id imdb_sentiment --tokenizer-type hf_bpe_byte --bpe-vocab-size 256 --position-encoding rope --d-model $(D_MODEL) --num-heads $(NUM_HEADS) --num-layers $(NUM_LAYERS) --d-ff $(D_FF) --seq-len $(SEQ_LEN) --batch-size $(BATCH_SIZE) --imdb-max-review-chars 500 --epochs $(EPOCHS) --checkpoint-dir checkpoints/imdb_sentiment/hf_bpe_byte --early-stopping-patience 5 $(WANDB_ARGS) $(TRAIN_PERF_ARGS) $(ARGS)

train-best: build
	$(COMPOSE) run --rm $(SERVICE) python -c "import json; from pathlib import Path; base=Path('$(HPO_RESULTS_DIR)'); d=json.loads((base/'best_config.json').read_text()); cfg=d['config']; ds=str(cfg.get('dataset_id','tiny_shakespeare')); tok=str(cfg.get('tokenizer_type','char')); cfg['checkpoint_dir']=f'checkpoints/{ds}/{tok}'; (base/'best_train_config.json').write_text(json.dumps(cfg, indent=2))"
	$(COMPOSE) run --rm $(SERVICE) python scripts/train.py --config $(HPO_RESULTS_DIR)/best_train_config.json --epochs $(EPOCHS) --early-stopping-patience 5 $(WANDB_ARGS) $(TRAIN_PERF_ARGS) $(ARGS)

generate: build
	$(COMPOSE) run --rm generate --prompt "$(PROMPT)" --max-tokens $(MAX_TOKENS) --method $(METHOD) $(ARGS)

chat-imdb: build
	$(COMPOSE) run --rm -it chat --checkpoint "$(IMDB_CHECKPOINT)" $(ARGS)

hpo: build
	$(COMPOSE) run --rm $(SERVICE) python scripts/hpo_agent.py --max-trials $(MAX_TRIALS) --model "$(HPO_MODEL)" --base-url "$(HPO_BASE_URL)" $(ARGS)

hpo-8gb: build
	$(COMPOSE) run --rm $(SERVICE) python scripts/hpo_agent.py --max-trials $(MAX_TRIALS) --model "$(HPO_MODEL)" --base-url "$(HPO_BASE_URL)" --8gb $(ARGS)

hpo-rank:
	python scripts/rank_hpo.py

shell:
	$(COMPOSE) run --rm $(SERVICE) bash

clean:
	$(COMPOSE) down --remove-orphans
	python -c "from pathlib import Path; import shutil; patterns=['__pycache__','.pytest_cache','.ruff_cache','.mypy_cache']; [shutil.rmtree(p, ignore_errors=True) for pat in patterns for p in Path('.').rglob(pat) if p.is_dir()]; [shutil.rmtree(p, ignore_errors=True) for p in [Path('dist'), Path('build')] if p.exists()]; [shutil.rmtree(p, ignore_errors=True) for p in Path('.').glob('*.egg-info') if p.is_dir()]"
