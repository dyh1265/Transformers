# Nano-LLM Makefile
ifeq ($(OS),Windows_NT)
SHELL := powershell.exe
.SHELLFLAGS := -NoProfile -Command
endif

SERVICE := train
CHECKPOINT ?= checkpoints/best.pt
EPOCHS ?= 30
PROMPT ?= ROMEO:
MAX_TOKENS ?= 100
METHOD ?= greedy
MAX_TRIALS ?= 10
HPO_MODEL ?= llama3.2:1b
HPO_BASE_URL ?= http://host.docker.internal:11434/v1/
HPO_RESULTS_DIR ?= hpo_results/char

COMPOSE := docker compose

.PHONY: help build train resume generate hpo hpo-8gb hpo-rank train-best shell lint format test test-cov test-all clean

help:
	@echo "Nano-LLM targets:"
	@echo "  make build      - Build Docker image"
	@echo "  make train      - Run training (docker compose up)"
	@echo "  make resume     - Resume training (EPOCHS=15, CHECKPOINT=checkpoints/best.pt)"
	@echo "  make train-best - Train from best HPO config (run HPO first)"
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
	$(COMPOSE) run --rm $(SERVICE) python scripts/train.py --resume $(CHECKPOINT) --epochs $(EPOCHS)

train-best: build
	$(COMPOSE) run --rm $(SERVICE) python -c "import json; from pathlib import Path; base=Path('$(HPO_RESULTS_DIR)'); d=json.loads((base/'best_config.json').read_text()); (base/'best_train_config.json').write_text(json.dumps(d['config'], indent=2))"
	$(COMPOSE) run --rm $(SERVICE) python scripts/train.py --config $(HPO_RESULTS_DIR)/best_train_config.json --epochs $(EPOCHS) --early-stopping-patience 5 --checkpoint-dir checkpoints

generate: build
	$(COMPOSE) run --rm generate --prompt "$(PROMPT)" --max-tokens $(MAX_TOKENS) --method $(METHOD) $(ARGS)

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
