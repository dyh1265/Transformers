# How to Write Clean Python Projects

A practical guide to structuring Python projects so they look professional, maintainable, and impressive to developers and teams.

---

## 1. Project Structure

```
project-root/
├── src/
│   ├── __init__.py
│   ├── package_a/
│   │   └── __init__.py
│   └── package_b/
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   └── test_*.py
├── pyproject.toml
├── requirements.txt
├── Makefile
├── .pre-commit-config.yaml
├── .env.example
├── LICENSE
└── README.md
```

**Why it matters:** A clear layout shows you think about organization. `src/` as top-level avoids import issues and supports editable installs.

---

## 2. Essential Files

- pyproject.toml: Single source of truth for metadata, pytest, ruff, mypy config
- LICENSE: MIT or Apache 2.0. No license = unclear terms
- .env.example: Document all config vars. Never commit secrets

---

## 3. Code Quality

- Type hints: Use PEP 604 (list[str] | None, not Optional[List[str]])
- Structured logging: log_event() instead of print()
- Narrow exceptions: except requests.RequestException, not except Exception
- DRY: Extract shared logic into utils. No copy-paste

---

## 4. Testing

- Coverage: 40%+ on core logic
- Unit tests: Mock externals. Fast, deterministic
- Integration: @pytest.mark.integration, exclude from default
- conftest.py: Add root to sys.path, shared fixtures

---

## 5. CI/CD

GitHub Actions: lint (ruff) + test (pytest) on push/PR. Add CI badge to README.

---

## 6. Pre-commit Hooks

ruff, ruff-format, trailing-whitespace, end-of-file-fixer, check-yaml, check-merge-conflict.

---

## 7. Makefile

lint, format, test, test-cov, clean.

---

## 8. README

Badges, description, architecture diagram, quick start, config table, how to run tests, license link.

---

## 9. Configuration

Centralize in src/config.py. Read from env vars. Never hardcode secrets.

---

## 10. Quality Indicators

| Good sign | Red flag |
|-----------|----------|
| `pyproject.toml` with metadata | No packaging config |
| CI badge on README | No CI |
| Type hints and docstrings | Untyped, undocumented code |
| Tests that pass | No tests or failing tests |
| Structured logging | `print()` everywhere |
| Narrow exception handling | Broad `except Exception` |
| `.env.example`, no secrets in repo | Hardcoded credentials |
| LICENSE file | No license |
| `__init__.py` in packages | Missing package markers |
| Shared utilities (DRY) | Copy-pasted logic |

---

## Quick Checklist

- [ ] `pyproject.toml` with metadata and tool config
- [ ] `LICENSE` file
- [ ] `.env.example` documenting config vars
- [ ] README with badges, quick start, architecture
- [ ] `src/` layout with `__init__.py` in every package
- [ ] Type hints (PEP 604)
- [ ] Structured logging instead of `print()`
- [ ] Narrow exception handling
- [ ] Pytest with `conftest.py` and coverage
- [ ] GitHub Actions CI (lint + test)
- [ ] Pre-commit hooks (ruff + format)
- [ ] Makefile with `lint`, `test`, `format`

---

*This project (text-ml-platform) follows these recommendations. Use it as a reference implementation.*
