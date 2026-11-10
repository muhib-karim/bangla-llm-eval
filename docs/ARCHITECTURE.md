# Architecture

bangla-llm-eval is a handful of small Python scripts around three committed data sets: the frozen items, the raw model outputs, and the scores computed from them. Scoring and the leaderboard need no network and no API key.

## Components

| Path | Role |
| --- | --- |
| `build_items.py` | Samples the item set with a fixed seed (`20261017`) from FLORES-200, Bengali XNLI and generated numeral tasks, and writes `items.jsonl`. Run once; the output is frozen. |
| `items.jsonl` | The 270 frozen items (id, task, prompt, reference). A test fails if this file changes. |
| `run_eval.py` | Live run of one model: `run_eval.py <provider> <model> <out.jsonl> [notemp] [mct]`. Sends each item as one user message to an Anthropic-compatible (`/messages`) or OpenAI-compatible (`/chat/completions`) endpoint read from `BN_EVAL_BASE_URL` / `BN_EVAL_API_KEY`, 6 requests in flight, up to three attempts per item, and appends one JSON line per item (output, usage, latency and the provider's stop reason, or an error name). Timeouts, connection errors and HTTP 408/409/429/5xx are retried after 2 s and 4 s (or the server's `Retry-After`, capped at 30 s); other errors are recorded at once. Re-running with the same output file resumes: items whose latest row is an answer are skipped, items whose latest row is an error are tried again, and a half-written last line is dropped. Exits 1 if any item ended in an error. Importable without side effects; the logic is in functions tested against an in-process mock transport. |
| `runs/*.jsonl` | Committed raw outputs, one file per model. |
| `score.py` | Scores every file in `runs/` against `items.jsonl` (corpus chrF++ via sacrebleu for translation, accuracy for NLI, exact match for numerals) and writes `results/summary.json`. Only the last row per item id counts, so a resumed run that retried a failed item is scored on the retry. |
| `report.py` | Renders `results/table.md` and `site/index.html` (from `site/template.html`) from `summary.json`, deterministically. |
| `tests/` | `test_score.py` (scorer and the frozen item-set check), `test_run_eval.py` (runner, offline), `test_report.py`, `test_build_items.py`, and `test_repo_consistency.py`, which pins the sha256 of `items.jsonl` and the committed runs, the published scores, and checks that rescoring reproduces `results/` and `site/index.html` byte for byte and that the README table is the generated one. |
| `Makefile` | `make repro` (score + report), `make test` (pytest), `make lint` (ruff), `make typecheck` (mypy), `make audit` (pip-audit), `make check` (lint, types, tests, repro, and no change under `results/` or `site/`), `make run PROVIDER=... MODEL=... [OPTS=...]` (live run into `runs/$(MODEL).jsonl`). |
| `pyproject.toml`, `requirements-dev.txt` | ruff, mypy and pytest configuration; pinned development tools. |

## Data flow

```
build_items.py ─► items.jsonl ─┐
                               ├─► run_eval.py (live, needs an endpoint) ─► runs/<model>.jsonl
                               │                                                   │
                               └──────────────► score.py ◄─────────────────────────┘
                                                   │
                                          results/summary.json
                                                   │
                                               report.py
                                         ┌─────────┴─────────┐
                                results/table.md      site/index.html
```

## CI and deploy

- `.github/workflows/ci.yml` runs on pull requests, pushes to `main` and weekly. On Python 3.11 and 3.12 it installs `requirements-dev.txt`, runs `make lint`, `make typecheck` and `make test`, then `make repro` to recompute every score from the committed raw outputs, and fails if the working tree is not clean afterwards. A second job scans the full git history with a pinned, checksum-verified gitleaks. `pip-audit` runs on the weekly schedule and on manual dispatch only.
- `.github/workflows/pages.yml` runs on pushes to `main`: it runs `make repro` and publishes the rendered page to GitHub Pages (the leaderboard linked from the README).
- Actions are pinned to commit SHAs; `.github/dependabot.yml` proposes updates to them and to the pinned Python requirements.

## Adding a model

1. `export BN_EVAL_BASE_URL=... BN_EVAL_API_KEY=...`
2. `make run PROVIDER=anthropic|openai MODEL=<model id>` (add `OPTS=notemp` if the endpoint rejects temperature 0, `OPTS=mct` for models that need `max_completion_tokens`).
3. If the run reports errors, run the same command again: only the failed items are retried. Check the warning about answers that stopped at the token limit.
4. `make repro` and commit `runs/<model>.jsonl`, `results/` and `site/index.html`, plus the README table if it changed (a test compares it with `results/table.md`).
