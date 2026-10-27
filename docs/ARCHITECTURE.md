# Architecture

bangla-llm-eval is a handful of small Python scripts around three committed data sets: the frozen items, the raw model outputs, and the scores computed from them. Scoring and the leaderboard need no network and no API key.

## Components

| Path | Role |
| --- | --- |
| `build_items.py` | Samples the item set with a fixed seed (`20261017`) from FLORES-200, Bengali XNLI and generated numeral tasks, and writes `items.jsonl`. Run once; the output is frozen. |
| `items.jsonl` | The 270 frozen items (id, task, prompt, reference). A test fails if this file changes. |
| `run_eval.py` | Live run of one model: `run_eval.py <provider> <model> <out.jsonl> [notemp] [mct]`. Sends each item as one user message to an Anthropic-compatible (`/messages`) or OpenAI-compatible (`/chat/completions`) endpoint read from `BN_EVAL_BASE_URL` / `BN_EVAL_API_KEY`, 6 requests in flight, up to three attempts per item, and appends one JSON line per item (output, usage, latency, or an error name). Items already in the output file are skipped, so a run can resume. |
| `runs/*.jsonl` | Committed raw outputs, one file per model. |
| `score.py` | Scores every file in `runs/` against `items.jsonl` (corpus chrF++ via sacrebleu for translation, accuracy for NLI, exact match for numerals) and writes `results/summary.json`. |
| `report.py` | Renders `results/table.md` and `site/index.html` (from `site/template.html`) from `summary.json`, deterministically. |
| `tests/test_score.py` | Scorer unit tests and the frozen item-set check. |
| `Makefile` | `make repro` (score + report), `make test` (pytest), `make run PROVIDER=... MODEL=... [OPTS=...]` (live run into `runs/$(MODEL).jsonl`). |

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

- `.github/workflows/ci.yml` installs `requirements.txt` plus pytest on Python 3.12, runs `make test`, then `make repro` to recompute every score from the committed raw outputs.
- `.github/workflows/pages.yml` runs on pushes to `main`: it runs `make repro` and publishes the rendered page to GitHub Pages (the leaderboard linked from the README).

## Adding a model

1. `export BN_EVAL_BASE_URL=... BN_EVAL_API_KEY=...`
2. `make run PROVIDER=anthropic|openai MODEL=<model id>` (add `OPTS=notemp` if the endpoint rejects temperature 0, `OPTS=mct` for models that need `max_completion_tokens`).
3. `make repro` and commit `runs/<model>.jsonl`, `results/` and `site/index.html`.
