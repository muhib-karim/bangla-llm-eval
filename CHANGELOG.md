# Changelog

## [Unreleased]

No measured result, raw output or item changed: `items.jsonl`, `runs/`, `results/` and `site/` are byte-identical to 1.0.0.

### Fixed

- `run_eval.py`: a re-run now retries items that ended in an error (they were skipped for ever); only timeouts, connection errors and HTTP 408/409/429/5xx are retried, with no sleep after the last attempt; the exit status is 1 if any item ended in an error; the provider's stop reason is recorded in each new row and truncated or empty answers are reported; unknown providers and options are rejected; blank lines in `items.jsonl` and a half-written last line in the output file no longer crash it; the module can be imported without side effects.
- `score.py`: only the last row per item id counts, and rows for unknown ids are ignored; a task with no items scores `None` instead of crashing; `n_answered` no longer counts empty replies; non-string outputs no longer crash; token usage reported under both alias names is not double-counted.
- `report.py`, `score.py`, `build_items.py`: files are read and written as UTF-8 with `\n` line endings whatever the locale (on a non-UTF-8 locale `report.py` used to empty `results/table.md` and fail); paths no longer depend on the working directory.
- `build_items.py`: no longer runs on import, and does not overwrite an existing item set with different content unless `--force` is given.

### Added

- Tests for the runner, report, item builder and the frozen artefacts (59 tests, all offline).
- `pyproject.toml` (ruff, mypy strict, pytest), `requirements-dev.txt` with pinned tools, `make lint`, `make typecheck`, `make audit`, `make check`.
- CI: lint, types, tests and the reproduction check on Python 3.11 and 3.12, a gitleaks scan of the full history, weekly `pip-audit`; actions pinned to commit SHAs; Dependabot configuration.

### Documentation

- README: quickstart, architecture summary, two new limits (answers cut off at the 512-token cap; the two translation directions use different FLORES-200 sentences), corrected install steps and the description of what `build_items.py` documents.

## [1.0.0] - 2026-10-17

First release: 270 frozen Bangla items (FLORES-200 translation both ways, Bengali XNLI, Bengali numerals), raw outputs for Claude Opus 5.5, Claude Sonnet 5, Claude Haiku 4.5 and GPT-6 Sol (live run, 2026-10-17), the scorer and the leaderboard. Reproduce with `make repro`.

[1.0.0]: https://github.com/muhib-karim/bangla-llm-eval/releases/tag/v1.0.0
