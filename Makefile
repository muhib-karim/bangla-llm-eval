PY ?= python

.PHONY: repro test run lint typecheck audit check
# Recompute every score, the table and the leaderboard page from the committed raw outputs.
repro:
	$(PY) score.py > /dev/null
	$(PY) report.py
# Unit tests for the scorer, runner, report and the frozen item set.
test:
	$(PY) -m pytest -q
# Live re-run of one model (needs BN_EVAL_BASE_URL and BN_EVAL_API_KEY for an
# OpenAI- or Anthropic-compatible endpoint), e.g. make run PROVIDER=anthropic MODEL=claude-haiku-4-5
run:
	$(PY) run_eval.py $(PROVIDER) $(MODEL) runs/$(MODEL).jsonl $(OPTS)
# Static checks (needs requirements-dev.txt).
lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .
typecheck:
	$(PY) -m mypy
# Known vulnerabilities in the pinned runtime dependencies (needs network).
audit:
	$(PY) -m pip_audit -r requirements.txt
# Everything CI gates on. Fails if make repro changes the committed results or leaderboard page.
check: lint typecheck test repro
	@test -z "$$(git status --porcelain -- results site)" || \
		{ git status --short -- results site; echo "make repro changed results/ or site/ (see above)"; exit 1; }
