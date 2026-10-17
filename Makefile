PY ?= python

.PHONY: repro test run
# Recompute every score, the table and the leaderboard page from the committed raw outputs.
repro:
	$(PY) score.py > /dev/null
	$(PY) report.py
# Unit tests for the scorer and the frozen item set.
test:
	$(PY) -m pytest -q
# Live re-run of one model (needs BN_EVAL_BASE_URL and BN_EVAL_API_KEY for an
# OpenAI- or Anthropic-compatible endpoint), e.g. make run PROVIDER=anthropic MODEL=claude-haiku-4-5
run:
	$(PY) run_eval.py $(PROVIDER) $(MODEL) runs/$(MODEL).jsonl $(OPTS)
