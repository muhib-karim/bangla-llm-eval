# bangla-llm-eval

[![CI](https://github.com/muhib-karim/bangla-llm-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/muhib-karim/bangla-llm-eval/actions/workflows/ci.yml)
[![Leaderboard](https://img.shields.io/badge/leaderboard-live-2dd4bf)](https://muhib-karim.github.io/bangla-llm-eval/)
[![License: MIT (code)](https://img.shields.io/badge/code-MIT-blue)](LICENSE)
[![Cite](https://img.shields.io/badge/cite-CITATION.cff-8a63d2)](CITATION.cff)

A small, fixed and reproducible evaluation of large language models on Bangla (Bengali):
translation in both directions, natural-language inference, and Bengali-digit numerals.
Every score in this README is recomputed in CI from the raw model outputs committed in `runs/`.

**Leaderboard:** https://muhib-karim.github.io/bangla-llm-eval/

## Problem

Bangla has more than 230 million speakers, but model reports rarely break results out for it, and
the numbers that do exist are hard to reproduce: prompts, item lists and raw outputs are usually
not published. This repository is a deliberately small, open alternative: 270 frozen items, the
exact prompts, every raw answer, and a scorer you can run in seconds.

## Measured results (live, 2026-10-17)

One run per model over all 270 items, through the providers' own APIs.

| Model | BN→EN chrF++ | EN→BN chrF++ | NLI acc. | Numerals exact | Median latency (s) | Cost (USD) |
|---|---:|---:|---:|---:|---:|---:|
| Claude Opus 5.5 | 75.40 | 45.39 | 90.0% | 100.0% | 2.24 | 0.978 |
| Claude Sonnet 5 | 64.88 | 47.42 | 77.8% | 100.0% | 1.19 | 0.257 |
| GPT-6 Sol | 61.98 | 42.10 | 77.8% | 100.0% | 1.56 | 0.312 |
| Claude Haiku 4.5 | 62.63 | 48.04 | 74.4% | 60.0% | 0.65 | 0.079 |

Rows are ordered by the sum of the four quality scores. Cost is the whole 270-item suite at list
prices, from the token counts each API returned. Machine-readable scores: [`results/summary.json`](results/summary.json).

**Findings**

- **Reading Bangla is easier than writing it.** Every model scores 14 to 30 chrF++ points lower
  translating into Bangla than out of it, and the order changes: Haiku 4.5 is highest into Bangla
  (48.0) and lowest-cost, while Opus 5.5 leads out of Bangla (75.4). With 60 sentences per
  direction, gaps of two or three points into Bangla are within noise.
- **Opus 5.5 is the only model clearly ahead on inference** (90.0% on 90 balanced Bengali XNLI
  items, against 74 to 78% for the others), at three to twelve times the cost.
- **Bengali digits are a real failure mode for the smallest model.** Haiku 4.5 got 24 of 60
  numeral items wrong: 16 of 30 plain digit transcriptions (for example ৭৩৩৯৭৮৯ written as
  733978, one digit dropped) and 8 of 30 arithmetic items. The other three models were exact on all 60.
- **Script compliance is not the problem.** Into-Bangla outputs were 98.9 to 100% Bengali letters
  for every model.

## Tasks

| Task | Items | Source | Metric |
|---|---:|---|---|
| Translation BN→EN | 60 | FLORES-200 devtest, `ben_Beng` → `eng_Latn` | corpus chrF++ (sacrebleu, word order 2) |
| Translation EN→BN | 60 | FLORES-200 devtest, `eng_Latn` → `ben_Beng` | corpus chrF++ |
| Natural-language inference | 90 | Bengali XNLI test split (`csebuetnlp/xnli_bn`), 30 per label | accuracy; first label word in the reply |
| Bangla numerals | 60 | generated: 30 arithmetic in Bengali digits, 30 Bengali→ASCII transcriptions | exact match after trimming spaces, commas and a trailing period |

Items are sampled with a fixed seed (`20261017`) by [`build_items.py`](build_items.py) and frozen
in [`items.jsonl`](items.jsonl); a test fails if the item set changes. Numerals are scored strictly:
an arithmetic answer must be written in Bengali digits as the prompt asks. `summary.json` also
reports a digit-normalised score; for this run the two are identical.

## Reproduce

```bash
pip install -r requirements.txt
make repro   # rescore all committed runs, rewrite results/ and the leaderboard page
make test    # scorer unit tests and the frozen item-set check
```

`make repro` needs no API key and no network. To re-run a model live against any OpenAI- or
Anthropic-compatible endpoint:

```bash
export BN_EVAL_BASE_URL=https://api.anthropic.com/v1 BN_EVAL_API_KEY=...
make run PROVIDER=anthropic MODEL=claude-haiku-4-5
```

Run settings: one user message per item, no system prompt, 6 requests in flight, up to three
attempts per item (no item ended in an error in this run). Temperature 0 where the endpoint
accepts it, otherwise the provider default; `max_tokens` 512, or `max_completion_tokens` 2048 for
GPT-6 Sol, whose replies include reasoning tokens.

## Limits

- One run per model. Repeat runs were not measured, so run-to-run variance is unknown.
- Small samples: 60 sentences per translation direction and 90 NLI items. On 90 items a 95%
  interval is roughly ±8 points, so only large gaps are meaningful.
- FLORES-200 and XNLI are public. Models may have seen them in training, which would inflate scores;
  this suite cannot rule that out.
- chrF++ rewards character overlap with one reference. It is not a human judgement of fluency.
- Bangla here is standard written Bangla. Dialects and romanised Bangla are not covered.

## Data and licences

Code is MIT ([LICENSE](LICENSE)). `items.jsonl` and `runs/` contain text from third-party datasets
under their own licences: FLORES-200 sentences are CC BY-SA 4.0, and Bengali XNLI items are
CC BY-NC-SA 4.0 (non-commercial). The numeral items are generated by this repository. Raw source
files are not redistributed; `build_items.py` documents where they come from.

## Citation

If you use this suite, please cite it ([CITATION.cff](CITATION.cff)):

```bibtex
@software{karim_bangla_llm_eval_2026,
  author  = {Karim, Muhib},
  title   = {bangla-llm-eval: a small reproducible Bangla evaluation for large language models},
  year    = {2026},
  url     = {https://github.com/muhib-karim/bangla-llm-eval}
}
```

and the datasets and metric it builds on:

- NLLB Team et al. (2022). *No Language Left Behind: Scaling Human-Centered Machine Translation.* (FLORES-200)
- Bhattacharjee et al. (2022). *BanglaBERT: Language Model Pretraining and Benchmarks for Low-Resource Language Understanding Evaluation in Bangla.* Findings of NAACL. (Bengali XNLI)
- Popović, M. (2017). *chrF++: words helping character n-grams.* WMT.
- Post, M. (2018). *A Call for Clarity in Reporting BLEU Scores.* WMT. (sacrebleu)
