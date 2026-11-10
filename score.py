"""Score every runs/<model>.jsonl against items.jsonl. Pure function of committed files."""

import json
import statistics
import sys
from pathlib import Path
from typing import Any

from sacrebleu.metrics.chrf import CHRF

ROOT = Path(__file__).resolve().parent
BN_DIGITS = "০১২৩৪৫৬৭৮৯"
TO_ASCII = str.maketrans(BN_DIGITS, "0123456789")
LABELS = ("entailment", "contradiction", "neutral")

Record = dict[str, Any]


def as_text(value: object) -> str:
    """A model output as text: None (no answer) is empty, a non-string is its string form."""
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def nli_label(text: object) -> str | None:
    """The first label word mentioned in the reply (the documented rule), or None."""
    t = as_text(text).strip().lower()
    hits = [(t.find(label), label) for label in LABELS if label in t]
    return min(hits)[1] if hits else None


def clean_num(text: object) -> str:
    """Trim spaces, commas and a trailing period. Frozen for v1.0.0: ASCII punctuation only."""
    return as_text(text).strip().strip(".").replace(",", "").replace(" ", "")


def bengali_share(text: object) -> float:
    letters = [c for c in as_text(text) if c.isalpha()]
    # Share of alphabetic characters in the Bengali Unicode block (U+0980 to U+09FF).
    return sum(0x0980 <= ord(c) <= 0x09FF for c in letters) / len(letters) if letters else 0.0


def last_rows(items: list[Record], rows: list[Record]) -> dict[str, Record]:
    """The last row per item id.

    A resumed run appends a new row for an item that failed earlier, so only the latest row counts.
    Rows without an id, or for ids outside the item set, are ignored.
    """
    ids = {i["id"] for i in items}
    by: dict[str, Record] = {}
    for r in rows:
        rid = r.get("id")
        if isinstance(rid, str) and rid in ids:
            by[rid] = r
    return by


def usage_tokens(row: Record, *aliases: str) -> int:
    """Token count under the first alias the provider reported (OpenAI and Anthropic name them differently)."""
    usage = row.get("usage") or {}
    for key in aliases:
        if usage.get(key):
            return int(usage[key])
    return 0


def score_model(items: list[Record], rows: list[Record]) -> dict[str, Any]:
    """Scores for one model. A task with no items scores None rather than failing."""
    by = last_rows(items, rows)
    final = list(by.values())

    def output(item: Record) -> str:
        return as_text(by.get(item["id"], {}).get("output"))

    out: dict[str, Any] = {"n_items": len(items), "n_answered": sum(1 for i in items if output(i).strip())}
    chrf = CHRF(word_order=2)
    for task in ("translate_bn_en", "translate_en_bn"):
        its = [i for i in items if i["task"] == task]
        hyp = [output(i).strip() for i in its]
        out[task + "_chrfpp"] = round(chrf.corpus_score(hyp, [[i["reference"] for i in its]]).score, 2) if its else None
        if task == "translate_en_bn":
            share = round(statistics.mean(bengali_share(h) for h in hyp), 4) if its else None
            out["en_bn_bengali_script_share"] = share
    nli = [i for i in items if i["task"] == "nli"]
    preds = [nli_label(output(i)) for i in nli]
    correct = sum(p == i["reference"] for p, i in zip(preds, nli, strict=True))
    out["nli_accuracy"] = round(correct / len(nli), 4) if nli else None
    out["nli_unparsed"] = sum(p is None for p in preds)
    num = [i for i in items if i["task"] == "bangla_numerals"]
    strict = lenient = 0
    for i in num:
        o = clean_num(output(i))
        strict += o == i["reference"]
        lenient += o.translate(TO_ASCII) == i["reference"].translate(TO_ASCII)
    out["numerals_exact_strict"] = round(strict / len(num), 4) if num else None
    out["numerals_exact_digit_normalised"] = round(lenient / len(num), 4) if num else None
    lat = [r["latency_s"] for r in final if r.get("latency_s") is not None]
    out["latency_median_s"] = round(statistics.median(lat), 2) if lat else None
    out["input_tokens"] = sum(usage_tokens(r, "prompt_tokens", "input_tokens") for r in final)
    out["output_tokens"] = sum(usage_tokens(r, "completion_tokens", "output_tokens") for r in final)
    out["errors"] = sum(1 for r in final if r.get("error"))
    return out


def read_jsonl(path: Path) -> list[Record]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(run_dir: str | Path | None = None, out_dir: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Score runs/*.jsonl into results/summary.json. Defaults are next to this file, whatever the CWD."""
    runs = Path(run_dir) if run_dir is not None else ROOT / "runs"
    out = Path(out_dir) if out_dir is not None else ROOT / "results"
    items = read_jsonl(ROOT / "items.jsonl")
    res = {f.stem: score_model(items, read_jsonl(f)) for f in sorted(runs.glob("*.jsonl"))}
    out.mkdir(parents=True, exist_ok=True)
    text = json.dumps(res, indent=2, ensure_ascii=False) + "\n"
    (out / "summary.json").write_text(text, encoding="utf-8", newline="\n")
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    main(*sys.argv[1:])
