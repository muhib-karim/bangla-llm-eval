"""Score every runs/<model>.jsonl against items.jsonl. Pure function of committed files."""
import json, re, statistics, sys
from pathlib import Path
from sacrebleu.metrics import CHRF

BN_DIGITS = "০১২৩৪৫৬৭৮৯"
TO_ASCII = str.maketrans(BN_DIGITS, "0123456789")
LABELS = ("entailment", "contradiction", "neutral")

def nli_label(text):
    t = text.strip().lower()
    hits = [(t.find(l), l) for l in LABELS if l in t]
    return min(hits)[1] if hits else None

def clean_num(text):
    t = text.strip().strip(".").replace(",", "").replace(" ", "")
    return t

def bengali_share(text):
    letters = [c for c in text if c.isalpha()]
    return sum("\u0980" <= c <= "\u09ff" for c in letters) / len(letters) if letters else 0.0

def score_model(items, rows):
    by = {r["id"]: r for r in rows}
    out = {"n_items": len(items), "n_answered": sum(1 for i in items if by.get(i["id"], {}).get("output") is not None)}
    chrf = CHRF(word_order=2)
    for task in ("translate_bn_en", "translate_en_bn"):
        its = [i for i in items if i["task"] == task]
        hyp = [(by.get(i["id"], {}).get("output") or "").strip() for i in its]
        out[task + "_chrfpp"] = round(chrf.corpus_score(hyp, [[i["reference"] for i in its]]).score, 2)
        if task == "translate_en_bn":
            out["en_bn_bengali_script_share"] = round(statistics.mean(bengali_share(h) for h in hyp), 4)
    nli = [i for i in items if i["task"] == "nli"]
    preds = [nli_label(by.get(i["id"], {}).get("output") or "") for i in nli]
    out["nli_accuracy"] = round(sum(p == i["reference"] for p, i in zip(preds, nli)) / len(nli), 4)
    out["nli_unparsed"] = sum(p is None for p in preds)
    num = [i for i in items if i["task"] == "bangla_numerals"]
    strict = lenient = 0
    for i in num:
        o = clean_num(by.get(i["id"], {}).get("output") or "")
        strict += o == i["reference"]
        lenient += o.translate(TO_ASCII) == i["reference"].translate(TO_ASCII)
    out["numerals_exact_strict"] = round(strict / len(num), 4)
    out["numerals_exact_digit_normalised"] = round(lenient / len(num), 4)
    lat = [r["latency_s"] for r in rows if r.get("latency_s") is not None]
    out["latency_median_s"] = round(statistics.median(lat), 2) if lat else None
    def tok(r, *ks):
        u = r.get("usage") or {}
        return sum(u.get(k) or 0 for k in ks)
    out["input_tokens"] = sum(tok(r, "prompt_tokens", "input_tokens") for r in rows)
    out["output_tokens"] = sum(tok(r, "completion_tokens", "output_tokens") for r in rows)
    out["errors"] = sum(1 for r in rows if r.get("error"))
    return out

def main(run_dir="runs", out_dir="results"):
    items = [json.loads(l) for l in Path("items.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    res = {}
    for f in sorted(Path(run_dir).glob("*.jsonl")):
        rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        res[f.stem] = score_model(items, rows)
    Path(out_dir).mkdir(exist_ok=True)
    Path(out_dir, "summary.json").write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(res, indent=1))

if __name__ == "__main__":
    main(*sys.argv[1:])
