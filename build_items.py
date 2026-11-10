"""Build the fixed Bangla eval item set (seeded, reproducible).

Usage: build_items.py [data_dir] [out.jsonl] [--force]

Run once; the output is frozen in items.jsonl and a test fails if it changes. The source files are
not redistributed with this repository. The script reads them from data_dir (default: data/ next
to this file) in this layout:

  flores200_dataset/devtest/ben_Beng.devtest   FLORES-200 devtest, Bengali, one sentence per line
  flores200_dataset/devtest/eng_Latn.devtest   FLORES-200 devtest, English, line-aligned with the Bengali file
  xnli_0.json, xnli_100.json, xnli_200.json, xnli_300.json
      rows of the Bengali XNLI test split (csebuetnlp/xnli_bn), each file a JSON object
      {"rows": [{"row_idx": int, "row": {"sentence1": str, "sentence2": str, "label": int}}, ...]}
      with label 0 = contradiction, 1 = entailment, 2 = neutral

An existing output file with different content is left alone unless --force is given.
"""

import json
import random
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
SEED = 20261017
BN = "০১২৩৪৫৬৭৮৯"
NLI_LABELS = ["contradiction", "entailment", "neutral"]
BN_EN_PROMPT = "Translate this Bengali sentence into English. Reply with the translation only.\n\n"
EN_BN_PROMPT = "Translate this English sentence into Bengali (Bangla script). Reply with the translation only.\n\n"


def bn(n: int) -> str:
    """The number in Bengali digits (a minus sign is kept as is)."""
    return "".join(BN[int(c)] if c.isdigit() else c for c in str(n))


def nli_prompt(premise: str, hypothesis: str) -> str:
    return (
        "Premise (Bengali): "
        + premise
        + "\nHypothesis (Bengali): "
        + hypothesis
        + "\n\nDoes the premise entail the hypothesis, contradict it, or neither? "
        "Answer with exactly one word: entailment, contradiction, or neutral."
    )


def arith_prompt(a: int, op: str, b: int) -> str:
    return (
        f"Compute {bn(a)} {op} {bn(b)}. The numbers use Bengali digits. "
        "Reply with the result only, written in Bengali digits."
    )


def read_prompt(v: int) -> str:
    return (
        f"Write the number {bn(v)} (Bengali digits) using ASCII digits 0-9 only, no separators. "
        "Reply with the number only."
    )


def build(data: Path) -> list[dict[str, str]]:
    """The 270 items. The order of the random draws is part of the frozen set: do not reorder."""
    rng = random.Random(SEED)
    items: list[dict[str, str]] = []
    ben = (data / "flores200_dataset/devtest/ben_Beng.devtest").read_text(encoding="utf-8").splitlines()
    eng = (data / "flores200_dataset/devtest/eng_Latn.devtest").read_text(encoding="utf-8").splitlines()
    idx = sorted(rng.sample(range(len(ben)), 120))
    for i in idx[:60]:
        items.append(
            {
                "id": f"mt-bn-en-{i}",
                "task": "translate_bn_en",
                "source": ben[i],
                "reference": eng[i],
                "prompt": BN_EN_PROMPT + ben[i],
            }
        )
    for i in idx[60:]:
        items.append(
            {
                "id": f"mt-en-bn-{i}",
                "task": "translate_en_bn",
                "source": eng[i],
                "reference": ben[i],
                "prompt": EN_BN_PROMPT + eng[i],
            }
        )

    rows: list[dict[str, Any]] = []
    for off in (0, 100, 200, 300):
        page = json.loads((data / f"xnli_{off}.json").read_text(encoding="utf-8"))
        rows += [r["row"] | {"row_idx": r["row_idx"]} for r in page["rows"]]
    by: dict[int, list[dict[str, Any]]] = {0: [], 1: [], 2: []}
    for r in rows:
        by[r["label"]].append(r)
    for lab in (0, 1, 2):
        for r in rng.sample(by[lab], 30):
            items.append(
                {
                    "id": f"nli-{r['row_idx']}",
                    "task": "nli",
                    "reference": NLI_LABELS[lab],
                    "prompt": nli_prompt(r["sentence1"], r["sentence2"]),
                }
            )

    for k in range(30):
        a, b = rng.randint(100, 9999), rng.randint(10, 999)
        op = rng.choice(["+", "-", "×"])
        val = a + b if op == "+" else a - b if op == "-" else a * b
        items.append(
            {"id": f"num-arith-{k}", "task": "bangla_numerals", "reference": bn(val), "prompt": arith_prompt(a, op, b)}
        )
    for k in range(30):
        v = rng.randint(1000, 9_999_999)
        items.append({"id": f"num-read-{k}", "task": "bangla_numerals", "reference": str(v), "prompt": read_prompt(v)})
    rng.shuffle(items)
    return items


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    force = "--force" in args
    paths = [a for a in args if a != "--force"]
    data = Path(paths[0]) if paths else ROOT / "data"
    out = Path(paths[1]) if len(paths) > 1 else ROOT / "items.jsonl"
    items = build(data)
    text = "".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items)
    print(len(items), Counter(i["task"] for i in items))
    if out.exists() and out.read_text(encoding="utf-8") != text and not force:
        print(f"build_items.py: {out} exists with different content; not overwritten (use --force)", file=sys.stderr)
        return 1
    out.write_text(text, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
