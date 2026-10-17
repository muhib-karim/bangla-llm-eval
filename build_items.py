"""Build the fixed Bangla eval item set (seeded, reproducible)."""
import json, random
from pathlib import Path

R = random.Random(20261017)
D = Path("data")
BN = "০১২৩৪৫৬৭৮৯"
def bn(n: int) -> str:
    s = str(n)
    return "".join(BN[int(c)] if c.isdigit() else c for c in s)

items = []
ben = (D / "flores200_dataset/devtest/ben_Beng.devtest").read_text(encoding="utf-8").splitlines()
eng = (D / "flores200_dataset/devtest/eng_Latn.devtest").read_text(encoding="utf-8").splitlines()
idx = sorted(R.sample(range(len(ben)), 120))
for k, i in enumerate(idx[:60]):
    items.append({"id": f"mt-bn-en-{i}", "task": "translate_bn_en", "source": ben[i], "reference": eng[i],
                  "prompt": "Translate this Bengali sentence into English. Reply with the translation only.\n\n" + ben[i]})
for k, i in enumerate(idx[60:]):
    items.append({"id": f"mt-en-bn-{i}", "task": "translate_en_bn", "source": eng[i], "reference": ben[i],
                  "prompt": "Translate this English sentence into Bengali (Bangla script). Reply with the translation only.\n\n" + eng[i]})

names = ["contradiction", "entailment", "neutral"]
rows = []
for off in (0, 100, 200, 300):
    rows += [r["row"] | {"row_idx": r["row_idx"]} for r in json.loads((D / f"xnli_{off}.json").read_text())["rows"]]
by = {0: [], 1: [], 2: []}
for r in rows:
    by[r["label"]].append(r)
for lab in (0, 1, 2):
    for r in R.sample(by[lab], 30):
        items.append({"id": f"nli-{r['row_idx']}", "task": "nli", "reference": names[lab],
                      "prompt": ("Premise (Bengali): " + r["sentence1"] + "\nHypothesis (Bengali): " + r["sentence2"]
                                 + "\n\nDoes the premise entail the hypothesis, contradict it, or neither? "
                                   "Answer with exactly one word: entailment, contradiction, or neutral.")})

for k in range(30):
    a, b = R.randint(100, 9999), R.randint(10, 999)
    op = R.choice(["+", "-", "×"])
    val = a + b if op == "+" else a - b if op == "-" else a * b
    items.append({"id": f"num-arith-{k}", "task": "bangla_numerals", "reference": bn(val),
                  "prompt": f"Compute {bn(a)} {op} {bn(b)}. The numbers use Bengali digits. "
                            "Reply with the result only, written in Bengali digits."})
for k in range(30):
    v = R.randint(1000, 9_999_999)
    s = bn(v)
    items.append({"id": f"num-read-{k}", "task": "bangla_numerals", "reference": str(v),
                  "prompt": f"Write the number {s} (Bengali digits) using ASCII digits 0-9 only, no separators. "
                            "Reply with the number only."})
R.shuffle(items)
Path("items.jsonl").write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items), encoding="utf-8")
from collections import Counter
print(len(items), Counter(i["task"] for i in items))
