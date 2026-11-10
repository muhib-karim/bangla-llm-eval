import json
import re
import subprocess
import sys
from pathlib import Path

import build_items

ROOT = Path(__file__).resolve().parents[1]
FROM_BN = str.maketrans(build_items.BN, "0123456789")


def make_data(root: Path) -> Path:
    """A synthetic data/ directory in the layout build_items.py reads."""
    devtest = root / "flores200_dataset" / "devtest"
    devtest.mkdir(parents=True)
    (devtest / "ben_Beng.devtest").write_text("".join(f"বাক্য {n}\n" for n in range(200)), encoding="utf-8")
    (devtest / "eng_Latn.devtest").write_text("".join(f"sentence {n}\n" for n in range(200)), encoding="utf-8")
    for off in (0, 100, 200, 300):
        rows = [
            {"row_idx": n, "row": {"sentence1": f"ভিত্তি {n}", "sentence2": f"অনুমান {n}", "label": n % 3}}
            for n in range(off, off + 100)
        ]
        (root / f"xnli_{off}.json").write_text(json.dumps({"rows": rows}, ensure_ascii=False), encoding="utf-8")
    return root


def test_bn_digits() -> None:
    assert build_items.bn(2613750) == "২৬১৩৭৫০"
    assert build_items.bn(-45) == "-৪৫"


def test_build_on_fixture_is_deterministic(tmp_path: Path) -> None:
    data = make_data(tmp_path / "data")
    items = build_items.build(data)
    assert items == build_items.build(data)
    assert len(items) == 270 and len({i["id"] for i in items}) == 270
    tasks = [i["task"] for i in items]
    assert {t: tasks.count(t) for t in set(tasks)} == {
        "translate_bn_en": 60,
        "translate_en_bn": 60,
        "nli": 90,
        "bangla_numerals": 60,
    }
    labels = [i["reference"] for i in items if i["task"] == "nli"]
    assert all(labels.count(label) == 30 for label in build_items.NLI_LABELS)


def test_main_writes_utf8_and_protects_an_existing_file(tmp_path: Path) -> None:
    data, out = make_data(tmp_path / "data"), tmp_path / "items.jsonl"
    assert build_items.main([str(data), str(out)]) == 0
    built = out.read_bytes()
    assert "বাক্য".encode() in built and b"\r" not in built
    assert build_items.main([str(data), str(out)]) == 0  # identical content: fine
    out.write_text("something else\n", encoding="utf-8")
    assert build_items.main([str(data), str(out)]) == 1
    assert out.read_text(encoding="utf-8") == "something else\n"
    assert build_items.main([str(data), str(out), "--force"]) == 0
    assert out.read_bytes() == built


def test_import_has_no_side_effects(tmp_path: Path) -> None:
    """The old script built and overwrote items.jsonl as soon as it was imported."""
    code = "import sys; sys.path.insert(0, sys.argv[1]); import build_items"
    done = subprocess.run([sys.executable, "-c", code, str(ROOT)], cwd=tmp_path, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert list(tmp_path.iterdir()) == []


def test_prompt_templates_match_the_frozen_items() -> None:
    """The templates in build_items.py are the ones the committed items were built with."""
    items = [json.loads(line) for line in (ROOT / "items.jsonl").read_text(encoding="utf-8").splitlines()]
    seen = {"translate_bn_en": 0, "translate_en_bn": 0, "nli": 0, "arith": 0, "read": 0}
    for i in items:
        if i["task"] == "translate_bn_en":
            assert i["prompt"] == build_items.BN_EN_PROMPT + i["source"]
            seen["translate_bn_en"] += 1
        elif i["task"] == "translate_en_bn":
            assert i["prompt"] == build_items.EN_BN_PROMPT + i["source"]
            seen["translate_en_bn"] += 1
        elif i["task"] == "nli":
            premise, hypothesis = re.fullmatch(
                r"Premise \(Bengali\): (.*)\nHypothesis \(Bengali\): (.*)\n\n.*", i["prompt"], re.S
            ).groups()  # type: ignore[union-attr]
            assert i["prompt"] == build_items.nli_prompt(premise, hypothesis)
            assert i["reference"] in build_items.NLI_LABELS
            seen["nli"] += 1
        elif i["id"].startswith("num-arith"):
            a_bn, op, b_bn = re.match(r"Compute (\S+) (\S) (\S+)\. ", i["prompt"]).groups()  # type: ignore[union-attr]
            a, b = int(a_bn.translate(FROM_BN)), int(b_bn.translate(FROM_BN))
            assert i["prompt"] == build_items.arith_prompt(a, op, b)
            value = a + b if op == "+" else a - b if op == "-" else a * b
            assert op in "+-×" and i["reference"] == build_items.bn(value)
            seen["arith"] += 1
        else:
            assert i["id"].startswith("num-read")
            assert i["prompt"] == build_items.read_prompt(int(i["reference"]))
            seen["read"] += 1
    assert seen == {"translate_bn_en": 60, "translate_en_bn": 60, "nli": 90, "arith": 30, "read": 30}
