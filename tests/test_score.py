import json
from pathlib import Path

import score


def test_nli_label_takes_first_label_mentioned():
    assert score.nli_label("Entailment.") == "entailment"
    assert score.nli_label("neutral, not contradiction") == "neutral"
    assert score.nli_label("I am not sure") is None


def test_numeral_cleaning_keeps_script_strict():
    assert score.clean_num(" ২,৬১৩,৭৫০. ") == "২৬১৩৭৫০"
    assert score.clean_num("2613750") != "২৬১৩৭৫০"
    assert score.clean_num("2613750") == "২৬১৩৭৫০".translate(score.TO_ASCII)


def test_bengali_share():
    assert score.bengali_share("আমি ভাত খাই") == 1.0
    assert score.bengali_share("ami bhat khai") == 0.0


def test_score_model_perfect_answers():
    items = [
        {"id": "a", "task": "translate_bn_en", "reference": "The cat sat."},
        {"id": "b", "task": "translate_en_bn", "reference": "বিড়ালটি বসল।"},
        {"id": "c", "task": "nli", "reference": "neutral"},
        {"id": "d", "task": "bangla_numerals", "reference": "১২৩"},
    ]
    rows = [{"id": i["id"], "output": i["reference"], "usage": {"input_tokens": 3, "output_tokens": 2},
             "latency_s": 1.0} for i in items]
    r = score.score_model(items, rows)
    assert r["translate_bn_en_chrfpp"] == 100.0 and r["translate_en_bn_chrfpp"] == 100.0
    assert r["nli_accuracy"] == 1.0 and r["numerals_exact_strict"] == 1.0
    assert r["input_tokens"] == 12 and r["output_tokens"] == 8 and r["errors"] == 0


def test_item_set_is_the_frozen_one():
    items = [json.loads(line) for line in Path("items.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(items) == 270 and len({i["id"] for i in items}) == 270
    tasks = {}
    for i in items:
        tasks[i["task"]] = tasks.get(i["task"], 0) + 1
    assert tasks == {"translate_bn_en": 60, "translate_en_bn": 60, "nli": 90, "bangla_numerals": 60}
