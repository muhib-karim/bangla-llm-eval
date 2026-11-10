import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

import score

ROOT = Path(__file__).resolve().parents[1]


def test_nli_label_takes_first_label_mentioned() -> None:
    assert score.nli_label("Entailment.") == "entailment"
    assert score.nli_label("neutral, not contradiction") == "neutral"
    assert score.nli_label("I am not sure") is None


def test_numeral_cleaning_keeps_script_strict() -> None:
    assert score.clean_num(" ২,৬১৩,৭৫০. ") == "২৬১৩৭৫০"
    assert score.clean_num("2613750") != "২৬১৩৭৫০"
    assert score.clean_num("2613750") == "২৬১৩৭৫০".translate(score.TO_ASCII)


def test_bengali_share() -> None:
    assert score.bengali_share("আমি ভাত খাই") == 1.0
    assert score.bengali_share("ami bhat khai") == 0.0


def test_score_model_perfect_answers() -> None:
    items = [
        {"id": "a", "task": "translate_bn_en", "reference": "The cat sat."},
        {"id": "b", "task": "translate_en_bn", "reference": "বিড়ালটি বসল।"},
        {"id": "c", "task": "nli", "reference": "neutral"},
        {"id": "d", "task": "bangla_numerals", "reference": "১২৩"},
    ]
    rows = [
        {"id": i["id"], "output": i["reference"], "usage": {"input_tokens": 3, "output_tokens": 2}, "latency_s": 1.0}
        for i in items
    ]
    r = score.score_model(items, rows)
    assert r["translate_bn_en_chrfpp"] == 100.0 and r["translate_en_bn_chrfpp"] == 100.0
    assert r["nli_accuracy"] == 1.0 and r["numerals_exact_strict"] == 1.0
    assert r["input_tokens"] == 12 and r["output_tokens"] == 8 and r["errors"] == 0


def test_item_set_is_the_frozen_one() -> None:
    items = [json.loads(line) for line in (ROOT / "items.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(items) == 270 and len({i["id"] for i in items}) == 270
    tasks: dict[str, int] = {}
    for i in items:
        tasks[i["task"]] = tasks.get(i["task"], 0) + 1
    assert tasks == {"translate_bn_en": 60, "translate_en_bn": 60, "nli": 90, "bangla_numerals": 60}
    digest = hashlib.sha256(json.dumps(items, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    assert digest == "ab9120e59d72844ea98d6c003b6c08a520b77311e60b6189ed98461d81a60f45"


# --- regression tests -------------------------------------------------------------------------

ITEMS: list[dict[str, Any]] = [
    {"id": "a", "task": "translate_bn_en", "reference": "The cat sat."},
    {"id": "b", "task": "translate_en_bn", "reference": "বিড়ালটি বসল।"},
    {"id": "c", "task": "nli", "reference": "neutral"},
    {"id": "d", "task": "bangla_numerals", "reference": "১২৩"},
]


def answered(item: dict[str, Any], **extra: Any) -> dict[str, Any]:
    usage = {"input_tokens": 3, "output_tokens": 2}
    return {"id": item["id"], "output": item["reference"], "usage": usage, "latency_s": 1.0, **extra}


def test_last_row_per_id_wins() -> None:
    """A resumed run appends a success after an error row: the error must stop counting."""
    rows = [{"id": "c", "error": "ReadTimeout"}, *(answered(i) for i in ITEMS)]
    r = score.score_model(ITEMS, rows)
    assert r["errors"] == 0 and r["n_answered"] == 4 and r["nli_accuracy"] == 1.0
    doubled = score.score_model(ITEMS, [answered(i) for i in ITEMS] * 2)
    assert doubled["input_tokens"] == 12 and doubled["output_tokens"] == 8


def test_rows_for_unknown_ids_are_ignored() -> None:
    stray: list[dict[str, Any]] = [{"id": "zzz", "error": "X", "usage": {"input_tokens": 99}}, {"output": "no id"}]
    r = score.score_model(ITEMS, stray)
    assert r["errors"] == 0 and r["input_tokens"] == 0 and r["n_answered"] == 0


@pytest.mark.parametrize("missing", ["translate_bn_en", "translate_en_bn", "nli", "bangla_numerals"])
def test_score_model_handles_missing_tasks(missing: str) -> None:
    items = [i for i in ITEMS if i["task"] != missing]
    r = score.score_model(items, [answered(i) for i in items])
    key = {
        "translate_bn_en": "translate_bn_en_chrfpp",
        "translate_en_bn": "translate_en_bn_chrfpp",
        "nli": "nli_accuracy",
        "bangla_numerals": "numerals_exact_strict",
    }
    assert r[key[missing]] is None
    assert all(r[v] is not None for k, v in key.items() if k != missing)


def test_score_model_handles_no_items_at_all() -> None:
    r = score.score_model([], [])
    assert r["n_items"] == 0 and r["nli_accuracy"] is None and r["en_bn_bengali_script_share"] is None


def test_empty_output_is_not_answered() -> None:
    r = score.score_model(ITEMS, [{"id": i["id"], "output": "  "} for i in ITEMS])
    assert r["n_answered"] == 0 and r["nli_unparsed"] == 1


def test_none_and_non_string_outputs() -> None:
    assert score.nli_label(None) is None
    assert score.clean_num(None) == ""
    assert score.bengali_share(None) == 0.0
    numeral = [{"id": "n", "task": "bangla_numerals", "reference": "123"}]
    assert score.score_model(numeral, [{"id": "n", "output": 123}])["numerals_exact_strict"] == 1.0
    r = score.score_model(ITEMS, [{"id": i["id"], "output": None} for i in ITEMS])
    assert r["n_answered"] == 0 and r["translate_bn_en_chrfpp"] == 0.0


def test_usage_aliases_not_double_counted() -> None:
    usage = {"prompt_tokens": 10, "input_tokens": 10, "completion_tokens": 5, "output_tokens": 5}
    rows: list[dict[str, Any]] = [
        {"id": "a", "output": "x", "usage": usage},
        {"id": "b", "output": "y", "usage": None},
    ]
    r = score.score_model(ITEMS, rows)
    assert r["input_tokens"] == 10 and r["output_tokens"] == 5


def test_digit_normalisation_is_bengali_only() -> None:
    """Haiku's committed answer to num-arith-17 mixes in Devanagari digits: wrong under both metrics."""
    mixed = "১৫" + chr(0x0968) + chr(0x0966)  # Bengali 1, 5 then Devanagari 2, 0
    item = [{"id": "n", "task": "bangla_numerals", "reference": "১৫২০"}]
    r = score.score_model(item, [{"id": "n", "output": mixed}])
    assert r["numerals_exact_strict"] == 0.0 and r["numerals_exact_digit_normalised"] == 0.0
    ascii_answer = score.score_model(item, [{"id": "n", "output": "1520"}])
    assert ascii_answer["numerals_exact_strict"] == 0.0 and ascii_answer["numerals_exact_digit_normalised"] == 1.0


def test_numeral_metric_is_the_v1_definition() -> None:
    """Frozen for v1.0.0: only ASCII spaces, commas and periods are trimmed."""
    assert score.clean_num("১২৩।") == "১২৩।"
    assert score.clean_num("১২" + chr(0x00A0) + "৩৪৫") == "১২" + chr(0x00A0) + "৩৪৫"
    assert score.clean_num("1 234,567.") == "1234567"


def test_nli_rule_is_first_mention_without_negation_handling() -> None:
    assert score.nli_label("not entailment") == "entailment"
    assert score.nli_label("**Neutral**") == "neutral"
    assert score.nli_label("It is not a contradiction; it is neutral") == "contradiction"
    assert score.nli_label("neither") is None


def test_chrf_is_chrfpp_with_one_reference() -> None:
    r = score.score_model(ITEMS[:1], [{"id": "a", "output": "The cat sat on the mat."}])
    assert 0.0 < r["translate_bn_en_chrfpp"] < 100.0
    assert score.score_model(ITEMS[:1], [])["translate_bn_en_chrfpp"] == 0.0
