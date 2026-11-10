"""Guards for the published artefacts: raw inputs are frozen and every score reproduces byte for byte."""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import report
import score

ROOT = Path(__file__).resolve().parents[1]
GENERATED = ("results/summary.json", "results/table.md", "site/index.html")

# sha256 of the v1.0.0 item set and raw model outputs. New runs add files; these never change.
FROZEN_INPUTS = {
    "items.jsonl": "1e8f1fa53e0120b8e71dc48226f798c71e7003c8fab05228ca4138283f64230d",
    "runs/claude-haiku-4-5.jsonl": "f8d3120791d64c65b945062130fac5258c47d2d47e15186b95045870db2c4785",
    "runs/claude-opus-5-5.jsonl": "919dc3fc0ef257e9d5ca51f9160eb1048f49cf37e0ae2af12d98d8766071430b",
    "runs/claude-sonnet-5.jsonl": "affd2d90dd0e75c8f8d30db12d4d1d3da5b6c50a428da18218b1e8ba501936e3",
    "runs/gpt-6-sol.jsonl": "c5675e6bc8660de3ff5ed86cd835b5bdaeadecf844844747c12bfd201d65dca0",
}

# Published v1.0.0 scores: BN->EN chrF++, EN->BN chrF++, NLI accuracy, numerals strict, numerals digit-normalised.
PUBLISHED = {
    "claude-haiku-4-5": (62.63, 48.04, 0.7444, 0.6, 0.6),
    "claude-opus-5-5": (75.4, 45.39, 0.9, 1.0, 1.0),
    "claude-sonnet-5": (64.88, 47.42, 0.7778, 1.0, 1.0),
    "gpt-6-sol": (61.98, 42.1, 0.7778, 1.0, 1.0),
}
SCORE_KEYS = (
    "translate_bn_en_chrfpp",
    "translate_en_bn_chrfpp",
    "nli_accuracy",
    "numerals_exact_strict",
    "numerals_exact_digit_normalised",
)


@pytest.mark.parametrize("name", sorted(FROZEN_INPUTS))
def test_committed_inputs_are_frozen(name: str) -> None:
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == FROZEN_INPUTS[name]


def test_published_scores_are_unchanged() -> None:
    summary = json.loads((ROOT / "results/summary.json").read_text(encoding="utf-8"))
    for model, expected in PUBLISHED.items():
        assert tuple(summary[model][k] for k in SCORE_KEYS) == expected, model
        assert summary[model]["n_items"] == summary[model]["n_answered"] == 270
        assert summary[model]["errors"] == 0


def test_make_repro_reproduces_committed_outputs(tmp_path: Path) -> None:
    """What `make repro` does, in a copy: the two scripts must regenerate the committed files exactly."""
    for name in ("score.py", "report.py", "items.jsonl"):
        shutil.copy(ROOT / name, tmp_path / name)
    shutil.copytree(ROOT / "runs", tmp_path / "runs")
    (tmp_path / "site").mkdir()
    shutil.copy(ROOT / "site/template.html", tmp_path / "site/template.html")
    for script in ("score.py", "report.py"):
        subprocess.run([sys.executable, script], cwd=tmp_path, check=True, capture_output=True)
    for name in GENERATED:
        assert (tmp_path / name).read_bytes() == (ROOT / name).read_bytes(), name


def test_score_and_report_run_from_any_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    score.main(out_dir=tmp_path / "results")
    report.main(summary=tmp_path / "results/summary.json", table=tmp_path / "table.md", page=tmp_path / "index.html")
    assert (tmp_path / "results/summary.json").read_bytes() == (ROOT / "results/summary.json").read_bytes()
    assert (tmp_path / "table.md").read_bytes() == (ROOT / "results/table.md").read_bytes()
    assert (tmp_path / "index.html").read_bytes() == (ROOT / "site/index.html").read_bytes()


def test_readme_table_matches_results() -> None:
    """The README says its scores are recomputed in CI: the table there must be the generated one."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert (ROOT / "results/table.md").read_text(encoding="utf-8") in readme
