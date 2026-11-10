import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import report

ROOT = Path(__file__).resolve().parents[1]


def entry(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "translate_bn_en_chrfpp": 50.0,
        "translate_en_bn_chrfpp": 40.0,
        "nli_accuracy": 0.5,
        "numerals_exact_strict": 0.5,
        "latency_median_s": 1.0,
        "input_tokens": 1_000_000,
        "output_tokens": 1_000_000,
    }
    return {**base, **over}


def write_summary(tmp_path: Path, summary: dict[str, Any]) -> Path:
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    return path


def test_fmt() -> None:
    assert report.fmt("nli_accuracy", 0.7444) == "74.4%"
    assert report.fmt("cost", 0.0794) == "0.079"
    assert report.fmt("translate_bn_en_chrfpp", 75.4) == "75.40"
    assert report.fmt("latency_median_s", None) == "n/a"


def test_rows_sort_with_missing_metric(tmp_path: Path) -> None:
    """A model scored on a subset has None for a task: it sorts as 0 and renders as n/a."""
    summary = {"partial": entry(nli_accuracy=None, translate_en_bn_chrfpp=None), "claude-haiku-4-5": entry()}
    rs = report.rows(write_summary(tmp_path, summary))
    assert [r["model"] for r in rs] == ["claude-haiku-4-5", "partial"]
    report.main(summary=tmp_path / "summary.json", table=tmp_path / "t.md", page=tmp_path / "i.html")
    assert "| partial | 50.00 | n/a | n/a | 50.0% | 1.00 | n/a |" in (tmp_path / "t.md").read_text(encoding="utf-8")


def test_cost_uses_list_prices_and_is_absent_for_unknown_models(tmp_path: Path) -> None:
    rs = {r["model"]: r for r in report.rows(write_summary(tmp_path, {"claude-opus-5-5": entry(), "new": entry()}))}
    assert rs["claude-opus-5-5"]["cost"] == 24.0 and rs["claude-opus-5-5"]["name"] == "Claude Opus 5.5"
    assert rs["new"]["cost"] is None and rs["new"]["name"] == "new"


def test_model_names_are_html_escaped(tmp_path: Path) -> None:
    summary = write_summary(tmp_path, {"<b>x</b>": entry()})
    report.main(summary=summary, table=tmp_path / "t.md", page=tmp_path / "i.html")
    page = (tmp_path / "i.html").read_text(encoding="utf-8")
    assert "&lt;b&gt;x&lt;/b&gt;" in page and "<b>x</b>" not in page


def test_report_is_locale_independent(tmp_path: Path) -> None:
    """Under an ASCII locale the old code emptied results/table.md and then failed on the arrow."""
    code = (
        "import sys, pathlib; sys.path.insert(0, sys.argv[1]); import report; out = pathlib.Path(sys.argv[2]); "
        "report.main(table=out / 'table.md', page=out / 'index.html')"
    )
    env = {**os.environ, "LC_ALL": "C", "LANG": "C", "PYTHONUTF8": "0", "PYTHONCOERCECLOCALE": "0"}
    env.pop("PYTHONIOENCODING", None)
    done = subprocess.run([sys.executable, "-c", code, str(ROOT), str(tmp_path)], env=env, capture_output=True)
    assert done.returncode == 0, done.stderr
    assert "BN→EN chrF++" in done.stdout.decode("utf-8")
    assert (tmp_path / "table.md").read_bytes() == (ROOT / "results/table.md").read_bytes(), done.stderr
    assert (tmp_path / "index.html").read_bytes() == (ROOT / "site/index.html").read_bytes()
