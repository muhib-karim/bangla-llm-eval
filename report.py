"""Render results/table.md and site/index.html from results/summary.json (deterministic)."""

import html
import io
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
PRICES = {  # USD per million tokens (input, output), list prices on the run date, 2026-10-17
    "claude-haiku-4-5": (1, 5),
    "claude-sonnet-5": (2, 10),
    "claude-opus-5-5": (4, 20),
    "gpt-6-sol": (2, 10),
}
NAMES = {
    "claude-haiku-4-5": "Claude Haiku 4.5",
    "claude-sonnet-5": "Claude Sonnet 5",
    "claude-opus-5-5": "Claude Opus 5.5",
    "gpt-6-sol": "GPT-6 Sol",
}
COLS: list[tuple[str, str]] = [
    ("Model", "name"),
    ("BN→EN chrF++", "translate_bn_en_chrfpp"),
    ("EN→BN chrF++", "translate_en_bn_chrfpp"),
    ("NLI acc.", "nli_accuracy"),
    ("Numerals exact", "numerals_exact_strict"),
    ("Median latency (s)", "latency_median_s"),
    ("Cost (USD)", "cost"),
]


def quality_sum(r: dict[str, Any]) -> float:
    """Sum of the four quality scores on a 0-100 scale. A missing score counts as 0."""
    total: float = (
        (r.get("translate_bn_en_chrfpp") or 0)
        + (r.get("translate_en_bn_chrfpp") or 0)
        + 100 * (r.get("nli_accuracy") or 0)
        + 100 * (r.get("numerals_exact_strict") or 0)
    )
    return total


def rows(summary: Path | None = None) -> list[dict[str, Any]]:
    s = json.loads((summary or ROOT / "results" / "summary.json").read_text(encoding="utf-8"))
    out = []
    for m, r in s.items():
        price = PRICES.get(m)
        cost = None
        if price is not None:
            cost = round(r["input_tokens"] / 1e6 * price[0] + r["output_tokens"] / 1e6 * price[1], 3)
        out.append({**r, "model": m, "name": NAMES.get(m, m), "cost": cost})
    return sorted(out, key=lambda r: -quality_sum(r))


def fmt(k: str, v: Any) -> str:
    if v is None:
        return "n/a"
    if k in ("nli_accuracy", "numerals_exact_strict"):
        return f"{100 * v:.1f}%"
    if k == "cost":
        return f"{v:.3f}"
    return f"{v:.2f}" if isinstance(v, float) else str(v)


def main(
    summary: Path | None = None,
    table: Path | None = None,
    template: Path | None = None,
    page: Path | None = None,
) -> None:
    """Write the Markdown table and the leaderboard page. Defaults are next to this file, whatever the CWD."""
    rs = rows(summary)
    md = ["| " + " | ".join(c for c, _ in COLS) + " |", "|---|" + "---:|" * (len(COLS) - 1)]
    for r in rs:
        md.append("| " + " | ".join([r["name"]] + [fmt(k, r.get(k)) for _, k in COLS[1:]]) + " |")
    table = table or ROOT / "results" / "table.md"
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text("\n".join(md) + "\n", encoding="utf-8", newline="\n")
    head = "".join(f"<th>{html.escape(c)}</th>" for c, _ in COLS)
    body = "".join(
        "<tr><td>"
        + html.escape(r["name"])
        + "</td>"
        + "".join(f"<td>{fmt(k, r.get(k))}</td>" for _, k in COLS[1:])
        + "</tr>"
        for r in rs
    )
    source = (template or ROOT / "site" / "template.html").read_text(encoding="utf-8")
    page = page or ROOT / "site" / "index.html"
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(source.replace("{{HEAD}}", head).replace("{{BODY}}", body), encoding="utf-8", newline="\n")
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")  # the table has a non-ASCII arrow; do not depend on the locale
    print("\n".join(md))


if __name__ == "__main__":
    main()
