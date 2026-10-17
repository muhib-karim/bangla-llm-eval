"""Render results/table.md and site/index.html from results/summary.json (deterministic)."""
import html, json
from pathlib import Path

PRICES = {  # USD per million tokens (input, output), list prices on the run date, 2026-10-17
    "claude-haiku-4-5": (1, 5), "claude-sonnet-5": (2, 10), "claude-opus-5-5": (4, 20), "gpt-6-sol": (2, 10)}
NAMES = {"claude-haiku-4-5": "Claude Haiku 4.5", "claude-sonnet-5": "Claude Sonnet 5",
         "claude-opus-5-5": "Claude Opus 5.5", "gpt-6-sol": "GPT-6 Sol"}
COLS = [("Model", None), ("BN→EN chrF++", "translate_bn_en_chrfpp"), ("EN→BN chrF++", "translate_en_bn_chrfpp"),
        ("NLI acc.", "nli_accuracy"), ("Numerals exact", "numerals_exact_strict"),
        ("Median latency (s)", "latency_median_s"), ("Cost (USD)", "cost")]

def rows():
    s = json.loads(Path("results/summary.json").read_text())
    out = []
    for m, r in s.items():
        a, b = PRICES.get(m, (None, None))
        cost = None if a is None else round(r["input_tokens"] / 1e6 * a + r["output_tokens"] / 1e6 * b, 3)
        out.append({**r, "model": m, "name": NAMES.get(m, m), "cost": cost})
    return sorted(out, key=lambda r: -(r["translate_bn_en_chrfpp"] + r["translate_en_bn_chrfpp"]
                                       + 100 * r["nli_accuracy"] + 100 * r["numerals_exact_strict"]))

def fmt(k, v):
    if v is None: return "n/a"
    if k in ("nli_accuracy", "numerals_exact_strict"): return f"{100 * v:.1f}%"
    if k == "cost": return f"{v:.3f}"
    return f"{v:.2f}" if isinstance(v, float) else str(v)

def main():
    rs = rows()
    md = ["| " + " | ".join(c for c, _ in COLS) + " |", "|---|" + "---:|" * (len(COLS) - 1)]
    for r in rs:
        md.append("| " + " | ".join([r["name"]] + [fmt(k, r[k]) for _, k in COLS[1:]]) + " |")
    Path("results/table.md").write_text("\n".join(md) + "\n")
    head = "".join(f"<th>{html.escape(c)}</th>" for c, _ in COLS)
    body = "".join("<tr><td>" + html.escape(r["name"]) + "</td>" + "".join(
        f"<td>{fmt(k, r[k])}</td>" for _, k in COLS[1:]) + "</tr>" for r in rs)
    page = Path("site/template.html").read_text().replace("{{HEAD}}", head).replace("{{BODY}}", body)
    Path("site/index.html").write_text(page)
    print("\n".join(md))

if __name__ == "__main__":
    main()
