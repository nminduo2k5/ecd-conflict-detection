"""Compare two results folders macro by macro (the same numbers the paper prints).

    python experiments/compare_results.py results results_rerun [--tol 0.02]

For each folder the LaTeX macros are exported with export_numbers.py (into a temp file),
then every macro is compared.  Timing macros (ms, s, speed-ups) are listed separately:
they depend on the machine and are expected to differ; everything else (counts, recall,
AUC, proof obligations, p-values, effect sizes) should agree within --tol (relative).
A Markdown report is written to <second folder>/COMPARE_vs_<first folder name>.md.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / "experiments" / "export_numbers.py"
MACRO = re.compile(r"\\newcommand\{\\(\w+)\}\{(.*)\}")
TIMING = re.compile(r"(Ms|Time|Speedup|GnnEmbed|ReverifyS)$|^rqFour\w*(Exact|Approx|TierA)(Sd)?$")


def export(res_dir: Path) -> dict[str, str]:
    out = Path(tempfile.mkdtemp(prefix="ecd_cmp_")) / "numbers.tex"
    env = {**os.environ, "ECD_RESULTS_DIR": str(res_dir.resolve())}
    subprocess.run([sys.executable, str(EXPORT), str(out)], env=env, check=True, cwd=ROOT,
                   stdout=subprocess.DEVNULL)
    macros = {}
    for line in out.read_text(encoding="utf-8").splitlines():
        m = MACRO.match(line)
        if m:
            macros[m.group(1)] = m.group(2)
    return macros


def num(v: str):
    v = v.replace(r"\,", "").replace("$<$", "<").strip()
    if v.startswith("<"):
        v = v[1:]
    try:
        return float(v)
    except ValueError:
        return None


def rel_diff(a: str, b: str):
    x, y = num(a), num(b)
    if x is None or y is None:
        return None if a == b else float("inf")
    if x == y:
        return 0.0
    return abs(x - y) / max(abs(x), abs(y), 1e-12)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("first", help="reference results folder (e.g. results)")
    ap.add_argument("second", help="folder to check (e.g. results_rerun)")
    ap.add_argument("--tol", type=float, default=0.02, help="relative tolerance for non-timing macros (default 0.02)")
    a = ap.parse_args()
    d1 = Path(a.first) if Path(a.first).is_absolute() else ROOT / a.first
    d2 = Path(a.second) if Path(a.second).is_absolute() else ROOT / a.second
    m1, m2 = export(d1), export(d2)
    names = sorted(set(m1) | set(m2))
    pending = sorted(n for n in names if m2.get(n, r"\todo") == r"\todo" and m1.get(n) != r"\todo")
    rows_num, rows_time, equal = [], [], 0
    for n in names:
        v1, v2 = m1.get(n, "–"), m2.get(n, "–")
        if v1 == r"\todo" or v2 == r"\todo":
            continue
        d = rel_diff(v1, v2)
        if d == 0.0:
            equal += 1
            continue
        (rows_time if TIMING.search(n) else rows_num).append((n, v1, v2, d))
    rows_num.sort(key=lambda r: -(r[3] if r[3] is not None else 0))
    rows_time.sort(key=lambda r: -(r[3] if r[3] is not None else 0))
    beyond = [r for r in rows_num if r[3] is None or r[3] > a.tol]

    def table(rows):
        if not rows:
            return ["(none)"]
        out = ["| macro | " + d1.name + " | " + d2.name + " | rel. diff |", "|---|---|---|---|"]
        for n, v1, v2, d in rows:
            ds = "n/a" if d is None else ("∞" if d == float("inf") else f"{100 * d:.1f} %")
            out.append(f"| {n} | {v1} | {v2} | {ds} |")
        return out

    lines = [f"# {d2.name} vs {d1.name}", "",
             f"- macros compared: {len(names)}; identical: {equal}; "
             f"non-timing differences beyond {100 * a.tol:.0f} %: **{len(beyond)}**; "
             f"timing differences: {len(rows_time)}; not yet produced in {d2.name}: {len(pending)}", "",
             f"## Non-timing macros that differ (should agree within {100 * a.tol:.0f} %)", ""]
    lines += table(rows_num)
    lines += ["", "## Timing macros that differ (machine-dependent, expected)", ""]
    lines += table(rows_time)
    if pending:
        lines += ["", f"## Macros still \\todo in {d2.name} (experiment not run yet)", "",
                  ", ".join(pending)]
    report = d2 / f"COMPARE_vs_{d1.name}.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:4 + len(table(rows_num))]))
    print(f"\nfull report: {report}")
    return 1 if beyond else 0


if __name__ == "__main__":
    sys.exit(main())
