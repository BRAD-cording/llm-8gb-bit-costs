"""
Compare the manuscript's capacity table against the solver's JSON output, cell
by cell.

The numeric audit checks headline claims; this checks an entire table. Table VI
is generated from `results/capacity_table.json`, so any edit to either the
solver parameters or the manuscript that is not mirrored in the other shows up
here as a mismatch rather than as a silently inconsistent paper.

Usage:  python experiments/check_table_vs_data.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEX = ROOT / "paper" / "main.tex"
DATA = ROOT / "results" / "capacity_table.json"


def norm(s: str) -> str:
    return s.replace("{,}", "").replace(",", "").strip()


def main() -> int:
    if not DATA.exists():
        print(f"missing {DATA}; run dp_alloc.py --capacity first")
        return 1
    rows = json.loads(DATA.read_text(encoding="utf-8"))["rows"]
    tex = TEX.read_text(encoding="utf-8")

    # locate the tabular that follows label{tab:capacity}
    m = re.search(r"\\label\{tab:capacity\}(.*?)\\end\{tabular\}", tex, re.S)
    if not m:
        print("could not find tab:capacity tabular")
        return 1
    body = m.group(1)

    parsed = {}
    for line in body.split("\\\\"):
        cells = [c.strip() for c in line.split("&")]
        if len(cells) < 4:
            continue
        # the first data row carries a preceding \midrule in the same chunk
        eps_txt = re.sub(r"\\[a-zA-Z]+", "", cells[0]).strip()
        if not re.match(r"0\.\d+", eps_txt):
            continue
        eps = float(eps_txt)
        try:
            a = int(norm(re.sub(r"\$.*?\$", "", cells[1])))
            b = int(norm(re.sub(r"\$.*?\$", "", cells[2])))
        except ValueError:
            continue
        parsed[eps] = (a, b)

    if not parsed:
        print("no rows parsed from the table")
        return 1

    bad = 0
    print(f"{'eps':>6} {'paper 4060':>11} {'data':>8} {'paper Jetson':>13} {'data':>8}")
    for r in rows:
        if r["device"] != "RTX4060":
            continue
        eps = r["eps"]
        if eps not in parsed:
            print(f"{eps:6.3f}  not in paper table")
            continue
        p4060, pjets = parsed[eps]
        d4060 = r["max_L"]
        dj = next((x["max_L"] for x in rows
                   if x["eps"] == eps and x["device"] != "RTX4060"), None)
        ok = (p4060 == d4060) and (dj is None or pjets == dj)
        print(f"{eps:6.3f} {p4060:11d} {d4060:8d} {pjets:13d} {dj if dj else -1:8d}"
              f"   {'ok' if ok else 'MISMATCH'}")
        bad += 0 if ok else 1

    print(f"\n{'TABLE MATCHES DATA' if bad == 0 else str(bad) + ' MISMATCH(ES)'}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
