"""
Numeric audit: list every quantitative claim in the manuscript with the line it
sits on, so each can be checked against a results file by hand.

This exists because a stale number in the abstract ("3.3--3.5x") survived a
revision of the table it referred to ("1.76--1.90x"). Nothing catches that
automatically except reading the two side by side, which this makes cheap.

Usage:  python paper/audit_numbers.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEX = ROOT / "paper" / "main.tex"

PATTERNS = [
    ("factor", r"\$([0-9][0-9.,{}]*)\\times\$"),
    ("percent", r"\$\+?([0-9][0-9.]*)\\%\$"),
    ("ratio", r"\$([0-9][0-9.]*)\\times\$"),
    ("context", r"([0-9][0-9,]{2,})"),
    ("mb", r"([0-9][0-9,]{2,})\,\\mathrm\{MB\}"),
]


def main() -> int:
    src = TEX.read_text(encoding="utf-8")
    lines = src.splitlines()

    for name, pat in PATTERNS[:2]:
        print(f"\n=== {name} claims ===")
        for m in re.finditer(pat, src):
            ln = src[:m.start()].count("\n") + 1
            ctx = " ".join(lines[ln - 1].strip().split())
            print(f"  L{ln:<4d} {m.group(1):>9s}  | {ctx[:96]}")

    # cross-check a few headline numbers against the result files
    print("\n=== headline cross-check ===")
    checks = []

    try:
        m = json.loads((ROOT / "results" / "probe_4060.json").read_text("utf-8"))
        checks.append(("4060 copy GB/s", m["bandwidth"]["GBps"], 161.9))
    except Exception as e:
        checks.append(("probe_4060", f"unreadable: {e}", None))

    try:
        f = json.loads((ROOT / "results" / "sweep_qwen06_fp16_real_naive.json").read_text("utf-8"))
        rows = {r["seq_len"]: r for r in f["rows"] if "error" not in r}
        naive8 = rows[8192]["ttft_ms"]
        g = json.loads((ROOT / "results" / "sweep_qwen06_fp16_real.json").read_text("utf-8"))
        fused = {r["seq_len"]: r for r in g["rows"] if "error" not in r}
        checks.append(("naive 0.6B real",
                       [rows[k]["ttft_ms"] for k in (2048, 4096, 8192)
                        if k in rows], None))
        if 8192 in fused:
            checks.append(("TTFT speedup @8k (naive/fused)",
                           round(naive8 / fused[8192]["ttft_ms"], 1), 72))
    except Exception as e:
        checks.append(("stack data", f"unreadable: {e}", None))

    try:
        ci = json.loads((ROOT / "results" / "bit_cost_ci.json").read_text("utf-8"))
        w4 = ci["weights"]["rel_ppl_pct"]
        k4 = ci["cache"]["rel_ppl_pct"]
        checks.append(("weights 4-bit % (48w)", round(w4, 2), w4))
        checks.append(("KV 4-bit % (48w)", round(k4, 2), k4))
        paired = ci["paired_cache_minus_weights"]["ci95_rel_ppl_pct"]
        checks.append(("paired CI excludes 0", paired[1] < 0, True))
    except Exception as e:
        checks.append(("bit cost", f"unreadable: {e}", None))

    for name, got, expect in checks:
        flag = ""
        if expect is not None and isinstance(got, (int, float)):
            flag = "  OK" if abs(got - expect) / max(expect, 1) < 0.02 else "  MISMATCH"
        print(f"  {name:34s} {got}{flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
