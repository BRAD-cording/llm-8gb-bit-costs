"""Guard against superseded numbers creeping back into any document.

Every value below was correct at some point and was later replaced by a
measurement or a bug fix. They have a habit of surviving in the one file nobody
re-read -- the abstract, the cover letter, the results note. This lists them
with their replacements and fails if any reappears.

Usage:  python experiments/check_stale_numbers.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (stale token, replacement, why it is stale)
STALE = [
    ("32.3%", "34.65%", "6-window weight cost, superseded by the 48-window CI run"),
    ("$5.6\\%$", "$5.35\\%$", "6-window cache cost"),
    ("5.6%", "5.35%", "6-window cache cost (plain-text form)"),
    ("$6.9\\%$", "$6.97\\%$", "6-window tuned weight cost"),
    ("6.9%", "6.97%", "6-window tuned weight cost (plain-text form)"),
    ("$69\\times$", "$72\\times$", "speedup computed against synthetic naive data"),
    ("69x", "72x", "same, plain text"),
    ("$3.3$--$3.5\\times$", "1.63--1.82x", "capacity ratio before the solver fixes"),
    ("3.3--3.5", "1.63--1.82", "capacity ratio before the solver fixes"),
    ("$1.76$--$1.90", "$1.63$--$1.82", "ratio range before the per-model refit"),
    ("105,064", "111,184", "capacity before the per-model workspace refit"),
    ("55,594", "67,324", "same"),
    ("231,544", "243,784", "same"),
    ("$3.5\\times$", "$3.7\\times$", "exchange rate before the workspace fix"),
    ("9.08%", "34.65%", "weight cost from the buggy grouping axis"),
    ("$9.08\\%$", "$34.65\\%$", "weight cost from the buggy grouping axis"),
    ("$0.6$\\,GB workspace", "0.23 MB/token", "the constant-workspace model"),
    # RETIRED 2026-10-20: these banned the plural "real checkpoints" while only
    # the 0.6B weights existed. The real 4B checkpoint has since been measured
    # (Table IV), so the phrase is true again and the rules were removed rather
    # than kept as a false alarm.
    #   ("real checkpoints", ...), ("real-weight runs", ...)
]

TEXT_FILES = ["paper/main.tex", "SUBMISSION.md", "STATUS.md",
              "docs/ARTIFACT.md", "docs/ENVIRONMENT.md"]


def main() -> int:
    bad = 0
    for rel in TEXT_FILES:
        p = ROOT / rel
        if not p.exists():
            continue
        s = p.read_text(encoding="utf-8")
        for tok, repl, why in STALE:
            if tok in s:
                idx = s.index(tok)
                line = s[:idx].count("\n") + 1
                print(f"  STALE  {rel}:{line}  {tok!r} -> {repl!r}  ({why})")
                bad += 1
    print(f"\n{'NO STALE NUMBERS' if bad == 0 else str(bad) + ' STALE OCCURRENCE(S)'}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
