"""Structural lint for the manuscript: what a splice breaks, without a TeX engine.

Checks, in the order that a bad splice actually fails:
  1. brace balance            (a lost '}' swallows the rest of the document)
  2. begin/end environment match
  3. \\ref / \\eqref / \\autoref targets that have no \\label
  4. duplicate labels
  5. \\cite keys missing from the .bib
  6. math-mode $ parity per paragraph

Usage: python jsa/revamp/lint_tex.py [path/to/main.tex]
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

REF = re.compile(r"\\(?:ref|eqref|autoref|pageref|Cref|cref)\{([^}]*)\}")
LABEL = re.compile(r"\\label\{([^}]*)\}")
CITE = re.compile(r"\\cite[a-zA-Z]*\{([^}]*)\}")
BIBKEY = re.compile(r"@\w+\s*\{\s*([^,\s]+)")


def strip_comment(line: str) -> str:
    """Drop everything after an unescaped %."""
    out, i = [], 0
    while i < len(line):
        ch = line[i]
        if ch == "\\" and i + 1 < len(line):
            out.append(line[i:i + 2])
            i += 2
            continue
        if ch == "%":
            break
        out.append(ch)
        i += 1
    return "".join(out)


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else Path("jsa/main.tex")
    raw = path.read_text(encoding="utf-8").splitlines()
    lines = [strip_comment(l) for l in raw]
    problems = 0

    def report(msg: str) -> None:
        nonlocal problems
        problems += 1
        print("  ! " + msg)

    # 1. braces -------------------------------------------------------------
    print("[1] brace balance")
    depth, worst = 0, 0
    for n, line in enumerate(lines, 1):
        depth += line.count("{") - line.count("}")
        worst = min(worst, depth)
        if depth < 0:
            report(f"line {n}: brace depth went negative ({depth})")
            depth = 0
    if depth > 0:
        report(f"file ends with {depth} unclosed '{{'")
    if problems == 0:
        print("  ok")

    # 2. environments -------------------------------------------------------
    print("[2] environments")
    before = problems
    stack: list[tuple[str, int]] = []
    for n, line in enumerate(lines, 1):
        for m in re.finditer(r"\\begin\{([^}]*)\}", line):
            stack.append((m.group(1), n))
        for m in re.finditer(r"\\end\{([^}]*)\}", line):
            name = m.group(1)
            if not stack:
                report(f"line {n}: \\end{{{name}}} with nothing open")
            elif stack[-1][0] != name:
                report(f"line {n}: \\end{{{name}}} closes \\begin{{{stack[-1][0]}}} from line {stack[-1][1]}")
                stack.pop()
            else:
                stack.pop()
    for name, n in stack:
        report(f"\\begin{{{name}}} at line {n} never closed")
    if problems == before:
        print("  ok")

    # 3/4. labels ------------------------------------------------------------
    print("[3] references resolve")
    labels = [(m.group(1), n) for n, line in enumerate(lines, 1) for m in LABEL.finditer(line)]
    have = {name for name, _ in labels}
    before = problems
    for n, line in enumerate(lines, 1):
        for m in REF.finditer(line):
            for target in (t.strip() for t in m.group(1).split(",")):
                if target and target not in have:
                    report(f"line {n}: \\ref{{{target}}} has no \\label")
    dupes = {name: c for name, c in Counter(name for name, _ in labels).items() if c > 1}
    for name in dupes:
        where = [str(n) for candidate, n in labels if candidate == name]
        report(f"duplicate \\label{{{name}}} at lines {', '.join(where)}")
    if problems == before:
        print("  ok")

    # 5. citations -----------------------------------------------------------
    print("[4] citations")
    bibs = sorted(path.parent.glob("*.bib"))
    if not bibs:
        print("  (no .bib beside the document)")
    else:
        keys = set()
        for bib in bibs:
            keys |= set(BIBKEY.findall(bib.read_text(encoding="utf-8", errors="replace")))
        cited = {k.strip() for _, line in enumerate(lines, 1) for m in CITE.finditer(line) for k in m.group(1).split(",") if k.strip()}
        missing = sorted(cited - keys)
        for key in missing:
            report(f"\\cite{{{key}}} not in {', '.join(b.name for b in bibs)}")
        print(f"  {len(cited)} keys cited, {len(keys)} in bib, {len(missing)} missing")

    # 6. dollar parity -------------------------------------------------------
    print("[5] math delimiters")
    before = problems
    para_start, parity = 1, 0
    for n, line in enumerate(lines, 1):
        if not line.strip():
            if parity % 2:
                report(f"paragraph starting at line {para_start}: odd number of '$'")
            para_start, parity = n + 1, 0
            continue
        parity += line.count("$")
    if parity % 2:
        report(f"paragraph starting at line {para_start}: odd number of '$'")
    if problems == before:
        print("  ok")

    print()
    print(f"{problems} problem(s)" if problems else "clean")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
