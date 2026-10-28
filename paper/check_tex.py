"""
Static sanity checks for the manuscript when no TeX toolchain is installed.

This is not a replacement for compiling. It catches the failure modes that are
mechanical and cheap to detect: unbalanced environments, table column counts that
do not match the number of `&` separators, unbalanced math delimiters, undefined
references and citations, and duplicate labels. Those are exactly the errors that
waste a submission cycle.

Usage:
    python paper/check_tex.py paper/main.tex
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

ENV_RE = re.compile(r"\\(begin|end)\{([^}]+)\}")


def strip_comments(src: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", ln) for ln in src.splitlines())


def check_envs(src: str) -> list[str]:
    problems = []
    stack: list[tuple[str, int]] = []
    for lineno, line in enumerate(src.splitlines(), 1):
        for kind, env in ENV_RE.findall(line):
            if kind == "begin":
                stack.append((env, lineno))
            else:
                if not stack:
                    problems.append(f"line {lineno}: \\end{{{env}}} with no matching \\begin")
                elif stack[-1][0] != env:
                    open_env, open_ln = stack[-1]
                    problems.append(
                        f"line {lineno}: \\end{{{env}}} closes \\begin{{{open_env}}} "
                        f"from line {open_ln}")
                    stack.pop()
                else:
                    stack.pop()
    for env, ln in stack:
        problems.append(f"line {ln}: \\begin{{{env}}} never closed")
    return problems


def check_tables(src: str) -> list[str]:
    """Column spec vs & count, per tabular row."""
    problems = []
    for m in re.finditer(r"\\begin\{tabular\}\{([^}]*)\}(.*?)\\end\{tabular\}", src, re.S):
        spec, body = m.group(1), m.group(2)
        start_line = src[:m.start()].count("\n") + 1
        # count columns: each l/c/r/p/m/b column is one; ignore @{}, |, >{..}
        cleaned = re.sub(r">\{[^}]*\}", "", spec)
        cleaned = re.sub(r"@\{[^}]*\}", "", cleaned)
        ncols = len(re.findall(r"[lcrpmb]", cleaned))
        if ncols == 0:
            continue
        for row in body.split("\\\\"):
            row = row.strip()
            if not row or row.startswith("\\") and "&" not in row:
                continue
            if "\\multicolumn" in row:
                continue
            if "\\cmidrule" in row or "\\toprule" in row or "\\bottomrule" in row:
                continue
            amps = row.count("&")
            if amps and amps + 1 != ncols:
                problems.append(
                    f"tabular at line {start_line}: {ncols} columns in spec but "
                    f"{amps + 1} cells in row: {row[:60]!r}")
    return problems


def check_math(src: str) -> list[str]:
    problems = []
    # remove escaped dollars
    s = src.replace(r"\$", "")
    # display math
    for opener, closer, name in ((r"\[", r"\]", "\\[ \\]"), (r"$$", r"$$", "$$")):
        if s.count(opener) != s.count(closer):
            problems.append(f"unbalanced {name}: {s.count(opener)} vs {s.count(closer)}")
    # inline math: pairs of unescaped $
    if s.count("$") % 2 != 0:
        problems.append(f"odd number of $ delimiters ({s.count('$')})")
    # \left / \right
    if s.count(r"\left") != s.count(r"\right"):
        problems.append(f"\\left/\\right mismatch: {s.count(r'\\left')} vs {s.count(r'\\right')}")
    return problems


def check_labels(src: str) -> list[str]:
    labels = re.findall(r"\\label\{([^}]+)\}", src)
    refs = set(re.findall(r"\\ref\{([^}]+)\}", src))
    problems = []
    dupes = [k for k, v in Counter(labels).items() if v > 1]
    for d in dupes:
        problems.append(f"duplicate label: {d}")
    for r in sorted(refs - set(labels)):
        problems.append(f"undefined reference: {r}")
    return problems


def check_cites(src: str, bib: Path) -> list[str]:
    if not bib.exists():
        return [f"missing bibliography: {bib}"]
    keys = set(re.findall(r"@\w+\{([^,]+),", bib.read_text(encoding="utf-8")))
    cited = set()
    for m in re.findall(r"\\cite\{([^}]+)\}", src):
        cited.update(k.strip() for k in m.split(","))
    return [f"citation with no bib entry: {c}" for c in sorted(cited - keys)]


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "paper/main.tex")
    src = strip_comments(path.read_text(encoding="utf-8"))
    bib = path.parent / "refs.bib"

    checks = {
        "environments": check_envs(src),
        "tables": check_tables(src),
        "math": check_math(src),
        "labels": check_labels(src),
        "citations": check_cites(src, bib),
    }

    total = 0
    for name, problems in checks.items():
        status = "ok" if not problems else f"{len(problems)} problem(s)"
        print(f"{name:14s} {status}")
        for p in problems:
            print(f"    {p}")
        total += len(problems)

    print(f"\n{'CLEAN' if total == 0 else str(total) + ' PROBLEM(S)'}")
    return 0 if total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
