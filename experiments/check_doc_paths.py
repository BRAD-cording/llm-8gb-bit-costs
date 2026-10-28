"""
Check that every path named in the documentation actually exists.

An artifact whose README points at files that were renamed or never created is
worse than one with no README: the reader concludes the whole thing is
unmaintained. This walks the markdown docs, extracts back-ticked paths that look
like repository files, and verifies they resolve.

Usage:  python experiments/check_doc_paths.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DOCS = ["DELIVERABLES.md", "STATUS.md", "SUBMISSION.md", "PLAN.md",
        "docs/ARTIFACT.md", "docs/ENVIRONMENT.md"]

# a backticked token is treated as a path if it has a known extension and no
# wildcard, placeholder or URL
EXT = (".py", ".md", ".tex", ".bib", ".json", ".csv", ".pdf", ".txt", ".sh", ".png")
TOKEN = re.compile(r"`([^`\n]+)`")


def looks_like_path(tok: str) -> bool:
    if any(c in tok for c in ("*", "<", ">", "|", "=", " ")):
        return False
    if tok.startswith(("http://", "https://")):
        return False
    return tok.endswith(EXT)


def main() -> int:
    # index every file by basename: docs often cite a bare filename whose
    # directory is implied by the surrounding sentence
    by_name: dict[str, list[Path]] = {}
    for f in ROOT.rglob("*"):
        if f.is_file() and ".tmp" not in f.parts:
            by_name.setdefault(f.name, []).append(f)

    missing, checked = [], 0
    for rel in DOCS:
        p = ROOT / rel
        if not p.exists():
            print(f"  MISSING DOC  {rel}")
            missing.append(rel)
            continue
        for tok in TOKEN.findall(p.read_text(encoding="utf-8")):
            if not looks_like_path(tok):
                continue
            checked += 1
            cand = ROOT / tok
            if cand.exists() or (ROOT / "paper" / tok).exists():
                continue
            if tok.rsplit("/", 1)[-1] in by_name:
                continue
            missing.append(f"{rel}: {tok}")

    print(f"checked {checked} path references across {len(DOCS)} documents")
    if missing:
        print(f"\n{len(missing)} unresolved:")
        for m in sorted(set(missing)):
            print("  " + m)
    print(f"\n{'ALL PATHS RESOLVE' if not missing else str(len(missing)) + ' UNRESOLVED'}")
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
