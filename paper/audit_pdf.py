"""Comprehensive pre-submission audit of the PDFs.

Checks that need the rendered file rather than the source: text extraction
quality, unresolved references, placeholder text, author/email/affiliation
strings, abstract length, CRediT wording, and the data-availability link.
"""
from __future__ import annotations

import re
import sys

import fitz

TARGETS = [
    ("paper/main.pdf", "IEEE IoT-J"),
    ("paper/jsa/main.pdf", "Elsevier JSA"),
]

CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def audit(path: str, label: str) -> dict:
    doc = fitz.open(path)
    text = "".join(p.get_text() for p in doc)
    raw = text
    problems: list[str] = []

    # 1. extraction quality
    ctrl = CTRL.findall(raw)
    if ctrl:
        problems.append(f"{len(ctrl)} control characters in extracted text")

    # 2. unresolved cross-references
    qmarks = len(re.findall(r"\?\?", raw))
    if qmarks:
        problems.append(f'{qmarks} unresolved reference(s) ("??")')

    # 3. placeholders
    for token in ("TODO", "XXX", "FIXME", "Lorem"):
        if token in raw:
            problems.append(f"placeholder text: {token}")

    # 4. stray non-English working notes
    cjk = re.findall(r"[\u4e00-\u9fff]+", raw)
    if cjk:
        problems.append(f"{len(cjk)} CJK fragment(s), first: {cjk[0][:20]}")

    # 5. names
    names = {
        "Syafiq": "Muhammad Syafiq" in raw,
        "Ghazali": "Ghazali" in raw,
        "Pozi": "Pozi" in raw,
        "no truncated 'Syaq bin'": "Syaq bin" not in raw,
    }
    for k, ok in names.items():
        if not ok:
            problems.append(f"author name check failed: {k}")

    # 6. email
    if "zhang_yibo@ahsgs.uum.edu.my" not in raw:
        problems.append("corresponding e-mail missing")

    # 7. affiliation
    if "Universiti Utara Malaysia" not in raw:
        problems.append("affiliation missing")

    # 8. CRediT
    m = re.search(r"Pozi:\s*([^.]*)\.", raw)
    credit_coauthor = m.group(1).strip() if m else "<not found>"
    if m and credit_coauthor != "Supervision":
        problems.append(f"co-author CRediT is {credit_coauthor!r}, expected 'Supervision'")

    # 9. abstract length
    a0 = raw.find("Abstract")
    a1 = raw.find("Keywords") if raw.find("Keywords") > a0 else raw.find("Index Terms")
    abstract = raw[a0:a1] if a0 >= 0 and a1 > a0 else ""
    words = len(abstract.split())
    if not (120 <= words <= 260):
        problems.append(f"abstract length {words} words outside 120-260")

    # 10. data availability. A long \url breaks across lines, so normalise
    # hyphen-newline joins before matching, and confirm the link annotation too.
    flat = re.sub(r"-\s*\n\s*", "-", raw)
    flat = re.sub(r"\s+", "", flat)
    url = "https://github.com/BRAD-cording/llm-8gb-bit-costs"
    if url.replace(" ", "") not in flat:
        problems.append("data availability URL missing from the text")
    link_uris = {l.get("uri") for p in doc for l in p.get_links() if l.get("uri")}
    if not any("llm-8gb-bit-costs" in (u or "") for u in link_uris):
        problems.append("data availability URL is not a clickable link")

    return {
        "label": label,
        "pages": len(doc),
        "words": len(re.findall(r"\w+", raw)),
        "abstract_words": words,
        "credit_coauthor": credit_coauthor,
        "problems": problems,
    }


def main() -> int:
    failed = 0
    for path, label in TARGETS:
        try:
            r = audit(path, label)
        except FileNotFoundError:
            print(f"{label}: MISSING FILE {path}")
            failed += 1
            continue
        print(f"=== {label} ({path}) ===")
        print(f"  pages {r['pages']}, {r['words']} words, abstract {r['abstract_words']} words")
        print(f"  co-author CRediT: {r['credit_coauthor']}")
        if r["problems"]:
            failed += 1
            for p in r["problems"]:
                print(f"  PROBLEM: {p}")
        else:
            print("  no problems found")
        print()
    print("AUDIT", "FAILED" if failed else "CLEAN")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
