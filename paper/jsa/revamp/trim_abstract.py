"""Trim the abstract to 250 words while keeping the executed-policy clause.

Matches phrases independent of the source line wrapping.
"""
import re
from pathlib import Path

MASTER = Path(r"C:\Users\HP\OneDrive\Desktop\Dev\harness\llm_8gb_bit_costs\paper\main.tex")
t = MASTER.read_text(encoding="utf-8")


def sub_phrase(text, phrase, new):
    pattern = r"\s+".join(re.escape(tok) for tok in phrase.split())
    hits = list(re.finditer(pattern, text))
    assert len(hits) == 1, (phrase[:50], len(hits))
    return text[: hits[0].start()] + new + text[hits[0].end():]


edits = [
    (
        "On this device class both choices are wrong, and this paper replaces them with a measured decision.",
        "Both are wrong on this device class, and this paper replaces them with a measured decision.",
    ),
    (
        "Three inputs decide the weight/cache split, and none of them is reported in the work that asks the question.",
        "Three inputs decide the weight/cache split, and none is reported in the work that asks the question.",
    ),
    (
        "reproduces on WikiText-2, and a 2-bit passkey experiment locates where it stops.",
        "reproduces on WikiText-2.",
    ),
    (
        "The allocator that follows is textbook, and we say so; the inputs are not, and on both devices the resulting policy compresses the cache and leaves the weights at 8 bits.",
        "The allocator is textbook; the inputs are not, and on both devices its policy compresses the cache and keeps the weights at 8 bits.",
    ),
]

for old, new in edits:
    t = sub_phrase(t, old, new)

MASTER.write_text(t, encoding="utf-8")
print("abstract edited")
