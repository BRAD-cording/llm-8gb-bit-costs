"""Reorder Results so the executed-policy subsection does not adopt an unrelated
paragraph: move "Is the policy an artefact of a weak weight quantiser?" in front
of "The recommended policy, executed" instead of trailing it.

Reads and writes paper/main.tex; jsa/main.tex is regenerated from it afterwards.
"""

from pathlib import Path

MASTER = Path(r"C:\Users\HP\OneDrive\Desktop\Dev\harness\llm_8gb_bit_costs\paper\main.tex")
text = MASTER.read_text(encoding="utf-8")

SUB = "\\subsection{The recommended policy, executed}"
PARA = "\\paragraph*{Is the policy an artefact of a weak weight quantiser?}"
NEXT = "\\subsection{The cache is not the fragile side}"

assert text.count(SUB) == 1 and text.count(PARA) == 1 and text.count(NEXT) == 1

p0 = text.index(PARA)
p1 = text.index(NEXT)
block = text[p0:p1]
text = text[:p0] + text[p1:]

s0 = text.index(SUB)
text = text[:s0] + block + text[s0:]

MASTER.write_text(text, encoding="utf-8")

order = [text.index(SUB), text.index(PARA), text.index(NEXT)]
print("order subsection < paragraph < next:", order[0] < order[1] < order[2])
print("paragraph before subsection:", text.index(PARA) < text.index(SUB))
