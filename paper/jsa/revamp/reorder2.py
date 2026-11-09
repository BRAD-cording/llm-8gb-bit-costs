"""Move the "what makes the default the wrong choice" paragraph out of the new
executed-policy subsection (it belongs to the default-cost discussion, and it
forward-references Section VI-D, so it must precede it) and place it in front of
the robustness paragraph that closes subsection C.
"""

from pathlib import Path

MASTER = Path(r"C:\Users\HP\OneDrive\Desktop\Dev\harness\llm_8gb_bit_costs\paper\main.tex")
text = MASTER.read_text(encoding="utf-8")

P_DEFAULT = "What makes the default the wrong choice regardless is the length at which it stops"
P_ARTEFACT = "\\paragraph*{Is the policy an artefact of a weak weight quantiser?}"
SUB_E = "\\subsection{The cache is not the fragile side}"

for probe in (P_DEFAULT, P_ARTEFACT, SUB_E):
    assert text.count(probe) == 1, probe

d0 = text.index(P_DEFAULT)
e0 = text.index(SUB_E)
a0 = text.index(P_ARTEFACT)

block = text[d0:e0]
text = text[:d0] + text[e0:]
a1 = text.index(P_ARTEFACT)
text = text[:a1] + block + text[a1:]

MASTER.write_text(text, encoding="utf-8")

order = [
    ("artefact", P_ARTEFACT),
    ("default-wrong", P_DEFAULT),
    ("subsection D", "\\subsection{The recommended policy, executed}"),
    ("subsection E", SUB_E),
]
print({name: text.index(probe) for name, probe in order})
assert text.index(P_ARTEFACT) < text.index(P_DEFAULT) < text.index(SUB_E)
print("moved: default-wrong now precedes D")
