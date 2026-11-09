"""Port the v3 revamp into the IEEEtran master (paper/main.tex).

paper/main.tex is the source of truth: make_jsa.py copies its body verbatim into
jsa/main.tex. Editing jsa/main.tex alone would be lost on the next regeneration,
so the edits are applied here and the JSA copy is rebuilt from this file.

New text is lifted from the already-reviewed jsa/main.tex rather than retyped.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

PAPER = Path(r"C:\Users\HP\OneDrive\Desktop\Dev\harness\llm_8gb_bit_costs\paper")
MASTER = PAPER / "main.tex"
JSA = PAPER / "jsa" / "main.tex"
FRONT = PAPER / "jsa" / "revamp" / "FRONT_v3.tex"
INTRO = PAPER / "jsa" / "revamp" / "INTRO_v3.tex"

master = MASTER.read_text(encoding="utf-8")
jsa = JSA.read_text(encoding="utf-8")
shutil.copy2(MASTER, MASTER.with_suffix(".tex.bak"))

# ---- lift new prose out of the JSA copy -------------------------------------
exec_start = jsa.index("\\subsection{The recommended policy, executed}")
exec_end = jsa.index("What makes the default the wrong choice regardless is the length")
EXEC_BLOCK = jsa[exec_start:exec_end].rstrip() + "\n\n"

lim_start = jsa.index("\\item \\textbf{Two cache-compression measurements, neither of them our kernel.}")
lim_end = jsa.index("\\item \\textbf{No kernel engineering.}")
LIM_ITEM = jsa[lim_start:lim_end].rstrip() + "\n\n"

assert "tab:exec" in EXEC_BLOCK and "per-32-block scheme" in LIM_ITEM

# ---- 1. abstract ------------------------------------------------------------
front_lines = FRONT.read_text(encoding="utf-8").splitlines()
abs_body = []
inside = False
for line in front_lines:
    if line.startswith("\\begin{abstract}"):
        inside = True
        continue
    if line.startswith("\\end{abstract}"):
        break
    if inside:
        abs_body.append(line)
ABSTRACT = "\n".join(abs_body).strip()
assert ABSTRACT.startswith("Quantisation fits"), ABSTRACT[:40]
anchor = "leaves the weights at 8 bits."
ABSTRACT = ABSTRACT.replace(
    anchor,
    "leaves the weights at 8 bits. We execute that policy on a shipped runtime,\n"
    "where it is what makes a 32k context fit at all.",
    1,
)

a0 = master.index("\\begin{abstract}") + len("\\begin{abstract}")
a1 = master.index("\\end{abstract}")
master = master[:a0] + "\n" + ABSTRACT + "\n" + master[a1:]

# ---- 2. introduction --------------------------------------------------------
intro_lines = INTRO.read_text(encoding="utf-8").splitlines()
i0 = next(n for n, l in enumerate(intro_lines) if l.startswith("\\section{Introduction}"))
INTRO_TEXT = "\n".join(intro_lines[i0:]).strip() + "\n\n"

s0 = master.index("\\section{Introduction}")
s1 = master.index("exhaustive literature coverage")
s1 = master.index("\n\n", s1) + 2
master = master[:s0] + INTRO_TEXT + master[s1:]

# ---- 3. the executed-policy subsection --------------------------------------
anchor = "What makes the default the wrong choice regardless is the length"
assert anchor in master, "subsection insertion anchor not found in master"
master = master.replace(anchor, EXEC_BLOCK + anchor, 1)

# ---- 4. the hypothetical packed-cache sentence ------------------------------
old = ("A packed 4-bit cache would need\n"
       "$1.8$\\,GB where 16-bit needs $7.2$\\,GB at that length. The honest summary is")
new = ("Section~\\ref{sec:results-exec} measures the\n"
       "same crossing on a packed runtime rather than estimating it: a 4B model at 32k\n"
       "allocates 9{,}033\\,MiB with an fp16 cache and does not complete, and 6{,}873\\,MiB\n"
       "with the cache at q8\\_0 and does. The honest summary is")
assert old in master, "packed-cache sentence not found in master"
master = master.replace(old, new, 1)

# ---- 5. the limitations item ------------------------------------------------
old_start = master.index("\\item \\textbf{The cache-compression measurement is partly simulated.}")
old_end = master.index("\\item \\textbf{No kernel engineering.}")
master = master[:old_start] + LIM_ITEM + master[old_end:]

MASTER.write_text(master, encoding="utf-8")

# ---- report -----------------------------------------------------------------
print("lines:", len(master.splitlines()))
print("exec section:", master.count("\\subsection{The recommended policy, executed}"))
print("tab:exec label:", master.count("\\label{tab:exec}"))
print("stale limitation:", master.count("partly simulated.} Our"))
print("todo uses:", len(re.findall(r"\\todo\{", master)))
