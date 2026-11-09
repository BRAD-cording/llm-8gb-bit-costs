"""Splice the v3 front matter and introduction into jsa/main.tex."""

import re
import shutil
from pathlib import Path

ROOT = Path(r"C:\Users\HP\OneDrive\Desktop\Dev\harness\llm_8gb_bit_costs\paper")
MAIN = ROOT / "jsa" / "main.tex"
INTRO = ROOT / "jsa" / "revamp" / "INTRO_v3.tex"
FRONT = ROOT / "jsa" / "revamp" / "FRONT_v3.tex"

shutil.copy2(MAIN, MAIN.with_suffix(".tex.bak"))

main = MAIN.read_text(encoding="utf-8").splitlines()
intro = INTRO.read_text(encoding="utf-8").splitlines()
front = FRONT.read_text(encoding="utf-8").splitlines()


def block(lines, start, end, start_off=0, end_off=0):
    i = next(n for n, l in enumerate(lines) if l.startswith(start))
    j = next(n for n in range(i, len(lines)) if lines[n].startswith(end))
    return i + start_off, j + end_off


# --- abstract ---------------------------------------------------------------
new_abs = []
inside = False
for line in front:
    if line.startswith("\\begin{abstract}"):
        inside = True
    if inside:
        new_abs.append(line)
    if line.startswith("\\end{abstract}"):
        break

assert new_abs[0] == "\\begin{abstract}" and new_abs[-1] == "\\end{abstract}", new_abs[:2]

joined = "\n".join(new_abs)
anchor = "leaves the weights at 8 bits."
assert anchor in joined
joined = joined.replace(
    anchor,
    "leaves the weights at 8 bits. We execute that policy on a shipped runtime,\n"
    "where it is what makes a 32k context fit at all.",
    1,
)
new_abs = joined.splitlines()

a0, a1 = block(main, "\\begin{abstract}", "\\end{abstract}")
main[a0:a1 + 1] = new_abs

# --- highlights -------------------------------------------------------------
new_hl = []
inside = False
for line in front:
    if line.startswith("\\section*{Highlights}"):
        inside = True
    if inside:
        new_hl.append(line)
    if inside and line.startswith("\\end{itemize}"):
        break
assert new_hl[-1] == "\\end{itemize}"

h0, h1 = block(main, "\\section*{Highlights}", "\\end{itemize}")
main[h0:h1 + 1] = new_hl

# --- introduction -----------------------------------------------------------
new_intro = []
inside = False
for line in intro:
    if line.startswith("\\section{Introduction}"):
        inside = True
    if inside:
        new_intro.append(line)
assert new_intro[0] == "\\section{Introduction}"

i0, _ = block(main, "\\section{Introduction}", "% ---")
j0 = next(n for n, l in enumerate(main) if "exhaustive literature coverage" in l)
main[i0:j0 + 1] = new_intro

MAIN.write_text("\n".join(main) + "\n", encoding="utf-8")

text = MAIN.read_text(encoding="utf-8")
print("lines:", len(text.splitlines()))
print("todo left:", text.count("\\todo"))
print("abstract count:", text.count("\\begin{abstract}"))
print("intro count:", text.count("\\section{Introduction}"))
print("highlights count:", text.count("\\section*{Highlights}"))
print("results-exec refs:", text.count("sec:results-exec"))
print("tab:exec defs:", len(re.findall(r"\\label\{tab:exec\}", text)))
