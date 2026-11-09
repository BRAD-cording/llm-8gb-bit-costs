import re
import textwrap
from pathlib import Path

p = Path(r"C:\Users\HP\OneDrive\Desktop\Dev\harness\llm_8gb_bit_costs\paper\main.tex")
t = p.read_text(encoding="utf-8")

BEGIN, END = "\\begin{abstract}\n", "\\end{abstract}"
i = t.index(BEGIN) + len(BEGIN)
j = t.index(END, i)
body = t[i:j].strip()

body = body.replace("on published implementations, one cache bit costs",
                    "across published implementations, one cache bit costs")

paras = [re.sub(r"\s+", " ", para).strip() for para in body.split("\n\n") if para.strip()]
wrapped = "\n\n".join(textwrap.fill(para, width=79) for para in paras)

t = t[:i] + wrapped + "\n\n\n" + t[j:].lstrip("\n")
p.write_text(t, encoding="utf-8")

print("lines:", len(wrapped.splitlines()), "| max width:", max(len(x) for x in wrapped.splitlines()))
