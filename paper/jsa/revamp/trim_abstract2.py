import re
from pathlib import Path

p = Path(r"C:\Users\HP\OneDrive\Desktop\Dev\harness\llm_8gb_bit_costs\paper\main.tex")
t = p.read_text(encoding="utf-8")

BEGIN, END = "\\begin{abstract}", "\\end{abstract}"
i = t.index(BEGIN) + len(BEGIN)
j = t.index(END, i)
head, abstract, tail = t[:i], t[i:j], t[j:]


def sub(text, phrase, new):
    pat = r"\s+".join(re.escape(tok) for tok in phrase.split())
    hits = list(re.finditer(pat, text))
    assert len(hits) == 1, (phrase[:45], len(hits))
    return text[: hits[0].start()] + new + text[hits[0].end():]


abstract = sub(
    abstract,
    "with published implementations on both sides, one cache bit costs",
    "on published implementations, one cache bit costs",
)
abstract = sub(
    abstract,
    "a $3.15\\times$ difference in the fraction lost, which moves the price of a weight bit by",
    "a $3.15\\times$ difference that moves the price of a weight bit by",
)
p.write_text(head + abstract + tail, encoding="utf-8")
print("written")
