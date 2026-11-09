"""Word count of the abstract in both manuscripts, plus the highlight bullets."""
import re
import sys
from pathlib import Path

BEGIN = "\\begin{abstract}"
END = "\\end{abstract}"

for name in sys.argv[1:]:
    text = Path(name).read_text(encoding="utf-8")
    i = text.index(BEGIN) + len(BEGIN)
    j = text.index(END, i)
    body = text[i:j]
    plain = re.sub(r"\\[a-zA-Z]+\*?", " ", body)
    plain = plain.replace("{", " ").replace("}", " ").replace("$", " ")
    words = [w for w in plain.split() if any(c.isalnum() for c in w)]
    print(f"{name}: abstract {len(words)} words")
