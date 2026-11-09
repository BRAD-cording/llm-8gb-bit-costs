from pathlib import Path

p = Path(r"C:\Users\HP\OneDrive\Desktop\Dev\harness\llm_8gb_bit_costs\paper\main.tex")
t = p.read_bytes().decode("utf-8")
i = t.find("which is why Table")
print(repr(t[i:i + 90]))
print([hex(ord(c)) for c in t[i + 16:i + 24]])
