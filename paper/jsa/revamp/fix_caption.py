from pathlib import Path

p = Path(r"C:\Users\HP\OneDrive\Desktop\Dev\harness\llm_8gb_bit_costs\paper\main.tex")
t = p.read_bytes().decode("utf-8")
bad = "an em dash means the allocation was refused in all three attempts."
good = "``refused'' means the allocation failed in all three attempts."
assert t.count(bad) == 1, t.count(bad)
p.write_bytes(t.replace(bad, good, 1).encode("utf-8"))
print("caption fixed")
