from pathlib import Path

p = Path(r"C:\Users\HP\OneDrive\Desktop\Dev\harness\llm_8gb_bit_costs\paper\main.tex")
t = p.read_bytes().decode("utf-8")

bad = "Table~" + chr(13) + "ef{tab:exec-orin}"
good = "Table~" + chr(92) + "ref{tab:exec-orin}"
assert t.count(bad) == 1, t.count(bad)
t = t.replace(bad, good, 1)

stray = sum(1 for i, ch in enumerate(t) if ch == chr(13) and (i + 1 >= len(t) or t[i + 1] != chr(10)))
p.write_bytes(t.encode("utf-8"))
print("fixed; stray CRs now", stray)
