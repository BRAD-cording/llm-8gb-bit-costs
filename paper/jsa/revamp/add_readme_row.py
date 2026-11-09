from pathlib import Path

p = Path(r"C:\Users\HP\OneDrive\Desktop\Dev\harness\llm_8gb_bit_costs\README.md")
t = p.read_text(encoding="utf-8")

anchor = "| Device-dependent capacity, and the edge module's measured 3k ceiling |"
row = (
    "| The same policy on the unified-memory device: each halving of the per-token cache buys one "
    "context doubling (8k/16k/32k at fp16/q8_0/q4_0), and the device refuses a 1,792 MiB allocation "
    "while its peak sits at 5,235 of 7,620 MB | `tools/jetson_c2/*`, `experiments/make_c2_jetson_summary.py` "
    "| `results/c2_jetson/`, `results/c2_jetson_summary.json`, `results/C2_JETSON.md` |\n"
)

assert t.count(anchor) == 1
assert "c2_jetson_summary.json" not in t.split("## Layout")[0]
t = t.replace(anchor, row + anchor)
p.write_text(t, encoding="utf-8")
print("rows now:", t.count("\n| ") )
