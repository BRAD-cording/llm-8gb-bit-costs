"""Cross-check the exchange-rate numbers against tools/feasible_region.py.

The audit script lists claims; this checks that the three places quoting the
"price of a weight bit" agree with each other and with the computation.
A stale 3.5x survived next to a fresh 3.7x until this existed.
"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

import feasible_region as fr  # noqa: E402

data = json.loads((ROOT / "results" / "kv_crossover.json").read_text("utf-8"))
info = data["Qwen/Qwen3-4B"]["info"]

b4060 = fr.DEVICES["RTX4060"]["free_GiB"] * 1.0737
bjets = fr.DEVICES["JetsonOrinNanoSuper"]["free_GiB"] * 1.0737

r4060 = fr.l_max(b4060, info["params_M"], info["kv_bytes_per_token"], 4, 16) / \
    fr.l_max(b4060, info["params_M"], info["kv_bytes_per_token"], 8, 16)
rjets = fr.l_max(bjets, info["params_M"], info["kv_bytes_per_token"], 4, 16) / \
    fr.l_max(bjets, info["params_M"], info["kv_bytes_per_token"], 8, 16)
cross = rjets / r4060

print(f"computed: laptop x{r4060:.2f}  jetson x{rjets:.2f}  ratio {cross:.2f}x")

tex = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
expect = {
    "laptop rate 1.57": "$1.57\\times$",
    "jetson rate 5.76": "$5.76\\times$",
    "cross-device 3.7": "$3.7\\times$",
}
bad = 0
for label, s in expect.items():
    n = tex.count(s)
    ok = n >= 1
    print(f"  {label:22s} appears {n}x  {'ok' if ok else 'MISSING'}")
    bad += 0 if ok else 1

# no stale 3.5x should remain
stale = tex.count("$3.5\\times$")
print(f"  stale 3.5x occurrences  {stale}  {'ok' if stale == 0 else 'STALE'}")
bad += 0 if stale == 0 else 1

print(f"\n{'CONSISTENT' if bad == 0 else str(bad) + ' PROBLEM(S)'}")
sys.exit(0 if bad == 0 else 1)
