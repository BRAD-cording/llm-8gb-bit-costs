"""Robustness check on the 4B binding-point result.

The raw DP is free to send insensitive modules to 2 bits, and for those modules
the measured 2-bit cost is inside the noise floor, so the gain it reports is an
upper bound. This variant removes the 2-bit option (its cost is set equal to the
3-bit cost), which makes the comparison conservative.
"""
import json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = json.load(open(os.path.join(ROOT, 'results', 'items_4b_nf4_clean.json'), encoding='utf-8'))['items']

for it in src:
    e = it['err']
    e['2'] = max(e['2'], e['3'])

dst = os.path.join(ROOT, 'results', 'items_4b_nf4_nov2.json')
json.dump({'items': src}, open(dst, 'w', encoding='utf-8'), indent=1)

for L in (8192, 16384, 32768):
    cp = subprocess.run([sys.executable, os.path.join(ROOT, 'src', 'alloc', 'dp_from_items.py'),
                         '--items', dst, '--L', str(L)],
                        capture_output=True, text=True, cwd=ROOT)
    out = (cp.stdout or cp.stderr).strip()
    print(f'=== L={L} rc={cp.returncode}')
    print(out[-700:])
