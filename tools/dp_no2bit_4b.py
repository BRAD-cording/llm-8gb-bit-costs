"""Disable the 2-bit option outright and re-solve.

A large finite penalty is used instead of deleting the key so the solver keeps
its four-choice structure; any module still assigned 2 bits would show up as an
absurd cost, so the outcome is auditable.
"""
import json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
items = json.load(open(os.path.join(ROOT, 'results', 'items_4b_nf4_clean.json'), encoding='utf-8'))['items']

PENALTY = 1e6
for it in items:
    it['err']['2'] = PENALTY

dst = os.path.join(ROOT, 'results', 'items_4b_nf4_no2bit.json')
json.dump({'items': items}, open(dst, 'w', encoding='utf-8'), indent=1)

for L in (8192, 16384, 32768):
    cp = subprocess.run([sys.executable, os.path.join(ROOT, 'src', 'alloc', 'dp_from_items.py'),
                         '--items', dst, '--L', str(L)],
                        capture_output=True, text=True, cwd=ROOT)
    out = (cp.stdout or cp.stderr).strip()
    print(f'=== L={L} rc={cp.returncode}')
    print(out[-650:])
