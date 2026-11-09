"""Clean the 4B nf4 per-item table and solve the allocator at the binding point.

Two steps:
  1. isotonic projection over the width axis, so a noise-level inversion cannot
     make a wider width look cheaper than a narrower one (cumulative minimum).
     Modules actually changed are flagged.
  2. hand the cleaned table to the exact multiple-choice knapsack and report
     uniform vs DP at L = 8192, 16384, 32768 on Device A.
"""
import json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'results', 'items_4b_nf4.jsonl')
CLEAN = os.path.join(ROOT, 'results', 'items_4b_nf4_clean.json')
WIDTHS = ['2', '3', '4', '8']

items, fixed, viol = [], 0, 0
for line in open(SRC, encoding='utf-8'):
    r = json.loads(line)
    err = {k: float(v) for k, v in r['err'].items()}
    out, prev, touched = {}, None, False
    for w in WIDTHS:
        v = err[w]
        if prev is not None and v > prev:
            v, touched = prev, True
            viol += 1
        out[w] = round(v, 6)
        prev = v
    fixed += touched
    items.append({'name': r['name'], 'kind': r['kind'], 'n_params': r['n_params'],
                  'err': out, 'isotonic_applied': touched})

json.dump(items, open(CLEAN, 'w', encoding='utf-8'), indent=1)
print(f'items {len(items)} | modules touched {fixed} | inversions clipped {viol}')

for L in (8192, 16384, 32768):
    cp = subprocess.run([sys.executable, os.path.join(ROOT, 'src', 'alloc', 'dp_from_items.py'),
                         '--items', CLEAN, '--L', str(L)],
                        capture_output=True, text=True, cwd=ROOT)
    print(f'--- L={L} rc={cp.returncode}')
    print((cp.stdout or cp.stderr).strip()[-900:])
