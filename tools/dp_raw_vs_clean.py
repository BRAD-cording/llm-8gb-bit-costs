"""Solve the DP on the *raw* per-item table as well as the isotonic-cleaned one.

The isotonic projection used to clean the table is cumulative-min over the
four widths, which flattens 168 of 252 rows (err[8] is replaced by err[4]).
That is fine for deciding whether a module is sensitive, but it biases the
costs downward, and the DP solution depends on those costs.  This script
therefore solves the same 4B@32k allocation twice:

    raw        -- the jsonl as measured, monotonicity repaired only where the
                  violation exceeds 0.5 pp (one module)
    isotonic   -- results/items_4b_nf4_clean.json as used so far

and reports the two predicted costs so the size of the projection artefact is
visible.
"""
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from alloc.dp_from_items import budget_mb, load_items, solve

BITS = [2, 3, 4, 8]
LM_HEAD = 155582464


def repair(err):
    """Cumulative min, but only where the violation exceeds 0.5 pp."""
    out = dict(err)
    for i in range(len(BITS) - 1):
        lo, hi = str(BITS[i]), str(BITS[i + 1])
        if out[lo] > out[hi] + 0.5:
            out[lo] = out[hi]
    return out


def to_items(rows, repair_fn=None):
    items = []
    for r in rows:
        err = r['err']
        if repair_fn:
            err = repair_fn(err)
        items.append({'name': r['name'], 'kind': r['kind'],
                      'n_params': r['n_params'],
                      'err': {k: float(v) for k, v in err.items()}})
    return items


def cost_pp(items, assign):
    tot = 0.0
    for it in items:
        tot += it['err'].get(str(assign[it['name']]), 0.0)
    x = max(tot / 100.0, -0.95)
    return 100.0 * (math.exp(math.log1p(x)) - 1.0)


def main():
    raw = [json.loads(l) for l in open(os.path.join(
        ROOT, 'results', 'items_4b_nf4.jsonl'), encoding='utf-8') if l.strip()]
    clean = json.load(open(os.path.join(
        ROOT, 'results', 'items_4b_nf4_clean.json'), encoding='utf-8'))['items']
    L = 32768
    B = budget_mb(L, lm_head_params=LM_HEAD)
    print(f'items raw {len(raw)} clean {len(clean)} budget {B:.1f} MB')

    out = {'budget_mb': round(B, 1), 'L': L}
    for tag, rows, fn in (('raw_minimal_repair', raw, repair),
                          ('isotonic_clean', clean, None)):
        items = to_items(rows, fn)
        uni = {it['name']: 4 for it in items}
        names = [it['name'] for it in items]
        sol = solve(items, B, BITS)
        d = {'uniform4_pp': round(cost_pp(items, uni), 3),
             'dp_pp': round(cost_pp(items, {names[i]: b for i, b in sol['assign'].items()}), 3),
             'dp_gain_over_uniform_pp': round(
                 cost_pp(items, uni) - cost_pp(items, {names[i]: b for i, b in sol['assign'].items()}), 3),
             'dp_bytes_mb': round(sol['bytes_mb'], 1),
             'hist': None}
        print(tag, json.dumps(d, ensure_ascii=False))
        out[tag] = d
    json.dump(out, open(os.path.join(ROOT, 'results', 'dp_raw_vs_clean.json'),
                        'w', encoding='utf-8'), indent=1)
    print('WROTE results/dp_raw_vs_clean.json')


if __name__ == '__main__':
    main()
