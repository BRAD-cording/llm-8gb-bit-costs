"""Solve the 4B@32k allocation on the raw per-item table and on the cleaned one.

Cost aggregation is done in the nats domain, which is what the per-item err
values encode: err[b] = 100 * (exp(dNLL_b) - 1), so dNLL_b = log1p(err/100)
and the additive model sums dNLL, not the percentages.

A single module's 4-bit measurement can come out slightly negative (its PPL
is a hair better than the nf4 base), so the sum can be negative; the total is
reported in nats as the primary figure and in percent only when the exponent
is representable (|total| < 20).
"""
import json
import math
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from alloc.dp_from_items import budget_mb, solve

BITS = [2, 3, 4, 8]
LM_HEAD = 155582464
FLOOR = -99.0


def repair(err):
    """Enforce monotonicity only where the violation exceeds 0.5 pp."""
    out = {str(k): float(v) for k, v in err.items()}
    for i in range(len(BITS) - 1):
        lo, hi = str(BITS[i]), str(BITS[i + 1])
        if lo in out and hi in out and out[lo] > out[hi] + 0.5:
            out[lo] = out[hi]
    return out


def to_items(rows, fn=None):
    items = []
    for r in rows:
        err = fn(r['err']) if fn else {str(k): float(v) for k, v in r['err'].items()}
        items.append({'name': r['name'], 'kind': r['kind'],
                      'n_params': int(r['n_params']), 'err': err})
    return items


def total_nats(items, by_name):
    tot = 0.0
    for it in items:
        e = max(it['err'].get(str(by_name[it['name']]), 0.0), FLOOR)
        tot += math.log1p(e / 100.0)
    return tot


def report(tag, items, by_name, extra):
    nats = total_nats(items, by_name)
    d = {'total_nats': round(nats, 5)}
    if abs(nats) < 20:
        d['ppl_change_pct'] = round(100.0 * (math.exp(nats) - 1.0), 3)
    else:
        d['ppl_change_pct'] = 'off-scale'
    d.update(extra)
    d['hist'] = dict(Counter(by_name.values()))
    print(tag, json.dumps(d, ensure_ascii=False))
    return d


def main():
    raw = [json.loads(l) for l in open(os.path.join(
        ROOT, 'results', 'items_4b_nf4.jsonl'), encoding='utf-8') if l.strip()]
    clean = json.load(open(os.path.join(
        ROOT, 'results', 'items_4b_nf4_clean.json'), encoding='utf-8'))['items']
    L = 32768
    B = budget_mb(L, lm_head_params=LM_HEAD)
    print(f'raw {len(raw)} clean {len(clean)} budget {B:.1f} MB')

    out = {'L': L, 'budget_mb': round(B, 1), 'variants': {}}
    for tag, rows, fn in (('raw_minimal_repair', raw, repair),
                          ('isotonic_clean', clean, None)):
        items = to_items(rows, fn)
        names = [it['name'] for it in items]
        uni = {it['name']: 4 for it in items}
        u = report(tag + '/uniform4', items, uni, {'bytes_mb': None})
        sol = solve(items, B, BITS)
        if sol is None:
            print(tag, 'DP INFEASIBLE')
            continue
        by_name = {names[i]: b for i, b in sol['assign'].items()}
        d = report(tag + '/dp', items, by_name, {'bytes_mb': round(sol['bytes_mb'], 1)})
        out['variants'][tag] = {
            'uniform4': u, 'dp': d,
            'dp_gain_nats': round(u['total_nats'] - d['total_nats'], 5),
            'dp_gain_pct_points': (round(u['ppl_change_pct'] - d['ppl_change_pct'], 3)
                                   if isinstance(u['ppl_change_pct'], float)
                                   and isinstance(d['ppl_change_pct'], float) else None),
        }
    json.dump(out, open(os.path.join(ROOT, 'results', 'dp_raw_vs_clean.json'),
                        'w', encoding='utf-8'), indent=1)
    print('WROTE results/dp_raw_vs_clean.json')


if __name__ == '__main__':
    main()
