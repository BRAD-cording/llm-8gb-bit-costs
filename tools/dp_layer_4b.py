"""Layer-level allocation for Qwen3-4B at 32k, plus a checker for its prediction.

The per-layer table (results/per_layer_4b.jsonl) gives, for each decoder layer
and each width, the perplexity change with that whole layer quantised and the
rest left at nf4.  This script

  1. attaches per-layer parameter counts (summed from the per-module table),
  2. solves the multi-choice knapsack at the 32k budget on Device A,
  3. reports the allocation and the additive prediction in nats, and
  4. writes the chosen layer->bits map so a joint measurement can test it.

Costs are summed in nats: err = 100*(exp(dNLL)-1) => dNLL = log1p(err/100).
"""
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from alloc.dp_from_items import budget_mb

LM_HEAD = 155582464
BITS = [2, 3, 4, 8]


def main():
    rows = [json.loads(l) for l in open(os.path.join(
        ROOT, 'results', 'per_layer_4b.jsonl'), encoding='utf-8') if l.strip()]
    per = {}
    for r in rows:
        per.setdefault(r['layer'], {})[r['bits']] = r['pct']

    params = {}
    for l in open(os.path.join(ROOT, 'results', 'items_4b_nf4.jsonl'),
                  encoding='utf-8'):
        if not l.strip():
            continue
        r = json.loads(l)
        li = int(r['name'].split('.layers.')[1].split('.')[0])
        params[li] = params.get(li, 0) + int(r['n_params'])
    tot_params = sum(params.values())

    L = 32768
    B = budget_mb(L, lm_head_params=LM_HEAD)
    print(f'layers {len(per)} params {tot_params/1e9:.3f}B budget {B:.1f} MB')

    layers = sorted(per)
    nats = {li: {b: math.log1p(per[li][b] / 100.0) for b in BITS} for li in layers}

    def dp(bits_allow):
        # 1 MB grid over layers, exact DP on the same objective the paper uses
        unit = 1.0
        cap = int(B)
        items = []
        for li in layers:
            arr = []
            for b in bits_allow:
                mb = params[li] * b / 8 / 1e6
                arr.append((int(math.ceil(mb / unit)), nats[li][b], b))
            items.append((li, arr))
        NEG = float('inf')
        best = [NEG] * (cap + 2)
        best[0] = 0.0
        choice = []
        for li, arr in items:
            nb = [NEG] * (cap + 2)
            ch = {}
            for c in range(cap + 1):
                if best[c] == NEG:
                    continue
                for u, cost, b in arr:
                    nc = c + u
                    if nc > cap:
                        continue
                    v = best[c] + cost
                    if v < nb[nc]:
                        nb[nc] = v
                        ch[nc] = (c, b)
            choice.append(ch)
            best = nb
        bestc = min(range(cap + 1), key=lambda c: best[c])
        assign = {}
        c = bestc
        for i in range(len(items) - 1, -1, -1):
            c0, b = choice[i][c]
            assign[items[i][0]] = b
            c = c0
        return assign, best[bestc], bestc

    uni = {li: 4 for li in layers}
    uni_nats = sum(nats[li][4] for li in layers)
    uni_mb = sum(params[li] * 4 / 8 / 1e6 for li in layers)

    out = {'L': L, 'budget_mb': round(B, 1),
           'uniform4': {'mb': round(uni_mb, 1), 'nats': round(uni_nats, 5),
                        'pct': round(100 * (math.exp(uni_nats) - 1), 3)}}
    print('uniform4', json.dumps(out['uniform4']))

    for tag, allow in (('dp_2_3_4_8', BITS), ('dp_3_4_8', [3, 4, 8]),
                       ('dp_4_8', [4, 8])):
        assign, cost, used = dp(allow)
        mb = sum(params[li] * assign[li] / 8 / 1e6 for li in layers)
        hist = {}
        for li in layers:
            hist[str(assign[li])] = hist.get(str(assign[li]), 0) + 1
        rec = {'nats': round(cost, 5), 'pct': round(100 * (math.exp(cost) - 1), 3),
               'mb': round(mb, 1), 'hist': hist,
               'gain_vs_uniform_nats': round(uni_nats - cost, 5),
               'gain_vs_uniform_pct_points': round(
                   (uni_nats - cost) * 100, 3)}
        print(tag, json.dumps(rec))
        out[tag] = rec
        if tag == 'dp_3_4_8':
            out['assign_3_4_8'] = {str(k): v for k, v in assign.items()}
        if tag == 'dp_4_8':
            out['assign_4_8'] = {str(k): v for k, v in assign.items()}

    json.dump(out, open(os.path.join(ROOT, 'results', 'dp_layer_4b_32k.json'),
                        'w', encoding='utf-8'), indent=1)
    print('WROTE results/dp_layer_4b_32k.json')


if __name__ == '__main__':
    main()
