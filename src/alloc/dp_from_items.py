"""Solve the bit-allocation problem from a measured per-item table.

Reads a JSON (or JSONL) of items with per-bit relative perplexity costs and
solves the multiple-choice knapsack under the byte budget of Eq. (1), then
compares the optimum against uniform nf4 and a greedy rule.

Cost model (same as the paper):
    B(L) = M(1-rho) - W_act(L) - lm_head_bytes
    item bytes at b bits = n_params * b / 8
    total NLL cost = sum_i log(1 + err_i(b)/100)
"""
import argparse, json, math, os, sys


def load_items(path):
    items = []
    if path.endswith('.jsonl'):
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
    else:
        with open(path, encoding='utf-8') as f:
            items = json.load(f)['items']
    return [it for it in items if it.get('err')]


def budget_mb(L, mem_mib=8192, rho=0.133, lm_head_params=155582464):
    M = mem_mib * 1024 * 1024 / 1e6          # nominal, MB
    usable = M * (1 - rho)
    ws = 0.142 * L + 9
    lm = lm_head_params * 2 / 8 / 1e6        # fp16 head, MB
    return usable - ws - lm


def solve(items, B, bits, uniform_bit=4):
    # dp over units of 1 MB
    cap = int(B)
    opts = []
    for it in items:
        n = it['n_params']
        row = []
        for b in bits:
            e = it['err'].get(str(b))
            if e is None:
                continue
            cost = math.log(1 + e / 100.0)
            row.append((b, n * b / 8 / 1e6, cost))
        opts.append(row)

    NEG = float('inf')
    dp = [NEG] * (cap + 1)
    dp[0] = 0.0
    choice = [[None] * (cap + 1) for _ in opts]
    for i, row in enumerate(opts):
        nd = [NEG] * (cap + 1)
        for used in range(cap + 1):
            if dp[used] == NEG:
                continue
            for b, mb, c in row:
                nu = used + int(mb)
                if nu <= cap and dp[used] + c < nd[nu]:
                    nd[nu] = dp[used] + c
                    choice[i][nu] = (b, used)
        dp = nd
    best = min(range(cap + 1), key=lambda u: dp[u])
    if dp[best] == NEG:
        return None
    assign = {}
    used = best
    for i in range(len(opts) - 1, -1, -1):
        b, prev = choice[i][used]
        assign[i] = b
        used = prev
    return {'cost': dp[best], 'bytes_mb': best, 'assign': assign}


def uniform(items, bits, uniform_bit):
    tot = 0.0
    for it in items:
        e = it['err'].get(str(uniform_bit), 0.0)
        tot += math.log(1 + e / 100.0)
    mb = sum(it['n_params'] * uniform_bit / 8 / 1e6 for it in items)
    return {'cost': tot, 'bytes_mb': mb}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--items', required=True)
    ap.add_argument('--L', type=int, default=32768)
    ap.add_argument('--bits', default='2,3,4,8')
    ap.add_argument('--uniform-bit', type=int, default=4)
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    bits = [int(x) for x in a.bits.split(',')]
    items = load_items(a.items)
    B = budget_mb(a.L)
    u = uniform(items, bits, a.uniform_bit)
    s = solve(items, B, bits, a.uniform_bit)
    res = {'n_items': len(items), 'L': a.L, 'budget_mb': round(B, 1),
           'uniform': {'bit': a.uniform_bit, 'bytes_mb': round(u['bytes_mb'], 1),
                       'ppl_cost_pct': round(100 * (math.exp(u['cost']) - 1), 3)},
           'constraint_violated_by_uniform': u['bytes_mb'] > B}
    if s is None:
        res['dp'] = None
        res['note'] = 'no feasible assignment under budget'
    else:
        hist = {}
        for b in s['assign'].values():
            hist[b] = hist.get(b, 0) + 1
        res['dp'] = {'bytes_mb': round(s['bytes_mb'], 1),
                     'ppl_cost_pct': round(100 * (math.exp(s['cost']) - 1), 3),
                     'width_histogram': hist}
        res['delta_dp_minus_uniform_pp'] = round(
            res['dp']['ppl_cost_pct'] - res['uniform']['ppl_cost_pct'], 3)
    print(json.dumps(res, indent=1))
    if a.out:
        with open(a.out, 'w', encoding='utf-8') as f:
            json.dump(res, f, indent=1)


if __name__ == '__main__':
    main()
