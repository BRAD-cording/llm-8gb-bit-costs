"""Additivity test for the 4B per-item table.

The allocator assumes the cost of quantising a set of modules is the sum of
the per-module costs.  This script measures the residual of that assumption:
it quantises two modules to 4 bits *at the same time*, measures the joint
perplexity change, and compares it with the sum of the two single-module
changes measured under the identical protocol (4096 tokens, one window).

    interaction_pp = measured_joint_pct - predicted_pct
    predicted_pct  = 100 * (exp(ln(1+p_i/100) + ln(1+p_j/100)) - 1)

A residual of a few hundredths of a point supports additivity; a residual of
the same order as the individual effects breaks the model.
"""
import argparse
import json
import math
import os
import sys
import time
import types

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from eval.per_item_4b_nf4 import (build, dequant_module, nll, quant_rtn,
                                  iter_targets)
from eval.ppl import load_tokens
from bench.gqa_patch import enable_gqa_sdpa
from transformers import AutoTokenizer


def pct_to_nats(p):
    return math.log(1.0 + p / 100.0)


def nats_to_pct(d):
    return 100.0 * (math.exp(d) - 1.0)


def patch(mod, wq, b16):
    def fwd(self, x, _w=wq, _b=b16):
        y = F.linear(x, _w.to(x.dtype))
        return y if _b is None else y + _b.to(x.dtype)
    mod.forward = types.MethodType(fwd, mod)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--text', required=True)
    ap.add_argument('--items', required=True)
    ap.add_argument('--pairs', type=int, default=10)
    ap.add_argument('--bits', type=int, default=4)
    ap.add_argument('--seq-len', type=int, default=4096)
    ap.add_argument('--windows', type=int, default=1)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    table = json.load(open(args.items, encoding='utf-8'))
    rows = table['items'] if isinstance(table, dict) else table
    key = str(args.bits)
    ranked = sorted(rows, key=lambda r: abs(r['err'].get(key, 0.0)),
                    reverse=True)
    need = 2 * args.pairs
    chosen = ranked[:need]
    pairs = [(chosen[2 * i], chosen[2 * i + 1]) for i in range(len(chosen) // 2)]
    print(f'items {len(rows)} -> {len(pairs)} pairs of the {need} most '
          f'sensitive modules', flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    wins = load_tokens(tok, args.text, args.seq_len, args.windows)
    model = build(args.model)
    enable_gqa_sdpa(model)
    base = nll(model, wins)
    print(f'base nll {base:.6f} ppl {math.exp(base):.4f} '
          f'windows {len(wins)}', flush=True)

    mods = dict(iter_targets(model))
    records = []
    t0 = time.time()
    for k, (a, b) in enumerate(pairs, 1):
        ma, mb = mods[a['name']], mods[b['name']]
        fa, fb = ma.forward, mb.forward
        wa = dequant_module(ma)
        wb = dequant_module(mb)
        qa = (quant_rtn(wa, args.bits) if args.bits < 16 else wa).to('cuda')
        qb = (quant_rtn(wb, args.bits) if args.bits < 16 else wb).to('cuda')
        ba = None if getattr(ma, 'bias', None) is None else ma.bias.detach().clone()
        bb = None if getattr(mb, 'bias', None) is None else mb.bias.detach().clone()
        patch(ma, qa, ba)
        patch(mb, qb, bb)
        try:
            d_joint = nll(model, wins) - base
        finally:
            ma.forward = fa
            mb.forward = fb
        d_i = pct_to_nats(a['err'][key])
        d_j = pct_to_nats(b['err'][key])
        joint_pct = nats_to_pct(d_joint)
        pred_pct = nats_to_pct(d_i + d_j)
        rec = {
            'a': a['name'], 'b': b['name'],
            'single_pct_a': a['err'][key], 'single_pct_b': b['err'][key],
            'predicted_pct': round(pred_pct, 4),
            'joint_pct': round(joint_pct, 4),
            'interaction_pp': round(joint_pct - pred_pct, 4),
            'ratio_joint_over_pred': round(joint_pct / pred_pct, 3) if pred_pct else None,
        }
        records.append(rec)
        print(f'[{k}/{len(pairs)}] {a["name"]} + {b["name"]}: '
              f'pred {pred_pct:+.3f}% joint {joint_pct:+.3f}% '
              f'inter {rec["interaction_pp"]:+.3f} pp '
              f'({time.time()-t0:.0f}s)', flush=True)

    inter = [r['interaction_pp'] for r in records]
    summ = {
        'pairs': len(records),
        'bits': args.bits,
        'seq_len': args.seq_len,
        'windows': args.windows,
        'mean_interaction_pp': round(sum(inter) / len(inter), 4) if inter else None,
        'max_abs_interaction_pp': round(max(abs(x) for x in inter), 4) if inter else None,
        'median_abs_interaction_pp': round(
            sorted(abs(x) for x in inter)[len(inter) // 2], 4) if inter else None,
    }
    out = {'summary': summ, 'base_nll': base, 'base_ppl': math.exp(base),
           'records': records}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, 'w', encoding='utf-8'), indent=1)
    print('SUMMARY', json.dumps(summ), flush=True)
    print('WROTE', args.out, flush=True)


if __name__ == '__main__':
    main()
