"""Ground-truth check: quantise every module to 4 bits and measure PPL.

The allocator predicts the cost of an allocation by summing per-module costs.
That sum is only meaningful if (a) the interaction between modules is small,
which eight paired measurements support, and (b) the per-module numbers are
not dominated by measurement noise.  With 252 modules whose 4-bit costs are
mostly within +-0.4 pp, a drift in the noise can accumulate.

This script measures the one number that settles it: the perplexity of the
model with *all* weights at 4 bits, under the same protocol, to compare with
the additive prediction for the same allocation.
"""
import argparse
import json
import math
import os
import sys
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--text', required=True)
    ap.add_argument('--bits', type=int, default=4)
    ap.add_argument('--seq-len', type=int, default=4096)
    ap.add_argument('--windows', type=int, default=1)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    wins = load_tokens(tok, args.text, args.seq_len, args.windows)
    model = build(args.model)
    enable_gqa_sdpa(model)
    base = nll(model, wins)
    print(f'base nll {base:.6f} ppl {math.exp(base):.4f}', flush=True)

    saved = []
    n_mod = 0
    for name, mod in iter_targets(model):
        w16 = dequant_module(mod)
        bias = None if getattr(mod, 'bias', None) is None else mod.bias.detach().clone()
        saved.append((mod, mod.forward))
        wq = (quant_rtn(w16, args.bits) if args.bits < 16 else w16).to('cuda')

        def fwd(self, x, _w=wq, _b=bias):
            y = F.linear(x, _w.to(x.dtype))
            return y if _b is None else y + _b.to(x.dtype)

        mod.forward = types.MethodType(fwd, mod)
        n_mod += 1
    print(f'patched {n_mod} modules to {args.bits} bits', flush=True)

    joint = nll(model, wins)
    for mod, f0 in saved:
        mod.forward = f0
    for mod, _ in saved:
        pass

    d = joint - base
    out = {
        'bits': args.bits, 'seq_len': args.seq_len, 'windows': args.windows,
        'modules': n_mod, 'base_nll': base, 'base_ppl': math.exp(base),
        'joint_nll': joint, 'joint_ppl': math.exp(joint),
        'joint_delta_nats': d, 'joint_pct': 100.0 * (math.exp(d) - 1.0),
    }
    print('RESULT', json.dumps(out), flush=True)
    json.dump(out, open(args.out, 'w', encoding='utf-8'), indent=1)
    print('WROTE', args.out, flush=True)


if __name__ == '__main__':
    main()
