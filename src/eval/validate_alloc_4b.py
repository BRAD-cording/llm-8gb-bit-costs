"""Measure the perplexity of a layer->bits assignment on Qwen3-4B.

The allocator predicts cost by summing per-layer costs.  This script applies a
predicted assignment to every layer at once and measures the result, which is
the only way to know whether the prediction is right.
"""
import argparse
import json
import math
import os
import re
import sys
import types

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from eval.per_item_4b_nf4 import build, dequant_module, nll, quant_rtn
from eval.ppl import load_tokens
from bench.gqa_patch import enable_gqa_sdpa
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--text', required=True)
    ap.add_argument('--assign', required=True, help='JSON {layer: bits}')
    ap.add_argument('--seq-len', type=int, default=4096)
    ap.add_argument('--windows', type=int, default=1)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    assign = {int(k): int(v) for k, v in json.load(open(args.assign)).items()}
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    wins = load_tokens(tok, args.text, args.seq_len, args.windows)
    model = build(args.model)
    enable_gqa_sdpa(model)
    base = nll(model, wins)

    n = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear) or 'lm_head' in name:
            continue
        m = re.search(r'\.layers\.(\d+)\.', name)
        if not m:
            continue
        b = assign.get(int(m.group(1)), 4)
        w16 = dequant_module(mod)
        bias = None if getattr(mod, 'bias', None) is None else mod.bias.detach().clone()
        wq = (quant_rtn(w16, b) if b < 16 else w16).to('cuda')

        def fwd(self, x, _w=wq, _b=bias):
            y = F.linear(x, _w.to(x.dtype))
            return y if _b is None else y + _b.to(x.dtype)

        mod.forward = types.MethodType(fwd, mod)
        n += 1

    joint = nll(model, wins)
    d = joint - base
    out = {'modules': n, 'base_ppl': math.exp(base), 'joint_ppl': math.exp(joint),
           'delta_nats': d, 'pct': 100.0 * (math.exp(d) - 1.0),
           'assign': {str(k): v for k, v in sorted(assign.items())}}
    print('RESULT', json.dumps(out), flush=True)
    json.dump(out, open(args.out, 'w', encoding='utf-8'), indent=1)
    print('WROTE', args.out, flush=True)


if __name__ == '__main__':
    main()
