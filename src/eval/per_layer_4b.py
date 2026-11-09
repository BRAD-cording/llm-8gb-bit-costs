"""Per-layer-group bit-cost table for Qwen3-4B on a 4-bit base.

Single-module costs on this checkpoint average 0.11% against a 0.42 pp
measurement noise, so a per-module table is not measurable.  A whole decoder
layer carries roughly 30x that signal, which puts it above the noise floor,
and a per-layer allocation is still a non-trivial instance of the same
problem (36 groups instead of 252 modules).

Each group is every Linear inside one decoder layer.  For every group and
every width in BITS we patch the group to that width, keep the rest at nf4,
and record the perplexity change.  Output is one JSON row per
(group, bits) so a long run can be resumed.
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

BITS = [2, 3, 4, 8]


def groups_of(model):
    """name -> layer index, plus the list of (name, module) per layer."""
    by_layer = {}
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        if 'lm_head' in name:
            continue
        m = re.search(r'\.layers\.(\d+)\.', name)
        if not m:
            continue
        by_layer.setdefault(int(m.group(1)), []).append((name, mod))
    return by_layer


def patch(mod, wq, bias):
    def fwd(self, x, _w=wq, _b=bias):
        y = F.linear(x, _w.to(x.dtype))
        return y if _b is None else y + _b.to(x.dtype)
    mod.forward = types.MethodType(fwd, mod)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--text', required=True)
    ap.add_argument('--seq-len', type=int, default=4096)
    ap.add_argument('--windows', type=int, default=1)
    ap.add_argument('--bits', default='2,3,4,8')
    ap.add_argument('--out', required=True)
    ap.add_argument('--layers', default='')
    args = ap.parse_args()

    bits = [int(b) for b in args.bits.split(',')]
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    wins = load_tokens(tok, args.text, args.seq_len, args.windows)
    model = build(args.model)
    enable_gqa_sdpa(model)
    base = nll(model, wins)
    print(f'base nll {base:.6f} ppl {math.exp(base):.4f}', flush=True)

    by_layer = groups_of(model)
    todo = sorted(int(x) for x in args.layers.split(',')) if args.layers else sorted(by_layer)
    print(f'layers {len(todo)} of {len(by_layer)}, bits {bits}', flush=True)

    out = open(args.out, 'w', encoding='utf-8')
    rows = 0
    for li in todo:
        members = by_layer[li]
        saved = [(mod, mod.forward) for _, mod in members]
        for b in bits:
            wq = {}
            for name, mod in members:
                w16 = dequant_module(mod)
                bias = None if getattr(mod, 'bias', None) is None else mod.bias.detach().clone()
                wq[name] = ((quant_rtn(w16, b) if b < 16 else w16).to('cuda'), bias)
            for name, mod in members:
                patch(mod, *wq[name])
            j = nll(model, wins)
            for mod, f0 in saved:
                mod.forward = f0
            d = j - base
            rec = {'layer': li, 'kind': 'layer', 'bits': b,
                   'n_modules': len(members), 'delta_nats': d,
                   'pct': 100.0 * (math.exp(d) - 1.0)}
            out.write(json.dumps(rec) + chr(10))
            out.flush()
            rows += 1
            print(f'L{li:02d} b{b} {rec["pct"]:+.3f}%', flush=True)
    out.close()
    print(f'WROTE {args.out} rows={rows}', flush=True)


if __name__ == '__main__':
    main()
