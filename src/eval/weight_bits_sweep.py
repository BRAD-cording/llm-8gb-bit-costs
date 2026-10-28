"""
Whole-model weight quantisation sweep: PPL versus weight bit-width.

This is the counterpart to kv_bits_sweep.py.  Together they give the two curves
the allocator needs: what a weight bit costs and what a KV bit costs.

`lm_head` is kept in fp16 unless --include-lm-head is passed, because the
per-layer calibration found it to be the single most sensitive module.

Usage:
    python src/eval/weight_bits_sweep.py --model C:/models/Qwen3-0.6B \
        --text data/calib_arxiv.txt --bits 8,4,3,2 --out results/w_bits.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.ppl import load_tokens, perplexity, quantised_weights  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--bits", default="8,4,3,2")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--max-windows", type=int, default=6)
    ap.add_argument("--include-lm-head", action="store_true")
    ap.add_argument("--out", default="results/w_bits.json")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="cuda",
        trust_remote_code=True).eval()

    layers = [(n, m) for n, m in model.named_modules()
              if isinstance(m, nn.Linear) and m.weight.dim() == 2]
    if not args.include_lm_head:
        layers = [(n, m) for n, m in layers if not n.endswith("lm_head")]
    print(f"quantising {len(layers)} linear layers "
          f"(lm_head {'included' if args.include_lm_head else 'kept fp16'})")

    windows = load_tokens(tok, args.text, args.seq_len, args.max_windows)
    base = perplexity(model, windows)
    print(f"baseline PPL = {base:.4f}")

    rows = []
    for bits in [int(b) for b in args.bits.split(",")]:
        if bits >= 16:
            rows.append({"w_bits": bits, "ppl": round(base, 5), "relative": 0.0})
            print(f"  W{bits:2d}  PPL {base:9.4f}   +0.00%")
            continue
        with quantised_weights(layers, bits):
            ppl = perplexity(model, windows)
        rel = (ppl - base) / base
        rows.append({"w_bits": bits, "ppl": round(ppl, 5), "relative": round(rel, 6)})
        print(f"  W{bits:2d}  PPL {ppl:9.4f}   {100*rel:+8.2f}%", flush=True)

    payload = {"model": args.model, "baseline_ppl": round(base, 6),
               "include_lm_head": args.include_lm_head,
               "n_layers": len(layers), "window_tokens": args.seq_len,
               "n_windows": args.max_windows, "rows": rows}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
