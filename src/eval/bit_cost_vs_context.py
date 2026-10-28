"""
Does the bit cost depend on context length?

Weights are quantised once; the cache is quantised for every token, so error
could accumulate with context. If the cache's relative cost grows with L while
the weight's does not, then the allocation policy of Section 6.3 has a boundary
and should be stated with one -- at long enough context the cache stops being
the cheap side.

Measures, at several window lengths: relative perplexity increase for all-weights
4-bit and for all-cache 4-bit.

Usage:
    python src/eval/bit_cost_vs_context.py --model C:/models/Qwen3-0.6B \
        --text data/calib_arxiv.txt --lengths 512,1024,2048,4096 \
        --out results/bit_cost_vs_context.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.gqa_patch import enable_gqa_sdpa  # noqa: E402
from eval.ppl import quantised_weights  # noqa: E402


def windows_of(tok, path: str, seq_len: int, n: int) -> torch.Tensor:
    text = open(path, encoding="utf-8", errors="ignore").read()
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    n = min(len(ids) // seq_len, n)
    return ids[: n * seq_len].view(n, seq_len)


@torch.inference_mode()
def mean_nll(model, win: torch.Tensor) -> float:
    tot = 0.0
    for i in range(win.shape[0]):
        x = win[i: i + 1].to("cuda")
        tot += float(model(x, labels=x).loss)
    return tot / win.shape[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--lengths", default="512,1024,2048,4096")
    ap.add_argument("--windows", type=int, default=8,
                    help="windows per length (tokens = windows x length)")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--out", default="results/bit_cost_vs_context.json")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="cuda",
        trust_remote_code=True).eval()
    enable_gqa_sdpa(model, kv_bits=None)

    layers = [(n, m) for n, m in model.named_modules()
              if isinstance(m, nn.Linear) and m.weight.dim() == 2
              and not n.endswith("lm_head")]

    rows = []
    for L in [int(x) for x in args.lengths.split(",")]:
        win = windows_of(tok, args.text, L, args.windows)
        base = mean_nll(model, win)

        with quantised_weights(layers, args.bits):
            w = mean_nll(model, win)

        enable_gqa_sdpa(model, kv_bits=args.bits)
        k = mean_nll(model, win)
        enable_gqa_sdpa(model, kv_bits=None)

        ew = 100 * (pow(2.718281828459045, w - base) - 1)
        ek = 100 * (pow(2.718281828459045, k - base) - 1)
        rows.append({"seq_len": L, "windows": int(win.shape[0]),
                     "baseline_nll": round(base, 5),
                     "weights_pct": round(ew, 3), "cache_pct": round(ek, 3),
                     "ratio_weights_over_cache": round(ew / ek, 3) if ek else None})
        print(f"L={L:5d}  tokens={win.numel():7d}  weights {ew:+7.2f}%  "
              f"cache {ek:+7.2f}%  ratio {ew/ek if ek else float('nan'):5.2f}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "bits": args.bits, "rows": rows}, f, indent=2)
    print(f"wrote {args.out}")

    if len(rows) >= 2:
        grow_k = rows[-1]["cache_pct"] - rows[0]["cache_pct"]
        grow_w = rows[-1]["weights_pct"] - rows[0]["weights_pct"]
        print(f"\ncache cost grows by {grow_k:+.2f} pts from L={rows[0]['seq_len']} "
              f"to L={rows[-1]['seq_len']}; weights by {grow_w:+.2f} pts")
        print("cache cost is context-sensitive" if abs(grow_k) > 5 * max(abs(grow_w), 0.5)
              else "no strong context dependence detected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
