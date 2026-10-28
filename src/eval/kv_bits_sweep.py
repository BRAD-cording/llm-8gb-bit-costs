"""
Correct KV-cache quantisation sensitivity: quantise K/V inside the attention
call, i.e. after RoPE, which is what a quantised cache actually stores.

Supersedes src/eval/kv_sensitivity.py, whose hook sits on k_proj/v_proj and
therefore measures a PRE-RoPE quantity.  Kept for comparison; the numbers here
are the ones to report.

Usage:
    python src/eval/kv_bits_sweep.py --model C:/models/Qwen3-0.6B \
        --text data/calib_arxiv.txt --bits 8,4,3,2 --out results/kv_bits.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.gqa_patch import enable_gqa_sdpa  # noqa: E402
from eval.ppl import load_tokens, perplexity  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--bits", default="8,4,3,2")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--max-windows", type=int, default=6)
    ap.add_argument("--out", default="results/kv_bits.json")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="cuda",
        trust_remote_code=True).eval()

    windows = load_tokens(tok, args.text, args.seq_len, args.max_windows)

    rows = []
    base = None
    for bits in [int(b) for b in args.bits.split(",")]:
        enable_gqa_sdpa(model, kv_bits=None if bits >= 8 else bits)
        ppl = perplexity(model, windows)
        if base is None:
            base = ppl
        rel = (ppl - base) / base
        rows.append({"kv_bits": bits, "attn": model.config._attn_implementation,
                     "ppl": round(ppl, 5), "relative": round(rel, 6)})
        print(f"  KV {bits:2d}-bit  PPL {ppl:9.4f}   {100*rel:+8.2f}%  "
              f"[{model.config._attn_implementation}]", flush=True)

    payload = {"model": args.model, "baseline_ppl": round(base, 6),
               "window_tokens": args.seq_len, "n_windows": args.max_windows,
               "rows": rows}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
