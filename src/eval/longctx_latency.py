"""
Latency against context length for the competing configurations.

Why this exists: on this laptop the driver spills VRAM into system memory instead
of raising OOM, so a configuration that "does not fit" still runs --- slowly. That
makes latency, not feasibility, the observable that separates the configurations
at long context, and it also locates the crossover: \texttt{nf4} weights read a
quarter of the weight bytes per decode step, so they win while the weights
dominate, and lose once the cache dominates.

Usage:
    python src/eval/longctx_latency.py --model C:/models/Qwen3-0.6B \
        --lengths 4096,16384,32768 --out results/longctx_06b.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _ROOT)

from bench.gqa_patch import enable_gqa_sdpa  # noqa: E402


def build(model_path: str, precision: str):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    if precision == "nf4":
        cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.float16,
                                 bnb_4bit_use_double_quant=True)
        return AutoModelForCausalLM.from_pretrained(
            model_path, device_map="cuda", trust_remote_code=True,
            quantization_config=cfg).eval()
    return AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="cuda",
        trust_remote_code=True).eval()


@torch.inference_mode()
def measure(model, seq_len: int, new_tokens: int = 8):
    ids = torch.randint(100, 2000, (1, seq_len), device="cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    out = model(ids, use_cache=True)
    torch.cuda.synchronize()
    ttft = (time.time() - t0) * 1000
    peak = torch.cuda.max_memory_allocated() / 1024 ** 2

    past, cur = out.past_key_values, ids[:, -1:]
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(new_tokens):
        o = model(cur, past_key_values=past, use_cache=True)
        past, cur = o.past_key_values, o.logits[:, -1:].argmax(-1)
    torch.cuda.synchronize()
    tpot = (time.time() - t0) * 1000 / new_tokens
    del out, past, cur
    torch.cuda.empty_cache()
    return ttft, tpot, peak


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lengths", default="4096,16384")
    ap.add_argument("--configs", default="default,policy")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    plan = {"default": ("nf4", 16, "none"), "policy": ("fp16", 4, "kivi"),
            "naive16": ("fp16", 16, "none"), "rtn4": ("fp16", 4, "rtn")}
    out = {"model": args.model, "rows": []}

    for name in args.configs.split(","):
        precision, kv_bits, cache_q = plan[name]
        m = build(args.model, precision)
        if cache_q == "kivi":
            enable_gqa_sdpa(m, kv_bits=kv_bits, kv_residual=128, kv_official=True)
        elif cache_q == "rtn":
            enable_gqa_sdpa(m, kv_bits=kv_bits)
        else:
            enable_gqa_sdpa(m, kv_bits=None)

        for L in [int(x) for x in args.lengths.split(",")]:
            try:
                ttft, tpot, peak = measure(m, L)
                row = {"config": name, "seq_len": L, "ttft_ms": round(ttft, 1),
                       "tpot_ms": round(tpot, 2), "peak_MB": round(peak, 0)}
            except RuntimeError as e:
                row = {"config": name, "seq_len": L,
                       "error": f"{type(e).__name__}: {str(e)[:100]}"}
                torch.cuda.empty_cache()
            out["rows"].append(row)
            print(json.dumps(row), flush=True)
        del m
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
