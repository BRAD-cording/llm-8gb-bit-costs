"""
Deploy both configurations for real instead of comparing them on curves.

The paper argues that at a fixed context the published practice --- 4-bit weights
with a 16-bit cache --- pays accuracy for memory it does not need. That argument
was made analytically from the measured cost curves. This script runs the two
configurations end to end on the same device, at the same context, and reports
perplexity, peak memory and latency for each:

    default   bnb nf4 weights, 16-bit cache          (what weight-only PTQ ships)
    policy    fp16 weights, 4-bit cache               (what our allocator picks)
    all4      bnb nf4 weights AND 4-bit cache         (the obvious "compress both")

The cache quantiser is KIVI's own (vendored), so the comparison is not against a
quantiser we wrote.

Usage:
    python src/eval/deploy_compare.py --model C:/models/Qwen3-0.6B --lengths 2048,4096
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
        m = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="cuda", trust_remote_code=True,
            quantization_config=cfg).eval()
    else:
        m = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="cuda",
            trust_remote_code=True).eval()
    return m


def token_windows(tok, text_path: str, seq_len: int, n: int) -> torch.Tensor:
    text = open(text_path, encoding="utf-8", errors="ignore").read()
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    n = min(len(ids) // seq_len, n)
    return ids[: n * seq_len].view(n, seq_len)


@torch.inference_mode()
def measure_ppl(model, win):
    return [float(model(win[i:i + 1].to("cuda"),
                        labels=win[i:i + 1].to("cuda")).loss)
            for i in range(win.shape[0])]


@torch.inference_mode()
def measure_latency(model, seq_len: int, new_tokens: int = 8):
    ids = torch.randint(100, 1000, (1, seq_len), device="cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    out = model(ids, use_cache=True)
    torch.cuda.synchronize()
    ttft = (time.time() - t0) * 1000
    peak = torch.cuda.max_memory_allocated() / 1024 ** 2

    past = out.past_key_values
    cur = ids[:, -1:]
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(new_tokens):
        o = model(cur, past_key_values=past, use_cache=True)
        past = o.past_key_values
        cur = o.logits[:, -1:].argmax(-1)
    torch.cuda.synchronize()
    tpot = (time.time() - t0) * 1000 / new_tokens
    del out, past
    torch.cuda.empty_cache()
    return ttft, tpot, peak


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", default="data/calib_arxiv.txt")
    ap.add_argument("--lengths", default="2048,4096")
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=True)

    configs = [
        ("default", "nf4", 16),
        ("policy", "fp16", 4),
        ("all4", "nf4", 4),
        # fp16 weights with an untouched cache: the clean way to isolate the
        # weight-precision effect, because no software dequantisation is on the
        # path (our cache quantiser simulates rather than storing packed bits).
        ("naive16", "fp16", 16),
    ]

    out = {"model": args.model, "rows": []}
    for L in [int(x) for x in args.lengths.split(",")]:
        win = token_windows(tok, args.text, L, args.windows)
        base = None
        for name, precision, kv_bits in configs:
            try:
                m = build(args.model, precision)
                if kv_bits < 16:
                    enable_gqa_sdpa(m, kv_bits=kv_bits, kv_residual=128,
                                    kv_official=True)
                else:
                    enable_gqa_sdpa(m, kv_bits=None)
                vals = measure_ppl(m, win)
                mean = sum(vals) / len(vals)
                if base is None:
                    base = mean          # the nf4/16-bit row is the reference
                ttft, tpot, peak = measure_latency(m, L)
                rel = 100 * (pow(2.718281828459045, mean - base) - 1)
                row = {"seq_len": L, "config": name, "precision": precision,
                       "kv_bits": kv_bits, "mean_nll": round(mean, 6),
                       "rel_pct_vs_default": round(rel, 3),
                       "ttft_ms": round(ttft, 1), "tpot_ms": round(tpot, 1),
                       "peak_prefill_MB": round(peak, 0)}
                out["rows"].append(row)
                print(json.dumps(row), flush=True)
                del m
                torch.cuda.empty_cache()
            except Exception as e:
                row = {"seq_len": L, "config": name, "error":
                       f"{type(e).__name__}: {str(e)[:120]}"}
                out["rows"].append(row)
                print(json.dumps(row), flush=True)
                torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
