"""
Batch>1 end-to-end runtime baseline (P1).

Why this exists: the paper argues one cache bit costs less accuracy than one
weight bit.  A reviewer (R1-M4 / R2) may reply that cache compression either
slows decode (unfused dequantisation adds work to every token, cf. kv4) or
that the "cache is cheaper" ordering does not survive a real serving loop.
This script measures how the competing weight-vs-cache configurations behave
under a batched decode, where the matmul is thick enough that dequantisation
overhead hides and the KV-bandwidth argument is put to the test.

It reuses the discipline of regime_sweep.py (inference_mode, spilled-detect,
median timing, prefill on the base model without per-position lm_head) and the
configuration matrix of longctx_latency.py:

    default  = nf4 weights, 16-bit KV
    policy   = fp16 weights, 4-bit KIVI KV (official, 128-token residual)
    naive16  = fp16 weights, 16-bit KV
    rtn4     = fp16 weights, 4-bit round-to-nearest KV

For each (config, context length, batch) it reports:
    TTFT  prefilling B sequences of length L in one forward
    TPOT  one batched decode step against a KV cache of length L
          (reported per-sequence and as aggregate tokens/s)

Usage (run on the laptop 4060):
    python src/bench/batch_sweep.py --model C:/models/Qwen3-0.6B \\
        --configs default,policy --lengths 1024,4096 --batches 1,4,16 \\
        --out results/batch_06b.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _ROOT)

from bench.regime_sweep import weight_bytes, kv_bytes_per_token  # noqa: E402
from bench.gqa_patch import enable_gqa_sdpa  # noqa: E402

# --- model loading (mirrors longctx_latency.build) -----------------------------

def build(model_path: str, precision: str):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    kwargs = dict(trust_remote_code=True)
    if precision == "nf4":
        kwargs["device_map"] = "cuda"
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True)
    elif precision == "fp16":
        kwargs["device_map"] = "cuda"
        kwargs["torch_dtype"] = torch.float16
        kwargs["low_cpu_mem_usage"] = True
    else:
        raise ValueError(precision)
    m = AutoModelForCausalLM.from_pretrained(model_path, **kwargs).eval()
    return m


# --- timing (mirrors regime_sweep.median_time) --------------------------------

def median_time(fn, warmup: int = 2, iters: int = 5) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


@torch.inference_mode()
def measure_one(model, base, length: int, batch: int, new_tokens: int = 8):
    total_vram = torch.cuda.get_device_properties(0).total_memory
    ids = torch.randint(100, 30000, (batch, length), device="cuda")
    nxt = torch.randint(100, 30000, (batch, 1), device="cuda")

    def prefill():
        return base(ids, use_cache=True).past_key_values

    torch.cuda.empty_cache()
    past = prefill()
    torch.cuda.reset_peak_memory_stats()
    ttft = median_time(prefill, warmup=1, iters=3)
    peak_prefill = torch.cuda.max_memory_allocated()

    def one_decode():
        model(nxt, past_key_values=past, use_cache=True)

    torch.cuda.reset_peak_memory_stats()
    tpot = median_time(one_decode, warmup=2, iters=new_tokens)
    peak_decode = torch.cuda.max_memory_allocated()

    spilled = max(peak_prefill, peak_decode) > 0.92 * total_vram
    return {
        "batch": batch,
        "seq_len": length,
        "ttft_ms": round(ttft * 1e3, 2),
        "tpot_ms": round(tpot * 1e3, 3),
        "tokens_per_s_total": round(batch / tpot, 2),
        "tokens_per_s_per_seq": round(1.0 / tpot, 3),
        "peak_prefill_MB": round(peak_prefill / 1e6, 1),
        "peak_decode_MB": round(peak_decode / 1e6, 1),
        "spilled": spilled,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--configs", default="default,policy")
    ap.add_argument("--lengths", default="1024,4096")
    ap.add_argument("--batches", default="1,4,16")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required", file=sys.stderr)
        return 1

    plan = {"default": ("nf4", None), "policy": ("fp16", ("kivi", 4, 128)),
            "naive16": ("fp16", None), "rtn4": ("fp16", ("rtn", 4, None))}
    batches = [int(x) for x in args.batches.split(",")]

    out = {"model": args.model,
           "device": torch.cuda.get_device_name(0),
           "torch": torch.__version__,
           "configs": args.configs, "rows": []}

    for name in args.configs.split(","):
        precision, kv = plan[name]
        m = build(args.model, precision)
        base = getattr(m, "model", None) or m
        if kv is not None:
            kind, bits, residual = kv
            if kind == "kivi":
                enable_gqa_sdpa(m, kv_bits=bits, kv_residual=residual,
                                kv_official=True)
            else:
                # rtn: round-to-nearest per-channel cache quantisation
                enable_gqa_sdpa(m, kv_bits=bits)
        else:
            enable_gqa_sdpa(m, kv_bits=None)

        wb = weight_bytes(m)
        kvb = kv_bytes_per_token(m)
        for L in [int(x) for x in args.lengths.split(",")]:
            for B in batches:
                try:
                    row = measure_one(m, base, L, B)
                except torch.cuda.OutOfMemoryError:
                    row = {"config": name, "batch": B, "seq_len": L,
                           "error": "OOM"}
                    torch.cuda.empty_cache()
                except RuntimeError as e:
                    row = {"config": name, "batch": B, "seq_len": L,
                           "error": f"{type(e).__name__}: {str(e)[:120]}"}
                    torch.cuda.empty_cache()
                row["config"] = name
                row["weights_MB"] = round(wb / 1e6, 1)
                row["kv_bytes_per_token"] = kvb
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