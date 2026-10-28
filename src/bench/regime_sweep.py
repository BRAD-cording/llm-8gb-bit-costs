"""
Regime sweep: the experiment that decides whether this paper's thesis is real.

For each (model, weight-precision, prompt-length) it measures, separately:

    TTFT   time-to-first-token  = one prefill forward over the whole prompt
    TPOT   time-per-output-token = one decode forward with a KV cache of length L

and derives the quantities the cost model needs:

    bytes_per_token   weights + KV, from the actual stored dtypes
    achieved GB/s     bytes_per_token / TPOT      (decode: should saturate B_eff)
    arithmetic_intensity = 2 * params / bytes_per_token
    regime            "memory" if intensity < b* else "compute"

b* = P_eff / B_eff comes from device_probe.py.  The point of the sweep is that
large prompts push the *prefill* forward into the compute-bound regime while the
decode forward stays bandwidth-bound at every L -- and that this crossover sits
at a different L on a 4060 than on a Jetson, which is what makes the optimal
weight/KV split reverse between the two machines.

Device-agnostic on purpose: the SAME script must run on the Jetson, otherwise the
two numbers are not comparable and the paper's central figure is worthless.

Usage:
    python src/bench/regime_sweep.py --model C:/models/Qwen/Qwen3-0.6B \
        --precision nf4 --lengths 128,512,2048,8192 --out results/sweep_qwen06_nf4.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch

# (version shim removed: transformers 4.51 loads the JetPack wheel as-is)


def build_model(path: str, precision: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # The sweep feeds random token ids, so the tokenizer is never used.  It is
    # loaded best-effort only because some ModelScope snapshots ship both
    # tokenizer.json and vocab.json/merges.txt, which makes the slow Qwen2
    # tokenizer raise "vocab and merges must both be from memory or both
    # filenames" on transformers 5.x.
    tok = None
    try:
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True,
                                            use_fast=True)
    except Exception as e:
        print(f"[warn] tokenizer unavailable ({type(e).__name__}); "
              f"continuing without it", file=sys.stderr)
    # device_map pulls in accelerate's parallel-state machinery, which needs a
    # version match with transformers and fails obscurely on a Jetson L4T
    # image ("argument of type 'NoneType' is not iterable"). fp16 does not need
    # it: load on CPU and move. Quantised configs still require device_map.
    kwargs = dict(trust_remote_code=True)
    if precision == "nf4":
        from transformers import BitsAndBytesConfig
        kwargs["device_map"] = device
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    elif precision == "int8":
        from transformers import BitsAndBytesConfig
        kwargs["device_map"] = device
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif precision == "fp16":
        # device_map matters on the Jetson: a single .to("cuda") asks for one
        # 1.2 GB NvMap block and fails with error 12, while per-module dispatch
        # succeeds. Requires PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.
        kwargs["device_map"] = device
        kwargs["torch_dtype"] = torch.float16
        kwargs["low_cpu_mem_usage"] = True
    else:
        raise ValueError(f"unknown precision {precision}")

    model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    if "device_map" not in kwargs:
        model = model.to(device)
    return model.eval(), tok


def weight_bytes(model) -> int:
    """Actual stored bytes of every parameter (post-quantisation)."""
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    # bitsandbytes stores 4-bit weights packed plus fp16 absmax scales
    if hasattr(model, "get_memory_footprint"):
        try:
            return int(model.get_memory_footprint())
        except Exception:
            pass
    return total


def kv_bytes_per_token(model) -> int:
    """K and V elements per token, across all layers and kv heads."""
    cfg = model.config
    n_layers = getattr(cfg, "num_hidden_layers", 0)
    n_kv = getattr(cfg, "num_key_value_heads", None) or getattr(cfg, "num_attention_heads", 0)
    head_dim = getattr(cfg, "head_dim", None)
    if head_dim is None:
        head_dim = cfg.hidden_size // max(1, getattr(cfg, "num_attention_heads", 1))
    return 2 * n_layers * n_kv * head_dim * 2  # fp16 KV as stored by llama/hf cache


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


def sweep_one(model, tok, length: int, device: str, p_eff_tflops: float,
              bw_gbps: float, decode_steps: int = 8) -> dict:
    """Measure TTFT and TPOT for one context length.

    Two traps this function is written to avoid, both of which produced garbage
    in the first version of this file:

    1. Computing logits for EVERY prompt position during prefill.  For a
       151936-token vocabulary and L=8192 that is a 2.5 GB intermediate, several
       copies of it, and the measurement then reports the allocator rather than
       the model.  Real engines project only the last position, so we pass
       `logits_to_keep=1`.
    2. Silent spill to host memory.  Windows WDDM does not raise OOM when VRAM is
       exhausted -- it backs the allocation with system RAM and throughput
       collapses by ~60x.  We detect that by comparing the peak against the
       device total and never report such a row as a valid measurement.
    """
    total_vram = torch.cuda.get_device_properties(0).total_memory
    ids = torch.randint(100, 30000, (1, length), device=device)
    nxt = torch.randint(100, 30000, (1, 1), device=device)

    # Prefill is timed on the BASE model only.  HuggingFace's CausalLM forward
    # ends with lm_head over every position (logits_to_keep is applied after the
    # matmul in this version), and for a 151936-token vocabulary that single
    # buffer is ~2.7 GB at L=4096 -- larger than the weights and the KV cache
    # combined, and it grows linearly in L.  Measured with a depth sweep, the
    # overhead is constant in depth (2712 MB at 1 layer, 2838 MB at 8), which
    # proves it is one buffer, not per-layer accumulation.
    # Real engines project only the final position, so we do the same.
    base = getattr(model, "model", None) or model

    def prefill():
        return base(ids, use_cache=True).past_key_values

    with torch.inference_mode():
        past = prefill()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    # NOTE: the timed calls MUST run under inference_mode.  Outside it, autograd
    # retains activations for a backward pass that never happens, which inflated
    # the measured footprint by ~4x and caused spurious OOMs in an earlier
    # revision of this file.
    with torch.inference_mode():
        ttft = median_time(prefill, warmup=1, iters=3)
        peak_prefill = torch.cuda.max_memory_allocated()

    def one_decode():
        model(nxt, past_key_values=past, use_cache=True)

    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        tpot = median_time(one_decode, warmup=2, iters=decode_steps)
        peak_decode = torch.cuda.max_memory_allocated()

    peak = max(peak_prefill, peak_decode)
    spilled = peak > 0.92 * total_vram

    wb = weight_bytes(model)
    kvb = kv_bytes_per_token(model)
    total_per_token = wb + kvb * length
    achieved_gbps = total_per_token / tpot / 1e9
    intensity = 2 * sum(p.numel() for p in model.parameters()) / total_per_token
    b_star = p_eff_tflops * 1e12 / (bw_gbps * 1e9)

    row = {
        "seq_len": length,
        "ttft_ms": round(ttft * 1e3, 2),
        "tpot_ms": round(tpot * 1e3, 3),
        "tok_per_s": round(1.0 / tpot, 2),
        "weights_MB": round(wb / 1e6, 1),
        "kv_MB": round(kvb * length / 1e6, 1),
        "peak_prefill_MB": round(peak_prefill / 1e6, 1),
        "peak_decode_MB": round(peak_decode / 1e6, 1),
        "achieved_GBps": round(achieved_gbps, 1),
        "arith_intensity": round(intensity, 2),
        "crossover_b": round(b_star, 1),
        "decode_regime": "compute" if intensity > b_star else "memory",
        "spilled": spilled,
    }
    if spilled:
        row["note"] = "EXCEEDED VRAM AND SPILLED TO HOST MEMORY - not a valid measurement"
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--precision", default="nf4", choices=["nf4", "int8", "fp16"])
    ap.add_argument("--lengths", default="128,512,2048,8192")
    ap.add_argument("--out", default="results/sweep.json")
    ap.add_argument("--p-eff", type=float, default=22.36)
    ap.add_argument("--bw", type=float, default=179.8)
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "gqa_sdpa"],
                    help="gqa_sdpa materialises KV heads so a fused kernel runs")
    ap.add_argument("--interleave", action="store_true",
                    help="run each length alternately across precisions so both "
                         "see the same background load (see "
                         "results/REAL_WEIGHT_VALIDATION.md)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required", file=sys.stderr)
        return 1

    model, tok = build_model(args.model, args.precision, "cuda")
    if args.attn == "gqa_sdpa":
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from gqa_patch import enable_gqa_sdpa
        enable_gqa_sdpa(model)
    meta = {
        "model": args.model,
        "precision": args.precision,
        "device": torch.cuda.get_device_name(0),
        "total_mem_GB": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
        "torch": torch.__version__,
        "p_eff_tflops": args.p_eff,
        "bw_gbps": args.bw,
        "rows": [],
    }
    for L in [int(x) for x in args.lengths.split(",")]:
        try:
            row = sweep_one(model, tok, L, "cuda", args.p_eff, args.bw)
        except torch.cuda.OutOfMemoryError:
            row = {"seq_len": L, "error": "OOM"}
            torch.cuda.empty_cache()
        meta["rows"].append(row)
        print(json.dumps(row), flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
