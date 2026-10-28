"""
Device probe: measure the three quantities the paper's cost model depends on.

  B_eff   effective streaming bandwidth (GB/s)        -> decode is bandwidth-bound
  P_eff   effective matmul throughput (TFLOP/s)       -> prefill is compute-bound
  rho     framework/allocator overhead ratio          -> the term everyone omits

Runs on any torch+CUDA device, including Jetson Orin Nano (aarch64) and a laptop
RTX 4060.  Writes JSON so the two machines can be compared directly.

Usage:  python src/bench/device_probe.py --out results/probe_4060.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time

import torch


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timed(fn, warmup: int = 3, iters: int = 10) -> float:
    """Median wall-clock seconds over `iters` runs after `warmup` runs."""
    for _ in range(warmup):
        fn()
    _sync()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        _sync()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def measure_bandwidth(mb: int = 512) -> dict:
    """Copy bandwidth: read + write of an N-byte buffer."""
    n = mb * 1024 * 1024 // 4
    src = torch.empty(n, dtype=torch.float32, device="cuda")
    dst = torch.empty_like(src)
    t = _timed(lambda: dst.copy_(src))
    moved = 2 * n * 4  # read + write
    return {"GBps": round(moved / t / 1e9, 2), "bytes_moved": moved, "seconds": t}


def measure_decode_gemv(d_model: int = 4096, d_ff: int = 11008) -> dict:
    """Decode proxy: batch-1 GEMV is pure memory traffic over the weights."""
    w = torch.empty(d_ff, d_model, dtype=torch.float16, device="cuda")
    w.normal_(0, 0.02)
    x = torch.empty(1, d_model, dtype=torch.float16, device="cuda")
    x.normal_()
    w32 = w.to(torch.float32)

    def run():
        x @ w.t()

    t = _timed(run)
    weight_bytes = w.numel() * 2
    # fp16 tensor-core GEMV is often emulated; also time an fp32 path for reference
    t32 = _timed(lambda: x.float() @ w32.t())
    return {
        "fp16_GBps": round(weight_bytes / t / 1e9, 2),
        "fp32_GBps": round(weight_bytes / t32 / 1e9, 2),
        "weight_MB": round(weight_bytes / 1e6, 1),
    }


def measure_prefill_matmul(d_model: int = 4096, d_ff: int = 11008) -> list:
    """Prefill proxy: batched matmul, FLOPs scale with batch, bytes do not."""
    w = torch.empty(d_ff, d_model, dtype=torch.float16, device="cuda")
    w.normal_(0, 0.02)
    rows = []
    for bs in (1, 2, 4, 8, 16, 32, 64, 128):
        x = torch.empty(bs, d_model, dtype=torch.float16, device="cuda")
        x.normal_()
        t = _timed(lambda: x @ w.t())
        flops = 2 * bs * d_model * d_ff
        rows.append({
            "batch": bs,
            "TFLOPs": round(flops / t / 1e12, 3),
            "GBps": round(w.numel() * 2 / t / 1e9, 1),
        })
    return rows


def measure_alloc_overhead(total_mb: int = 2048) -> dict:
    """rho: bytes actually reserved vs bytes requested, plus allocator granularity.

    On a unified-memory device a single large allocation can fail with NvMap
    ENOMEM even when `free` reports enough, because the carve-out is fragmented
    or the desktop holds pages. We therefore shrink on failure instead of
    aborting the whole probe -- an earlier version crashed here and lost the
    bandwidth and throughput numbers that had already been measured.
    """
    for mb in (total_mb, total_mb // 2, total_mb // 4, 64):
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_reserved()
            n = mb * 1024 * 1024 // 2
            buf = torch.empty(n, dtype=torch.float16, device="cuda")
            requested = n * 2
            reserved = torch.cuda.memory_reserved() - base
            del buf
            torch.cuda.empty_cache()
            return {
                "requested_MB": round(requested / 1e6, 1),
                "reserved_MB": round(reserved / 1e6, 1),
                "rho_alloc": round((reserved - requested) / requested, 4),
                "succeeded_at_MB": mb,
            }
        except Exception as e:
            torch.cuda.empty_cache()
            last = f"{type(e).__name__}: {str(e)[:80]}"
    return {"error": last, "note": "every allocation size failed"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/probe.json")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="shrink test buffers (0.25 suits a 4GB-usable device)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA device; this probe requires one", file=sys.stderr)
        return 1

    props = torch.cuda.get_device_properties(0)
    report = {
        "host": platform.node(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": props.name,
        "total_mem_GB": round(props.total_memory / 1024**3, 2),
        "sm_count": props.multi_processor_count,
        "capability": f"{props.major}.{props.minor}",
        "bandwidth": measure_bandwidth(int(512 * args.scale)),
        "decode_gemv": measure_decode_gemv(),
        "prefill_matmul": measure_prefill_matmul(),
        "alloc_overhead": measure_alloc_overhead(int(2048 * args.scale)),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps({k: report[k] for k in
                      ("device", "capability", "sm_count", "total_mem_GB",
                       "bandwidth", "decode_gemv", "alloc_overhead")}, indent=2))
    print("prefill TFLOPs by batch:", [(r["batch"], r["TFLOPs"]) for r in report["prefill_matmul"]])
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
