"""Is GQA fusion unavailable on this stack, or is it a calling-convention artefact?

Reviewer question: PyTorch exposes an `enable_gqa` argument to
`scaled_dot_product_attention`, and transformers' sdpa path may expand KV heads
itself. If either route fuses, the claim that "every fused attention kernel is
disabled" is a statement about the default call pattern rather than about the
stack.

This script builds one attention layer's tensors (16 query heads, 8 KV heads,
head_dim 128) and measures peak allocated memory and time for:
  as-is        q[1,16,L,128], k/v[1,8,L,128]            (what GQA looks like)
  repeat       k/v expanded to 16 heads, then SDPA      (the fix the paper uses)
  enable_gqa   enable_gqa=True passed to SDPA           (torch's own GQA path)

Math backend materialises an L x L score matrix per head (~1 GB at L=4096, fp16),
so peak memory identifies the backend without a profiler.

Usage: python tools/check_gqa_sdpa.py --length 4096
"""
from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn.functional as F


def peak_and_time(fn, iters: int = 3):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn()  # warm
    torch.cuda.synchronize()
    start = torch.cuda.Event(True)
    end = torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 2 ** 20
    return peak, start.elapsed_time(end) / iters


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=4096)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--kv-heads", type=int, default=8)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--out", default="results/gqa_sdpa_control.json")
    args = ap.parse_args()

    L, H, KV, D = args.length, args.heads, args.kv_heads, args.dim
    dtype = torch.float16
    q = torch.randn(1, H, L, D, device="cuda", dtype=dtype)
    k = torch.randn(1, KV, L, D, device="cuda", dtype=dtype)
    v = torch.randn(1, KV, L, D, device="cuda", dtype=dtype)

    rows = []

    def run(name, fn):
        try:
            peak, ms = peak_and_time(fn)
            rows.append({"case": name, "ok": True, "peak_MB": round(peak, 1),
                         "ms": round(ms, 1)})
        except Exception as exc:  # noqa: BLE001
            rows.append({"case": name, "ok": False,
                         "error": f"{type(exc).__name__}: {str(exc)[:160]}"})
        print(json.dumps(rows[-1]), flush=True)

    run("as-is", lambda: F.scaled_dot_product_attention(q, k, v))
    run("repeat", lambda: F.scaled_dot_product_attention(
        q, k.repeat_interleave(H // KV, dim=1), v.repeat_interleave(H // KV, dim=1)))
    run("enable_gqa", lambda: F.scaled_dot_product_attention(
        q, k, v, enable_gqa=True))

    payload = {"torch": torch.__version__, "length": L,
               "heads": [H, KV], "head_dim": D, "dtype": "float16", "rows": rows}
    try:
        import transformers
        payload["transformers"] = transformers.__version__
    except Exception:  # noqa: BLE001
        pass
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
    print("WROTE", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
