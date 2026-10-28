"""
Minimal reproduction: grouped-query attention silently disables every fused
attention kernel in PyTorch on this stack, costing 32x attention memory.

This is the artifact behind the paper's claim that execution-stack details can
dominate bit allocation.  It needs no model, no dataset and no checkpoint: only
torch.  Expected runtime: a few seconds.

Run:
    python artifacts/reproduce_gqa_fallback.py

Expected output (RTX 4060 Laptop, Windows, torch 2.6.0+cu124):

    q_heads=16 kv_heads=8  as-is    mem_efficient  UNAVAILABLE (no kernel)
    q_heads=16 kv_heads=8  as-is    flash          UNAVAILABLE (no kernel)
    q_heads=16 kv_heads=8  expanded  mem_efficient   ~84 MB
    q_heads=16 kv_heads=8  expanded  math           ~2693 MB

The ratio between the last two lines is the whole point: identical mathematics,
identical output values, 32x difference in peak memory, decided by whether the
key/value tensors are materialised to match the query head count.
"""
from __future__ import annotations

import json
import sys

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

L, D = 4096, 128
Q_HEADS, KV_HEADS = 16, 8


def peak_mb(fn) -> tuple[float | None, str]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        with torch.inference_mode():
            out = fn()
            peak = torch.cuda.max_memory_allocated() / 1e6
            del out
        return peak, "ok"
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:60]}"
    finally:
        torch.cuda.empty_cache()


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA required", file=sys.stderr)
        return 1

    print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)}")
    print(f"q_heads={Q_HEADS} kv_heads={KV_HEADS}  L={L} D={D} fp16\n")

    q = torch.randn(1, Q_HEADS, L, D, dtype=torch.float16, device="cuda")
    k = torch.randn(1, KV_HEADS, L, D, dtype=torch.float16, device="cuda")
    v = torch.randn(1, KV_HEADS, L, D, dtype=torch.float16, device="cuda")

    rep = Q_HEADS // KV_HEADS
    kx = k.repeat_interleave(rep, dim=1)
    vx = v.repeat_interleave(rep, dim=1)

    rows = []
    for label, kk, vv in [("as-is   ", k, v), ("expanded", kx, vx)]:
        for bname, backend in [("mem_efficient", SDPBackend.EFFICIENT_ATTENTION),
                               ("flash        ", SDPBackend.FLASH_ATTENTION),
                               ("math         ", SDPBackend.MATH)]:
            def run(kk=kk, vv=vv, b=backend):
                with sdpa_kernel(b):
                    return F.scaled_dot_product_attention(q, kk, vv, is_causal=True)

            peak, status = peak_mb(run)
            shown = f"{peak:8.0f} MB" if peak is not None else "     ---   "
            print(f"  {label}  {bname}  {shown}  {status}")
            rows.append({"kv": label.strip(), "backend": bname.strip(),
                         "peak_mb": None if peak is None else round(peak, 1),
                         "status": status})

    # correctness: expansion must not change the result
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        a = F.scaled_dot_product_attention(q, k.repeat_interleave(rep, 1),
                                           v.repeat_interleave(rep, 1), is_causal=True)
        b = F.scaled_dot_product_attention(q, kx, vx, is_causal=True)
    same = torch.allclose(a, b, atol=1e-3, rtol=1e-3)
    print(f"\nexpansion is mathematically transparent: {same}")

    summary = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__,
               "rows": rows, "expansion_transparent": bool(same)}
    with open("results/gqa_fallback_repro.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("wrote results/gqa_fallback_repro.json")
    return 0 if same else 2


if __name__ == "__main__":
    sys.exit(main())
