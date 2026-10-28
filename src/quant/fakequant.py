"""
Reference (simulated) quantizers for weights and KV cache.

These are *fake-quant* operators: they store the value in the target grid and
immediately dequantise back to fp16/fp32, so the model runs unchanged but with
the numerical damage of the target precision.  That is the right tool for
measuring ACCURACY sensitivity per layer; wall-clock numbers must come from real
kernels (bnb nf4 / llama.cpp GGUF) and are measured separately.

Conventions (chosen to match deployed kernels, not papers):
  * weights: symmetric, per-group (group_size contiguous elements along the
    input dim), fp16 scale per group -- this is the llama.cpp / GPTQ layout.
  * KV: K is quantised per-channel (per head-dim), V per-token, asymmetric with
    zero-point -- this is the KIVI convention.
"""
from __future__ import annotations

import math

import torch


def _group_view(w: torch.Tensor, group_size: int, dim: int) -> torch.Tensor:
    """Reshape so the last dim is the quantisation group.

    `dim` is normalised to a positive index FIRST.  Using a negative dim
    directly made ``shape.insert(dim + 1, group_size)`` insert at position 0, so
    the group axis silently became the leading axis and the tensor was
    quantised along the wrong dimension.  The bug was invisible in the output
    shapes (the result is reshaped back) and only showed up as a quantiser that
    got *worse* with finer groups -- see results/ADVERSARIAL_WEIGHT_TEST.md.
    """
    dim = dim % w.dim()
    size = w.shape[dim]
    if group_size <= 0 or size % group_size != 0:
        group_size = size
    shape = list(w.shape)
    shape[dim] = size // group_size
    shape.insert(dim + 1, group_size)
    return w.reshape(shape)


def qdq_symmetric(w: torch.Tensor, bits: int, group_size: int = 128,
                  dim: int = -1) -> torch.Tensor:
    """Symmetric per-group fake quantisation (weights)."""
    qmax = 2 ** (bits - 1) - 1
    x = _group_view(w.float(), group_size, dim)
    scale = x.abs().amax(dim=-1, keepdim=True) / qmax
    scale = scale.clamp(min=1e-8)
    q = torch.round(x / scale).clamp(-qmax, qmax)
    out = (q * scale).reshape(w.shape)
    return out.to(w.dtype)


def qdq_asymmetric(x: torch.Tensor, bits: int, dim: int = -1) -> torch.Tensor:
    """Asymmetric min-max fake quantisation with zero point (KV cache)."""
    qmax = 2 ** bits - 1
    xf = x.float()
    mn = xf.amin(dim=dim, keepdim=True)
    mx = xf.amax(dim=dim, keepdim=True)
    scale = ((mx - mn) / qmax).clamp(min=1e-8)
    zp = torch.round(-mn / scale).clamp(0, qmax)
    q = torch.round(xf / scale + zp).clamp(0, qmax)
    out = (q - zp) * scale
    return out.to(x.dtype)


def qdq_kv(k: torch.Tensor, v: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    """KIVI-style: K per-channel (dim=-1 is head_dim), V per-token."""
    # k: [..., heads, seq, head_dim] -> per-channel = along head_dim
    kq = qdq_asymmetric(k, bits, dim=-1)
    # v: per-token = along the token axis (dim=-2)
    vq = qdq_asymmetric(v, bits, dim=-2)
    return kq, vq


def storage_bytes(n_elements: int, bits: int, group_size: int = 128,
                  scale_bytes: int = 2) -> float:
    """Stored bytes including per-group fp16 scale (weights)."""
    groups = max(1, math.ceil(n_elements / group_size))
    return n_elements * bits / 8.0 + groups * scale_bytes


def auto_group(hidden: int) -> int:
    """Group size that divides the hidden dim, preferring 128."""
    for g in (128, 64, 32, 16, 8):
        if hidden % g == 0:
            return g
    return 1


# --------------------------------------------------------------------------- #
# self-test: no model or dataset required
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    torch.manual_seed(0)
    ok = True
    w = torch.randn(256, 1024)

    print("weights: relative error should fall monotonically with bits")
    prev = 1e9
    for bits in (2, 3, 4, 8):
        q = qdq_symmetric(w, bits, group_size=128)
        rel = ((q - w).norm() / w.norm()).item()
        mono = rel < prev
        ok &= mono
        print(f"  {bits:2d} bit  rel_err={rel:.5f}  bytes={storage_bytes(w.numel(), bits)/w.numel():.3f}/elem"
              f"  {'ok' if mono else 'NON-MONOTONE'}")
        prev = rel
    # 8-bit should be close to lossless
    q8 = qdq_symmetric(w, 8, 128)
    assert ((q8 - w).norm() / w.norm()).item() < 0.02, "8-bit should be near lossless"
    print("  8-bit near-lossless: ok")

    print("\nkv: K per-channel, V per-token")
    k = torch.randn(1, 8, 64, 128)   # [b, heads, seq, head_dim]
    v = torch.randn(1, 8, 64, 128)
    for bits in (2, 4, 8):
        kq, vq = qdq_kv(k, v, bits)
        print(f"  {bits:2d} bit  K_err={((kq-k).norm()/k.norm()).item():.5f}  "
              f"V_err={((vq-v).norm()/v.norm()).item():.5f}")

    print("\nshape preservation and dtype")
    for shp in [(64, 128), (1, 3, 256), (7, 33)]:
        t = torch.randn(*shp)
        q = qdq_symmetric(t, 4, group_size=8, dim=-1)
        assert q.shape == t.shape and q.dtype == t.dtype, f"shape/dtype broken for {shp}"
    print("  ok")

    # regression guard: finer groups must not make a symmetric quantiser worse.
    # The absence of this assertion is how the negative-dim grouping bug
    # (results/BUG_group_view_axis.md) survived into three measurement runs.
    print("\ngroup-size monotonicity (regression guard for the dim=-1 bug)")
    w2 = torch.randn(256, 512)
    w2[:, ::97] *= 10
    errs = [((qdq_symmetric(w2, 4, g, dim=-1) - w2).norm() / w2.norm()).item()
            for g in (128, 64, 32, 16)]
    for g, e in zip((128, 64, 32, 16), errs):
        print(f"  g={g:3d} rel_err={e:.5f}")
    mono = all(errs[i] > errs[i + 1] for i in range(len(errs) - 1))
    if not mono:
        ok = False
    print(f"  monotone: {mono}")

    print(f"\nSELF-TEST {'PASSED' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_self_test())
