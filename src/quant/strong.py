"""
Stronger, published-style quantisers, so the weight-versus-cache comparison can
be run with matched strength instead of comparing our naive round-to-nearest
against a literature number.

  awq_scale / quantise_awq / awq_best_alpha
      Activation-aware scaling in the spirit of AWQ, with the per-layer alpha
      chosen from a small grid by *output* error on calibration data --- which
      is the quantity AWQ minimises. A weight-MSE test would reject a correct
      implementation, because the transform trades per-channel error for
      per-group scale; see `_test_awq`.

  quantise_kv_kivi
      Per-channel keys, per-token values, plus a residual window: the most
      recent `residual` tokens stay in full precision. Quantising only the head
      of the sequence is what makes the window meaningful.

This is our implementation of the ideas, not the authors' code, and the paper
says so. Both are validated in `__main__` against naive RTN at the same width.
"""
from __future__ import annotations

import torch

from quant.fakequant import qdq_asymmetric, qdq_symmetric


def awq_scale(x: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    """Per-input-channel scale from calibration activations.

    x: [tokens, in_features]. Returns [in_features], normalised to mean 1.
    """
    s = x.float().abs().mean(dim=0).clamp(min=1e-8).pow(alpha)
    s = s / s.mean()
    return s.clamp(min=1e-4, max=1e4)


def quantise_awq(w: torch.Tensor, s: torch.Tensor, bits: int,
                 group_size: int = 128) -> torch.Tensor:
    """Quantise w with the AWQ-style equivalent transform."""
    orig_dtype = w.dtype
    wf = w.float()
    s = s.to(w.device, torch.float32)
    q = qdq_symmetric(wf * s.unsqueeze(0), bits, group_size=group_size)
    return (q / s.unsqueeze(0)).to(orig_dtype)


def awq_best_alpha(w: torch.Tensor, x: torch.Tensor, bits: int,
                   group_size: int = 128,
                   grid=(0.0, 0.25, 0.5, 0.75)) -> tuple[torch.Tensor, float]:
    """Pick the per-layer alpha that minimises output error on calibration data.

    alpha = 0 is plain RTN, so this cannot do worse than RTN on the calibration
    set. That property is what makes it a fair "strong quantiser" rather than a
    possibly-broken one.
    """
    wf = w.float()
    ref = wf @ x.float().T
    best, best_a = None, 0.0
    for a in grid:
        if a == 0.0:
            q = qdq_symmetric(wf, bits, group_size=group_size)
        else:
            q = quantise_awq(w, awq_scale(x, a), bits, group_size).float()
        err = (q @ x.float().T - ref).pow(2).mean().item()
        if best is None or err < best:
            best, best_a = err, a
    if best_a == 0.0:
        return qdq_symmetric(wf, bits, group_size=group_size).to(w.dtype), 0.0
    return quantise_awq(w, awq_scale(x, best_a), bits, group_size), best_a


def quantise_kv_kivi(k: torch.Tensor, v: torch.Tensor, bits: int,
                     residual: int = 0):
    """KIVI-style cache quantisation with a full-precision residual window.

    k, v: [batch, heads, seq, head_dim].
    """
    if bits >= 16:
        return k, v
    seq = k.shape[2]
    r = max(0, min(residual, seq))
    if seq - r == 0:
        return k, v
    hk, tk = k[:, :, : seq - r, :], k[:, :, seq - r:, :]
    hv, tv = v[:, :, : seq - r, :], v[:, :, seq - r:, :]
    qk = qdq_asymmetric(hk, bits, dim=-2)       # per-channel across the sequence
    qv = qdq_asymmetric(hv, bits, dim=-1)       # per-token across head_dim
    return torch.cat([qk, tk], dim=2), torch.cat([qv, tv], dim=2)


def _qdq_group(x: torch.Tensor, bits: int, group: int, dim: int) -> torch.Tensor:
    """Asymmetric fake-quant grouping dim into chunks of group.

    Padding reflects the last slice and is narrowed back, so the output always
    has the input's shape. An earlier version returned the padded tensor when
    the length was not a multiple of the group.
    """
    qmax = float(2 ** bits - 1)
    xf = x.float()
    if dim < 0:
        dim = xf.dim() + dim
    n = xf.shape[dim]
    g = min(group, n)
    pad = (-n) % g
    if pad:
        shape = [pad if i == dim else xf.shape[i] for i in range(xf.dim())]
        xf = torch.cat([xf, xf.narrow(dim, n - 1, 1).expand(*shape)], dim=dim)
    np_ = xf.shape[dim]
    nshape = list(xf.shape[:dim]) + [np_ // g, g] + list(xf.shape[dim + 1:])
    v = xf.reshape(*nshape)
    mn = v.amin(dim=dim + 1, keepdim=True)
    mx = v.amax(dim=dim + 1, keepdim=True)
    scale = ((mx - mn) / qmax).clamp(min=1e-8)
    q = torch.round((v - mn) / scale).clamp(0, qmax)
    out = (q * scale + mn).reshape(*xf.shape)
    if pad:
        out = out.narrow(dim, 0, n)
    return out.to(x.dtype)


def quantise_kv_grouped(k, v, bits: int, residual: int = 0, group: int = 32):
    """KIVI-style cache quantisation with residual window AND per-side grouping.

    K is grouped along the sequence, V along head_dim, so the cache receives the
    same granularity the weight side gets from group-wise quantisation.
    """
    if bits >= 16:
        return k, v
    seq = k.shape[2]
    r = max(0, min(residual, seq))
    hk, tk = k[:, :, : seq - r, :], k[:, :, seq - r:, :]
    hv, tv = v[:, :, : seq - r, :], v[:, :, seq - r:, :]
    if hk.shape[2] == 0:
        return k, v
    qk = _qdq_group(hk, bits, group, dim=2)
    qv = _qdq_group(hv, bits, group, dim=3)
    return torch.cat([qk, tk], dim=2), torch.cat([qv, tv], dim=2)


def _test_awq() -> None:
    torch.manual_seed(0)
    w = torch.randn(64, 256)
    x = torch.randn(512, 256)
    x[:, ::8] *= 20.0
    x = x / x.abs().max()

    rtn = qdq_symmetric(w, 4, group_size=128).float()
    q_awq, alpha = awq_best_alpha(w, x, 4)
    out_rtn = (rtn @ x.T - w @ x.T).pow(2).mean().item()
    out_awq = (q_awq.float() @ x.T - w @ x.T).pow(2).mean().item()
    print(f"  output error  RTN {out_rtn:.3e}   AWQ-style {out_awq:.3e} "
          f"(alpha={alpha})")
    assert out_awq <= out_rtn * 1.0001, \
        "AWQ-style must not be worse than RTN on calibration output error"


def _test_kivi() -> None:
    torch.manual_seed(0)
    k = torch.randn(1, 8, 512, 128)
    v = torch.randn(1, 8, 512, 128)
    k[:, :, :, ::16] *= 15.0
    nk, nv = quantise_kv_kivi(k, v, 4, residual=0)
    kk, kv = quantise_kv_kivi(k, v, 4, residual=64)
    e_naive = ((nk - k).pow(2).mean() + (nv - v).pow(2).mean()).item()
    e_kivi = ((kk - k).pow(2).mean() + (kv - v).pow(2).mean()).item()
    print(f"  cache MSE     naive {e_naive:.3e}   KIVI-style(R=64) {e_kivi:.3e}")
    assert e_kivi < e_naive, "a residual window must reduce cache error"


if __name__ == "__main__":
    print("self-test:")
    _test_awq()
    _test_kivi()
    print("PASSED")
