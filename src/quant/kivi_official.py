"""
The cache-side quantisation from KIVI, applied through their own code.

Why this file exists: the paper prices a cache bit against a weight bit, and both
sides should come from someone else's implementation where possible. The weight
side uses the published gptqmodel; this is the cache side.

  theirs  -- third_party/KIVI/utils_quant.py, vendored unmodified from
             github.com/jy-yuan/KIVI at commit 876b4d2 (Apache-2.0, licence kept
             alongside). It supplies the grouping, the ranges and the rounding.
  ours    -- the adapter: it maps KIVI's simulate-mode outputs back to a
             dequantised tensor and wraps the residual window around them.

Not used: KIVI's Triton kernels (`quant/new_pack.py`). There is no Triton on
Windows and their kernels target a 2024 Triton; the simulate path is the same
arithmetic without the packing. The comparison is of the quantisation scheme,
not of kernel speed.

One trap worth stating, because it is easy to get wrong and still produces
plausible-looking numbers: the two paths in their file use **opposite scale
conventions**. Keys (`quantize_by_channel_and_pack_cache`) compute
`scale = (mx-mn)/B` and quantise by dividing, so dequantisation multiplies.
Values (`quantize_and_pack`) compute `scale = B/(mx-mn)` and quantise by
multiplying, so dequantisation divides. Their own commented-out code and
`dequantize_and_unpack` confirm both.

KIVI ships no Qwen3 model file (only llama and mistral), so the attention
plumbing is ours either way; see `src/bench/gqa_patch.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "third_party" / "KIVI"))
import utils_quant as _uq  # noqa: E402  (vendored, unmodified)


def quantise_keys_kivi(k: torch.Tensor, bits: int, group_size: int = 32,
                       residual: int = 128) -> torch.Tensor:
    """Keys: per-channel ranges over groups of `group_size` sequence positions."""
    if bits >= 16 or k.shape[2] == 0:
        return k
    seq = k.shape[2]
    r = min(residual, seq)
    head, tail = k[:, :, : seq - r, :], k[:, :, seq - r:, :]
    if head.shape[2] == 0:
        return k

    bsz, n_heads, n, dh = head.shape
    h = head.contiguous()
    if h.device.type != "cuda":
        h = h.cuda()

    rounded, scale, mn = _uq.quantize_by_channel_and_pack_cache(
        h, group_size, bits, simulate=True)
    if scale.dim() == 2:
        scale = scale.unsqueeze(-1)
    if mn.dim() == 2:
        mn = mn.unsqueeze(-1)
    g = rounded.shape[-1]
    out = rounded.float() * scale + mn                 # key path: multiply back
    out = out.reshape(n_heads * dh, -1, g)
    flat = out.reshape(n_heads * dh, -1)[:, : bsz * n]
    q = flat.transpose(0, 1).reshape(bsz, n, n_heads, dh).transpose(1, 2)
    q = q.to(device=k.device, dtype=k.dtype)
    return torch.cat([q, tail], dim=2)


def quantise_values_kivi(v: torch.Tensor, bits: int, group_size: int = 32,
                         residual: int = 128) -> torch.Tensor:
    """Values: per-token, grouped along head_dim (`quantize_and_pack`)."""
    if bits >= 16 or v.shape[2] == 0:
        return v
    seq = v.shape[2]
    r = min(residual, seq)
    head, tail = v[:, :, : seq - r, :], v[:, :, seq - r:, :]
    if head.shape[2] == 0:
        return v

    bsz, n_heads, n, dh = head.shape

    # Their value path is called per token on (bsz, heads, 1, head_dim), and
    # process_input groups the flattened heads*head_dim vector in chunks of
    # group_size. Batching the tokens into the leading dimension is the same
    # arithmetic in one call, which matters because a Python loop over sequence
    # positions would make the measurement unusable.
    vt = head.permute(0, 2, 1, 3).contiguous().reshape(bsz * n, n_heads, 1, dh)
    if vt.device.type != "cuda":
        vt = vt.cuda()

    out_int, scale, mn = _uq.quantize_and_pack(vt, group_size, int(bits),
                                               simulate=True)
    # Their quantize_and_pack transposes the batch axis internally, so the
    # returned tensors are (groups, tokens, g) and (groups, tokens), not the
    # (tokens, groups, ...) that the call site suggests.
    if mn.dim() == 2:
        mn = mn.unsqueeze(-1)
    if scale.dim() == 2:
        scale = scale.unsqueeze(-1)
    q = out_int.float() / scale + mn                   # value path: divide back
    q = q.transpose(0, 1).reshape(bsz * n, -1)[:, : n_heads * dh]
    q = q.reshape(bsz, n, n_heads, dh).permute(0, 2, 1, 3)
    q = q.to(device=v.device, dtype=v.dtype)
    return torch.cat([q, tail], dim=2)


def quantise_kv_kivi_official(k: torch.Tensor, v: torch.Tensor, bits: int,
                              group_size: int = 32, residual: int = 128):
    return (quantise_keys_kivi(k, bits, group_size, residual),
            quantise_values_kivi(v, bits, group_size, residual))


def _test() -> None:
    torch.manual_seed(0)
    k = torch.randn(1, 8, 512, 128)
    v = torch.randn(1, 8, 512, 128)
    k[:, :, :, ::16] *= 15.0

    qk, qv = quantise_kv_kivi_official(k, v, 4, group_size=32, residual=128)
    assert qk.shape == k.shape and qv.shape == v.shape, (qk.shape, qv.shape)
    assert torch.equal(qk[:, :, -128:], k[:, :, -128:]), "residual touched (keys)"
    assert torch.equal(qv[:, :, -128:], v[:, :, -128:]), "residual touched (values)"
    assert not torch.equal(qk[:, :, :-128], k[:, :, :-128]), "keys not quantised"
    assert not torch.equal(qv[:, :, :-128], v[:, :, :-128]), "values not quantised"

    e_k = (qk[:, :, :-128] - k[:, :, :-128]).pow(2).mean().item()
    e_v = (qv[:, :, :-128] - v[:, :, :-128]).pow(2).mean().item()
    var_k, var_v = k.var().item(), v.var().item()
    print(f"  kivi official: key MSE {e_k:.4e} (var {var_k:.4e})"
          f"  value MSE {e_v:.4e} (var {var_v:.4e})")
    # A sensible 4-bit quantiser sits far below the signal variance. A reversed
    # scale convention yields an enormous error, so this catches that trap.
    assert e_k < 0.25 * var_k, "key MSE implausible for 4 bits"
    assert e_v < 0.25 * var_v, "value MSE implausible for 4 bits"

    from quant.fakequant import qdq_asymmetric
    nk = qdq_asymmetric(k[:, :, :-128], 4, dim=-2)
    nv = qdq_asymmetric(v[:, :, :-128], 4, dim=-1)
    e_naive = ((nk - k[:, :, :-128]).pow(2).mean()
               + (nv - v[:, :, :-128]).pow(2).mean()).item()
    print(f"  grouped part: naive {e_naive:.4e}  kivi {e_k + e_v:.4e}")
    assert e_k + e_v <= e_naive * 1.05, "KIVI grouping should not be much worse than RTN"


if __name__ == "__main__":
    print("self-test:")
    _test()
    print("PASSED")
