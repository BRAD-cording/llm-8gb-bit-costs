"""
GPTQ-style second-order weight quantisation (Hessian / OBQ error compensation).

Why this exists: the paper's strength-matched comparison uses our own AWQ-style
activation scaling. A reviewer can argue that our AWQ is weaker than the
published one. GPTQ works by a different mechanism --- it uses second-order
information to compensate the error each column introduces --- so if the
ordering survives GPTQ too, the conclusion stops depending on any single
quantiser family.

Algorithm (per linear layer), following the published outline:
  1. H = X^T X from calibration inputs, damped by `damp * mean(diag(H))`.
  2. Hinv = inverse Cholesky factor.
  3. For each column j in order:
        q_j = round-to-nearest of W[:, j] on its group grid
        err = (W[:, j] - q_j) / Hinv[j, j]
        W[:, j+1:] -= err[:, None] * Hinv[j, j+1:][None, :]
This is the standard error-feedback form; we omit dynamic clamping and
act-order, and say so in the paper.

Correctness gate: `__main__` asserts GPTQ beats RTN on the same layer at the same
width and group size. If that assertion fails, the implementation is wrong and
must not be used to make any claim.
"""
from __future__ import annotations

import torch

from quant.fakequant import qdq_symmetric


def _group_grid(w: torch.Tensor, bits: int, group_size: int):
    """Asymmetric per-group (min, scale) along the input axis, from the weights."""
    out_f, in_f = w.shape
    g = min(group_size, in_f)
    pad = (-in_f) % g
    wf = w.float()
    if pad:
        wf = torch.cat([wf, wf[:, -1:].expand(out_f, pad)], dim=1)
    v = wf.reshape(out_f, wf.shape[1] // g, g)
    mn = v.amin(dim=2, keepdim=True)
    mx = v.amax(dim=2, keepdim=True)
    qmax = float(2 ** bits - 1)
    scale = ((mx - mn) / qmax).clamp(min=1e-8)
    return mn, scale, g, pad


def _quant_group(w: torch.Tensor, mn, scale, g, pad, bits: int) -> torch.Tensor:
    """Round-to-nearest onto the group grid, shape-preserving."""
    out_f, in_f = w.shape
    wf = w.float()
    if pad:
        wf = torch.cat([wf, wf[:, -1:].expand(out_f, pad)], dim=1)
    v = wf.reshape(out_f, wf.shape[1] // g, g)
    qmax = float(2 ** bits - 1)
    q = torch.round((v - mn) / scale).clamp(0, qmax)
    out = (q * scale + mn).reshape(out_f, wf.shape[1])
    if pad:
        out = out[:, :in_f]
    return out


def gptq_quantise(w: torch.Tensor, x: torch.Tensor | None = None, bits: int = 4,
                  group_size: int = 128, damp: float = 0.01,
                  blocksize: int = 128,
                  H: torch.Tensor | None = None) -> torch.Tensor:
    """Quantise `w` [out, in] from calibration inputs `x` [tokens, in] or a
    precomputed Hessian ``H = x^T x`` [in, in].

    The Hessian form exists because a forward hook can accumulate H cheaply,
    whereas storing activations for every layer would cost about a gigabyte on
    the 1.7B model.
    """
    orig_dtype = w.dtype
    W = w.detach().float().clone()
    in_f = W.shape[1]
    if H is None:
        if x is None:
            raise ValueError("need either x or H")
        X = x.detach().float()
        if X.shape[-1] != W.shape[1]:
            raise ValueError(
                f"calibration width {X.shape[-1]} != weight width {W.shape[1]}")
        H = X.T @ X
    else:
        H = H.detach().float().clone()
        if tuple(H.shape) != (in_f, in_f):
            raise ValueError(f"H shape {tuple(H.shape)} != {(in_f, in_f)}")

    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0
    H += damp * torch.diag(H).mean() * torch.eye(in_f, device=H.device, dtype=H.dtype)

    Hinv = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(Hinv)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)   # upper Cholesky of H^-1

    mn, scale, g, pad = _group_grid(W, bits, group_size)
    Q = torch.zeros_like(W)

    for i1 in range(0, in_f, blocksize):
        i2 = min(i1 + blocksize, in_f)
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        E1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]

        for i in range(i2 - i1):
            col = i1 + i
            w_col = W1[:, i]
            m = mn[:, col // g, :]              # [out, 1] for this group
            sc = scale[:, col // g, :]
            qmax = float(2 ** bits - 1)
            q_col = (torch.round((w_col.unsqueeze(1) - m) / sc)
                     .clamp(0, qmax) * sc + m).squeeze(1)
            Q1[:, i] = q_col
            err = (w_col - q_col) / Hinv1[i, i]
            W1[:, i:] -= err.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
            E1[:, i] = err
        Q[:, i1:i2] = Q1
        if i2 < in_f:
            W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]

    return Q.to(orig_dtype)


def _test() -> None:
    torch.manual_seed(0)
    out_f, in_f, n = 64, 256, 512
    w = torch.randn(out_f, in_f)
    x = torch.randn(n, in_f)
    bits, g = 4, 32

    rtn = qdq_symmetric(w, bits, group_size=g).float()
    gq = gptq_quantise(w, x, bits, g).float()

    ref = w @ x.T
    e_rtn = (rtn @ x.T - ref).pow(2).mean().item()
    e_gptq = (gq @ x.T - ref).pow(2).mean().item()
    print(f"  output error  RTN {e_rtn:.4e}   GPTQ {e_gptq:.4e}  "
          f"({e_rtn/e_gptq:.2f}x better)")
    assert e_gptq < e_rtn, "GPTQ must beat RTN; implementation is wrong otherwise"

    # group size 128 too, since that is what the paper's tables use
    rtn2 = qdq_symmetric(w, bits, group_size=128).float()
    gq2 = gptq_quantise(w, x, bits, 128).float()
    e_rtn2 = (rtn2 @ x.T - ref).pow(2).mean().item()
    e_gq2 = (gq2 @ x.T - ref).pow(2).mean().item()
    print(f"  g128          RTN {e_rtn2:.4e}   GPTQ {e_gq2:.4e}  "
          f"({e_rtn2/e_gq2:.2f}x better)")
    assert e_gq2 < e_rtn2


if __name__ == "__main__":
    print("self-test:")
    _test()
    print("PASSED")
