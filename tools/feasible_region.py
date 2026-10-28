"""
Feasible-region computation: what can actually run on each device?

This is the paper's Figure 1.  It combines three MEASURED quantities:

    M_total        nominal device memory
    rho            fraction unavailable to the model (CUDA context, desktop, carveout)
    KV bytes/token per model        (from the real config.json)

with a workspace allowance, and reports, for every (model, weight bits, KV bits),
the maximum context length that fits -- on the RTX 4060 and on the Jetson.

The point: rho differs by 3x between the two devices (0.133 vs 0.419), so the
EFFECTIVE budget ratio is ~1.6x even though both are nominally 8 GB.  Any paper
that says "we test on 8GB devices" without reporting rho is reporting a number
that does not transfer.

Usage:  python tools/feasible_region.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CROSS = ROOT / "results" / "kv_crossover.json"

# Measured on 2026-10-20 (see results/rho_findings.md for the raw numbers)
DEVICES = {
    "RTX4060": {"total_GiB": 8.00, "free_GiB": 6.93, "rho": 0.133},
    "JetsonOrinNanoSuper": {"total_GiB": 7619 / 1024, "free_GiB": 4426 / 1024,
                            "rho": 1 - 4426 / 7619},
}

# MEASURED, not assumed: results/QWEN3_4B_8GB_LIMIT.md fits the prefill workspace
# at 0.23 MB per token in flight, with a ~170 MB stack-constant.  An earlier
# version of this file used a flat 0.6 GB and therefore overestimated the
# attainable context by ~3x -- see docs/ARTIFACT.md section 4.
WS_MB_PER_TOKEN = 0.23
WS_CONST_MB = 170.0
PREFILL_CHUNK = 2048

KV_BITS = (16, 8, 4, 2)
W_BITS = (8, 4, 3, 2)


def workspace_gb(seq_len: int) -> float:
    tokens_in_flight = min(seq_len, PREFILL_CHUNK)
    return (WS_MB_PER_TOKEN * tokens_in_flight + WS_CONST_MB) / 1024


def l_max(budget_gb: float, params_M: float, kv_bytes_per_token: float,
          w_bits: int, kv_bits: int) -> int:
    w_gb = params_M * 1e6 * w_bits / 8 / 1e9
    kv_per_tok = kv_bytes_per_token * kv_bits / 16 / 1e9
    left = budget_gb - w_gb
    if left <= 0:
        return 0
    # workspace depends on L only through min(L, chunk); solve directly
    lo, hi = 0, 2_000_000
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if kv_per_tok * mid + workspace_gb(mid) <= left:
            lo = mid
        else:
            hi = mid - 1
    return lo


def main() -> int:
    data = json.loads(CROSS.read_text(encoding="utf-8"))
    print(f"workspace model: {WS_MB_PER_TOKEN} MB/token in flight, "
          f"+{WS_CONST_MB:.0f} MB stack constant, chunk={PREFILL_CHUNK}\n")
    for dev, d in DEVICES.items():
        print(f"### {dev}: total {d['total_GiB']:.2f} GiB, free {d['free_GiB']:.2f} GiB, "
              f"rho={d['rho']:.3f}  -> budget {d['free_GiB']*1.0737:.2f} GB")

    for model, m in data.items():
        info = m["info"]
        print(f"\n## {model}  (params~{info['params_M']}M, "
              f"KV={info['kv_bytes_per_token']/1024:.0f} KiB/token @16bit)")
        print(f"{'Wbits':>5} {'KVbits':>6} {'W(GB)':>6} | "
              f"{'Lmax 4060':>10} {'Lmax Jetson':>12} | {'ratio':>6}")
        for wb in W_BITS:
            for kb in KV_BITS:
                l1 = l_max(DEVICES["RTX4060"]["free_GiB"] * 1.0737,
                           info["params_M"], info["kv_bytes_per_token"], wb, kb)
                l2 = l_max(DEVICES["JetsonOrinNanoSuper"]["free_GiB"] * 1.0737,
                           info["params_M"], info["kv_bytes_per_token"], wb, kb)
                if l1 == 0 and l2 == 0:
                    continue
                ratio = (l1 / l2) if l2 else float("inf")
                w_gb = info["params_M"] * 1e6 * wb / 8 / 1e9
                print(f"{wb:>5} {kb:>6} {w_gb:>6.2f} | {l1:>10,} {l2:>12,} | {ratio:>6.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
