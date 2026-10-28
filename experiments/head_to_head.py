"""
Head-to-head at fixed context: the literature default versus the allocation.

The paper argues from per-bit costs. A reviewer will ask the blunter question:
at a given context length on this device, how much accuracy does the published
practice actually give up? This computes it directly from the measured curves,
with no new measurement:

  default   : 4-bit weights, 16-bit cache  (what weight-only quantisation does)
  allocated : the error-minimising choice under the same byte budget
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from alloc.dp_alloc import BIT_CHOICES, Device, kv_bytes, load_items, \
    min_error_allocation, weight_bytes  # noqa: E402


def default_cost(items, bits_w=4, bits_kv=8, seq_len=32768) -> float:
    """Error of the published practice: uniform weight width, full cache."""
    return sum(it.err.get(bits_w if it.kind == "w" else bits_kv, 0.0)
               for it in items)


def default_bytes(items, bits_w=4, bits_kv=8, seq_len=32768) -> float:
    tot = 0.0
    for it in items:
        b = (weight_bytes(it.n_params, bits_w) if it.kind == "w"
             else kv_bytes(it.n_params, bits_kv, seq_len))
        tot += b
    return tot / 1024**2


def main() -> int:
    items, meta = load_items(str(ROOT / "results" / "items_qwen06_v2.json"))
    devs = {
        "RTX4060": Device("RTX4060", 22.36, 179.8, 8.0, rho=0.133,
                          ws_mb_per_token=0.142, ws_const_mb=9.0),
        "Jetson": Device("Jetson", 12.0, 60.0, 7.44, rho=0.419,
                         ws_mb_per_token=0.142, ws_const_mb=9.0),
    }
    rows = []
    for L in (16384, 32768, 65536):
        for name, dev in devs.items():
            budget = dev.usable_mb(L)
            dflt_err = default_cost(items, 4, 8, L)
            dflt_mb = default_bytes(items, 4, 8, L)
            best = min_error_allocation(items, dev, L, nbuckets=2500)
            if best is None:
                rows.append({"device": name, "seq_len": L, "fits": False})
                print(f"{name:9s} L={L:6d}  allocation infeasible")
                continue
            ratio = (best["error"] / dflt_err) if dflt_err > 0 else 0.0
            rows.append({"device": name, "seq_len": L, "budget_MB": round(budget, 1),
                         "default_err_pct": round(100 * dflt_err, 2),
                         "default_MB": round(dflt_mb, 1),
                         "default_fits": dflt_mb <= budget,
                         "alloc_err_pct": round(100 * best["error"], 2),
                         "alloc_MB": best["bytes_MB"],
                         "alloc_W": best["weight_bits_mean"],
                         "alloc_KV": best["kv_bits_mean"],
                         "error_ratio": round(ratio, 3)})
            print(f"{name:9s} L={L:6d}  budget {budget:6.0f}MB  "
                  f"default W4/KV16: {100*dflt_err:6.2f}% ({dflt_mb:5.0f}MB"
                  f"{'' if dflt_mb <= budget else ' OVER'})  "
                  f"allocated: {100*best['error']:6.2f}% "
                  f"(W{best['weight_bits_mean']}/KV{best['kv_bits_mean']}, "
                  f"{best['bytes_MB']:5.0f}MB)  ratio {ratio:.2f}")

    out = ROOT / "results" / "head_to_head.json"
    out.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
