"""
Refit the workspace model from the measurements it is supposed to describe.

The 0.23 MB/token coefficient in Device came from a Qwen3-4B nf4 run, but the
capacity table is computed for Qwen3-0.6B. The coefficient is not
model-independent: it scales with hidden size and layer count. Using the 4B value
for a 0.6B model over-reserves workspace and therefore under-reports attainable
context.

This fits c and c0 from the measured peak memory of the fused 0.6B runs and
prints the implied coefficients, so the capacity table can be recomputed with the
matching ones.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def fit(points: list[tuple[int, float]]) -> tuple[float, float]:
    """Least squares for ws(L) = c*L + c0."""
    n = len(points)
    sx = sum(p[0] for p in points)
    sy = sum(p[1] for p in points)
    sxx = sum(p[0] ** 2 for p in points)
    sxy = sum(p[0] * p[1] for p in points)
    denom = n * sxx - sx * sx
    c = (n * sxy - sx * sy) / denom
    c0 = (sy - c * sx) / n
    return c, c0


def main() -> int:
    runs = {
        "Qwen3-0.6B fp16 (fused)": "sweep_qwen06_fp16_real.json",
        "Qwen3-0.6B nf4  (fused)": "sweep_qwen06_nf4_real.json",
        "Qwen3-4B    nf4  (fused)": "synth_qwen3_4b_nf4_gqa.json",
    }
    out = {}
    for label, fname in runs.items():
        p = RES / fname
        if not p.exists():
            print(f"  {label}: missing {fname}")
            continue
        d = json.loads(p.read_text("utf-8"))
        pts = []
        print(f"\n{label}")
        for r in d["rows"]:
            if "error" in r or r.get("spilled"):
                continue
            ws = r["peak_prefill_MB"] - r["weights_MB"] - r["kv_MB"]
            pts.append((r["seq_len"], ws))
            print(f"  L={r['seq_len']:6d}  peak={r['peak_prefill_MB']:8.0f} "
                  f"w={r['weights_MB']:7.0f} kv={r['kv_MB']:7.0f}  ws={ws:8.0f} MB")
        if len(pts) >= 2:
            c, c0 = fit(pts)
            per_token_kb = c * 1024
            print(f"  fit: ws(L) = {c:.4f} MB/token * L + {c0:.0f} MB "
                  f"({per_token_kb:.0f} KiB/token)")
            out[label] = {"c_mb_per_token": round(c, 5),
                          "c0_mb": round(c0, 2), "n_points": len(pts)}

    print("\n=== implied coefficient by model ===")
    for k, v in out.items():
        print(f"  {k:26s} {v['c_mb_per_token']:.4f} MB/token  (+{v['c0_mb']:.0f} MB)")

    (RES / "workspace_fit.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nwrote results/workspace_fit.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
