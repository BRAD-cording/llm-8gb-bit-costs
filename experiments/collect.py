"""
Collect every measurement JSON in results/ into one tidy CSV.

The paper's system tables are generated from this file, so a table can never
drift from the raw data.  Runs offline; the JSONs are produced by
`src/bench/regime_sweep.py` and `src/bench/synth_sweep.py`.

Usage:
    python experiments/collect.py --out results/system_matrix.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

FIELDS = ["source", "model", "precision", "attn", "device", "seq_len",
          "ttft_ms", "tpot_ms", "tok_per_s", "weights_MB", "kv_MB",
          "peak_prefill_MB", "peak_decode_MB", "achieved_GBps",
          "arith_intensity", "decode_regime", "spilled", "error"]


def infer_attn(path: Path, meta: dict) -> str:
    """The attention path is part of the file name by convention."""
    name = path.stem
    if "gqa" in name:
        return "gqa_sdpa"
    return "sdpa"


def rows_from(path: Path) -> list[dict]:
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if "rows" not in meta:
        return []
    out = []
    attn = infer_attn(path, meta)
    for r in meta["rows"]:
        row = {k: r.get(k, "") for k in FIELDS}
        row["source"] = path.name
        row["model"] = meta.get("model", "")
        row["precision"] = meta.get("precision", "")
        row["attn"] = attn
        row["device"] = meta.get("device", "")
        out.append(row)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/system_matrix.csv")
    args = ap.parse_args()

    paths = sorted(list(RESULTS.glob("synth_*.json")) + list(RESULTS.glob("sweep_*.json")),
                   key=lambda p: p.stat().st_mtime)
    rows = []
    # newest file wins for a given (model, precision, attn, L): superseded runs
    # must not silently mix with the current protocol
    seen: dict[tuple, int] = {}
    for p in paths:
        got = rows_from(p)
        if not got:
            continue
        print(f"  {p.name}: {len(got)} rows")
        for r in got:
            key = (r["model"], r["precision"], r["attn"], r["seq_len"])
            if key in seen:
                rows[seen[key]] = r
            else:
                seen[key] = len(rows)
                rows.append(r)

    if not rows:
        print("no measurement rows found")
        return 1

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    valid = [r for r in rows if not r["spilled"] and not r["error"]]
    print(f"\n{len(rows)} rows total ({len(valid)} valid, "
          f"{len(rows)-len(valid)} flagged)")
    print(f"wrote {out}")

    # pivot per (model, precision) so each ablation is visible, flagged runs included
    groups: dict[tuple[str, str], dict[int, dict[str, str]]] = {}
    for r in rows:
        key = (str(r["model"]), str(r["precision"]))
        L = int(r["seq_len"])
        if r["error"]:
            cell = "OOM"
        elif str(r["spilled"]).lower() == "true":
            cell = "spill"
        else:
            cell = r["ttft_ms"]
        groups.setdefault(key, {}).setdefault(L, {})[r["attn"]] = cell

    for (model, prec), piv in sorted(groups.items()):
        if not any("gqa_sdpa" in v for v in piv.values()):
            continue
        print(f"\n=== stack ablation: {model} / {prec} (TTFT ms) ===")
        print(f"{'L':>7} {'sdpa':>10} {'gqa_sdpa':>10} {'speedup':>8}")
        for L in sorted(piv):
            d = piv[L]
            a, b = d.get("sdpa", "-"), d.get("gqa_sdpa", "-")
            try:
                sp = f"{float(a)/float(b):.1f}x"
            except (TypeError, ValueError):
                sp = "-"
            print(f"{L:>7} {str(a):>10} {str(b):>10} {sp:>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
