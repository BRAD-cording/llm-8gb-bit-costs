"""Allocator baselines: is the exact DP actually better than simple rules?

The paper argues the contribution is the measured inputs, not the optimiser, and
reports no allocation-method baseline. A reviewer is entitled to ask whether a
textbook rule would have produced the same answer. This script compares, on the
same measured item table and the same byte budget:

  shipped   weights 4-bit, cache fp16            (what published recipes do)
  uniform   one width for every item, widest that fits
  greedy    start everywhere at 2 bits, repeatedly take the upgrade with the
            best error reduction per byte until the budget is exhausted
  dp        exact multiple-choice knapsack (src/alloc/dp_alloc.py)

Reported at several context lengths; error is the summed per-item proxy the
allocator itself minimises, so this isolates the optimiser, not the model.

Usage:
  python src/alloc/alloc_baselines.py --items results/items_qwen06_v2.json \
      --lengths 4096,8192,32768 --out results/alloc_baselines_06b.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))

from alloc.dp_alloc import (  # noqa: E402
    BIT_CHOICES, Device, Item, kv_bytes, load_items, min_error_allocation,
    weight_bytes,
)

BITS = list(BIT_CHOICES)


def item_mb(it: Item, bits: int, seq_len: int) -> float:
    b = (weight_bytes(it.n_params, bits) if it.kind == "w"
         else kv_bytes(it.n_params, bits, seq_len))
    return b / 1024 ** 2


def total_err(items, assign) -> float:
    # a 16-bit cache is the reference: zero accuracy cost, no curve needed
    return float(sum(it.err.get(assign[it.name], 0.0) for it in items))


def budget_mb(device: Device, seq_len: int) -> float:
    return device.usable_mb(seq_len)


def pack(items, device, seq_len, assign_bits) -> dict:
    """Assign every item one bit width from assign_bits(kind) and report."""
    assign = {it.name: assign_bits(it) for it in items}
    used = sum(item_mb(it, assign[it.name], seq_len) for it in items)
    return {"bits": assign, "used_mb": round(used, 1),
            "err": round(total_err(items, assign), 6)}


def uniform(items, device, seq_len) -> dict | None:
    limit = budget_mb(device, seq_len)
    best = None
    for b in sorted(BITS, reverse=True):
        cand = {it.name: b for it in items}
        used = sum(item_mb(it, b, seq_len) for it in items)
        if used <= limit:
            best = {"bits": cand, "used_mb": round(used, 1),
                    "err": round(total_err(items, cand), 6)}
            break
    return best


def greedy(items, device, seq_len) -> dict | None:
    limit = budget_mb(device, seq_len)
    assign = {it.name: BITS[0] for it in items}
    used = sum(item_mb(it, BITS[0], seq_len) for it in items)
    if used > limit:
        return None
    while True:
        best = None
        for it in items:
            cur = assign[it.name]
            nxt = [b for b in BITS if b > cur]
            if not nxt:
                continue
            n = nxt[0]
            db = item_mb(it, n, seq_len) - item_mb(it, cur, seq_len)
            if db <= 0 or used + db > limit:
                continue
            gain = it.err[cur] - it.err[n]
            if gain <= 0:
                continue
            score = gain / db
            if best is None or score > best[0]:
                best = (score, it, n, db)
        if best is None:
            break
        _, it, n, db = best
        assign[it.name] = n
        used += db
    return {"bits": assign, "used_mb": round(used, 1),
            "err": round(total_err(items, assign), 6)}


def shipped(items, device, seq_len) -> dict:
    assign = {it.name: (4 if it.kind == "w" else 16) for it in items}
    used = sum(item_mb(it, assign[it.name], seq_len) for it in items)
    return {"bits": assign, "used_mb": round(used, 1),
            "err": round(total_err(items, assign), 6)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="results/items_qwen06_v2.json")
    ap.add_argument("--lengths", default="4096,8192,32768")
    ap.add_argument("--out", default="results/alloc_baselines_06b.json")
    ap.add_argument("--device", default="laptop")
    args = ap.parse_args()

    items, payload = load_items(args.items)
    if args.device == "laptop":
        dev = Device(name="RTX4060Laptop", p_eff_tflops=22.36, bw_gbps=161.9,
                     mem_gib=7.996, rho=0.133,
                     ws_mb_per_token=0.142, ws_const_mb=9.0)
    else:
        dev = Device(name="JetsonOrinNanoSuper", p_eff_tflops=4.51,
                     bw_gbps=54.12, mem_gib=7.44, rho=0.419,
                     ws_mb_per_token=0.142, ws_const_mb=9.0)

    rows = []
    for L in [int(x) for x in args.lengths.split(",") if x.strip()]:
        budget = budget_mb(dev, L)
        dp = min_error_allocation(items, dev, L)
        rec = {"length": L, "budget_mb": round(budget, 1),
               "shipped": shipped(items, dev, L),
               "uniform": uniform(items, dev, L),
               "greedy": greedy(items, dev, L)}
        if dp:
            # dp returns its own bookkeeping; recompute the summed error the
            # same way the baselines are scored
            assign = dp.get("choices") or dp.get("bits") or {}
            if assign:
                rec["dp"] = {"bits": assign,
                             "err": round(total_err(items, assign), 6),
                             "used_mb": round(sum(
                                 item_mb(it, assign[it.name], L) for it in items
                                 if it.name in assign), 1),
                             "budget_mb": round(dp.get("budget_MB", 0), 1),
                             "w_bits_mean": dp.get("weight_bits_mean"),
                             "kv_bits_mean": dp.get("kv_bits_mean")}
            else:
                rec["dp"] = {"raw": {k: v for k, v in dp.items() if k != "choices"}}
        for k in ("shipped", "uniform", "greedy", "dp"):
            r = rec.get(k)
            if isinstance(r, dict) and "err" in r:
                r["err_rel_dp"] = (
                    round(100 * (r["err"] / rec["dp"]["err"] - 1), 2)
                    if rec.get("dp", {}).get("err") else None)
        rows.append(rec)
        print(L, json.dumps({k: (v.get("used_mb"), v.get("err"), v.get("err_rel_dp"))
                             if isinstance(v, dict) else None
                             for k, v in rec.items() if k in
                             ("shipped", "uniform", "greedy", "dp")}), flush=True)

    out = {"items": args.items, "device": dev.name, "rho": dev.rho, "rows": rows}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("WROTE", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
