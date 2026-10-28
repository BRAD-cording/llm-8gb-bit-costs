"""
Budget-constrained bit allocation for weights + KV, with regime-aware objective.

    minimise   alpha * T_prefill(b) + (1-alpha) * T_decode(b)
    s.t.       bytes_weights(b) + bytes_kv(b) <= M * (1 - rho)
               sum_l err_l(b_l) <= epsilon

WHY THIS IS NOT TRIVIAL
-----------------------
If both T_prefill and T_decode were plain "bytes / B_eff", the two objectives
would be collinear, alpha would cancel, and the whole regime story would be
vacuous.  The regime only exists because low bit-widths buy bandwidth but COST
compute (dequantisation, no INT4 tensor-core path on Ada SM89 / Ampere SM87).

So the cost model carries two measured, bit-dependent efficiency factors:

    eta_c(b)  compute efficiency  : effective TFLOP/s relative to FP16
    eta_b(b)  bandwidth efficiency: effective GB/s relative to FP16 traffic

Both are to be measured on each device (W4); the defaults below are documented
placeholders chosen from the probe results, NOT paper numbers.

Decode  (batch=1, bandwidth-bound):  t_dec = sum_i bytes_i(b_i) / (B * eta_b(b_i))
Prefill (batch=B, compute-bound):    t_pre = max( bytes/(B*eta_b), flops/(P*eta_c(b_w)) )

Because b* = P_eff/B_eff differs per device, the alpha at which the optimal
allocation flips differs per device -> the reversal this paper claims.
"""
from __future__ import annotations

import math

from dataclasses import dataclass, field
from typing import Sequence

BIT_CHOICES = (2, 3, 4, 8)

# Placeholder efficiency tables -- replace with measured curves in W4.
# fp16 reference = 1.0 for both.
ETA_C = {8: 1.00, 4: 0.62, 3: 0.40, 2: 0.26}   # low bits cost compute (dequant)
ETA_B = {8: 1.00, 4: 0.95, 3: 0.90, 2: 0.85}   # low bits buy bandwidth, not linearly


@dataclass
class Item:
    name: str
    kind: str          # "w" | "kv"
    n_params: int      # weight: element count; kv: elements per token per group
    err: dict[int, float] = field(default_factory=dict)
    flops_per_elem: float = 2.0


def weight_bytes(n: int, bits: int, group_size: int = 128) -> float:
    groups = max(1, n // group_size)
    return n * bits / 8.0 + groups * 2 * 2


def kv_bytes(n_per_token: int, bits: int, seq_len: int) -> float:
    """n_per_token already counts K and V (= 2*d_model/n_groups). No extra x2."""
    return n_per_token * seq_len * bits / 8.0


@dataclass
class Device:
    name: str
    p_eff_tflops: float
    bw_gbps: float
    mem_gib: float
    rho: float = 0.133          # measured unusable fraction (see RHO_FINDINGS.md)
    ws_mb_per_token: float = 0.23   # measured workspace growth (QWEN3_4B_8GB_LIMIT.md)
    ws_const_mb: float = 170.0      # stack-related fixed workspace
    prefill_chunk: int = 2048       # chunked prefill bounds the workspace
    eta_c: dict[int, float] = field(default_factory=lambda: dict(ETA_C))
    eta_b: dict[int, float] = field(default_factory=lambda: dict(ETA_B))

    @property
    def crossover_batch(self) -> float:
        return self.p_eff_tflops * 1e12 / (self.bw_gbps * 1e9)

    def usable_mb(self, seq_len: int) -> float:
        """Bytes actually available for weights + KV, after OS overhead and the
        measured activation/workspace curve.

        The workspace is proportional to the number of tokens in ONE forward
        pass, not to the conversation length: with chunked prefill it is
        min(L, chunk).  Fitting the full prompt in a single pass (chunk = L) is
        what makes 32k look infeasible, and that is a property of the serving
        stack, not of the device.
        """
        tokens_in_flight = min(seq_len, self.prefill_chunk)
        return (self.mem_gib * (1 - self.rho) * 1024
                - self.ws_mb_per_token * tokens_in_flight - self.ws_const_mb)

    def t_decode(self, w_bytes: float, kv_byt: float, choices: dict[str, int],
                 items: dict[str, Item]) -> float:
        t = 0.0
        for name, bits in choices.items():
            it = items[name]
            b = w_bytes if it.kind == "w" else kv_byt
            # recompute this item's own bytes rather than guessing shares
            nb = (weight_bytes(it.n_params, bits) if it.kind == "w"
                  else kv_bytes(it.n_params, bits, self._seq_len))
            t += nb / (self.bw_gbps * 1e9 * self.eta_b[bits])
        return t

    def t_prefill(self, choices: dict[str, int], items: dict[str, Item],
                  batch: int) -> float:
        total_bytes = 0.0
        flops = 0.0
        flops_scaled = 0.0
        for name, bits in choices.items():
            it = items[name]
            nb = (weight_bytes(it.n_params, bits) if it.kind == "w"
                  else kv_bytes(it.n_params, bits, self._seq_len))
            total_bytes += nb
            if it.kind == "w":
                f = batch * it.n_params * it.flops_per_elem
                flops += f
                flops_scaled += f / self.eta_c[bits]
        t_mem = total_bytes / (self.bw_gbps * 1e9)
        t_compute = flops_scaled / (self.p_eff_tflops * 1e12)
        return max(t_mem, t_compute)


def min_error_allocation(items: Sequence[Item], device: Device, seq_len: int,
                         nbuckets: int = 600) -> dict | None:
    """Smallest achievable error at a given context length.

    This is the objective the measurements actually justify: 4-bit weights gave
    a 1.00x prefill speedup and a *slower* decode (see
    results/P2_FALSIFIED_NO_SPEEDUP.md), so latency is not a function of bit
    width on this stack.  What bit width does buy is capacity, so the allocation
    problem is: given a context length that must fit, minimise the accuracy loss.
    """
    device._seq_len = seq_len
    total_budget_mb = device.usable_mb(seq_len)
    if total_budget_mb <= 0:
        return None
    # every item costs at least one bucket, so the grid must be finer than the
    # item count or a feasible problem is reported as infeasible (bug found by
    # tests/test_alloc.py case 7)
    nbuckets = max(nbuckets, 4 * len(items))
    quantum_mb = total_budget_mb / nbuckets
    by_name = {it.name: it for it in items}

    def item_mb(it: Item, bits: int) -> float:
        b = (weight_bytes(it.n_params, bits) if it.kind == "w"
             else kv_bytes(it.n_params, bits, seq_len))
        return b / 1024**2

    states: dict[int, tuple[float, dict[str, int]]] = {0: (0.0, {})}
    for it in items:
        nxt: dict[int, tuple[float, dict[str, int]]] = {}
        for used, (err, ch) in states.items():
            for bits in BIT_CHOICES:
                if bits not in it.err:
                    continue
                # ceil, never round: rounding down lets the solver quietly
                # exceed the budget, and zeroing an item smaller than one
                # bucket loses its memory entirely (both were real bugs found
                # by tests/test_alloc.py)
                nb = used + max(1, math.ceil(item_mb(it, bits) / quantum_mb))
                if nb > nbuckets:
                    continue
                ne = err + it.err[bits]
                cur = nxt.get(nb)
                if cur is None or ne < cur[0] - 1e-12:
                    nch = dict(ch)
                    nch[it.name] = bits
                    nxt[nb] = (ne, nch)
        pruned: dict[int, tuple[float, dict[str, int]]] = {}
        best_err = float("inf")
        for nb, (ne, nch) in sorted(nxt.items()):
            if ne < best_err - 1e-12:
                pruned[nb] = (ne, nch)
                best_err = ne
        states = pruned
        if not states:
            return None

    nb, (err, ch) = min(states.items(), key=lambda kv: kv[1][0])
    kinds = {it.name: it.kind for it in items}
    wb = [v for k, v in ch.items() if kinds.get(k, "w") == "w"]
    kb = [v for k, v in ch.items() if kinds.get(k) == "kv"]
    if not wb or not kb:
        return {"seq_len": seq_len, "error": round(err, 5),
                "bytes_MB": 0.0, "budget_MB": round(total_budget_mb, 1),
                "weight_bits_mean": float("nan"),
                "kv_bits_mean": float("nan"), "choices": ch,
                "note": "item list lacks weight or KV units"}
    actual_mb = sum(item_mb(it, ch[it.name]) for it in items)
    return {"seq_len": seq_len, "error": round(err, 5),
            "bytes_MB": round(actual_mb, 1),
            "bucket_MB": round(nb * quantum_mb, 1),
            "budget_MB": round(total_budget_mb, 1),
            "weight_bits_mean": round(sum(wb) / len(wb), 2),
            "kv_bits_mean": round(sum(kb) / len(kb), 2),
            "choices": ch}


def max_context(items: Sequence[Item], device: Device, eps: float,
                lo: int = 1024, hi: int = 262144, nbuckets: int = 2500) -> dict:
    """Largest context length whose minimum achievable error stays under eps."""
    lo_r = min_error_allocation(items, device, lo, nbuckets)
    if lo_r is None or lo_r["error"] > eps:
        return {"device": device.name, "eps": eps, "max_L": 0,
                "note": "even the shortest context exceeds the error budget"}
    best = lo_r
    while hi - lo > max(256, lo // 64):   # relative tolerance was 1/8: at L~1e5 that left a 6k-wide band
        mid = (lo + hi) // 2
        r = min_error_allocation(items, device, mid, nbuckets)
        if r is not None and r["error"] <= eps:
            lo, best = mid, r
        else:
            hi = mid
    return {"device": device.name, "eps": eps, "max_L": lo,
            "error_at_max": best["error"],
            "weight_bits_mean": best["weight_bits_mean"],
            "kv_bits_mean": best["kv_bits_mean"],
            "bytes_MB": best["bytes_MB"], "budget_MB": best["budget_MB"]}


def load_items(path: str) -> tuple[list[Item], dict]:
    """Load the measured item list produced by src/alloc/build_items.py."""
    import json
    payload = json.loads(open(path, encoding="utf-8").read())
    items = [Item(it["name"], it["kind"], int(it["n_params"]),
                  {int(b): float(v) for b, v in it["err"].items()})
             for it in payload["items"]]
    return items, payload


def allocate(items: Sequence[Item], device: Device, seq_len: int, alpha: float,
             rho: float | None = None, eps: float = 0.05, prefill_batch: int = 1024,
             nbuckets: int = 1200, topk: int = 400) -> dict:
    device._seq_len = seq_len
    batch = prefill_batch  # prefill processes the prompt as one batch
    by_name = {it.name: it for it in items}
    total_budget_mb = device.usable_mb(seq_len)
    if total_budget_mb <= 0:
        return {"feasible": False, "reason": "workspace exceeds the whole budget"}
    quantum_mb = total_budget_mb / nbuckets

    def item_mb(it: Item, bits: int) -> float:
        b = (weight_bytes(it.n_params, bits) if it.kind == "w"
             else kv_bytes(it.n_params, bits, seq_len))
        return b / 1024**2

    # memory-vs-error DP, keeping the min-error choice for each byte bucket
    states: dict[int, tuple[float, dict[str, int]]] = {0: (0.0, {})}
    for it in items:
        nxt: dict[int, tuple[float, dict[str, int]]] = {}
        for used, (err, ch) in states.items():
            for bits in BIT_CHOICES:
                if bits not in it.err:
                    continue
                nb = used + int(round(item_mb(it, bits) / quantum_mb))
                if nb > nbuckets:
                    continue
                ne = err + it.err[bits]
                cur = nxt.get(nb)
                if cur is None or ne < cur[0] - 1e-12:
                    nch = dict(ch)
                    nch[it.name] = bits
                    nxt[nb] = (ne, nch)
        ordered = sorted(nxt.items())
        pruned: dict[int, tuple[float, dict[str, int]]] = {}
        best_err = float("inf")
        for nb, (ne, nch) in ordered:
            if ne < best_err - 1e-12:
                pruned[nb] = (ne, nch)
                best_err = ne
        states = pruned

    feasible = [(nb, ne, nch) for nb, (ne, nch) in states.items() if ne <= eps]
    if not feasible:
        min_err = min(ne for ne, _ in states.values())
        return {"feasible": False, "min_error_found": min_err,
                "reason": f"min error {min_err:.4f} > eps={eps} within budget"}

    scored = []
    for nb, ne, nch in feasible:
        t = (alpha * device.t_prefill(nch, by_name, batch)
             + (1 - alpha) * device.t_decode(0, 0, nch, by_name))
        scored.append({"bytes_MB": nb * quantum_mb, "error": ne,
                       "time_s": t, "choices": nch})
    # collapse to the time-Pareto front, keep a bounded number
    scored.sort(key=lambda r: r["time_s"])
    front, best_err = [], float("inf")
    for r in scored:
        if r["error"] < best_err - 1e-12:
            front.append(r)
            best_err = r["error"]
        if len(front) >= topk:
            break
    win = front[0]
    kinds = {it.name: it.kind for it in items}
    wbits = [v for k, v in win["choices"].items() if kinds.get(k, "w") == "w"]
    kbits = [v for k, v in win["choices"].items() if kinds.get(k) == "kv"]
    wbits = wbits or [0]
    kbits = kbits or [0]
    return {
        "feasible": True,
        "device": device.name,
        "seq_len": seq_len,
        "alpha": alpha,
        "bytes_MB": round(win["bytes_MB"], 1),
        "error": round(win["error"], 5),
        "time_ms": round(win["time_s"] * 1e3, 3),
        "crossover_batch": round(device.crossover_batch, 1),
        "weight_bits_mean": round(sum(wbits) / len(wbits), 2),
        "kv_bits_mean": round(sum(kbits) / len(kbits), 2),
        "weight_bits_hist": {b: wbits.count(b) for b in sorted(set(wbits))},
        "kv_bits_hist": {b: kbits.count(b) for b in sorted(set(kbits))},
    }


def synthetic_items(n_layers: int = 32, d_model: int = 4096, d_ff: int = 11008,
                    n_kv_groups: int = 8, seed: int = 0) -> list[Item]:
    """Stand-in sensitivity curves until per-layer calibration (W5-W6) lands."""
    import random
    rng = random.Random(seed)
    items = []
    for l in range(n_layers):
        s = 0.15 + 0.85 * (l / max(1, n_layers - 1))
        items.append(Item(f"L{l}.qkv", "w", d_model * 3 * d_model,
                          {2: 3.0 * s, 3: 1.2 * s, 4: 0.3 * s, 8: 0.0}))
        items.append(Item(f"L{l}.mlp", "w", 2 * d_model * d_ff,
                          {2: 2.6 * s, 3: 1.0 * s, 4: 0.25 * s, 8: 0.0}))
        for g in range(n_kv_groups):
            gs = rng.uniform(0.5, 1.5)
            items.append(Item(f"L{l}.kv{g}", "kv", d_model * 2 // n_kv_groups,
                              {2: 0.8 * gs, 3: 0.3 * gs, 4: 0.05 * gs, 8: 0.0}))
    # normalise so eps is interpretable as relative perplexity increase
    tot = len(items)
    for it in items:
        for b in it.err:
            it.err[b] /= tot * 4.0
    return items


if __name__ == "__main__":
    import sys
    if "--capacity" in sys.argv:
        items_path = None
        if "--items" in sys.argv:
            items_path = sys.argv[sys.argv.index("--items") + 1]
        nb = 2500
        if "--buckets" in sys.argv:
            nb = int(sys.argv[sys.argv.index("--buckets") + 1])
        ws_tok, ws_const = 0.23, 170.0
        if "--ws-token" in sys.argv:
            ws_tok = float(sys.argv[sys.argv.index("--ws-token") + 1])
        if "--ws-const" in sys.argv:
            ws_const = float(sys.argv[sys.argv.index("--ws-const") + 1])
        dev4060 = Device("RTX4060", p_eff_tflops=22.36, bw_gbps=179.8, mem_gib=8.0,
                         rho=0.133, ws_mb_per_token=ws_tok, ws_const_mb=ws_const)
        devjetson = Device("OrinNanoSuper", p_eff_tflops=12.0, bw_gbps=60.0,
                           mem_gib=7.44, rho=0.419, ws_mb_per_token=ws_tok,
                           ws_const_mb=ws_const)
        if items_path:
            items, meta = load_items(items_path)
            print(f"measured items: {len(items)} from {items_path}")
            print(f"  E_w  = {meta['E_w']}")
            print(f"  E_kv = {meta['E_kv']}\n")
        else:
            items = synthetic_items(n_layers=36, d_model=2560, d_ff=9728,
                                    n_kv_groups=8)
            print("synthetic items (placeholder)\n")
        print(f"{'device':14s} {'eps':>7s} {'max_L':>8s} {'err':>8s} "
              f"{'Wbits':>6s} {'KVbits':>6s} {'used/budget MB':>18s}")
        eps_list = [0.01, 0.02, 0.05, 0.10, 0.20] if items_path else \
                   [0.001, 0.005, 0.01, 0.02, 0.05]
        table_rows = []
        for eps in eps_list:
            for dev in (dev4060, devjetson):
                r = max_context(items, dev, eps, nbuckets=nb)
                table_rows.append({"device": dev.name, "eps": eps,
                                   **{k: r.get(k) for k in
                                      ("max_L", "error_at_max", "weight_bits_mean",
                                       "kv_bits_mean", "bytes_MB", "budget_MB")}})
                print(f"{dev.name:14s} {eps:7.3f} {r['max_L']:8d} "
                      f"{r.get('error_at_max', float('nan')):8.4f} "
                      f"{r.get('weight_bits_mean', float('nan')):6.2f} "
                      f"{r.get('kv_bits_mean', float('nan')):6.2f} "
                      f"{r.get('bytes_MB', 0):8.0f}/{r.get('budget_MB', 0):8.0f}")
        import json as _json, os as _os
        _out = "results/capacity_table.json"
        _os.makedirs("results", exist_ok=True)
        _json.dump({"items": items_path, "ws_token": ws_tok, "ws_const": ws_const,
                    "buckets": nb, "rows": table_rows},
                   open(_out, "w", encoding="utf-8"), indent=2)
        print(f"wrote {_out}")
        raise SystemExit(0)

    dev4060 = Device("RTX4060", p_eff_tflops=22.36, bw_gbps=179.8, mem_gib=8.0,
                     rho=0.133)
    devjetson = Device("OrinNanoSuper", p_eff_tflops=12.0, bw_gbps=60.0, mem_gib=7.44,
                       rho=0.419)
    items = synthetic_items(n_layers=36, d_model=2560, d_ff=9728, n_kv_groups=8)
    print(f"{'device':14s} {'L':>6s} {'alpha':>5s} {'MB':>8s} {'err':>7s} "
          f"{'ms':>9s} {'Wbits':>6s} {'KVbits':>6s}  b*")
    for dev in (dev4060, devjetson):
        for L in (2048, 32768):
            for alpha in (0.05, 0.5, 0.95):
                r = allocate(items, dev, seq_len=L, alpha=alpha)
                if not r.get("feasible"):
                    print(f"{dev.name:14s} {L:6d} {alpha:5.2f} INFEASIBLE: {r['reason']}")
                    continue
                print(f"{dev.name:14s} {L:6d} {alpha:5.2f} {r['bytes_MB']:8.0f} "
                      f"{r['error']:7.4f} {r['time_ms']:9.2f} "
                      f"{r['weight_bits_mean']:6.2f} {r['kv_bits_mean']:6.2f}  "
                      f"{r['crossover_batch']:.0f}")
