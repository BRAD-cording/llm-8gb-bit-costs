"""Property tests for the allocation solver.

The solver is the paper's method, so its invariants are checked rather than
assumed.  Run:  python tests/test_alloc.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from alloc.dp_alloc import (BIT_CHOICES, Device, Item, allocate, kv_bytes,
                            min_error_allocation, synthetic_items, weight_bytes)

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' — ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


def small_items() -> list[Item]:
    # 4 layers x 2 weight matrices + 2 KV groups, errors normalised so eps bites
    items = []
    for l in range(4):
        items.append(Item(f"L{l}.qkv", "w", 1024 * 1024,
                          {2: 0.30, 3: 0.12, 4: 0.03, 8: 0.0}))
        items.append(Item(f"L{l}.mlp", "w", 2 * 1024 * 1024,
                          {2: 0.26, 3: 0.10, 4: 0.025, 8: 0.0}))
        for g in range(2):
            items.append(Item(f"L{l}.kv{g}", "kv", 512,
                              {2: 0.08, 3: 0.03, 4: 0.005, 8: 0.0}))
    return items


def main() -> int:
    items = small_items()

    print("1. budget accounting")
    dev = Device("test", p_eff_tflops=20.0, bw_gbps=180.0, mem_gib=2.0, rho=0.1)
    r = min_error_allocation(items, dev, seq_len=2048, nbuckets=200)
    check("allocation is produced", r is not None)
    if r:
        check("never exceeds the usable budget",
              r["bytes_MB"] <= dev.usable_mb(2048) + 1e-6,
              f"{r['bytes_MB']:.1f} <= {dev.usable_mb(2048):.1f}")

    print("\n2. monotonicity in budget")
    tight = Device("tight", p_eff_tflops=20.0, bw_gbps=180.0, mem_gib=1.2, rho=0.1)
    loose = Device("loose", p_eff_tflops=20.0, bw_gbps=180.0, mem_gib=8.0, rho=0.1)
    rt = min_error_allocation(items, tight, seq_len=2048, nbuckets=200)
    rl = min_error_allocation(items, loose, seq_len=2048, nbuckets=200)
    check("both budgets produce an allocation", rt is not None and rl is not None)
    if rt and rl:
        check("loose budget never has higher error", rl["error"] <= rt["error"] + 1e-9,
              f"{rl['error']:.4f} <= {rt['error']:.4f}")
        check("loose budget uses wider weights on average",
              rl["weight_bits_mean"] >= rt["weight_bits_mean"] - 1e-9,
              f"{rl['weight_bits_mean']} >= {rt['weight_bits_mean']}")

    print("\n3. error falls monotonically as the budget grows")
    errs = []
    for m in (0.3, 0.5, 1.0, 2.0, 4.0, 8.0):
        d = Device("d", p_eff_tflops=20.0, bw_gbps=180.0, mem_gib=m, rho=0.1)
        res = min_error_allocation(items, d, seq_len=2048, nbuckets=200)
        errs.append(None if res is None else res["error"])
    seq = [e for e in errs if e is not None]
    check("errors are non-increasing in budget",
          all(seq[i] >= seq[i + 1] - 1e-9 for i in range(len(seq) - 1)), str(seq))

    print("\n4. bit widths are legal")
    if r:
        legal = all(v in BIT_CHOICES for v in r["choices"].values())
        check("every chosen width is in BIT_CHOICES", legal)

    print("\n5. impossible budget is reported, not approximated")
    tiny = Device("tiny", p_eff_tflops=20.0, bw_gbps=180.0, mem_gib=0.05, rho=0.1)
    check("infeasible budget returns None",
          min_error_allocation(items, tiny, seq_len=2048, nbuckets=100) is None)

    print("\n6. byte helpers agree with the hand formula")
    n = 128 * 128
    check("weight_bytes(4bit) ~ n/2 + scales",
          abs(weight_bytes(n, 4) - (n * 4 / 8 + (n // 128) * 4)) < 1e-6)
    check("kv_bytes scales linearly",
          abs(kv_bytes(1024, 4, 2048) / kv_bytes(1024, 4, 1024) - 2.0) < 1e-9)

    print("\n7. real-size problem still solves")
    big = synthetic_items(n_layers=36, d_model=2560, d_ff=9728, n_kv_groups=8)
    d = Device("RTX4060", p_eff_tflops=22.36, bw_gbps=179.8, mem_gib=8.0, rho=0.133)
    rb = min_error_allocation(big, d, seq_len=16384, nbuckets=300)
    check("16384-context allocation is feasible", rb is not None)
    if rb:
        check("stays inside the measured usable budget",
              rb["bytes_MB"] <= d.usable_mb(16384) + 1e-6,
              f"{rb['bytes_MB']:.0f} <= {d.usable_mb(16384):.0f}")

    print(f"\n{'ALL TESTS PASSED' if not FAILS else str(len(FAILS)) + ' FAILED: ' + ', '.join(FAILS)}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    raise SystemExit(main())
