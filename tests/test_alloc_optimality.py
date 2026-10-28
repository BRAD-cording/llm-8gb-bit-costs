"""Brute-force optimality check for the allocation DP.

Property tests (tests/test_alloc.py) check invariants; they do not prove the DP
finds the optimum. This enumerates every assignment for a small instance and
compares against the solver, which is the only way to know the DP is not
silently suboptimal.

Run:  python tests/test_alloc_optimality.py
"""
from __future__ import annotations

import itertools
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from alloc.dp_alloc import (BIT_CHOICES, Device, Item, kv_bytes,
                            min_error_allocation, weight_bytes)


def brute_force(items, device, seq_len):
    """Enumerate all assignments; return the minimum error that fits."""
    budget = device.usable_mb(seq_len)
    best = None
    choices = [[b for b in BIT_CHOICES if b in it.err] for it in items]
    for combo in itertools.product(*choices):
        mb = 0.0
        err = 0.0
        for it, bits in zip(items, combo):
            b = (weight_bytes(it.n_params, bits) if it.kind == "w"
                 else kv_bytes(it.n_params, bits, seq_len))
            mb += b / 1024**2
            err += it.err[bits]
        if mb <= budget and (best is None or err < best):
            best = err
    return best


def main() -> int:
    rng = random.Random(7)
    seq_len = 1024

    # The budget must BIND, otherwise every assignment fits at 8 bits, both
    # methods return zero error, and the test proves nothing.  A first version
    # of this file had that flaw: six "passing" trials that were all vacuous.
    fails = 0
    tested = 0
    for trial in range(8):
        n_w = rng.randint(3, 4)
        n_kv = rng.randint(2, 3)
        items = []
        for i in range(n_w):
            items.append(Item(f"w{i}", "w", rng.randint(200_000, 400_000),
                              {2: rng.uniform(0.2, 0.5), 3: rng.uniform(0.05, 0.2),
                               4: rng.uniform(0.005, 0.05), 8: 0.0}))
        for i in range(n_kv):
            items.append(Item(f"kv{i}", "kv", rng.randint(400, 1200),
                              {2: rng.uniform(0.1, 0.4), 3: rng.uniform(0.02, 0.1),
                               4: rng.uniform(0.002, 0.02), 8: 0.0}))

        # size the budget to a fraction of the all-8bit requirement
        full_mb = sum((weight_bytes(it.n_params, 8) if it.kind == "w"
                       else kv_bytes(it.n_params, 8, seq_len)) / 1024**2
                      for it in items)
        frac = rng.uniform(0.35, 0.8)
        target = full_mb * frac
        dev = Device("test", p_eff_tflops=20.0, bw_gbps=180.0,
                     mem_gib=(target + 1.0) / (0.9 * 1024), rho=0.1,
                     ws_mb_per_token=0.001, ws_const_mb=0.5, prefill_chunk=512)
        budget = dev.usable_mb(seq_len)

        bf = brute_force(items, dev, seq_len)
        dp = min_error_allocation(items, dev, seq_len, nbuckets=4000)

        if bf is None and dp is None:
            print(f"trial {trial}: budget {budget:6.2f}MB of {full_mb:6.2f}MB "
                  f"— both infeasible")
            continue
        if bf is None or dp is None:
            print(f"trial {trial}: MISMATCH feasibility bf={bf} dp={dp}")
            fails += 1
            continue
        if bf <= 1e-12 and dp["error"] <= 1e-12:
            print(f"trial {trial}: budget {budget:6.2f}MB — non-binding, skipped")
            continue

        tested += 1
        ok = dp["error"] >= bf - 1e-4   # dp error is rounded to 5 dp
        print(f"trial {trial}: budget {budget:6.2f}/{full_mb:6.2f}MB "
              f"brute={bf:.5f} dp={dp['error']:.5f} gap={dp['error']-bf:+.5f} "
              f"{'ok' if ok else 'DP BEATS BRUTE FORCE (impossible)'}")
        if not ok:
            fails += 1

    print(f"\n{tested} binding trials, "
          f"{'OPTIMALITY OK' if fails == 0 and tested else 'INCONCLUSIVE/VACUOUS'}"
          f"{'' if fails == 0 else ' — ' + str(fails) + ' failure(s)'}")
    return 0 if fails == 0 and tested else 1


if __name__ == "__main__":
    sys.exit(main())
