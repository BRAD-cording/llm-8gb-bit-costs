"""Policy comparison: uniform / widest-greedy / marginal-ratio water-filling /
WF+1-opt repair / (DP if importable), on the measured per-item curves.

Writes results/alloc_policies.json and prints a table.
"""
import json
import os
import sys
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))

from alloc import alloc_baselines as ab  # noqa: E402

DATA = os.path.join(ROOT, "results", "items_qwen06_v2.json")
# Device A: 7.996 GiB nominal, rho 0.133, workspace 0.142 MB/token + 9 MB
MEM_MB, RHO, WS_A, WS_B = 7.996 * 1024.0, 0.133, 0.142, 9.0


def budget(L):
    return MEM_MB * (1.0 - RHO) - (WS_A * L + WS_B)


def load():
    d = json.load(open(DATA, encoding="utf-8"))
    out = []
    for x in d["items"]:
        err = {int(k): v for k, v in x["err"].items()}
        out.append(SimpleNamespace(name=x["name"], kind=x["kind"],
                                   n_params=x["n_params"], err=err))
    return out


def mb(it, b, L):
    try:
        return ab.item_mb(it, b, L)
    except Exception:
        n = it.n_params
        return n * b / 8.0 / 1e6 if it.kind == "w" else n * b / 8.0 * L / 1e6


def err(arr, assign):
    return sum(it.err[assign[it.name]] for it in arr)


def uniform(arr, L):
    B = budget(L)
    for b in (8, 4, 3, 2):
        a = {it.name: b for it in arr}
        if sum(mb(it, b, L) for it in arr) <= B:
            return a, b
    return None, None


def greedy_widest(arr, L):
    B = budget(L)
    a = {it.name: 2 for it in arr}
    use = sum(mb(it, 2, L) for it in arr)
    for b in (8, 4, 3):
        for it in sorted(arr, key=lambda z: -z.n_params):
            d = mb(it, b, L) - mb(it, a[it.name], L)
            if use + d <= B:
                use += d
                a[it.name] = b
    return a


def waterfill(arr, L, order=(2, 3, 4, 8)):
    """Marginal ratio: repeatedly buy the cheapest accuracy-per-MB upgrade."""
    B, a = budget(L), {it.name: order[0] for it in arr}
    use = sum(mb(it, a[it.name], L) for it in arr)
    while True:
        best, best_r = None, None
        for it in arr:
            i = order.index(a[it.name])
            if i + 1 >= len(order):
                continue
            b1, b2 = order[i], order[i + 1]
            gain = it.err[b1] - it.err[b2]
            cost = mb(it, b2, L) - mb(it, b1, L)
            if cost <= 0 or gain <= 0 or use + cost > B:
                continue
            r = gain / cost
            if best_r is None or r > best_r:
                best, best_r = (it, b2, cost), r
        if best is None:
            return a
        it, b2, cost = best
        a[it.name] = b2
        use += cost


def repair(arr, L, a):
    B = budget(L)
    a = dict(a)
    use = sum(mb(it, a[it.name], L) for it in arr)
    improved = True
    while improved:
        improved = False
        for it in arr:
            cur = a[it.name]
            for nb in (2, 3, 4, 8):
                if nb == cur:
                    continue
                d = mb(it, nb, L) - mb(it, cur, L)
                de = it.err[nb] - it.err[cur]
                if de < 0 and use + d <= B:
                    use += d
                    a[it.name] = nb
                    improved = True
                    break
            if improved:
                break
    return a


def main():
    arr = load()
    rows = []
    for L in (4096, 32768):
        B = round(budget(L))
        pol = {"uniform": uniform(arr, L)[0], "greedy": greedy_widest(arr, L),
               "waterfill": waterfill(arr, L)}
        pol["wf+1opt"] = repair(arr, L, pol["waterfill"])
        for k, a in pol.items():
            rows.append({"L": L, "policy": k, "budget_mb": B,
                         "used_mb": round(sum(mb(it, a[it.name], L) for it in arr), 1),
                         "err": round(err(arr, a), 4),
                         "hist": {b: sum(1 for v in a.values() if v == b)
                                  for b in (2, 3, 4, 8)}})
        print(f"L={L} budget={B}MB", flush=True)
        for r in rows[-4:]:
            print(f"  {r['policy']:10s} used={r['used_mb']:8.1f} err={r['err']:8.4f} {r['hist']}")
    json.dump({"device": "A", "rows": rows}, open(os.path.join(ROOT, "results", "alloc_policies.json"), "w"), indent=1)
    print("WROTE results/alloc_policies.json")


if __name__ == "__main__":
    main()
