"""Plot the weight-bit x cache-bit cross product as a unified-budget Pareto.

x: attainable context on Device A (laptop) from the measured budget equation
y: relative perplexity cost (%)
Each marker is one (w_bits, k_bits) cell measured by src/eval/bit_cost_cross.py.
Points toward the lower right are better; the staircase is the Pareto frontier.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def lmax(B_mb, weight_mb, k_bits, workspace_per_tok=0.142, workspace_fixed=9.0):
    kv_mb_per_tok = 0.007168 * k_bits      # 114688 bytes/token at 16 bits / 1e6
    return (B_mb - weight_mb - workspace_fixed) / (workspace_per_tok + kv_mb_per_tok)


def main(path, out, budget_mb=7096.0):
    d = json.load(open(path))
    fig, ax = plt.subplots(figsize=(3.4, 2.5), dpi=200)
    pts = []
    for r in d["rows"]:
        x = lmax(budget_mb, r["weight_MB"], r["k_bits"])
        y = r["rel_pct"]
        pts.append((x, y, r))
    # Pareto staircase: sort by x, keep points that improve y monotonically
    rows = sorted(pts, key=lambda t: -t[0])
    frontier, best = [], float("inf")
    for x, y, r in rows:
        if y < best:
            frontier.append((x, y, r))
            best = y
    fx = [p[0] / 1000 for p in frontier]
    fy = [p[1] for p in frontier]
    ax.step(fx, fy, where="post", color="0.35", lw=1.0, ls="--", zorder=1)
    for x, y, r in pts:
        if y > 1000:
            continue
        ax.scatter(x / 1000, y, s=22, zorder=3,
                   c=["#1f77b4" if r["k_bits"] == 4 else "#d62728"])
    for lab, w, k in (("W4/K4", 4, 4), ("W6/K4", 6, 4), ("W8/K8", 8, 8)):
        for x, y, r in pts:
            if r["w_bits"] == w and r["k_bits"] == k:
                ax.annotate(lab, (x / 1000, y), textcoords="offset points",
                            xytext=(4, 4), fontsize=6)
    ax.set_xlabel("attainable context on device A (k tokens)", fontsize=7)
    ax.set_ylabel(r"relative PPL cost (%)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_ylim(-2, 60)
    ax.grid(alpha=0.25, lw=0.4)
    fig.tight_layout(pad=0.3)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    print("wrote", out, "frontier points:", [(round(a, 1), round(b, 1)) for a, b, _ in frontier])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "cross_06b_rtn.json"),
         sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "paper", "figures", "fig_cross.pdf"))
