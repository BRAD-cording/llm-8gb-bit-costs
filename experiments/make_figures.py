"""
Generate the paper's figures from the measurement JSON/CSV.

Every figure is produced from data files, never hand-drawn, so a figure cannot
drift from the numbers in the text. Outputs PDF (vector) for LaTeX.

Usage:
    python experiments/make_figures.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.dpi": 200,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def fig_stack() -> None:
    """TTFT versus context, naive versus fused attention, both models.

    Failed runs are drawn, not dropped: a reader has to see where the naive path
    stops working, otherwise the surviving line looks merely "a bit slower".
    """
    rows = list(csv.DictReader(open(RES / "system_matrix.csv", encoding="utf-8")))

    def collect(model_key: str, prec_sub: str, attn: str):
        good, bad = [], []
        for r in rows:
            if model_key not in r["model"] or r["attn"] != attn:
                continue
            if prec_sub not in r["precision"]:
                continue
            L = int(r["seq_len"])
            if r["error"] or r["spilled"].lower() == "true":
                bad.append((L, r["error"] or "spill"))
            else:
                good.append((L, float(r["ttft_ms"])))
        return sorted(good), sorted(bad)

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5))
    specs = [("0.6B", "fp16", "Qwen3-0.6B (fp16)"),
             ("4B", "nf4", "Qwen3-4B (nf4)")]
    for ax, (model_key, prec, title) in zip(axes, specs):
        for attn, color, marker, label in [
                ("sdpa", "#c0392b", "X", "math fallback"),
                ("gqa_sdpa", "#2471a3", "o", "fused")]:
            good, bad = collect(model_key, prec, attn)
            if good:
                ax.plot([g[0] for g in good], [g[1] for g in good],
                        marker=marker, color=color, ls="--", ms=4, lw=1.2,
                        label=label)
            for L, why in bad:
                ax.plot([L], [good[-1][1] if good else 100], marker="X",
                        color=color, ms=6, mfc="none", ls="none")
                ax.annotate(why.upper() if why == "OOM" else "spill",
                            (L, good[-1][1] if good else 100),
                            textcoords="offset points", xytext=(4, 4),
                            fontsize=6, color=color)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("context length $L$")
        ax.set_ylabel("TTFT (ms)")
        ax.set_title(title)
        ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "fig_stack.pdf")
    plt.close(fig)
    print("  fig_stack.pdf")


def fig_bitcost() -> None:
    """Perplexity cost of quantising weights versus the cache, as grouped bars.

    A log line plot was tried first and was wrong: a 0% point cannot be drawn on
    a log axis, so matplotlib silently dropped it and the surviving line read as
    if the cache cost rose with bit width. Bars with printed values cannot be
    misread that way.
    """
    w = json.loads((RES / "w_bits_qwen06_v2.json").read_text(encoding="utf-8"))
    kv = json.loads((RES / "kv_bits_qwen06.json").read_text(encoding="utf-8"))
    wm = {r["w_bits"]: r["relative"] * 100 for r in w["rows"]}
    km = {r["kv_bits"]: r["relative"] * 100 for r in kv["rows"]}

    bits = [8, 4, 3]
    wv = [max(wm.get(b, 0.0), 0.05) for b in bits]     # floor so bars are visible
    kvv = [max(km.get(b, 0.0), 0.05) for b in bits]

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    x = range(len(bits))
    width = 0.36
    ax.bar([i - width / 2 for i in x], wv, width, label="all weights",
           color="#c0392b")
    ax.bar([i + width / 2 for i in x], kvv, width, label="KV cache",
           color="#2471a3")
    ax.set_yscale("log")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{b}-bit" for b in bits])
    ax.set_ylabel("perplexity increase (%, log)")
    ax.set_ylim(0.04, 5000)
    for i, b in enumerate(bits):
        for off, val, raw in [(-width / 2, wv[i], wm.get(b, 0.0)),
                              (width / 2, kvv[i], km.get(b, 0.0))]:
            txt = "0" if raw <= 0.0 else (f"{raw:.1f}" if raw < 100 else f"{raw:.0f}")
            ax.annotate(txt, (i + off, val), textcoords="offset points",
                        xytext=(0, 2), ha="center", fontsize=5.5)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "fig_bitcost.pdf")
    plt.close(fig)
    print("  fig_bitcost.pdf")


def fig_exchange() -> None:
    """Context length attainable versus weight width, both devices."""
    cross = json.loads((RES / "kv_crossover.json").read_text(encoding="utf-8"))
    info = cross["Qwen/Qwen3-4B"]["info"]
    budgets = {"RTX 4060": 6.93 * 1.0737, "Jetson": 4.32 * 1.0737}
    ws_tok, ws_const, chunk = 0.23, 170.0, 2048
    kv_tok = info["kv_bytes_per_token"] / 1e9  # GB per token at 16 bit

    def lmax(budget_gb: float, w_bits: int, kv_bits: int) -> int:
        left = budget_gb - info["params_M"] * 1e6 * w_bits / 8 / 1e9
        if left <= 0:
            return 0
        per = kv_tok * kv_bits / 16
        lo, hi = 0, 2_000_000
        while lo < hi:
            mid = (lo + hi + 1) // 2
            ws = (ws_tok * min(mid, chunk) + ws_const) / 1024
            if per * mid + ws <= left:
                lo = mid
            else:
                hi = mid - 1
        return lo

    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    for (name, budget), style in zip(budgets.items(), ("o-", "s-")):
        ys = [lmax(budget, wb, 16) for wb in (8, 4, 3, 2)]
        ax.plot([8, 4, 3, 2], ys, style, label=name, ms=4)
    ax.set_xscale("log", base=2)
    ax.set_xticks([8, 4, 3, 2])
    ax.set_xticklabels(["8", "4", "3", "2"])
    ax.invert_xaxis()   # heavy precision on the left, so the slope reads left-to-right
    ax.set_xlabel("weight bit-width (cache fixed at 16)")
    ax.set_ylabel("max context length")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "fig_exchange.pdf")
    plt.close(fig)
    print("  fig_exchange.pdf")


def main() -> int:
    print("writing figures to", FIG)
    for fn in (fig_stack, fig_bitcost, fig_exchange):
        try:
            fn()
        except Exception as e:
            print(f"  {fn.__name__} FAILED: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
