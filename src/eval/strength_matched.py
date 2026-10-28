"""
Strength-matched weight-versus-cache comparison.

The paper's ordering claim was measured with a naive round-to-nearest weight
quantiser and a naive cache quantiser. A reviewer is entitled to ask whether the
ordering is an artefact of that pair. This runs the comparison with a stronger
quantiser on each side --- AWQ-style activation-aware scaling for the weights,
KIVI-style residual window for the cache --- so the claim can be made at matched
strength rather than against the literature's numbers.

Usage:
    python src/eval/strength_matched.py --model C:/models/Qwen3-0.6B \
        --text data/calib_arxiv.txt --windows 32 --out results/strength_06b.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.gqa_patch import enable_gqa_sdpa  # noqa: E402
from quant.fakequant import qdq_symmetric  # noqa: E402
from quant.strong import awq_best_alpha, awq_scale, quantise_awq  # noqa: E402


def load_windows(tok, path, seq_len, n):
    text = open(path, encoding="utf-8", errors="ignore").read()
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    n = min(len(ids) // seq_len, n)
    return ids[: n * seq_len].view(n, seq_len)


@torch.inference_mode()
def per_window_nll(model, win):
    return [float(model(win[i: i + 1].to("cuda"),
                        labels=win[i: i + 1].to("cuda")).loss)
            for i in range(win.shape[0])]


def rel(base, new):
    n = len(base)
    d = sum(b - a for a, b in zip(base, new)) / n
    return 100 * (pow(2.718281828459045, d) - 1)


def boot_ci(base, new, n_boot=2000, seed=0):
    rng = random.Random(seed)
    n = len(base)
    out = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        d = sum(new[i] - base[i] for i in idx) / n
        out.append(100 * (pow(2.718281828459045, d) - 1))
    out.sort()
    return out[int(0.025 * n_boot)], out[int(0.975 * n_boot)]


def linear_layers(model):
    return [(n, m) for n, m in model.named_modules()
            if isinstance(m, nn.Linear) and m.weight.dim() == 2
            and not n.endswith("lm_head")]


@torch.inference_mode()
def collect_acts(model, layers, win, n_windows=4):
    """Mean |activation| per input channel for every linear layer."""
    acts = {}
    hooks = []

    def mk(name):
        def hook(_m, inp, _out):
            x = inp[0].detach()
            x = x.reshape(-1, x.shape[-1]).float()
            a = x.abs().mean(0)
            acts[name] = a if name not in acts else acts[name] + a
        return hook

    for name, m in layers:
        hooks.append(m.register_forward_hook(mk(name)))
    for i in range(min(n_windows, win.shape[0])):
        model(win[i: i + 1].to("cuda"))
    for h in hooks:
        h.remove()
    return acts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--windows", type=int, default=32)
    ap.add_argument("--residual", type=int, default=128)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--kv-group", type=int, default=0)
    ap.add_argument("--out", default="results/strength_matched.json")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="cuda",
        trust_remote_code=True).eval()
    enable_gqa_sdpa(model, kv_bits=None)

    win = load_windows(tok, args.text, args.seq_len, args.windows)
    # Disjoint calibration and evaluation. Calibrating AWQ on the same windows
    # we score would let it fit the test set and would overstate the weight side
    # --- a contamination that is easy to miss because the numbers stay plausible.
    n_cal = max(2, args.windows // 8)
    cal, ev = win[-n_cal:], win[:-n_cal]
    print(f"{ev.shape[0]} eval windows + {cal.shape[0]} held-out calibration windows")
    base = per_window_nll(model, ev)
    print(f"baseline NLL {sum(base)/len(base):.4f}")

    layers = linear_layers(model)
    acts = collect_acts(model, layers, cal)

    # ---- weights: RTN vs AWQ-style -------------------------------------
    saved = [(m, m.weight.data.clone()) for _, m in layers]
    alphas = {}
    for name, m in layers:
        g = m.weight.shape[-1]
        m.weight.data = qdq_symmetric(m.weight.float(), args.bits,
                                      group_size=min(args.group, g)).to(m.weight.dtype)
    rtn = per_window_nll(model, ev)
    for m, w0 in saved:
        m.weight.data = w0

    for name, m in layers:
        x = acts[name].unsqueeze(0)
        q, a = awq_best_alpha(m.weight.data, x, args.bits,
                              group_size=min(args.group, m.weight.shape[-1]))
        m.weight.data = q
        alphas[name] = a
    awq = per_window_nll(model, ev)
    for m, w0 in saved:
        m.weight.data = w0
    n_nonzero = sum(1 for a in alphas.values() if a > 0)
    print(f"AWQ-style picked alpha>0 for {n_nonzero}/{len(layers)} layers")

    # ---- cache: naive vs KIVI-style ------------------------------------
    enable_gqa_sdpa(model, kv_bits=args.bits)
    kv_naive = per_window_nll(model, ev)
    enable_gqa_sdpa(model, kv_bits=args.bits, kv_residual=args.residual)
    kv_kivi = per_window_nll(model, ev)
    enable_gqa_sdpa(model, kv_bits=args.bits, kv_residual=args.residual,
                    kv_group=args.kv_group or args.group)
    kv_kivi_g = per_window_nll(model, ev)
    enable_gqa_sdpa(model, kv_bits=None)

    rows = {}
    for label, vals in (("weights_rtn", rtn), ("weights_awq", awq),
                        ("cache_naive", kv_naive), ("cache_kivi", kv_kivi),
                        ("cache_kivi_grouped", kv_kivi_g)):
        rows[label] = {"rel_pct": round(rel(base, vals), 3),
                       "ci95": [round(c, 3) for c in boot_ci(base, vals)]}
        print(f"  {label:14s} {rows[label]['rel_pct']:+7.2f}%  "
              f"CI [{rows[label]['ci95'][0]:+.2f}, {rows[label]['ci95'][1]:+.2f}]")

    w = rows["weights_awq"]["rel_pct"]
    k = rows["cache_kivi_grouped"]["rel_pct"]
    print(f"\nstrength-matched: weights {w:+.2f}%  cache {k:+.2f}%  "
          f"ratio {w/k if k else float('nan'):.2f}x")
    print("ordering (cache cheaper) holds:", k < w)

    payload = {"model": args.model, "bits": args.bits,
               "windows": int(ev.shape[0]), "cal_windows": n_cal, "seq_len": args.seq_len,
               "residual": args.residual, "group": args.group, "kv_group": args.kv_group or args.group, "rows": rows,
               "awq_layers_using_scaling": n_nonzero,
               "ordering_holds": bool(k < w)}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
