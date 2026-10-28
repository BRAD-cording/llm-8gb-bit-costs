"""
Error bars for the headline comparison: 4-bit weights versus 4-bit cache.

The paper's central ordering claim rests on two perplexity numbers measured on
six windows. A reviewer is entitled to ask how stable that ordering is. This
script recomputes both costs over many windows, keeps the per-window negative
log-likelihood, and bootstraps a 95% confidence interval for the relative
increase -- so the claim can be stated with an interval instead of a point.

Usage:
    python src/eval/bit_cost_ci.py --model C:/models/Qwen3-0.6B \
        --text data/calib_arxiv.txt --windows 48 --out results/bit_cost_ci.json
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
from eval.ppl import quantised_weights  # noqa: E402


def load_windows(tok, path: str, seq_len: int, n: int) -> torch.Tensor:
    text = open(path, encoding="utf-8", errors="ignore").read()
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    n = min(len(ids) // seq_len, n)
    return ids[: n * seq_len].view(n, seq_len)


@torch.inference_mode()
def per_window_nll(model, windows: torch.Tensor) -> list[float]:
    out = []
    for i in range(windows.shape[0]):
        ids = windows[i: i + 1].to("cuda")
        loss = model(ids, labels=ids).loss
        out.append(float(loss))
    return out


def relative(nll_base: list[float], nll_new: list[float],
             n_boot: int = 2000, seed: int = 0) -> dict:
    rng = random.Random(seed)
    n = len(nll_base)
    point = (sum(nll_new) / n) - (sum(nll_base) / n)   # in nats, then PPL-ish
    boots = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        b = sum(nll_base[i] for i in idx) / n
        v = sum(nll_new[i] for i in idx) / n
        boots.append(v - b)
    boots.sort()
    return {
        "delta_nats": round(point, 5),
        "rel_ppl_pct": round(100 * (pow(2.718281828459045, point) - 1), 3),
        "ci95_nats": [round(boots[int(0.025 * n_boot)], 5),
                      round(boots[int(0.975 * n_boot)], 5)],
        "ci95_rel_ppl_pct": [
            round(100 * (pow(2.718281828459045, boots[int(0.025 * n_boot)]) - 1), 3),
            round(100 * (pow(2.718281828459045, boots[int(0.975 * n_boot)]) - 1), 3)],
        "n_windows": n,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--windows", type=int, default=48)
    ap.add_argument("--group", type=int, default=None,
                    help="group size for the weight quantiser (default auto)")
    ap.add_argument("--asym", action="store_true",
                    help="use an asymmetric weight grid")
    ap.add_argument("--precision", default="fp16",
                    choices=["fp16", "int8"],
                    help="reference precision; int8 lets a 4B model fit in 8 GB")
    ap.add_argument("--out", default="results/bit_cost_ci.json")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
    if args.precision == "int8":
        from transformers import BitsAndBytesConfig
        model = AutoModelForCausalLM.from_pretrained(
            args.model, device_map="cuda", trust_remote_code=True,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True)).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16, device_map="cuda",
            trust_remote_code=True).eval()
    enable_gqa_sdpa(model, kv_bits=None)

    windows = load_windows(tok, args.text, args.seq_len, args.windows)
    print(f"{windows.shape[0]} windows x {args.seq_len} tokens")

    base = per_window_nll(model, windows)
    print(f"baseline mean NLL = {sum(base)/len(base):.4f}")

    layers = [(n, m) for n, m in model.named_modules()
              if isinstance(m, nn.Linear) and m.weight.dim() == 2
              and not n.endswith("lm_head")]

    if args.group or args.asym:
        from eval.adversarial_weight import quantise
        saved = [(m, m.weight.data.clone()) for _, m in layers]
        try:
            for _, m in layers:
                g = args.group or m.weight.shape[-1]
                m.weight.data = quantise(m.weight.data, args.bits, g, args.asym)
            w_nll = per_window_nll(model, windows)
        finally:
            for m, w0 in saved:
                m.weight.data = w0
    else:
        with quantised_weights(layers, args.bits):
            w_nll = per_window_nll(model, windows)
    print(f"weights {args.bits}-bit mean NLL = {sum(w_nll)/len(w_nll):.4f}")

    enable_gqa_sdpa(model, kv_bits=args.bits)
    kv_nll = per_window_nll(model, windows)
    print(f"cache   {args.bits}-bit mean NLL = {sum(kv_nll)/len(kv_nll):.4f}")

    w = relative(base, w_nll)
    k = relative(base, kv_nll)
    paired = relative(w_nll, kv_nll)   # cache minus weights, paired by window

    print(f"\nweights {args.bits}-bit: {w['rel_ppl_pct']:+.2f}%  "
          f"95% CI [{w['ci95_rel_ppl_pct'][0]:+.2f}, {w['ci95_rel_ppl_pct'][1]:+.2f}]")
    print(f"cache   {args.bits}-bit: {k['rel_ppl_pct']:+.2f}%  "
          f"95% CI [{k['ci95_rel_ppl_pct'][0]:+.2f}, {k['ci95_rel_ppl_pct'][1]:+.2f}]")
    print(f"paired difference (cache - weights): {paired['rel_ppl_pct']:+.2f}%  "
          f"95% CI [{paired['ci95_rel_ppl_pct'][0]:+.2f}, "
          f"{paired['ci95_rel_ppl_pct'][1]:+.2f}]")
    excl = paired["ci95_rel_ppl_pct"][1] < 0
    print(f"\nordering claim (cache cheaper) supported by the CI: {excl}")

    payload = {"model": args.model, "bits": args.bits, "precision": args.precision,
               "windows": int(windows.shape[0]), "seq_len": args.seq_len,
               "baseline_mean_nll": round(sum(base) / len(base), 6),
               "weights": w, "cache": k, "paired_cache_minus_weights": paired,
               "ordering_supported": bool(excl)}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
