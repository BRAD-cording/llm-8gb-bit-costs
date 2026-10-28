"""
GPTQ weights on a real model, measured on the same held-out windows as the
strength-matched comparison so the numbers are directly comparable.

Layers are quantised in blocks: a forward pass collects the Hessian for the
layers in the current block only, those layers are replaced, and the next block
runs on the partially quantised model. Blocking keeps the Hessian memory bounded
(sum of in^2 over a block instead of over every layer) and costs one forward
pass per block.

Usage:
    python src/eval/gptq_weights.py --model C:/models/Qwen3-0.6B \
        --text data/calib_arxiv.txt --windows 32 --bits 4 --group 128 \
        --out results/gptq_06b_g128.json
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
from quant.gptq import gptq_quantise  # noqa: E402
from quant.fakequant import qdq_symmetric  # noqa: E402


def load_windows(tok, path, seq_len, n):
    text = open(path, encoding="utf-8", errors="ignore").read()
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    n = min(len(ids) // seq_len, n)
    return ids[: n * seq_len].view(n, seq_len)


@torch.inference_mode()
def nll(model, win):
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


@torch.inference_mode()
def collect_acts(model, layers, cal, max_tokens=1024):
    """Capture each layer's input from the *unquantised* model, one pass.

    Blocking the capture and quantising in between (an earlier version) feeds
    later blocks activations from an already-perturbed model. On the 0.6B model
    that still worked, but on 1.7B it produced end-to-end perplexity worse than
    plain RTN --- the compensation was fitted to a distribution that no longer
    existed. Capture first, then quantise, is the standard order.
    """
    acts, hooks = {}, []
    for name, m in layers:
        def mk(n):
            def hook(_m, inp, _out):
                x = inp[0].detach().reshape(-1, inp[0].shape[-1])[:max_tokens]
                acts[n] = x.float()
            return hook
        hooks.append(m.register_forward_hook(mk(name)))
    for i in range(cal.shape[0]):
        model(cal[i: i + 1].to("cuda"))
    for h in hooks:
        h.remove()
    return acts


def quantise_block(model, layers, cal, bits, group, block=16, max_tokens=1024):
    acts = collect_acts(model, layers, cal, max_tokens)
    done = 0
    for name, m in layers:
        x = acts.get(name)
        if x is None or x.shape[0] < 2:
            continue
        m.weight.data = gptq_quantise(m.weight.data, x, bits, group)
        done += 1
    print(f"GPTQ applied to {done}/{len(layers)} layers")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--windows", type=int, default=32)
    ap.add_argument("--block", type=int, default=16,
                    help="layers per Hessian block; lower it for larger layers")
    ap.add_argument("--out", default="results/gptq_weights.json")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="cuda",
        trust_remote_code=True).eval()
    enable_gqa_sdpa(model, kv_bits=None)

    win = load_windows(tok, args.text, args.seq_len, args.windows)
    n_cal = max(2, args.windows // 8)
    cal, ev = win[-n_cal:], win[:-n_cal]
    print(f"{ev.shape[0]} eval windows + {cal.shape[0]} calibration windows")

    base = nll(model, ev)
    print(f"baseline NLL {sum(base)/len(base):.4f}")

    layers = [(n, m) for n, m in model.named_modules()
              if isinstance(m, nn.Linear) and m.weight.dim() == 2
              and not n.endswith("lm_head")]
    print(f"{len(layers)} linear layers")

    saved = [(m, m.weight.data.clone()) for _, m in layers]
    for name, m in layers:
        g = m.weight.shape[-1]
        m.weight.data = qdq_symmetric(m.weight.float(), args.bits,
                                      group_size=min(args.group, g)).to(m.weight.dtype)
    rtn = nll(model, ev)
    for m, w0 in saved:
        m.weight.data = w0

    quantise_block(model, layers, cal, args.bits, args.group, block=args.block)
    gq = nll(model, ev)

    out = {"model": args.model, "bits": args.bits, "group": args.group,
           "windows": int(ev.shape[0]), "cal_windows": n_cal,
           "baseline_nll": round(sum(base) / len(base), 6)}
    for label, vals in (("weights_rtn", rtn), ("weights_gptq", gq)):
        out[label] = {"rel_pct": round(rel(base, vals), 3),
                      "ci95": [round(c, 3) for c in boot_ci(base, vals)]}
        print(f"  {label:14s} {out[label]['rel_pct']:+7.2f}%  "
              f"CI [{out[label]['ci95'][0]:+.2f}, {out[label]['ci95'][1]:+.2f}]")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
