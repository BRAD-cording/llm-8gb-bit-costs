"""
Weight-bit x cache-bit cross product under a unified memory budget.

The paper's exchange-rate claim rests on a single 4-bit point per side. A
reviewer can reasonably ask whether the ordering is a property of the bit axis
itself or an artefact of comparing 4 bits against 4 bits. This script measures
the whole small grid (weights in W_BITS, cache in K_BITS) with both quantisers
applied in the same run, records the joint perplexity, and reports the byte
cost of each cell so the exchange rate can be refitted from several points
instead of one.

Two budgets are recorded per cell:
  * weight_bytes  = (non-lm_head params) * wb / 8 + (lm_head params) * 2
  * kv_bytes      = 2 * n_layers * n_kv * head_dim * kb / 8 * seq_len
so a cell's total footprint at a given context length is weight_bytes + kv_bytes.

Usage:
    python src/eval/bit_cost_cross.py --model C:/models/Qwen3-0.6B \\
        --text data/calib_arxiv.txt --w-bits 2,3,4,6,8 --k-bits 2,3,4,8 \\
        --windows 8 --out results/bit_cost_cross_06b.json
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
    if n == 0:
        raise SystemExit(f"corpus too small for seq_len={seq_len}")
    return ids[: n * seq_len].view(n, seq_len)


CHUNK = [0]


def _chunk_of(x: torch.Tensor, chunk: int) -> bool:
    return 0 < chunk < x.shape[1]


@torch.inference_mode()
def chunked_nll(model, x: torch.Tensor, chunk: int) -> float:
    """Token-mean NLL, fed in chunks with a carried KV cache.

    Mathematically the same quantity as ``model(x, labels=x).loss``, but the
    logits are never materialised for the whole window: at 32k tokens a single
    float32 logits tensor is 19 GB, which is why the naive call OOMs on 8 GB.
    """
    total, count, past = 0.0, 0, None
    prev_row, prev_tok = None, None
    for start in range(0, x.shape[1], chunk):
        seg = x[:, start: start + chunk]
        out = model(seg, past_key_values=past, use_cache=True)
        logits = out.logits.float()
        past = out.past_key_values
        if start > 0:
            # the last row of the previous chunk predicts this chunk's first token
            total += float(torch.nn.functional.cross_entropy(
                prev_row, prev_tok))
            count += 1
        if logits.shape[1] > 1:
            total += float(torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                seg[:, 1:].reshape(-1), reduction="sum"))
            count += seg.shape[1] - 1
        prev_row = logits[:, -1].clone()
        prev_tok = seg[:, -1].clone()
        del logits, out
    return total / count


@torch.inference_mode()
def per_window_nll(model, win: torch.Tensor) -> list[float]:
    out = []
    for i in range(win.shape[0]):
        x = win[i: i + 1].to("cuda")
        if _chunk_of(x, CHUNK[0]):
            out.append(float(chunked_nll(model, x, CHUNK[0])))
        else:
            out.append(float(model(x, labels=x).loss))
    return out


def rel_pct(base: list[float], new: list[float]) -> float:
    n = len(base)
    d = sum(b - a for a, b in zip(base, new)) / n
    return 100.0 * (pow(2.718281828459045, d) - 1.0)


def folds(base: list[float], new: list[float], k: int = 4) -> dict:
    """Split windows into k folds; report the spread of the relative delta."""
    n = len(base)
    k = max(2, min(k, n))
    vals = []
    for j in range(k):
        idx = [i for i in range(n) if i % k == j]
        vals.append(rel_pct([base[i] for i in idx], [new[i] for i in idx]))
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return {"mean": round(mean, 3), "std": round(var ** 0.5, 3),
            "fold_values": [round(v, 3) for v in vals], "n_folds": k}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--w-bits", default="2,3,4,6,8")
    ap.add_argument("--k-bits", default="2,3,4,8")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--residual", type=int, default=128)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--w-quant", default="rtn", choices=["rtn", "awq"])
    ap.add_argument("--kv-official", action="store_true",
                    help="use the vendored KIVI official quantiser for the cache")
    ap.add_argument("--chunk", type=int, default=0,
                    help="tokens per forward when scoring long windows")
    ap.add_argument("--precision", default="fp16", choices=["fp16", "int8"])
    ap.add_argument("--out", default="results/bit_cost_cross.json")
    args = ap.parse_args()
    CHUNK[0] = args.chunk

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

    win = load_windows(tok, args.text, args.seq_len, args.windows)
    print(f"{win.shape[0]} windows x {args.seq_len} tokens")
    base = per_window_nll(model, win)
    base_mean = sum(base) / len(base)
    print(f"baseline mean NLL {base_mean:.4f}  PPL {pow(2.718281828459045, base_mean):.4f}")

    layers = [(n, m) for n, m in model.named_modules()
              if isinstance(m, nn.Linear) and m.weight.dim() == 2
              and not n.endswith("lm_head")]
    lm_head = [(n, m) for n, m in model.named_modules()
               if isinstance(m, nn.Linear) and m.weight.dim() == 2
               and n.endswith("lm_head")]

    n_quant_params = sum(m.weight.numel() for _, m in layers)
    n_lm_params = sum(m.weight.numel() for _, m in lm_head)
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = cfg.hidden_size // cfg.num_attention_heads

    def weight_bytes(bits: int) -> int:
        return int(n_quant_params * bits / 8.0 + n_lm_params * 2)

    def kv_bytes(bits: int) -> int:
        return int(2 * n_layers * n_kv * head_dim * bits / 8.0 * args.seq_len)

    w_bits = [int(x) for x in args.w_bits.split(",")]
    k_bits = [int(x) for x in args.k_bits.split(",")]

    # Optional AWQ: fit alphas once on held-out windows, then reuse per weight bit.
    awq_acts = None
    if args.w_quant == "awq":
        from quant.strong import awq_best_alpha
        n_cal = max(2, args.windows // 8)
        cal = win[-n_cal:]
        acts = {}

        def mk(name):
            def hook(_m, inp, _out):
                x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
                a = x.abs().mean(0)
                acts[name] = a if name not in acts else acts[name] + a
            return hook

        hooks = [m.register_forward_hook(mk(n)) for n, m in layers]
        for i in range(cal.shape[0]):
            model(cal[i: i + 1].to("cuda"))
        for h in hooks:
            h.remove()
        awq_acts = acts
        print(f"collected AWQ activations on {cal.shape[0]} windows")

    rows = []
    for wb in w_bits:
        # materialise the weight quantisation once per weight bit
        saved = [(m, m.weight.data.clone()) for _, m in layers]
        try:
            if args.w_quant == "awq":
                for n, m in layers:
                    q, _a = awq_best_alpha(m.weight.data, awq_acts[n].unsqueeze(0),
                                           wb, group_size=min(args.group, m.weight.shape[-1]))
                    m.weight.data = q
            else:
                with quantised_weights(layers, wb, group_size=args.group):
                    pass  # applied below, per cell

            for kb in k_bits:
                if args.w_quant == "awq":
                    enable_gqa_sdpa(model, kv_bits=None if kb >= 16 else kb,
                                    kv_residual=args.residual,
                                    kv_official=args.kv_official)
                    nll = per_window_nll(model, win)
                else:
                    enable_gqa_sdpa(model, kv_bits=None if kb >= 16 else kb,
                                    kv_residual=args.residual,
                                    kv_official=args.kv_official)
                    with quantised_weights(layers, wb, group_size=args.group):
                        nll = per_window_nll(model, win)
                r = rel_pct(base, nll)
                f = folds(base, nll)
                cell = {
                    "w_bits": wb, "k_bits": kb,
                    "mean_nll": round(sum(nll) / len(nll), 5),
                    "ppl": round(pow(2.718281828459045, sum(nll) / len(nll)), 4),
                    "rel_pct": round(r, 3),
                    "fold_mean": f["mean"], "fold_std": f["std"],
                    "folds": f["fold_values"],
                    "weight_MB": round(weight_bytes(wb) / 1e6, 1),
                    "kv_MB": round(kv_bytes(kb) / 1e6, 1),
                    "total_MB": round((weight_bytes(wb) + kv_bytes(kb)) / 1e6, 1),
                }
                rows.append(cell)
                print(f"  W{wb} K{kb}: ppl {cell['ppl']:9.4f}  rel {r:+8.2f}%  "
                      f"fold {f['mean']:+.2f}+-{f['std']:.2f}  "
                      f"mem {cell['weight_MB']:.0f}+{cell['kv_MB']:.0f}={cell['total_MB']:.0f}MB",
                      flush=True)
        finally:
            for m, w0 in saved:
                m.weight.data = w0
            enable_gqa_sdpa(model, kv_bits=None)

    payload = {
        "model": args.model, "text": args.text, "w_quant": args.w_quant,
        "seq_len": args.seq_len, "windows": int(win.shape[0]),
        "group": args.group, "residual": args.residual,
        "precision": args.precision,
        "baseline_mean_nll": round(base_mean, 6),
        "baseline_ppl": round(pow(2.718281828459045, base_mean), 4),
        "n_quant_params": int(n_quant_params), "n_lm_params": int(n_lm_params),
        "rows": rows,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {args.out} ({len(rows)} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
