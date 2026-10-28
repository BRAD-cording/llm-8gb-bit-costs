"""
KV-cache sensitivity: how much does the model suffer when the cache is quantised?

Weight sensitivity is measured per linear layer by `src/eval/ppl.py`.  The
allocation problem needs the other half: what a KV bit costs.  Rather than
reach into transformers' cache implementation (whose API moves between
versions), we quantise the OUTPUT of `k_proj` and `v_proj`.  Those tensors are
exactly what gets written into the cache, so this is quantising the cache at
write time, which is the same numerical operation the deployed kernels do.

  K is quantised per-channel (along head_dim), V per-token -- the KIVI
  convention, matching src/quant/fakequant.qdq_kv.

Usage:
    python src/eval/kv_sensitivity.py --model C:/models/Qwen3-0.6B \
        --text data/calib_arxiv.txt --bits 2,3,4 --out results/kv_sens.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.ppl import load_tokens, perplexity  # noqa: E402
from quant.fakequant import qdq_asymmetric  # noqa: E402


def find_kv_projs(model) -> list[tuple[str, nn.Module, str]]:
    out = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and m.weight.dim() == 2:
            tail = name.rsplit(".", 1)[-1]
            if tail in ("k_proj", "v_proj"):
                out.append((name, m, "k" if tail == "k_proj" else "v"))
    return out


@contextmanager
def quantised_kv(projs: list[tuple[str, nn.Module, str]], bits: int):
    """Quantise k_proj/v_proj outputs on the fly; restore on exit."""
    handles = []

    def make_hook(kind: str):
        def hook(module, inputs, output):
            # output: [batch, seq, heads*head_dim]; reshape to per-head for the
            # channel/token axes to mean what they mean inside attention
            b, s, h = output.shape
            cfg = getattr(module, "_kv_shape", None)
            if cfg is None:
                return output
            heads, head_dim = cfg
            x = output.view(b, s, heads, head_dim)
            if kind == "k":
                # per-channel along head_dim
                x = qdq_asymmetric(x, bits, dim=-1)
            else:
                # per-token along the sequence axis
                x = qdq_asymmetric(x, bits, dim=-2)
            return x.reshape(b, s, h)
        return hook

    for name, mod, kind in projs:
        handles.append(mod.register_forward_hook(make_hook(kind)))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def attach_shape(model) -> None:
    """Tell each projection how many heads and what head_dim it produces."""
    cfg = model.config
    heads = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    for _, mod, _ in find_kv_projs(model):
        mod._kv_shape = (heads, head_dim)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--bits", default="2,3,4")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--max-windows", type=int, default=6)
    ap.add_argument("--out", default="results/kv_sens.json")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="cuda",
        trust_remote_code=True).eval()
    attach_shape(model)

    windows = load_tokens(tok, args.text, args.seq_len, args.max_windows)
    base = perplexity(model, windows)
    print(f"baseline PPL = {base:.4f}")

    rows = []
    for bits in [int(b) for b in args.bits.split(",")]:
        with quantised_kv(find_kv_projs(model), bits):
            ppl = perplexity(model, windows)
        rel = (ppl - base) / base
        rows.append({"bits": bits, "ppl": round(ppl, 5),
                     "delta": round(ppl - base, 5), "relative": round(rel, 6)})
        print(f"  KV {bits}-bit: PPL {ppl:8.4f}  (+{ppl-base:7.4f}, "
              f"+{100*rel:5.2f}%)")

    payload = {"model": args.model, "baseline_ppl": round(base, 6),
               "window_tokens": args.seq_len, "n_windows": args.max_windows,
               "rows": rows}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
