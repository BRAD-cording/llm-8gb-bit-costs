"""
Adversarial test of the "a cache bit is cheaper than a weight bit" result.

The obvious objection is that our weight quantiser is plain round-to-nearest
(RTN), which is weaker than GPTQ or AWQ, so the weight side may look
artificially expensive. This script strengthens the WEIGHT side as far as
cheap quantiser tuning allows -- finer groups, asymmetric grids, and excluding
the most sensitive module -- and asks whether the weight cost at 4 bits falls
below the measured cache cost of 5.58%.

If it does, the paper's central ordering claim is a quantiser artefact and must
be withdrawn. If it does not, the claim survives the strongest objection we can
raise against it without a full GPTQ implementation (which is future work and
is flagged as such).

Usage:
    python src/eval/adversarial_weight.py --model C:/models/Qwen3-0.6B \
        --text data/calib_arxiv.txt --out results/adversarial_weight.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.ppl import load_tokens, perplexity  # noqa: E402
from quant.fakequant import qdq_asymmetric, qdq_symmetric  # noqa: E402


def quantise(w: torch.Tensor, bits: int, group: int, asym: bool) -> torch.Tensor:
    if asym:
        # reshape into groups along the input dim, then per-group asymmetric
        n = w.shape[-1]
        g = group if (group and n % group == 0) else n
        x = w.reshape(*w.shape[:-1], n // g, g)
        return qdq_asymmetric(x, bits, dim=-1).reshape(w.shape)
    return qdq_symmetric(w, bits, group, dim=-1)


@torch.inference_mode()
def ppl_with(model, layers, windows, bits, group, asym):
    saved = [(m, m.weight.data.clone()) for _, m in layers]
    try:
        for _, m in layers:
            m.weight.data = quantise(m.weight.data, bits, group, asym)
        return perplexity(model, windows)
    finally:
        for m, w in saved:
            m.weight.data = w


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--max-windows", type=int, default=6)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--kv-reference", type=float, default=0.0558,
                    help="measured cache cost at the same width")
    ap.add_argument("--out", default="results/adversarial_weight.json")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="cuda",
        trust_remote_code=True).eval()

    all_layers = [(n, m) for n, m in model.named_modules()
                  if isinstance(m, nn.Linear) and m.weight.dim() == 2]
    no_head = [(n, m) for n, m in all_layers if not n.endswith("lm_head")]

    windows = load_tokens(tok, args.text, args.seq_len, args.max_windows)
    base = perplexity(model, windows)
    print(f"baseline PPL = {base:.4f}\n")

    rows = []
    cases = [
        ("rtn  g=128 sym  (as published)", no_head, 128, False),
        ("rtn  g=64  sym",                 no_head, 64, False),
        ("rtn  g=32  sym",                 no_head, 32, False),
        ("rtn  g=32  asym",                no_head, 32, True),
        ("rtn  g=16  asym",                no_head, 16, True),
    ]
    for label, layers, group, asym in cases:
        ppl = ppl_with(model, layers, windows, args.bits, group, asym)
        rel = (ppl - base) / base
        verdict = "BELOW cache" if rel < args.kv_reference else "above cache"
        rows.append({"case": label, "bits": args.bits, "group": group,
                     "asym": asym, "ppl": round(ppl, 5),
                     "relative": round(rel, 6), "verdict": verdict})
        print(f"  {label:32s} PPL {ppl:9.4f}  {100*rel:+7.2f}%   {verdict}")

    print(f"\ncache cost at {args.bits} bits = {100*args.kv_reference:.2f}% "
          f"(measured, results/kv_bits_qwen06.json)")
    survives = all(r["verdict"] == "above cache" for r in rows)
    print(f"ordering claim survives: {survives}")

    payload = {"model": args.model, "baseline_ppl": round(base, 6),
               "bits": args.bits, "kv_reference": args.kv_reference,
               "claim_survives": bool(survives), "rows": rows}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
