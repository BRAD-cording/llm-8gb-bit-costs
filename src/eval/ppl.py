"""
Perplexity evaluation + per-layer sensitivity calibration.

Two jobs:

1. `evaluate()`  -- perplexity of a model on a plain-text corpus, with an
   optional per-layer weight-quantisation override.  This is the accuracy side
   of every table in the paper.

2. `calibrate()` -- for every linear layer and every candidate bit-width,
   quantise ONLY that layer and measure the perplexity delta.  This produces the
   real `Item.err` curves that `src/alloc/dp_alloc.py` needs; the synthetic
   curves currently in that file are placeholders and must be replaced before
   any allocation result is reported.

The corpus is a plain text file (wikitext-2 raw train.txt is the standard
choice).  HuggingFace, GitHub and jsDelivr are all unreachable from this
network, so the file must be supplied locally -- see STATUS.md.

Usage:
    python src/eval/ppl.py --model C:/models/Qwen3-0.6B --text data/wikitext2_train.txt
    python src/eval/ppl.py --model C:/models/Qwen3-0.6B --text ... --calibrate \
        --out results/sensitivity_qwen06.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from contextlib import contextmanager

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quant.fakequant import auto_group, qdq_symmetric  # noqa: E402

BITS = (2, 3, 4)


# --------------------------------------------------------------------------- #
# corpus
# --------------------------------------------------------------------------- #
def load_tokens(tok, path: str, seq_len: int, max_windows: int) -> torch.Tensor:
    with open(path, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    n = min(len(ids) // seq_len, max_windows)
    if n == 0:
        raise SystemExit(f"corpus too small: {len(ids)} tokens < {seq_len}")
    out = ids[: n * seq_len].view(n, seq_len)
    print(f"[data] {len(ids):,} tokens -> {n} windows of {seq_len}")
    return out


# --------------------------------------------------------------------------- #
# quantisation hooks
# --------------------------------------------------------------------------- #
def linear_layers(model) -> list[tuple[str, nn.Linear]]:
    return [(n, m) for n, m in model.named_modules()
            if isinstance(m, nn.Linear) and m.weight.dim() == 2]


@contextmanager
def quantised_weights(layers: list[tuple[str, nn.Linear]], bits: int,
                      group_size: int | None = None):
    """Temporarily replace the listed layers' weights with fake-quantised copies."""
    saved = []
    try:
        for _, m in layers:
            g = group_size or auto_group(m.weight.shape[-1])
            saved.append((m, m.weight.data.clone()))
            m.weight.data = qdq_symmetric(m.weight.data, bits, g, dim=-1)
        yield
    finally:
        for m, w in saved:
            m.weight.data = w


# --------------------------------------------------------------------------- #
# perplexity
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def perplexity(model, windows: torch.Tensor, device: str = "cuda") -> float:
    model.eval()
    total_nll, total_tok = 0.0, 0
    for i in range(windows.shape[0]):
        ids = windows[i: i + 1].to(device)
        out = model(ids, labels=ids)
        # HF returns mean over the window; weight by token count
        total_nll += float(out.loss) * (ids.shape[1] - 1)
        total_tok += ids.shape[1] - 1
    return math.exp(total_nll / total_tok)


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #
def calibrate(model, windows: torch.Tensor, device: str, out_path: str,
              max_layers: int | None = None) -> dict:
    layers = linear_layers(model)
    print(f"[calib] {len(layers)} linear layers")
    if max_layers:
        layers = layers[:max_layers]

    t0 = time.time()
    base = perplexity(model, windows, device)
    print(f"[calib] fp32 baseline PPL = {base:.4f}  ({time.time()-t0:.1f}s)")

    rows = []
    for idx, (name, module) in enumerate(layers):
        for bits in BITS:
            with quantised_weights([(name, module)], bits):
                ppl = perplexity(model, windows, device)
            rows.append({"layer": name, "bits": bits, "ppl": round(ppl, 5),
                         "delta": round(ppl - base, 6),
                         "n_params": int(module.weight.numel())})
            if bits == 4:
                print(f"  [{idx+1}/{len(layers)}] {name:44s} "
                      f"p4={ppl:8.4f} (+{ppl-base:7.4f})  {time.time()-t0:.0f}s",
                      flush=True)

    payload = {"model": str(getattr(model, "name_or_path", "")),
               "baseline_ppl": round(base, 6),
               "group_size": auto_group(layers[0][1].weight.shape[-1]),
               "window_tokens": int(windows.shape[1]),
               "n_windows": int(windows.shape[0]),
               "rows": rows,
               "seconds": round(time.time() - t0, 1)}
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[calib] wrote {out_path}  ({len(rows)} rows)")

    top = sorted([r for r in rows if r["bits"] == 4], key=lambda r: -r["delta"])[:8]
    print("\nmost sensitive layers @4bit:")
    for r in top:
        print(f"  {r['layer']:44s} +{r['delta']:.3f}")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--precision", default="fp16", choices=["fp16", "nf4"])
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--max-windows", type=int, default=8)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--max-layers", type=int, default=None)
    ap.add_argument("--out", default="results/sensitivity.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required", file=sys.stderr)
        return 1

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    if args.calibrate and args.precision != "fp16":
        print("calibration needs an unquantised model; forcing fp16", file=sys.stderr)
        args.precision = "fp16"

    kwargs = dict(trust_remote_code=True, device_map="cuda")
    if args.precision == "nf4":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    else:
        kwargs["torch_dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs).eval()

    windows = load_tokens(tok, args.text, args.seq_len, args.max_windows)
    if args.calibrate:
        calibrate(model, windows, "cuda", args.out, args.max_layers)
    else:
        ppl = perplexity(model, windows, "cuda")
        print(json.dumps({"model": args.model, "precision": args.precision,
                          "ppl": round(ppl, 4), "windows": int(windows.shape[0])},
                         indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
