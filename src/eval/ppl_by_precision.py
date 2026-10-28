"""Mean NLL of a model loaded with a real bnb precision, over the same windows.

Used to price 4-bit weights on a 4B model: fp16 does not fit in 8 GB, so the
reference is bnb int8 and the treatment is bnb nf4. Both are real kernels, so
this is a 4-bit-versus-8-bit comparison rather than a quantiser-strength one.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.gqa_patch import enable_gqa_sdpa  # noqa: E402


@torch.inference_mode()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--precision", default="nf4", choices=["nf4", "int8"])
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--windows", type=int, default=32)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
    text = open(args.text, encoding="utf-8", errors="ignore").read()
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    n = min(len(ids) // args.seq_len, args.windows)
    win = ids[: n * args.seq_len].view(n, args.seq_len)

    if args.precision == "nf4":
        cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.float16,
                                 bnb_4bit_use_double_quant=True)
    else:
        cfg = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="cuda", trust_remote_code=True,
        quantization_config=cfg).eval()
    enable_gqa_sdpa(model, kv_bits=None)

    tot = 0.0
    for i in range(win.shape[0]):
        x = win[i: i + 1].to("cuda")
        tot += float(model(x, labels=x).loss)
    mean = tot / win.shape[0]
    print(f"{args.precision}: mean NLL = {mean:.6f} over {win.shape[0]} windows")
    if args.out:
        json.dump({"model": args.model, "precision": args.precision,
                   "windows": int(win.shape[0]), "mean_nll": round(mean, 6)},
                  open(args.out, "w", encoding="utf-8"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
