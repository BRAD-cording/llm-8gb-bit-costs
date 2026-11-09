"""Scaled passkey retrieval: n examples per depth, Wilson intervals.

The published passkey set has five prompts, which cannot support a claim about a
long-context capability. This script generates its own needles at a given context
length, runs the same configuration's cache quantiser as the perplexity
experiments, and reports the fraction of keys recovered with a 95% Wilson
interval.

Configurations:
  fp16   no cache quantisation
  kNg    Kv cache at N bits, KIVI-style (per-channel keys, per-token values,
         128-token residual window)
  kNu    Kv cache at N bits, plain per-channel/per-token rounding, no window

Usage:
  python src/eval/passkey_scale.py --model C:/models/Qwen3-0.6B \
      --lengths 11000,16000 --n 50 --out results/passkey_scale_06b.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _ROOT)

from bench.gqa_patch import enable_gqa_sdpa  # noqa: E402

FILLER = ("The grass is green. The sky is blue. The sun is yellow. "
          "Here we go. There and back again. ")
PREFIX = ("There is an important info hidden inside a lot of irrelevant text. "
          "Find it and memorize it. I will quiz you about the important information there.\n")
POSTFIX = "What is the pass key? The pass key is "


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - half) / d, (c + half) / d)


def make_example(tok, length_tokens: int, rng: random.Random) -> tuple[str, str]:
    key = f"{rng.randint(100000, 999999)}"
    needle = f"The pass key is {key}. Remember it. "
    # build haystack of roughly length_tokens tokens
    unit = len(tok(FILLER, add_special_tokens=False).input_ids)
    reps = max(1, length_tokens // unit)
    depth = rng.uniform(0.1, 0.9)
    cut = int(reps * depth)
    body = PREFIX + FILLER * cut + needle + FILLER * (reps - cut)
    return body, key


def build(model_path: str):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="cuda",
        trust_remote_code=True).eval()
    return m


@torch.inference_mode()
def ask(model, tok, prompt: str, max_new: int = 12) -> str:
    ids = tok(prompt + POSTFIX, return_tensors="pt",
              add_special_tokens=False).input_ids.to("cuda")
    out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.eos_token_id or 0)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def apply_config(model, name: str):
    if name == "fp16":
        enable_gqa_sdpa(model, kv_bits=None)
    elif name.startswith("k"):
        bits = int(name[1])
        windowed = name.endswith("g")
        enable_gqa_sdpa(model, kv_bits=bits, kv_residual=128 if windowed else 0)
    else:
        raise ValueError(name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lengths", default="11000,16000")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--configs", default="fp16,k2g,k2u,k4g,k4u")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    lengths = [int(x) for x in args.lengths.split(",")]
    names = [x.strip() for x in args.configs.split(",")]

    rng = random.Random(args.seed)
    cases = {}
    for L in lengths:
        cases[L] = [make_example(tok, L, rng) for _ in range(args.n)]

    model = build(args.model)
    rows = []
    for name in names:
        apply_config(model, name)
        for L in lengths:
            hits = 0
            for body, key in cases[L]:
                ans = ask(model, tok, body)
                if key in ans:
                    hits += 1
            lo, hi = wilson(hits, len(cases[L]))
            row = {"config": name, "length": L, "hit": hits, "n": len(cases[L]),
                   "rate": hits / len(cases[L]), "wilson95": [round(lo, 4), round(hi, 4)]}
            rows.append(row)
            print(json.dumps(row), flush=True)

    payload = {"model": args.model, "seed": args.seed, "rows": rows}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
    print("WROTE", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
