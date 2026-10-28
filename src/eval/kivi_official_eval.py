"""
Cache cost with KIVI's own quantiser, on the same windows as the GPTQ numbers.

This is the counterpart to `gptq_win.py`: the weight side of those runs came from
the published gptqmodel, and here the cache side comes from KIVI's own code
(`third_party/KIVI/utils_quant.py`, simulate mode).

    python src/eval/kivi_official_eval.py --model C:/models/Qwen3-0.6B
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

# run as a script from the repo root: the package roots are src/ and .
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _ROOT)
from bench.gqa_patch import enable_gqa_sdpa  # noqa: E402


def load_windows(tok, path, seq_len, n):
    text = open(path, encoding="utf-8", errors="ignore").read()
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    n = min(len(ids) // seq_len, n)
    return ids[: n * seq_len].view(n, seq_len)


@torch.inference_mode()
def nll(model, win):
    out = []
    for i in range(win.shape[0]):
        x = win[i: i + 1].to("cuda")
        out.append(float(model(x, labels=x).loss))
    return out


def rel(a, b):
    d = sum(y - x for x, y in zip(a, b)) / len(a)
    return 100 * (pow(2.718281828459045, d) - 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", default="data/calib_arxiv.txt")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--windows", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--residual", type=int, default=128)
    ap.add_argument("--group", type=int, default=32)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
    win = load_windows(tok, args.text, args.seq_len, args.windows)
    n_cal = max(2, args.windows // 8)
    ev = win[:-n_cal]
    print(f"{ev.shape[0]} eval windows", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="cuda",
        trust_remote_code=True).eval()
    enable_gqa_sdpa(model, kv_bits=None)
    base = nll(model, ev)
    print("baseline NLL", round(sum(base) / len(base), 6), flush=True)

    # KIVI's own quantiser, with its residual window and grouping
    enable_gqa_sdpa(model, kv_bits=args.bits, kv_residual=args.residual,
                    kv_official=True)
    kivi = nll(model, ev)
    print("kivi official", round(rel(base, kivi), 3), flush=True)

    # their grouping switched off, to separate the two effects
    enable_gqa_sdpa(model, kv_bits=args.bits, kv_residual=args.residual,
                    kv_group=args.bits * 0 + 1, kv_official=True)
    kivi_g1 = nll(model, ev)
    print("kivi official, group=1 (residual only)", round(rel(base, kivi_g1), 3))

    payload = {"model": args.model, "bits": args.bits, "windows": int(ev.shape[0]),
               "baseline_nll": round(sum(base) / len(base), 6),
               "cache_kivi_official_group32": round(rel(base, kivi), 3)}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(payload, open(args.out, "w"), indent=2)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
