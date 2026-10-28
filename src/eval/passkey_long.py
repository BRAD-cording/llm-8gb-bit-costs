"""
Passkey retrieval at lengths where the configurations actually differ.

The five KIVI prompts are about 11k tokens, and at that length every configuration
recovers the passkey, so the task cannot separate them. This script builds the
same style of prompt at chosen lengths (the filler is periodic, the passkey
sentence is lifted from KIVI's own examples, the question template is theirs) and
reports accuracy, peak memory and whether the configuration runs at all.

What matters is expected to be the failure to run rather than a wrong answer:
a 16-bit cache grows with the context while a 4-bit one grows at a quarter of the
rate, so at some length the 16-bit configuration stops fitting and the compressed
one does not. That is a capability difference in the only sense that matters for
deployment.

Usage:
    python src/eval/passkey_long.py --model C:/models/Qwen3-0.6B --lengths 16384,32768
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _ROOT)

from bench.gqa_patch import enable_gqa_sdpa  # noqa: E402

HEADER = ("There is an important info hidden inside a lot of irrelevant text. "
          "Find it and memorize it. I will quiz you about the important "
          "information there.\n")
FILLER = ("The grass is green. The sky is blue. The sun is yellow. "
          "Here we go. There and back again.\n")
POSTFIX = "What is the pass key? The pass key is "


def build_prompt(tok, target_len: int, passkey: str, depth: float) -> tuple[str, int]:
    """Filler text with one passkey sentence at `depth` (0 = start, 1 = end)."""
    head_ids = tok(HEADER, add_special_tokens=False).input_ids
    fill_ids = tok(FILLER, add_special_tokens=False).input_ids
    key_text = f"The pass key is {passkey}. Remember it. {passkey} is the pass key.\n"
    key_ids = tok(key_text, add_special_tokens=False).input_ids
    post_ids = tok(POSTFIX, add_special_tokens=False).input_ids

    budget = target_len - len(head_ids) - len(key_ids) - len(post_ids)
    n_fill = max(1, budget // len(fill_ids))
    split = int(n_fill * depth)
    ids = (head_ids
           + fill_ids * split
           + key_ids
           + fill_ids * (n_fill - split)
           + post_ids)
    return tok.decode(ids), len(ids)


def build(model_path: str, precision: str):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    if precision == "nf4":
        cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.float16,
                                 bnb_4bit_use_double_quant=True)
        return AutoModelForCausalLM.from_pretrained(
            model_path, device_map="cuda", trust_remote_code=True,
            quantization_config=cfg).eval()
    return AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="cuda",
        trust_remote_code=True).eval()


@torch.inference_mode()
def ask(model, tok, prompt: str, max_new: int = 8) -> str:
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
    out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.eos_token_id or 0)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lengths", default="16384")
    ap.add_argument("--configs", default="naive16,default,policy,rtn4")
    ap.add_argument("--depths", default="0.5")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    plan = {
        "default": ("nf4", 16, "none"),
        "policy": ("fp16", 4, "kivi"),
        "naive16": ("fp16", 16, "none"),
        "rtn4": ("fp16", 4, "rtn"),
        "kivi2": ("fp16", 2, "kivi"),
        "rtn2": ("fp16", 2, "rtn"),
    }
    depths = [float(x) for x in args.depths.split(",")]
    lengths = [int(x) for x in args.lengths.split(",")]
    rng = random.Random(0)
    cases = []
    for L in lengths:
        for d in depths:
            for _ in range(args.repeats):
                key = str(rng.randrange(10000, 99999))
                prompt, real_len = build_prompt(tok, L, key, d)
                cases.append({"len": L, "depth": d, "key": key,
                              "prompt": prompt, "real_len": real_len})

    out = {"model": args.model, "rows": []}
    for name in args.configs.split(","):
        precision, kv_bits, cache_q = plan[name]
        row = {"config": name, "precision": precision, "kv_bits": kv_bits,
               "runs": [], "accuracy": None, "fits": True}
        try:
            m = build(args.model, precision)
        except Exception as e:
            row["fits"] = False
            row["error"] = f"load: {type(e).__name__}: {str(e)[:100]}"
            out["rows"].append(row)
            print(json.dumps(row), flush=True)
            continue
        if cache_q == "kivi":
            enable_gqa_sdpa(m, kv_bits=kv_bits, kv_residual=128, kv_official=True)
        elif cache_q == "rtn":
            enable_gqa_sdpa(m, kv_bits=kv_bits)
        else:
            enable_gqa_sdpa(m, kv_bits=None)

        hits = 0
        for c in cases:
            torch.cuda.reset_peak_memory_stats()
            try:
                ans = ask(m, tok, c["prompt"])
                ok = c["key"] in ans
                err = None
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                ans, ok, err = "<oom>", False, "OOM"
                row["fits"] = False
            hits += int(ok)
            row["runs"].append({"len": c["real_len"], "depth": c["depth"],
                                "key": c["key"], "answer": ans.strip()[:30],
                                "ok": ok, "error": err,
                                "peak_MB": round(torch.cuda.max_memory_allocated() / 1024 ** 2)})
            print(f"  {name:8s} L={c['real_len']:6d} d={c['depth']:.1f} "
                  f"{'OK' if ok else 'MISS' if not err else err}", flush=True)
        row["accuracy"] = round(100.0 * hits / len(cases), 1)
        out["rows"].append(row)
        print(f"{name}: {row['accuracy']}%  fits={row['fits']}", flush=True)
        del m
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
