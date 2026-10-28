"""
Passkey retrieval: a task-level metric for the cache quantisation comparison.

Perplexity says how much a quantiser disturbs next-token prediction on average;
it says little about whether a model can still *find* something it was told
11,000 tokens ago, which is exactly what a compressed cache threatens. KIVI ships
a passkey set for this purpose (`passkey_examples.jsonl`, five prompts of about
44,000 characters each, one hidden passkey per prompt, with its position given).

Each configuration is scored by generating a short answer and checking whether the
passkey appears in it.

Usage:
    python src/eval/passkey_eval.py --model C:/models/Qwen3-0.6B --out results/passkey_06b.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _ROOT)

from bench.gqa_patch import enable_gqa_sdpa  # noqa: E402


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


PASSKEY_POSTFIX = "What is the pass key? The pass key is "


@torch.inference_mode()
def ask(model, tok, prompt: str, max_new: int = 10) -> str:
    text = prompt + PASSKEY_POSTFIX
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
    out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.eos_token_id or 0)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--examples", default=".tmp/KIVI/passkey_examples.jsonl")
    ap.add_argument("--configs", default="default,policy,naive16")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    rows = [json.loads(l) for l in open(args.examples, encoding="utf-8")]

    # cache quantiser: "kivi" is KIVI's own, "rtn" is our ungrouped
    # per-channel/per-token rounding, which costs about 5% perplexity --- a
    # deliberately weaker cache that might actually break the task.
    plan = {
        "default":  ("nf4", 16, "none"),
        "policy":   ("fp16", 4, "kivi"),
        "naive16":  ("fp16", 16, "none"),
        "rtn4":     ("fp16", 4, "rtn"),
        "kivi2":    ("fp16", 2, "kivi"),
        "rtn2":     ("fp16", 2, "rtn"),
    }
    out = {"model": args.model, "rows": []}
    for name in args.configs.split(","):
        precision, kv_bits, cache_q = plan[name]
        m = build(args.model, precision)
        if cache_q == "kivi":
            enable_gqa_sdpa(m, kv_bits=kv_bits, kv_residual=128, kv_official=True)
        elif cache_q == "rtn":
            enable_gqa_sdpa(m, kv_bits=kv_bits)
        else:
            enable_gqa_sdpa(m, kv_bits=None)

        hits, details = 0, []
        for r in rows:
            torch.cuda.reset_peak_memory_stats()
            try:
                text = ask(m, tok, r["input"])
                ok = str(r["target"]) in text
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                text, ok = "<oom>", False
            hits += int(ok)
            details.append({"target": r["target"], "pos": r["passkey_position"],
                            "answer": text.strip()[:40], "ok": ok})
            print(f"  {name:8s} target={r['target']} -> {text.strip()[:24]!r} "
                  f"{'OK' if ok else 'MISS'}", flush=True)
        acc = 100.0 * hits / len(rows)
        peak = torch.cuda.max_memory_allocated() / 1024 ** 2
        row = {"config": name, "precision": precision, "kv_bits": kv_bits,
               "accuracy": round(acc, 1), "n": len(rows),
               "peak_MB": round(peak, 0), "details": details}
        out["rows"].append(row)
        print(f"{name}: {acc:.0f}% ({hits}/{len(rows)}), peak {peak:.0f} MB", flush=True)
        del m
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
