"""Sensor-log QA: a real IoT text task under two bit budgets.

Windows are cut from the public LogHub 2k samples (eight systems). Each window
is 2,048 tokens. Two questions per window, both with programmatic answers:

  A  count  "How many lines contain the token X?"          -> integer match
  B  system "Which system produced this log?"              -> word match

Run: python src/eval/logqa.py --conf shipped --out results/logqa_shipped.json
"""
import argparse
import glob
import io
import json
import os
import random
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from bench.gqa_patch import enable_gqa_sdpa  # noqa: E402
from contextlib import nullcontext  # noqa: E402
from eval.ppl import linear_layers, quantised_weights  # noqa: E402

CONFS = {"shipped": (4, 16), "ours": (8, 4), "fp16": (16, 16)}


def windows(path, tok, size, n):
    text = io.open(path, encoding="utf-8", errors="replace").read()
    lines = [ln for ln in text.splitlines() if ln.strip()]
    rng = random.Random(7)
    out = []
    while len(out) < n and len(lines) > 60:
        start = rng.randrange(0, len(lines) - 60)
        chunk = lines[start:start + 60]
        ids = tok("\n".join(chunk))["input_ids"]
        if len(ids) < size * 0.7:
            continue
        out.append((chunk, ids[:size]))
    return out


def build_questions(tok, sets, size, per_set):
    items = []
    for path in sets:
        name = os.path.basename(path).split("_")[0].lower()
        for chunk, ids in windows(path, tok, size, per_set):
            body = "\n".join(chunk)
            toks = re.findall(r"[A-Za-z][A-Za-z_.-]{3,}", body)
            counts = {}
            for t in toks:
                counts[t] = counts.get(t, 0) + 1
            pick = [t for t, c in counts.items() if 2 <= c <= 9]
            if not pick:
                continue
            t = pick[len(items) % len(pick)]
            items.append({"ids": ids, "qtype": "count",
                          "q": f"How many lines above contain the word {t}?"
                               " Answer with one number.",
                          "a": str(counts[t]), "set": name})
            items.append({"ids": ids, "qtype": "system",
                          "q": "Which system produced the log above?"
                               " Answer with one word.",
                          "a": name, "set": name})
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", default="shipped", choices=list(CONFS))
    ap.add_argument("--model", default=r"C:\models\Qwen3-1.7B")
    ap.add_argument("--size", type=int, default=2048)
    ap.add_argument("--per-set", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    wb, kb = CONFS[args.conf]
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.float16,
        low_cpu_mem_usage=True).eval()
    model = model.cuda()
    enable_gqa_sdpa(model, kv_bits=None if kb >= 16 else kb, kv_residual=128)

    sets = sorted(glob.glob(os.path.join(ROOT, "data", "loghub", "*_2k.log")))
    items = build_questions(tok, sets, args.size, args.per_set)
    layers = [(n, m) for n, m in linear_layers(model)
              if 'lm_head' not in n]

    rows, hits = [], 0
    for i, it in enumerate(items):
        prompt = ("You are given a log excerpt.\n\n"
                  + tok.decode(it["ids"], skip_special_tokens=True)
                  + "\n\nQuestion: " + it["q"] + "\nAnswer:")
        enc = tok(prompt, return_tensors="pt", truncation=True,
                  max_length=args.size + 256).to("cuda")
        cm = quantised_weights(layers, wb) if wb < 16 else nullcontext()
        with cm, torch.inference_mode():
            gen = model.generate(**enc, max_new_tokens=24, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        text = tok.decode(gen[0][enc["input_ids"].shape[1]:],
                          skip_special_tokens=True).strip().lower()
        if it["qtype"] == "count":
            m = re.search(r"\d+", text)
            ok = bool(m) and int(m.group()) == int(it["a"])
        else:
            aliases = {"hdfs": ["hdfs", "dfs"], "openstack": ["openstack", "nova", "open"],
                       "healthapp": ["health"], "mac": ["mac"], "linux": ["linux"],
                       "zookeeper": ["zookeeper", "zoo"], "apache": ["apache"],
                       "bgl": ["bgl", "blue gene"]}
            ok = any(a in text for a in aliases.get(it["a"], [it["a"]]))
        hits += ok
        rows.append({"i": i, "qtype": it["qtype"], "answer": it["a"],
                     "said": text[:60], "ok": ok})
        if i % 5 == 0:
            print(i, len(items), hits, flush=True)

    res = {"conf": args.conf, "weights_bits": wb, "kv_bits": kb,
           "n": len(rows), "correct": hits,
           "acc": hits / max(1, len(rows)), "rows": rows}
    io.open(args.out, "w", encoding="utf-8").write(json.dumps(res, indent=1))
    print("WROTE", args.out, "acc", res["acc"])


if __name__ == "__main__":
    main()
