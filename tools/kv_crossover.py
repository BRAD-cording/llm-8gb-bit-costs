"""
Compute the KV-vs-weights memory crossover for candidate models, from their REAL
config.json (fetched from ModelScope, since HuggingFace is unreachable).

This produces the numbers behind results/PREREGISTRATION_W7.md: at which context
length the KV cache overtakes the weights, and therefore at which L the optimal
weight/KV bit split can plausibly reverse.

Usage:
    python tools/kv_crossover.py --models Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B Qwen/Qwen3-4B Qwen/Qwen3-8B
"""
from __future__ import annotations

import argparse
import json
import urllib.request

BLOB = "https://modelscope.cn/api/v1/models/{mid}/repo?Revision=master&FilePath=config.json"
UA = {"User-Agent": "cfg/1.0"}


def fetch_config(mid: str) -> dict:
    with urllib.request.urlopen(urllib.request.Request(BLOB.format(mid=mid), headers=UA), timeout=45) as r:
        return json.loads(r.read().decode("utf-8"))


def analyse(cfg: dict) -> dict:
    n_layers = cfg.get("num_hidden_layers")
    hidden = cfg.get("hidden_size")
    n_heads = cfg.get("num_attention_heads")
    n_kv = cfg.get("num_key_value_heads") or n_heads
    head_dim = cfg.get("head_dim") or hidden // n_heads
    inter = cfg.get("intermediate_size")
    tie = cfg.get("tie_word_embeddings", False)

    # parameter count for the decoder body (approx, ignoring norms)
    attn = 2 * hidden * (n_heads * head_dim) + 2 * hidden * (n_kv * head_dim)
    mlp = 3 * hidden * inter if cfg.get("hidden_act") != "moe" else 3 * hidden * inter
    per_layer = attn + mlp
    params = n_layers * per_layer + (0 if tie else 2 * hidden * cfg.get("vocab_size", 0))

    kv_per_token = 2 * n_layers * n_kv * head_dim * 2  # bytes, fp16 K and V
    return {
        "layers": n_layers, "hidden": hidden, "heads": n_heads, "kv_heads": n_kv,
        "head_dim": head_dim, "params_M": round(params / 1e6, 1),
        "kv_bytes_per_token": kv_per_token,
    }


def crossover_rows(info: dict, bit_widths=(2, 3, 4, 8)) -> list[dict]:
    rows = []
    params = info["params_M"] * 1e6
    kv_tok = info["kv_bytes_per_token"]
    for b in bit_widths:
        w_bytes = params * b / 8
        # L at which KV bytes equal weight bytes
        l_eq = w_bytes / kv_tok
        rows.append({
            "bits": b,
            "weights_GB": round(w_bytes / 1e9, 3),
            "L_equal_KB": int(l_eq),
            "kv_GB_at_8k": round(kv_tok * 8192 / 1e9, 3),
            "kv_GB_at_32k": round(kv_tok * 32768 / 1e9, 3),
            "total_GB_at_32k": round((w_bytes + kv_tok * 32768) / 1e9, 3),
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=[
        "Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B", "Qwen/Qwen3-8B"])
    ap.add_argument("--out", default="results/kv_crossover.json")
    args = ap.parse_args()

    all_rows = {}
    for mid in args.models:
        try:
            cfg = fetch_config(mid)
        except Exception as e:
            print(f"{mid}: CONFIG FETCH FAILED {type(e).__name__}: {str(e)[:80]}")
            continue
        info = analyse(cfg)
        rows = crossover_rows(info)
        all_rows[mid] = {"info": info, "rows": rows}
        print(f"\n{mid}")
        print(f"  layers={info['layers']} hidden={info['hidden']} "
              f"heads={info['heads']} kv_heads={info['kv_heads']} "
              f"params~{info['params_M']}M  KV={info['kv_bytes_per_token']/1024:.0f} KiB/token")
        print(f"  {'bits':>4} {'W(GB)':>7} {'L@W==KV':>9} {'KV@8k':>7} {'KV@32k':>7} {'tot@32k':>8}")
        for r in rows:
            print(f"  {r['bits']:>4} {r['weights_GB']:>7.2f} {r['L_equal_KB']:>9d} "
                  f"{r['kv_GB_at_8k']:>7.2f} {r['kv_GB_at_32k']:>7.2f} {r['total_GB_at_32k']:>8.2f}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
