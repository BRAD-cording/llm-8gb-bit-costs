"""
Run the regime sweep on a synthetic, architecture-exact model.

Why this is sound: TTFT, TPOT, the achievable bandwidth and the memory footprint
depend on tensor SHAPES and DTYPES, not on the values in the weights. A model
built from the real published config therefore reproduces the system behaviour of
the real model exactly, without waiting for a multi-gigabyte download.

This gives the memory/latency half of the paper immediately. Accuracy (PPL,
sensitivity) still requires real weights -- there is no substitute for those.

Usage:
    python src/bench/synth_sweep.py --config-from Qwen/Qwen3-4B --precision nf4 \
        --lengths 128,512,2048,8192 --out results/synth_qwen3_4b_nf4.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.regime_sweep import build_model, sweep_one  # noqa: E402
from bench.gqa_patch import enable_gqa_sdpa  # noqa: E402

CFG_URL = ("https://modelscope.cn/api/v1/models/{mid}/repo"
           "?Revision=master&FilePath=config.json")
UA = {"User-Agent": "synth/1.0"}


def fetch_config(mid: str) -> dict:
    with urllib.request.urlopen(urllib.request.Request(
            CFG_URL.format(mid=mid), headers=UA), timeout=45) as r:
        return json.loads(r.read().decode("utf-8"))


def build_synth(cfg_dict: dict, model_type: str, dtype=torch.float16):
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg_dict = dict(cfg_dict)
    cfg_dict["model_type"] = model_type
    cfg = AutoConfig.for_model(model_type, **{
        k: v for k, v in cfg_dict.items()
        if k not in ("architectures", "model_type", "transformers_version",
                     "_name_or_path", "torch_dtype", "dtype")
    })
    t0 = time.time()
    model = AutoModelForCausalLM.from_config(cfg, dtype=dtype)
    print(f"[synth] built {sum(p.numel() for p in model.parameters())/1e9:.2f} B params "
          f"in {time.time()-t0:.1f}s", flush=True)
    return model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-from", required=True)
    ap.add_argument("--model-type", default="qwen3")
    ap.add_argument("--precision", default="nf4", choices=["nf4", "int8", "fp16"])
    ap.add_argument("--lengths", default="128,512,2048,8192")
    ap.add_argument("--out", default="results/synth_sweep.json")
    ap.add_argument("--save-dir", default=None)
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "gqa_sdpa"])
    args = ap.parse_args()

    cfg = fetch_config(args.config_from)
    print(f"[synth] config: L={cfg.get('num_hidden_layers')} H={cfg.get('hidden_size')} "
          f"heads={cfg.get('num_attention_heads')} kv={cfg.get('num_key_value_heads')} "
          f"ffn={cfg.get('intermediate_size')} vocab={cfg.get('vocab_size')}", flush=True)

    model = build_synth(cfg, args.model_type)

    if args.precision in ("nf4", "int8"):
        # quantised loading needs a checkpoint on disk; fp16 weights would not
        # fit in 8 GB for a 4B model
        save_dir = Path(args.save_dir or f"C:/models/_synth_{args.config_from.split('/')[-1]}")
        save_dir.mkdir(parents=True, exist_ok=True)
        cfg_txt = save_dir / "config.json"
        if not cfg_txt.exists():
            t0 = time.time()
            model.save_pretrained(str(save_dir), safe_serialization=True)
            print(f"[synth] saved to {save_dir} in {time.time()-t0:.0f}s", flush=True)
        del model
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        # NOTE: this line used to hardcode "nf4", so an int8 run silently
        # produced nf4 numbers -- identical footprint, identical timings. The
        # requested precision must be passed through.
        model, _ = build_model(str(save_dir), args.precision, "cuda")
    else:
        model = model.to("cuda").eval()

    total_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
    print(f"[synth] on-device weight footprint {total_gb:.2f} GB", flush=True)

    if args.attn == "gqa_sdpa":
        enable_gqa_sdpa(model)
        print(f"[synth] attention implementation -> "
              f"{model.config._attn_implementation}", flush=True)

    meta = {"model": f"SYNTH:{args.config_from}", "precision": args.precision,
            "device": torch.cuda.get_device_name(0),
            "config": {k: cfg[k] for k in ("num_hidden_layers", "hidden_size",
                                           "num_attention_heads", "num_key_value_heads",
                                           "intermediate_size", "vocab_size")
                       if k in cfg},
            "rows": []}
    for L in [int(x) for x in args.lengths.split(",")]:
        try:
            row = sweep_one(model, None, L, "cuda", 22.36, 179.8)
        except torch.cuda.OutOfMemoryError:
            row = {"seq_len": L, "error": "OOM"}
            torch.cuda.empty_cache()
        meta["rows"].append(row)
        print(json.dumps(row), flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
