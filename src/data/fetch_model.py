"""Fetch a model snapshot from ModelScope (HuggingFace is unreachable in this network).

ModelScope is the only working weight host here: huggingface.co fails DNS and
hf-mirror.com fails TLS.  Recorded in docs/ENVIRONMENT.md.

Usage:
    python src/data/fetch_model.py Qwen/Qwen3-0.6B
    python src/data/fetch_model.py Qwen/Qwen3-4B --out D:/models
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

DEFAULT_CACHE = Path("C:/models")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_id")
    ap.add_argument("--out", default=str(DEFAULT_CACHE))
    ap.add_argument("--revision", default=None)
    args = ap.parse_args()

    cache = Path(args.out)
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MODELSCOPE_CACHE", str(cache))

    from modelscope import snapshot_download

    print(f"[fetch] {args.model_id} -> {cache}", flush=True)
    path = snapshot_download(args.model_id, cache_dir=str(cache),
                             revision=args.revision)
    print(json.dumps({"model_id": args.model_id, "path": path}, indent=2))

    p = Path(path)
    total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    print(f"[fetch] done, {total / 1e9:.2f} GB at {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
