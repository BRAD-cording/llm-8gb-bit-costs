"""C2: run the recommended policy on a shipped runtime that stores the cache packed.

The manuscript's policy rows (compress the KV cache, keep weights at 8 bits) were
previously *simulated*: the attention patch quantises and immediately dequantises.
This script replaces those rows with a real measurement on llama.cpp, whose
``--cache-type-k/--cache-type-v`` options genuinely store the cache in packed form.

What it measures, per (context length, cache type):
  * prefill and decode throughput (llama-bench)
  * peak device memory (nvidia-smi sampled while the process runs)
  * perplexity on WikiText-2 (llama-perplexity), i.e. accuracy with a packed cache

Usage:
    python src/eval/c2_llamacpp.py --bin C:/Users/HP/c2/bin \
        --model C:/Users/HP/c2/models/Qwen3-1.7B-Q8_0.gguf \
        --contexts 4096,8192,16384,32768 \
        --cache-types f16,q8_0,q4_0 \
        --out results/c2_llamacpp_17b.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

USABLE_MB = 6.9 * 1024        # the manuscript's measured usable budget on device A
PHYSICAL_MB = 8187.0         # nvidia-smi total VRAM on device A

_SMI = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"]


def device_used_mb() -> float:
    out = subprocess.run(_SMI, capture_output=True, text=True, timeout=10).stdout.strip()
    first = out.splitlines()[0].replace("MiB", "").strip()
    return float(first)


class PeakSampler:
    """Sample device memory in the background; report peak above a baseline."""

    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.baseline = device_used_mb()
        self.peak = self.baseline
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                self.peak = max(self.peak, device_used_mb())
            except Exception:
                pass
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=2)

    @property
    def delta_mb(self) -> float:
        return max(0.0, self.peak - self.baseline)


def run_capture(cmd: list[str], log: Path, timeout: int) -> tuple[int, str, float]:
    t0 = time.perf_counter()
    with PeakSampler() as sampler:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            rc, out = proc.returncode, (proc.stdout or "") + (proc.stderr or "")
        except subprocess.TimeoutExpired as exc:
            rc, out = -9, (exc.stdout or "") + (exc.stderr or "") + "\n[timeout]"
    elapsed = time.perf_counter() - t0
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(out, encoding="utf-8", errors="replace")
    record = {"rc": rc, "seconds": round(elapsed, 2), "peak_mb": round(sampler.delta_mb, 1),
              "baseline_mb": round(sampler.baseline, 1)}
    return rc, out, record


PP_RE = re.compile(r"pp(\d+)\s*\|\s*([0-9.]+)\s*±\s*([0-9.]+)")
TG_RE = re.compile(r"tg(\d+)\s*\|\s*([0-9.]+)\s*±\s*([0-9.]+)")
PPL_RE = re.compile(r"Final estimate:\s*PPL\s*=\s*([0-9.]+)", re.I)


def bench(binary_dir: Path, model: Path, ctx: int, ctk: str, ctv: str,
          threads: int, ngl: int, log_dir: Path, timeout: int = 900) -> dict:
    exe = binary_dir / "llama-bench.exe"
    cmd = [str(exe), "-m", str(model), "-ngl", str(ngl), "-t", str(threads),
           "-p", str(ctx), "-n", "128", "-r", "2",
           "-ctk", ctk, "-ctv", ctv, "-o", "json"]
    rc, out, rec = run_capture(cmd, log_dir / f"bench_{model.stem}_L{ctx}_{ctk}.log", timeout=timeout)
    rec["prefill_tps"] = rec["decode_tps"] = None
    if rc == 0:
        try:
            # llama-bench prints the JSON array followed by a build banner; raw_decode
            # stops at the end of the array instead of demanding a single document.
            start = out.index("[")
            data, _ = json.JSONDecoder().raw_decode(out[start:])
            for row in data:
                n_prompt = int(row.get("n_prompt", 0))
                n_gen = int(row.get("n_gen", 0))
                tps = row.get("avg_ts")
                if n_prompt and not n_gen:
                    rec["prefill_tps"] = round(tps, 2)
                elif n_gen and not n_prompt:
                    rec["decode_tps"] = round(tps, 2)
        except Exception as exc:  # keep the raw log; the number is simply missing
            rec["parse_error"] = str(exc)
    elif m := PP_RE.search(out):
        rec["prefill_tps"] = float(m.group(2))
    return rec


def perplexity(binary_dir: Path, model: Path, corpus: Path, ctx: int,
               ctk: str, ctv: str, ngl: int, log_dir: Path) -> dict:
    exe = binary_dir / "llama-perplexity.exe"
    cmd = [str(exe), "-m", str(model), "-f", str(corpus), "-c", str(ctx),
           "-ngl", str(ngl), "-ctk", ctk, "-ctv", ctv, "--chunks", "8"]
    rc, out, rec = run_capture(cmd, log_dir / f"ppl_{model.stem}_L{ctx}_{ctk}.log", timeout=5400)
    rec["ppl"] = None
    if m := PPL_RE.search(out):
        rec["ppl"] = float(m.group(1))
    elif "error" in out.lower() or rc != 0:
        rec["error"] = out[-400:]
    return rec


def _save(out_path: Path, model: Path, rows: list) -> None:
    payload = {"model": str(model), "usable_mb": USABLE_MB, "physical_mb": PHYSICAL_MB,
               "device": "RTX 4060 Laptop (device A)", "runtime": "llama.cpp", "rows": rows}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", default="data/wikitext2_test.txt")
    ap.add_argument("--contexts", default="4096,8192,16384,32768")
    ap.add_argument("--cache-types", default="f16,q8_0,q4_0")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--row-timeout", type=int, default=900)
    ap.add_argument("--skip-bench", action="store_true")
    ap.add_argument("--skip-ppl", action="store_true")
    ap.add_argument("--out", default="results/c2_llamacpp.json")
    args = ap.parse_args()

    binary_dir = Path(args.bin)
    model = Path(args.model)
    corpus = Path(args.corpus)
    out_path = Path(args.out)
    log_dir = out_path.parent / "c2_logs"

    rows = []
    for ctx in [int(c) for c in args.contexts.split(",")]:
        for cache_type in [c.strip() for c in args.cache_types.split(",")]:
            print(f"--- L={ctx} cache={cache_type}", flush=True)
            row = {"model": model.stem, "context": ctx, "cache_type": cache_type}
            if not args.skip_bench:
                row.update(bench(binary_dir, model, ctx, cache_type, cache_type,
                                 args.threads, args.ngl, log_dir, args.row_timeout))
            if not args.skip_ppl and corpus.exists():
                ppl = perplexity(binary_dir, model, corpus, ctx, cache_type,
                                 cache_type, args.ngl, log_dir)
                row["ppl"] = ppl["ppl"]
                row["ppl_peak_mb"] = ppl["peak_mb"]
                row["ppl_seconds"] = ppl["seconds"]
                if ppl.get("error"):
                    row["ppl_error"] = ppl["error"][:300]
            row["total_mb"] = round(row.get("peak_mb", 0) + row.get("baseline_mb", 0), 1)
            row["fits"] = row["total_mb"] < PHYSICAL_MB
            rows.append(row)
            _save(out_path, model, rows)   # survive a killed cell
            print(json.dumps(row), flush=True)

    _save(out_path, model, rows)
    print(f"wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
