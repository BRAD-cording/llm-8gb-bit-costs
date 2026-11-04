"""Collapse the Jetson artifacts into one JSON the manuscript can cite.

Inputs (results/c2_jetson/):
  bench_L{L}_{t}.json   llama-bench output for a context length and cache type
  rep_L{L}_{t}_{i}.log  instantiation attempts (three per cell)
  ppl_{t}.log           llama-perplexity, WikiText-2, L=4096, 16 chunks
  ram_L{L}_{t}.log      tegrastats line stream during one cell
  grid.log              the run's own log, for the repeat verdicts
Output: results/c2_jetson_summary.json
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "c2_jetson"
OUT = ROOT / "results" / "c2_jetson_summary.json"
CACHES = ("f16", "q8_0", "q4_0")
LENGTHS = (4096, 8192, 16384, 32768)


def bench_array(path):
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    start = text.find("[")
    if start < 0:
        return None
    try:
        return json.JSONDecoder().raw_decode(text[start:])[0]
    except json.JSONDecodeError:
        return None


def throughput():
    rows = {}
    for L in LENGTHS:
        for t in CACHES:
            arr = bench_array(SRC / f"bench_L{L}_{t}.json")
            if not arr or len(arr) < 2:
                continue
            pp, tg = arr[0], arr[1]
            rows[f"{L}/{t}"] = {
                "prefill": round(pp["avg_ts"], 1),
                "prefill_stddev": round(pp.get("stddev_ts", 0.0), 1),
                "decode": round(tg["avg_ts"], 1),
                "decode_stddev": round(tg.get("stddev_ts", 0.0), 1),
            }
    return rows


def fits():
    verdict = {}
    log = SRC / "grid.log"
    if log.exists():
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.search(r"L=(\d+) (\w+) ok=(\d)/3", line)
            if m:
                verdict[f"{m.group(1)}/{m.group(2)}"] = f"{m.group(3)}/3"
        return verdict
    for L in LENGTHS:                      # fall back to the attempt logs
        for t in CACHES:
            logs = sorted(SRC.glob(f"rep_L{L}_{t}_*.log"))
            if logs:
                verdict[f"{L}/{t}"] = f"{sum('error' not in p.read_text(errors='replace').lower() for p in logs)}/{len(logs)}"
    return verdict


def perplexity():
    out = {}
    base = None
    for t in CACHES:
        path = SRC / f"ppl_{t}.log"
        if not path.exists():
            continue
        m = re.search(r"Final estimate: PPL = ([0-9.]+)", path.read_text(errors="replace"))
        if not m:
            continue
        value = float(m.group(1))
        if base is None:
            base = value
        out[t] = {"ppl": round(value, 4), "delta_pct": round((value / base - 1.0) * 100.0, 3)}
    return out


def peak_ram():
    out = {}
    for path in sorted(SRC.glob("ram_L*_*.log")):
        m = re.match(r"ram_L(\d+)_(\w+)\.log", path.name)
        if not m:
            continue
        values = [int(v) for v in re.findall(r"RAM (\d+)/(\d+)MB", path.read_text(errors="replace")) for v in v[:1]]
        if values:
            out[f"{m.group(1)}/{m.group(2)}"] = max(values)
    return out


def main():
    data = {
        "device": "Jetson Orin Nano Super, 8 GB unified, 25 W, desktop running",
        "runtime": "llama.cpp b12063, CUDA 12.6.68, sm_87",
        "weights": "Qwen3-1.7B-Q8_0.gguf (1,834,426,016 B, byte-identical to the laptop copy)",
        "fits": fits(),
        "throughput": throughput(),
        "perplexity_L4096_16chunk": perplexity(),
        "peak_ram_mb": peak_ram(),
    }
    OUT.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"  fit cells {len(data['fits'])}, throughput cells {len(data['throughput'])}, "
          f"ppl {len(data['perplexity_L4096_16chunk'])}, ram {len(data['peak_ram_mb'])}")


if __name__ == "__main__":
    main()
