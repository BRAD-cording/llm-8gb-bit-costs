"""Merge the C2 measurement files into one table the manuscript can cite.

Inputs (all produced by ``src/eval/c2_llamacpp.py`` and ``alloc_census``):
  results/c2_llamacpp_17b.json, results/c2_llamacpp_4b.json   throughput + device peaks
  results/c2_alloc.txt                                        llama.cpp's own allocation logs
  perplexity values below                                     16-chunk WikiText-2 at L=4096

Output: results/c2_summary.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = [ROOT / "results" / "c2_llamacpp_17b.json", ROOT / "results" / "c2_llamacpp_4b.json"]
ALLOC = ROOT / "results" / "c2_alloc.txt"
OUT = ROOT / "results" / "c2_summary.json"

# 16 chunks x 4096 tokens of data/wikitext2_test.txt, llama-perplexity, flash attention on
PPL_16 = {
    ("Qwen3-1.7B-Q8_0", "f16"): 15.6078,
    ("Qwen3-1.7B-Q8_0", "q8_0"): 15.6257,
    ("Qwen3-1.7B-Q8_0", "q4_0"): 19.6896,
    ("Qwen3-4B-Q8_0", "f16"): 11.8091,
    ("Qwen3-4B-Q8_0", "q8_0"): 11.8099,
    ("Qwen3-4B-Q8_0", "q4_0"): 11.9257,
}

ALLOC_RE = re.compile(
    r"(Qwen3-\S+) \| L=(\d+) \| cache=(\w+) \|.*size = ([\d.]+) MiB.*"
    r"compute buffer size = ([\d.]+) MiB; CUDA0 model buffer size = ([\d.]+) MiB"
)


def load_alloc() -> dict:
    out = {}
    for line in ALLOC.read_text(encoding="utf-8").splitlines():
        m = ALLOC_RE.match(line)
        if m:
            out[(m.group(1), int(m.group(2)), m.group(3))] = {
                "kv_mib": float(m.group(4)),
                "compute_mib": float(m.group(5)),
                "model_mib": float(m.group(6)),
            }
    return out


def main() -> int:
    alloc = load_alloc()
    models = {}
    for src in RAW:
        if not src.exists():
            continue
        data = json.loads(src.read_text(encoding="utf-8"))
        name = Path(data["model"].replace("\\", "/")).stem
        rows = []
        for row in data["rows"]:
            a = alloc.get((name, row["context"], row["cache_type"]), {})
            rows.append({
                "context": row["context"],
                "cache": row["cache_type"],
                "kv_mib": a.get("kv_mib"),
                "compute_mib": a.get("compute_mib"),
                "model_mib": a.get("model_mib"),
                "alloc_total_mib": round(a["kv_mib"] + a["compute_mib"] + a["model_mib"], 1) if a else None,
                "prefill_tps": row.get("prefill_tps"),
                "decode_tps": row.get("decode_tps"),
                "peak_delta_mib": row["peak_mb"],
                "peak_total_mib": row["total_mb"],
                "completed": row["rc"] == 0 and bool(row.get("decode_tps")),
                "seconds": row["seconds"],
            })
        base = PPL_16[(name, "f16")]
        models[name] = {
            "rows": sorted(rows, key=lambda r: (r["context"], r["cache"])),
            "perplexity_L4096_16chunk": {
                ct: {"ppl": v, "delta_pct": round(100 * (v - base) / base, 3)}
                for (n, ct), v in PPL_16.items() if n == name
            },
        }

    payload = {
        "device": "RTX 4060 Laptop (device A)",
        "total_vram_mib": 8188,
        "usable_mib": 7066,
        "runtime": "llama.cpp b12063, CUDA 12.4, flash attention enabled",
        "note": "alloc_* from llama.cpp own logs; peak_* from nvidia-smi sampling; "
                "perplexity on WikiText-2, L=4096, 16 chunks",
        "models": models,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    for name, m in models.items():
        print(f"== {name}")
        for r in m["rows"]:
            print(f"  L={r['context']:>6} {r['cache']:>5} alloc={r['alloc_total_mib'] or 0:>7.0f} "
                  f"peak={r['peak_total_mib']:>6.0f} pp={r['prefill_tps'] or 0:>7.0f} "
                  f"tg={r['decode_tps'] or 0:>5.1f} ok={r['completed']}")
        for ct, v in m["perplexity_L4096_16chunk"].items():
            print(f"   ppl {ct}: {v['ppl']} ({v['delta_pct']:+.2f}%)")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
