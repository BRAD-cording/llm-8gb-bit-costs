"""
Wait for a model download to complete, then immediately run the regime sweep.

Rationale: the model arrives in ~2 h; nobody should be awake to start the
experiment.  This polls the file size, then runs nf4 and fp16 sweeps back to
back and leaves JSON behind.

Usage:
    python tools/run_when_ready.py --model-dir C:/models/Qwen3-0.6B \
        --expected-mb 1503.3 --tag qwen06
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

WS = Path(__file__).resolve().parent.parent


def size_mb(p: Path) -> float:
    return p.stat().st_size / 1e6 if p.exists() else 0.0


def wait_for(model_dir: Path, expected_mb: float, tol: float = 0.02,
             timeout_h: float = 6.0) -> bool:
    f = model_dir / "model.safetensors"
    t0 = time.time()
    last = -1.0
    stable = 0
    while time.time() - t0 < timeout_h * 3600:
        cur = size_mb(f)
        if cur >= expected_mb * (1 - tol):
            return True
        if cur > 0 and abs(cur - last) < 0.05:
            stable += 1
            if stable >= 40:  # ~20 min with no progress
                print(f"[watch] stalled at {cur:.0f}/{expected_mb:.0f} MB", flush=True)
                return False
        else:
            stable = 0
        last = cur
        if int(time.time() - t0) % 300 < 31:
            print(f"[watch] {cur:.0f}/{expected_mb:.0f} MB", flush=True)
        time.sleep(30)
    return False


def run(cmd: list[str]) -> int:
    print(f"[run] {' '.join(cmd)}", flush=True)
    p = subprocess.run(cmd, cwd=str(WS))
    return p.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--expected-mb", type=float, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--lengths", default="128,512,2048,8192,16384")
    args = ap.parse_args()

    md = Path(args.model_dir)
    ok = wait_for(md, args.expected_mb)
    if not ok:
        print("[watch] giving up", flush=True)
        return 1
    print("[watch] model complete, starting sweeps", flush=True)
    time.sleep(10)  # let the writer flush

    py = sys.executable
    jobs = [
        ("nf4", args.lengths),
        ("fp16", "128,512,2048,8192"),
    ]
    results = {}
    for prec, lengths in jobs:
        out = f"results/sweep_{args.tag}_{prec}.json"
        rc = run([py, "src/bench/regime_sweep.py", "--model", str(md),
                  "--precision", prec, "--lengths", lengths, "--out", out])
        results[prec] = {"rc": rc, "out": out}
        time.sleep(5)

    print(json.dumps(results, indent=2), flush=True)
    Path("results").mkdir(exist_ok=True)
    (Path("results") / f"sweep_{args.tag}_manifest.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
