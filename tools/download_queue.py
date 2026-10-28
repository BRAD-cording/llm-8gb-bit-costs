"""
Sequential model-download queue.

The link is ~0.2 MB/s and parallel connections get RST, so downloads MUST be
serialised.  This process walks a list, skips what is already complete, and
stops on the first hard failure.

Started as a detached background job on the laptop so that the queue keeps
draining while experiments run.

Usage:
    python tools/download_queue.py --dest C:/models \
        --models Qwen/Qwen3-1.7B Qwen/Qwen3-4B
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

API_TREE = "https://modelscope.cn/api/v1/models/{mid}/repo/tree?Revision=master&Root=/"
API_FILES = "https://modelscope.cn/api/v1/models/{mid}/repo/files?Revision=master&Recursive=true"
UA = {"User-Agent": "queue/1.0"}


def expected_bytes(mid: str) -> int:
    with urllib.request.urlopen(urllib.request.Request(
            API_FILES.format(mid=mid), headers=UA), timeout=60) as r:
        files = json.loads(r.read().decode())["Data"]["Files"]
    return sum(f["Size"] for f in files if f.get("Type") == "blob")


def already_done(dest: Path, mid: str, total: int) -> bool:
    d = dest / mid.split("/")[-1]
    if not d.exists():
        return False
    have = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
    return have >= total * 0.98


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="C:/models")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--wait-for", default=None,
                    help="path whose completion must precede the first download")
    ap.add_argument("--wait-mb", type=float, default=0.0)
    ap.add_argument("--wait-hours", type=float, default=8.0)
    args = ap.parse_args()

    dest = Path(args.dest)
    log = Path("results/download_queue.log")
    log.parent.mkdir(exist_ok=True)

    def note(msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        with log.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    if args.wait_for:
        wf = Path(args.wait_for)
        target = args.wait_mb * 1e6
        note(f"waiting for {wf} to reach {args.wait_mb:.0f} MB before starting")
        t0 = time.time()
        while time.time() - t0 < args.wait_hours * 3600:
            cur = wf.stat().st_size if wf.exists() else 0
            if cur >= target * 0.98:
                note("prerequisite complete, starting queue")
                time.sleep(20)
                break
            time.sleep(60)
        else:
            note("timed out waiting; starting queue anyway")

    for mid in args.models:
        try:
            total = expected_bytes(mid)
        except Exception as e:
            note(f"[{mid}] size probe failed: {type(e).__name__}; starting anyway")
            total = 0
        if total and already_done(dest, mid, total):
            note(f"[{mid}] already complete ({total/1e9:.2f} GB), skip")
            continue
        note(f"[{mid}] start, {total/1e9:.2f} GB expected")
        t0 = time.time()
        rc = subprocess.run([sys.executable, "tools/ms_download.py", mid,
                             "--dest", str(dest / mid.split('/')[-1]),
                             "--only-model"]).returncode
        dt = (time.time() - t0) / 3600
        note(f"[{mid}] rc={rc} in {dt:.2f} h")
        if rc != 0:
            note(f"[{mid}] FAILED, stopping queue")
            return 1
    note("queue complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
