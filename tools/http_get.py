"""
Generic resumable HTTP downloader.

The links in this environment reset often enough that every long download needs
resume + retry.  This is the third place we needed it (ModelScope weights,
a CUDA wheel, and now a TeX distribution), so it lives on its own.

Usage:
    python tools/http_get.py URL DEST [--expect-bytes N]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.error
import urllib.request

UA = {"User-Agent": "resume-get/1.0"}


def fetch(url: str, dest: str, expect: int | None = None,
          max_retries: int = 300) -> bool:
    os.makedirs(os.path.dirname(os.path.abspath(dest)) or ".", exist_ok=True)
    attempts = 0
    while True:
        have = os.path.getsize(dest) if os.path.exists(dest) else 0
        if expect and have >= expect:
            break
        headers = dict(UA)
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as r:
                if not expect:
                    expect = have + int(r.headers.get("Content-Length") or 0)
                mode = "ab" if have else "wb"
                with open(dest, mode) as f:
                    last = time.time()
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        f.flush()
                        if time.time() - last > 60:
                            raise TimeoutError("stalled")
                        last = time.time()
        except Exception as e:
            attempts += 1
            if attempts > max_retries:
                print(f"[FAIL] {type(e).__name__} after {attempts}", file=sys.stderr)
                return False
            now = os.path.getsize(dest) if os.path.exists(dest) else 0
            wait = min(30, 2 + attempts // 8)
            pct = f"{100*now/expect:.1f}%" if expect else "?"
            print(f"  [retry {attempts}] {now/1e6:.1f} MB ({pct}) "
                  f"{type(e).__name__}, sleep {wait}s", flush=True)
            time.sleep(wait)
        else:
            attempts = 0

    size = os.path.getsize(dest)
    if expect and size != expect:
        print(f"[MISMATCH] {size} != {expect}", file=sys.stderr)
        return False
    print(f"[ok] {dest} ({size/1e6:.1f} MB)")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("dest")
    ap.add_argument("--expect-bytes", type=int, default=None)
    args = ap.parse_args()
    return 0 if fetch(args.url, args.dest, args.expect_bytes) else 1


if __name__ == "__main__":
    sys.exit(main())
