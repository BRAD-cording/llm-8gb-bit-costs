"""
Resumable single-connection downloader for ModelScope repos.

Why this exists: the egress here is throttled to ~0.23 MB/s and *resets* parallel
connections (ConnectionResetError 10054).  ModelScope's own downloader stalls
indefinitely after a reset.  So we do the only thing that works on this link:
ONE connection, Resume-from-offset, aggressive retry with backoff, and a
size check per file.

Usage:
    python tools/ms_download.py Qwen/Qwen3-0.6B --dest C:/models/Qwen3-0.6B
    python tools/ms_download.py Qwen/Qwen3-4B   --dest C:/models/Qwen3-4B --only-model
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

API_FILES = "https://modelscope.cn/api/v1/{kind}/{mid}/repo/files?Revision={rev}&Recursive=true"
API_TREE = "https://modelscope.cn/api/v1/{kind}/{mid}/repo/tree?Revision={rev}&Root={root}"
API_BLOB = "https://modelscope.cn/api/v1/{kind}/{mid}/repo?Revision={rev}&FilePath={path}"
UA = {"User-Agent": "ms-resume/1.0"}

KIND = "models"  # overridden by --kind


def _tree(mid: str, root: str, rev: str) -> list[dict]:
    url = API_TREE.format(kind=KIND, mid=mid, rev=rev, root=urllib.parse.quote(root))
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))["Data"]["Files"]


def list_files(mid: str, rev: str = "master") -> list[dict]:
    """List blobs. Datasets use the tree API and must be walked recursively."""
    if KIND == "datasets":
        out: list[dict] = []
        stack = ["/"]
        seen = set()
        while stack:
            root = stack.pop()
            if root in seen:
                continue
            seen.add(root)
            for f in _tree(mid, root, rev):
                if f.get("Type") == "tree":
                    stack.append("/" + f["Path"].strip("/"))
                elif f.get("Type") == "blob":
                    out.append(f)
        return out

    url = API_FILES.format(kind=KIND, mid=mid, rev=rev)
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        payload = json.loads(r.read().decode("utf-8"))
    return [f for f in payload["Data"]["Files"] if f.get("Type") == "blob"]


def download(mid: str, path: str, size: int, dest: str, rev: str = "master",
             max_retries: int = 200) -> bool:
    out = os.path.join(dest, path)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out) and os.path.getsize(out) == size:
        print(f"  [skip] {path} ({size/1e6:.1f} MB already complete)")
        return True

    url = API_BLOB.format(kind=KIND, mid=mid, rev=rev, path=urllib.parse.quote(path))
    attempts = 0
    stall = 0
    while True:
        have = os.path.getsize(out) if os.path.exists(out) else 0
        if have >= size:
            break
        headers = dict(UA)
        headers["Range"] = f"bytes={have}-"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as r:
                mode = "ab" if have else "wb"
                with open(out, mode) as f:
                    last = time.time()
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        f.flush()
                        if time.time() - last > 60:
                            stall += 1
                            raise TimeoutError("stalled")
        except Exception as e:
            attempts += 1
            if attempts > max_retries:
                print(f"  [FAIL] {path}: {type(e).__name__} after {attempts} attempts",
                      file=sys.stderr)
                return False
            backoff = min(30, 2 + attempts // 5)
            now = os.path.getsize(out) if os.path.exists(out) else 0
            print(f"  [retry {attempts}] {path} {now/1e6:.1f}/{size/1e6:.1f} MB "
                  f"({type(e).__name__}), sleep {backoff}s", flush=True)
            time.sleep(backoff)
        else:
            attempts = 0

    final = os.path.getsize(out) if os.path.exists(out) else 0
    if final == size:
        print(f"  [ok] {path} ({size/1e6:.1f} MB)")
        return True
    print(f"  [MISMATCH] {path}: {final} != {size}", file=sys.stderr)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_id")
    ap.add_argument("--dest", required=True)
    ap.add_argument("--rev", default="master")
    ap.add_argument("--only-model", action="store_true",
                    help="skip anything that is not config/tokenizer/weights")
    ap.add_argument("--kind", default="models", choices=["models", "datasets"])
    args = ap.parse_args()

    global KIND
    KIND = args.kind

    files = list_files(args.model_id, args.rev)
    total = sum(f["Size"] for f in files)
    print(f"{args.model_id}: {len(files)} files, {total/1e9:.2f} GB total", flush=True)

    keep_default = {"config.json", "generation_config.json", "tokenizer.json",
                    "tokenizer_config.json", "vocab.json", "merges.txt",
                    "special_tokens_map.json", "model.safetensors",
                    "model.safetensors.index.json", "chat_template.jinja"}
    ok = True
    for f in files:
        p, s = f["Path"], f["Size"]
        if args.only_model and p not in keep_default and not p.startswith("model-"):
            continue
        if not download(args.model_id, p, s, args.dest, args.rev):
            ok = False
            break
    print("DONE" if ok else "INCOMPLETE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
