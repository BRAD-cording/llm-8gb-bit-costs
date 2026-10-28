"""
Static file server with HTTP Range support.

`python -m http.server` ignores Range requests, so a resumed `curl -C -` silently
restarts the download. Serving 3.2 GB again to save a 0.5 GB gap is wasteful on a
link that runs at about 1.5 MB/s, hence this.

Usage:
    python tools/serve_range.py --root C:/models/Qwen3-1.7B --port 8779
"""
from __future__ import annotations

import argparse
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote


class Handler(BaseHTTPRequestHandler):
    root = "."

    def _resolve(self) -> str | None:
        rel = unquote(self.path.split("?", 1)[0]).lstrip("/")
        full = os.path.normpath(os.path.join(self.root, rel))
        if not full.startswith(os.path.abspath(self.root)):
            return None
        return full if os.path.isfile(full) else None

    def do_HEAD(self) -> None:  # noqa: N802
        full = self._resolve()
        if not full:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(os.path.getsize(full)))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        full = self._resolve()
        if not full:
            self.send_error(404)
            return
        size = os.path.getsize(full)
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        partial = False
        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)", rng)
            if m:
                if m.group(1):
                    start = int(m.group(1))
                    if m.group(2):
                        end = min(int(m.group(2)), size - 1)
                elif m.group(2):
                    start = max(0, size - int(m.group(2)))
                partial = True
        length = end - start + 1
        self.send_response(206 if partial else 200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with open(full, "rb") as fh:
            fh.seek(start)
            left = length
            while left > 0:
                chunk = fh.read(min(1 << 20, left))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                left -= len(chunk)

    def log_message(self, fmt, *args):  # quieter than the default
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--port", type=int, default=8779)
    args = ap.parse_args()
    Handler.root = os.path.abspath(args.root)
    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"serving {Handler.root} on :{args.port}", flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
