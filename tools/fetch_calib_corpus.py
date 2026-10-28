"""
Build a calibration corpus from arXiv abstracts.

Why: the allocation solver needs a per-layer sensitivity curve, and that curve
only has to be right up to a multiplicative constant -- the error budget epsilon
is a RELATIVE perplexity increase, which is scale-free.  A corpus of 2,000
abstracts therefore supports the relative ranking even though it cannot support
an absolute perplexity number comparable to the literature.

This is a documented fallback.  HuggingFace, GitHub and jsDelivr are all
unreachable from this network, so WikiText-2 cannot be obtained; when it can be,
`src/eval/ppl.py` takes it unchanged and the numbers become comparable.

Run on the Jetson (its network reaches export.arxiv.org):
    python3 fetch_calib_corpus.py calib.txt --papers 2000
"""
from __future__ import annotations

import argparse
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET

ATOM = "{http://www.w3.org/2005/Atom}"
UA = {"User-Agent": "calib-corpus/1.0"}
CATS = ("cs.CL", "cs.LG")


def fetch_page(cat: str, start: int, n: int) -> list[str]:
    url = (f"https://export.arxiv.org/api/query?search_query=cat:{cat}"
           f"&start={start}&max_results={n}&sortBy=submittedDate&sortOrder=descending")
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90) as r:
        root = ET.fromstring(r.read())
    out = []
    for e in root.findall(f"{ATOM}entry"):
        s = e.find(f"{ATOM}summary")
        t = e.find(f"{ATOM}title")
        if s is None:
            continue
        body = " ".join((s.text or "").split())
        head = " ".join((t.text or "").split()) if t is not None else ""
        if body:
            out.append(f"{head}. {body}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", default="calib.txt", nargs="?")
    ap.add_argument("--papers", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=100)
    args = ap.parse_args()

    chunks: list[str] = []
    per_cat = max(1, args.papers // len(CATS))
    for cat in CATS:
        got = 0
        while got < per_cat:
            n = min(args.batch, per_cat - got)
            try:
                page = fetch_page(cat, got, n)
            except Exception as e:
                print(f"  [{cat}] fetch failed at {got}: {type(e).__name__}", file=sys.stderr)
                time.sleep(5)
                continue
            if not page:
                break
            chunks.extend(page)
            got += len(page)
            print(f"  [{cat}] {got}/{per_cat}", flush=True)
            time.sleep(3)

    text = "\n\n".join(chunks)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    words = len(text.split())
    print(f"\nwrote {args.out}: {len(chunks)} abstracts, {words:,} words, "
          f"{len(text)/1e6:.2f} MB, ~{int(words*1.3):,} tokens")
    return 0


if __name__ == "__main__":
    sys.exit(main())
