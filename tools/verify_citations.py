"""
Systematic citation audit.

For every row in lit/corpus.csv:
  * if the URL contains an arXiv id, ask the arXiv API what that id's title IS,
    and compare it with the title we recorded;
  * otherwise, check whether the URL resolves (HTTP status).

Rationale: a single fabricated or mismatched reference is enough for a desk
rejection, and 55/108 rows in this corpus came back with `unverified` venues.
We do not guess which are right -- we ask the source.

Runs on the Jetson (its network reaches export.arxiv.org; the laptop's does not).
Stdlib only, no pip install needed.

Usage:
    python3 verify_citations.py corpus.csv report.csv
"""
from __future__ import annotations

import csv
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

ARXIV_ID = re.compile(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})(v\d+)?")
UA = {"User-Agent": "citation-audit/1.0 (research reproducibility check)"}
ATOM = "{http://www.w3.org/2005/Atom}"


def norm(t: str) -> str:
    t = t.lower()
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def similarity(a: str, b: str) -> float:
    """Token-set overlap -- robust to word order and minor edits."""
    sa, sb = set(norm(a).split()), set(norm(b).split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def arxiv_titles(ids: list[str]) -> dict[str, str]:
    q = ",".join(ids)
    url = f"https://export.arxiv.org/api/query?id_list={q}&max_results={len(ids) + 5}"
    out: dict[str, str] = {}
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                    timeout=60) as r:
            xml = r.read()
    except Exception as e:
        print(f"  [api error] {type(e).__name__}: {str(e)[:70]}", file=sys.stderr)
        return out
    root = ET.fromstring(xml)
    for entry in root.findall(f"{ATOM}entry"):
        eid = entry.find(f"{ATOM}id")
        ttl = entry.find(f"{ATOM}title")
        if eid is None or ttl is None:
            continue
        m = ARXIV_ID.search(eid.text or "")
        if m:
            out[m.group(1)] = " ".join((ttl.text or "").split())
    return out


def url_alive(url: str) -> str:
    try:
        req = urllib.request.Request(url, headers=UA, method="HEAD")
        with urllib.request.urlopen(req, timeout=25) as r:
            return f"http_{r.status}"
    except urllib.error.HTTPError as e:
        return f"http_{e.code}"
    except Exception as e:
        return f"err_{type(e).__name__}"


def main() -> int:
    src = sys.argv[1] if len(sys.argv) > 1 else "corpus.csv"
    dst = sys.argv[2] if len(sys.argv) > 2 else "citation_report.csv"

    rows = list(csv.DictReader(open(src, encoding="utf-8")))
    print(f"{len(rows)} rows", flush=True)

    by_arxiv: dict[str, list[dict]] = {}
    others: list[dict] = []
    for r in rows:
        m = ARXIV_ID.search(r.get("url", ""))
        (by_arxiv.setdefault(m.group(1), []).append(r) if m else others.append(r))

    api: dict[str, str] = {}
    ids = list(by_arxiv)
    for i in range(0, len(ids), 40):
        chunk = ids[i:i + 40]
        print(f"  arxiv batch {i//40 + 1}: {len(chunk)} ids", flush=True)
        api.update(arxiv_titles(chunk))
        time.sleep(3)

    out = []
    for aid, rs in by_arxiv.items():
        api_title = api.get(aid, "")
        for r in rs:
            if not api_title:
                status, sim = "NOT_FOUND_ON_ARXIV", 0.0
            else:
                sim = similarity(r["title"], api_title)
                status = "OK" if sim >= 0.6 else "TITLE_MISMATCH"
            out.append({"id": r["id"], "arxiv_id": aid, "status": status,
                        "similarity": round(sim, 3),
                        "corpus_title": r["title"], "api_title": api_title,
                        "url": r["url"]})

    for r in others:
        out.append({"id": r["id"], "arxiv_id": "", "status": url_alive(r["url"]),
                    "similarity": "", "corpus_title": r["title"],
                    "api_title": "", "url": r["url"]})

    with open(dst, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    from collections import Counter
    c = Counter(o["status"] for o in out)
    print("\n=== summary ===")
    for k, v in c.most_common():
        print(f"  {k:24s} {v}")
    bad = [o for o in out if o["status"] in ("TITLE_MISMATCH", "NOT_FOUND_ON_ARXIV")]
    print(f"\n=== {len(bad)} rows need attention ===")
    for o in bad[:15]:
        print(f"  {o['id']} {o['arxiv_id'] or o['url'][:50]} sim={o['similarity']}")
        print(f"     corpus: {o['corpus_title'][:90]}")
        print(f"     arxiv : {o['api_title'][:90]}")
    print(f"\nwrote {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
