"""
Generate refs.bib from lit/corpus.csv by pulling authoritative metadata from the
arXiv API (authors, date, journal reference).  Run on the Jetson: its network
reaches export.arxiv.org, the laptop's does not.

Rows whose URL is not arXiv are emitted from the corpus record alone, with a
TODO marker so they cannot silently pass as verified.

Usage:
    python3 make_bib.py corpus.csv refs.bib
"""
from __future__ import annotations

import csv
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET

ARXIV_ID = re.compile(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})")
ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"
UA = {"User-Agent": "bibgen/1.0"}

# Stable, readable citation keys for the papers the manuscript actually cites.
# Anything not listed gets an auto key (first author surname + year).
CURATED = {
    "QuaRot": "quarot",
    "SpinQuant": "spinquant",
    "DuQuant:": "duquant",
    "PrefixQuant": "prefixquant",
    "OstQuant": "ostquant",
    "SpectrumQuant": "spectrumquant",
    "GPTQ:": "gptq",
    "GPTAQ": "gptaq",
    "HAWQ-V2": "hawqv2",
    "MixLLM": "mixllm",
    "GAMMA": "gamma",
    "AlignQuant": "alignquant",
    "KIVI": "kivi",
    "KVQuant": "kvquant",
    "MixKVQ": "mixkvq",
    "KV Pareto": "kvpareto",
    "Disaggregated Quantization": "dq",
    "Mix-Quant": "mixquant",
    "Quantize What Counts": "kvcache-quant-what-counts",
    "AWQ": "awq",
    "SmoothQuant": "smoothquant",
    "LLM.int8": "llmint8",
    "RULER": "ruler",
    "InfiniteBench": "infinitebench",
}

STOP = {"a", "an", "the", "on", "for", "of", "towards", "toward", "via", "with"}


def key_for(title: str, first_author: str, year: str) -> str:
    for frag, k in CURATED.items():
        if frag.lower() in title.lower():
            return k
    surname = re.sub(r"[^A-Za-z]", "", (first_author.split()[-1] if first_author else "anon"))
    return f"{surname.lower()}{year}"


def fetch(ids: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for i in range(0, len(ids), 40):
        chunk = ids[i:i + 40]
        url = ("https://export.arxiv.org/api/query?id_list=" + ",".join(chunk)
               + f"&max_results={len(chunk) + 5}")
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
            root = ET.fromstring(r.read())
        for e in root.findall(f"{ATOM}entry"):
            eid = e.find(f"{ATOM}id")
            if eid is None:
                continue
            m = ARXIV_ID.search(eid.text or "")
            if not m:
                continue
            authors = [a.find(f"{ATOM}name").text for a in e.findall(f"{ATOM}author")
                       if a.find(f"{ATOM}name") is not None]
            jr = e.find(f"{ARXIV_NS}journal_ref")
            out[m.group(1)] = {
                "title": " ".join((e.find(f"{ATOM}title").text or "").split()),
                "authors": authors,
                "published": (e.find(f"{ATOM}published").text or "")[:10],
                "journal_ref": (jr.text.strip() if jr is not None else ""),
            }
        print(f"  batch {i//40+1}: {len(chunk)} ids", flush=True)
        time.sleep(3)
    return out


def esc(s: str) -> str:
    return s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def main() -> int:
    src = sys.argv[1] if len(sys.argv) > 1 else "corpus.csv"
    dst = sys.argv[2] if len(sys.argv) > 2 else "refs.bib"
    rows = list(csv.DictReader(open(src, encoding="utf-8")))

    ids, by_id = [], {}
    for r in rows:
        m = ARXIV_ID.search(r.get("url", ""))
        if m:
            if m.group(1) not in by_id:
                ids.append(m.group(1))
            by_id[m.group(1)] = r
    print(f"{len(rows)} rows, {len(ids)} unique arXiv ids", flush=True)
    meta = fetch(ids)

    entries, used, missing = [], {}, []
    for aid in ids:
        r, m = by_id[aid], meta.get(aid)
        if not m:
            missing.append(aid)
            continue
        year = m["published"][:4]
        key = key_for(r["title"], m["authors"][0] if m["authors"] else "", year)
        if key in used:
            key = f"{key}{chr(ord('a') + used[key])}"
            used[key] = 0
        used[key] = used.get(key, 0) + 1
        authors = " and ".join(m["authors"]) if m["authors"] else "Anonymous"
        note = m["journal_ref"] or "arXiv preprint"
        entries.append(
            f"@misc{{{key},\n"
            f"  title        = {{{esc(m['title'])}}},\n"
            f"  author       = {{{esc(authors)}}},\n"
            f"  year         = {{{year}}},\n"
            f"  eprint       = {{{aid}}},\n"
            f"  archivePrefix= {{arXiv}},\n"
            f"  note         = {{{esc(note)}}},\n"
            f"  url          = {{https://arxiv.org/abs/{aid}}}\n"
            f"}}\n")

    for r in rows:
        if ARXIV_ID.search(r.get("url", "")):
            continue
        year = r.get("year") or "n.d."
        entries.append(
            f"@misc{{{re.sub(r'[^a-z0-9]', '', r['id'].lower())},\n"
            f"  title  = {{{esc(r['title'])}}},\n"
            f"  year   = {{{year}}},\n"
            f"  note   = {{TODO verify venue and authors: {esc(r.get('venue',''))}}},\n"
            f"  url    = {{{r['url']}}}\n"
            f"}}\n")

    with open(dst, "w", encoding="utf-8") as f:
        f.write("\n".join(entries))
    print(f"\nwrote {dst}: {len(entries)} entries, {len(missing)} arxiv ids unresolved")
    if missing:
        print("unresolved:", missing)
    keys = sorted(used)
    print("keys:", " ".join(keys[:40]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
