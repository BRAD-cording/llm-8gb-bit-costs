"""Single source of truth for lit/corpus.csv (A1-A8).

A1-A4 come from the repaired sweep capture (build/../sweep.json, produced by
parse_sweep.py) plus the hand coding in CODING. A5-A8 come from
recovered_a5_a8.py (this session's incremental web recovery).
"""
import csv
import json
import pathlib
import re

from recovered_a5_a8 import A5, A6, A7, A8

BUILD = pathlib.Path(__file__).resolve().parent
LIT = BUILD.parent
HEADER = ["id", "title", "year", "venue", "url", "subarea", "method_one_line",
          "quant_target", "hw_evaluated", "regime_aware", "budget_aware", "gap_note"]
MAX_GAP = 180

# url -> (quant_target, hw_evaluated, regime_aware, budget_aware)
# A1-A4 codings, inferred from each record's method+gap text only.
CODING = {
    "https://arxiv.org/abs/2404.00456":  ("W+A+KV", "unspecified", "no", "no"),
    "https://arxiv.org/abs/2405.16406v4": ("W+A+KV", "GPU-unspecified", "no", "no"),
    "https://arxiv.org/abs/2406.01721v3": ("W+A", "unspecified", "no", "no"),
    "https://arxiv.org/abs/2410.09426v4": ("W+A", "unspecified", "no", "no"),
    "https://arxiv.org/abs/2410.05265v2": ("W+A+KV", "unspecified", "no", "no"),
    "https://arxiv.org/abs/2501.13987v1": ("W+A+KV", "unspecified", "no", "no"),
    "https://arxiv.org/abs/2501.16383v2": ("KV", "unspecified", "no", "no"),
    "https://arxiv.org/abs/2604.18556v2": ("W", "unspecified", "no", "no"),
    "https://arxiv.org/abs/2604.17789v2": ("W+A", "unspecified", "no", "no"),
    "https://arxiv.org/abs/2608.25188v1": ("unknown", "unspecified", "no", "no"),
    "https://arxiv.org/abs/2610.20303v1": ("unknown", "unspecified", "no", "no"),
    "https://arxiv.org/abs/2610.25675v1": ("W+A+KV", "GPU-unspecified", "no", "no"),
    "https://arxiv.org/abs/2605.29843v2": ("unknown", "unspecified", "no", "no"),
    "https://arxiv.org/abs/2606.02823v1": ("W+A+KV", "unspecified", "no", "no"),
    "https://arxiv.org/abs/2605.17757v1": ("KV", "GPU-unspecified", "no", "no"),
    "https://arxiv.org/abs/2605.20295v1": ("W+A", "mobile-NPU", "no", "no"),
    "https://arxiv.org/abs/2607.27694v1": ("unknown", "custom-ASIC", "no", "no"),
    "https://arxiv.org/abs/2605.26339v1": ("W", "unspecified", "no", "no"),
    "https://arxiv.org/abs/2210.17323":  ("W", "A100", "no", "no"),
    "https://arxiv.org/abs/2208.11580":  ("W", "unspecified", "no", "yes"),
    "https://arxiv.org/abs/2303.08302":  ("W+A", "GPU-unspecified", "no", "no"),
    "https://arxiv.org/abs/2407.11062":  ("W", "A100", "no", "no"),
    "https://arxiv.org/abs/2504.02692":  ("W", "unspecified", "no", "no"),
    "https://arxiv.org/abs/2412.01129":  ("W", "unspecified", "no", "no"),
    "https://arxiv.org/abs/2507.11017":  ("W+A", "GPU-unspecified", "no", "no"),
    "https://arxiv.org/abs/2605.11222":  ("W", "GPU-unspecified", "no", "no"),
    "https://arxiv.org/abs/2607.07964":  ("W", "unspecified", "no", "yes"),
    "https://arxiv.org/abs/2604.07955":  ("W", "unspecified", "no", "no"),
    "https://arxiv.org/abs/2603.08185":  ("W+A", "GPU-unspecified", "no", "no"),
    "https://arxiv.org/abs/2610.08164":  ("W", "unspecified", "no", "no"),
    "https://arxiv.org/abs/2606.01412":  ("W", "unspecified", "no", "no"),
    "https://arxiv.org/abs/2610.11315":  ("W+A", "GPU-unspecified", "no", "no"),
    "https://arxiv.org/abs/2605.18475":  ("W", "unspecified", "no", "yes"),
    "https://arxiv.org/abs/2607.23047":  ("W", "unspecified", "no", "yes"),
    "https://zenodo.org/records/21190586": ("W", "M1-Pro", "no", "yes"),
    "https://arxiv.org/abs/2402.02750":  ("KV", "GPU-unspecified", "no", "no"),
    "https://arxiv.org/abs/2401.18079":  ("KV", "A100", "no", "no"),
    "https://arxiv.org/abs/2405.14256":  ("KV", "A100", "no", "no"),
    "https://arxiv.org/abs/2405.06219":  ("KV", "GPU-unspecified", "no", "no"),
    "https://arxiv.org/abs/2403.05527":  ("KV", "GPU-unspecified", "no", "no"),
    "https://arxiv.org/abs/2502.04420":  ("KV", "unspecified", "no", "no"),
    "https://arxiv.org/abs/2503.23294":  ("KV", "GPU-unspecified", "no", "no"),
    "https://arxiv.org/abs/2510.05373":  ("KV", "unspecified", "no", "no"),
    "https://aclanthology.org/2026.eacl-industry.9/": ("W+KV", "unspecified", "no", "no"),
    "https://aclanthology.org/2026.acl-long.326/":     ("KV", "unspecified", "no", "no"),
    "https://arxiv.org/abs/2604.04722":  ("KV", "unspecified", "no", "no"),
    "https://arxiv.org/abs/2609.26333":  ("W+A", "unspecified", "yes", "no"),
    "https://arxiv.org/abs/2610.07457":  ("W", "GPU-unspecified", "yes", "yes"),
    "https://arxiv.org/abs/2605.20315":  ("W+A", "GPU-unspecified", "yes", "no"),
    "https://arxiv.org/abs/2605.17170":  ("KV", "GPU-unspecified", "no", "yes"),
    "https://arxiv.org/abs/2606.03458":  ("KV", "unspecified", "no", "no"),
    "https://arxiv.org/abs/2610.14475":  ("KV", "RaspberryPi5", "no", "no"),
    "https://arxiv.org/abs/1811.08886":  ("W+A", "simulator", "no", "no"),
    "https://arxiv.org/abs/1911.03852":  ("W+A", "unspecified", "no", "no"),
    "https://arxiv.org/abs/2011.10680":  ("W+A", "T4", "no", "yes"),
    "https://arxiv.org/abs/2412.14590":  ("W", "GPU-unspecified", "no", "yes"),
}

AMQ = {
    "title": "AMQ: Enabling AutoML for Mixed-precision Weight-Only Quantization of Large Language Models",
    "year": "2025", "venue": "EMNLP 2025",
    "url": "https://aclanthology.org/2025.emnlp-main.1799/",
    "method": "AutoML search over layer-wise weight bitwidths with search-space pruning, a "
              "format-conversion-free quantization proxy, and a learned quality predictor.",
    "gap": "Weight-only and layer-granular; the constraint is a memory limit but the objective "
           "never touches KV ca",
}
AMQ_CODE = ("W", "unspecified", "no", "yes")


def gap_note(text: str, partial: bool = False) -> str:
    text = " ".join(text.split())
    if partial:
        text += " [gap string truncated in capture]"
    if len(text) <= MAX_GAP:
        return text
    return text[: MAX_GAP - 3].rsplit(" ", 1)[0] + "..."


def _row(p, tag, code, partial=False):
    qt, hw, ra, ba = code
    return {
        "title": p["title"], "year": str(p["year"]), "venue": p["venue"],
        "url": p["url"].strip(), "subarea": tag,
        "method_one_line": " ".join(p["method"].split()),
        "quant_target": qt, "hw_evaluated": hw, "regime_aware": ra,
        "budget_aware": ba, "gap_note": gap_note(p["gap"], partial),
    }


def build_rows():
    sweep = json.loads((BUILD / "sweep.json").read_text(encoding="utf-8"))
    rows, seen = [], {}
    for ai, area in enumerate(sweep["areas"], 1):
        tag = f"A{ai}"
        for p in area["papers"]:
            url = p["url"].strip()
            if url in seen:
                idx = seen[url]
                have = rows[idx]["subarea"].split("|")
                if tag not in have:
                    rows[idx]["subarea"] = "|".join(have + [tag])
                continue
            seen[url] = len(rows)
            rows.append(_row(p, tag, CODING[url]))
    rows.append(_row(AMQ, "A4", AMQ_CODE, partial=True))   # 5th A4 paper, captured partially
    for tag, block in (("A5", A5), ("A6", A6), ("A7", A7), ("A8", A8)):
        for p in block["papers"]:
            url = p["url"].strip()
            if url in seen:                    # dedup by URL, join tags
                idx = seen[url]
                have = rows[idx]["subarea"].split("|")
                if tag not in have:
                    rows[idx]["subarea"] = "|".join(have + [tag])
                continue
            seen[url] = len(rows)
            rows.append(_row(p, tag, (p["qt"], p["hw"], p["ra"], p["ba"])))
    for i, r in enumerate(rows, 1):
        r["id"] = f"P{i:03d}"
    return rows


def write_csv(rows):
    out = LIT / "corpus.csv"
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=HEADER, quoting=csv.QUOTE_MINIMAL,
                           lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in HEADER})
    return out


def stats(rows):
    s = {"n_rows": len(rows), "by_subarea": {}, "by_hw": {}, "by_regime": {},
         "by_budget": {}, "cross": {}, "quant_targets": {}, "unverified": [],
         "y2026": [], "both": [], "unknown_year": [], "sources": {},
         "y2526": {}, "dup_titles": []}
    for r in rows:
        for key, col in (("by_subarea", "subarea"), ("by_hw", "hw_evaluated"),
                         ("by_regime", "regime_aware"), ("by_budget", "budget_aware"),
                         ("quant_targets", "quant_target")):
            for part in r[col].split("|"):
                s[key][part] = s[key].get(part, 0) + 1
        s["cross"][f'{r["regime_aware"]}/{r["budget_aware"]}'] = \
            s["cross"].get(f'{r["regime_aware"]}/{r["budget_aware"]}', 0) + 1
        if "unverified" in r["venue"].lower():
            s["unverified"].append(r["id"])
        if r["year"] == "2026":
            s["y2026"].append(r["id"])
        if r["year"] == "unknown":
            s["unknown_year"].append(r["id"])
        if "unverified" in r["venue"].lower() and r["year"] == "2026":
            s["both"].append(r["id"])
        for tag in r["subarea"].split("|"):
            s["sources"][tag] = "sweep" if tag in {"A1", "A2", "A3", "A4"} else "recovered"
            if r["year"] in {"2025", "2026"}:
                s["y2526"][tag] = s["y2526"].get(tag, 0) + 1
    low = {}
    for r in rows:
        low.setdefault(r["title"].lower()[:60], []).append((r["id"], r["url"]))
    s["dup_titles"] = [v for v in low.values() if len({u for _, u in v}) > 1]
    s["decisive"] = [r["id"] for r in rows if r["regime_aware"] == "yes"
                     and re.search(r"4060|Jetson", r["hw_evaluated"], re.I)]
    s["regime_yes"] = [r["id"] for r in rows if r["regime_aware"] == "yes"]
    s["budget_yes"] = [r["id"] for r in rows if r["budget_aware"] == "yes"]
    return s


def area_meta():
    sweep = json.loads((BUILD / "sweep.json").read_text(encoding="utf-8"))
    meta = []
    for ai, area in enumerate(sweep["areas"], 1):
        meta.append({"tag": f"A{ai}", "title": area["area"], "saturation": area["saturation"],
                     "area_gaps": area.get("area_gaps", []),
                     "open_questions": area.get("open_questions", []),
                     "source": "sweep"})
    for tag, block in (("A5", A5), ("A6", A6), ("A7", A7), ("A8", A8)):
        meta.append({"tag": tag, "title": block["area"], "saturation": block["saturation"],
                     "area_gaps": block["area_gaps"],
                     "open_questions": block["open_questions"], "source": "recovered"})
    return meta


if __name__ == "__main__":
    rows = build_rows()
    write_csv(rows)
    payload = {"stats": stats(rows), "areas": area_meta(),
               "ids": {r["id"]: r["title"] for r in rows}}
    (BUILD / "corpus_meta.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"rows={len(rows)}")
    print(json.dumps(payload["stats"], indent=1, ensure_ascii=False))
