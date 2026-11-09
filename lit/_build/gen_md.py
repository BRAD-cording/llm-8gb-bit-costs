"""Generate lit/GAP_MATRIX.md from corpus_core data (so tables cannot drift from the CSV)."""
import json
import pathlib

import corpus_core as core

BUILD = pathlib.Path(__file__).resolve().parent
LIT = BUILD.parent
rows = core.build_rows()
meta = core.area_meta()
st = core.stats(rows)
by_id = {r["id"]: r for r in rows}


def pid(needle: str) -> str:
    """Return the P-id of the first paper whose title contains `needle`."""
    hits = [r["id"] for r in rows if needle.lower() in r["title"].lower()]
    if not hits:
        raise SystemExit(f"no paper matching {needle!r}")
    return hits[0]


def pids(*needles) -> str:
    return ", ".join(pid(n) for n in needles)


def sort_desc(d):
    return sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))


L = []
w = L.append

w("# Gap Matrix")
w("")
w("**Scope.** This matrix is the evidence base for the thesis that, on 8 GB-class edge devices, "
  "the optimal split of a single memory budget between weight bits and KV-cache bits is not a "
  "static property but reverses with the prefill/decode regime. Every row is a paper retrieved "
  "under the PRISMA-style protocol frozen in [SEARCH_PROTOCOL.md](SEARCH_PROTOCOL.md) "
  "(window 2023-01-01 to 2026-10-20, the six arXiv/OpenAlex/DBLP/IEEE/ACM queries recorded there, "
  "and the inclusion criteria I1-I4). The corpus file is [corpus.csv](corpus.csv); each row's "
  "`subarea`, `quant_target`, `hw_evaluated`, `regime_aware`, `budget_aware` and `gap_note` "
  "columns are derived from the retrieved record, never from the reviewer's expectations. Where "
  "the record is silent the cell says `unknown`/`unspecified` rather than a plausible value. "
  "A1-A4 were captured by the original eight-agent sweep; A5-A8 were re-derived in a separate "
  "recovery pass because the sweep output was truncated in capture (see §7), so the corpus is a "
  "bounded sample, not an exhaustive census, and every claim below is a claim about this corpus.")
w("")

# ---------------------------------------------------------------- 1
w("## 1. Corpus overview")
w("")
w("| subarea | #papers | saturation (as reported) | 2025-2026 paper count | source |")
w("|---|---|---|---|---|")
titles = {m["tag"]: m["title"] for m in meta}
for tag in [f"A{i}" for i in range(1, 9)]:
    n = st["by_subarea"].get(tag, 0)
    sat = next(m["saturation"] for m in meta if m["tag"] == tag)
    rec = st["y2526"].get(tag, 0)
    src = "sweep" if tag in {"A1", "A2", "A3", "A4"} else "recovered"
    w(f"| {tag} | {n} | {sat} | {rec} | {src} |")
w(f"| **total** | **{st['n_rows']}** | — | **{sum(st['y2526'].values())}** | — |")
w("")
w("Notes on the accounting:")
w("")
w("- Saturation for A1-A4 is the value the sweep reported. Saturation for A5-A8 is **our** "
  "assessment at recovery time, not a sweep finding, because those sub-areas were not part of the "
  "captured run.")
w(f"- Three rows carry `year = unknown` ({', '.join(st['unknown_year'])}) because the IEEE/ACM "
  "records resolved to a landing page without extractable publication metadata; they are excluded "
  "from the 2025-2026 column rather than guessed into it.")
w(f"- {len(st['unverified'])} of {st['n_rows']} rows carry `venue = unverified`; §6 lists them and "
  "the rule that applies to them before submission.")
w(f"- `quant_target` is `unknown` for {st['quant_targets'].get('unknown', 0)} rows. That is a "
  "finding about reporting practice, not a coding failure: those records describe an allocation or "
  "a measurement without ever naming which tensor class is quantized.")
w("- Enum limitation: the schema has no `W+A+LR` value, so low-rank-compensation papers whose "
  "quantized tensors are weights+activations (the LoRC/RILQ/SERQ family) are coded `W+A` and their "
  "low-rank path is recorded in `method_one_line`. `W+A+KV+LR` is therefore unused in this corpus: "
  "no retrieved paper combines a KV decision variable with low-rank compensation.")
w("")

# ---------------------------------------------------------------- 2
w("## 2. The decisive cross-tabulation")
w("")
w("### Table A — hardware actually evaluated")
w("")
w("| hw_evaluated | papers |")
w("|---|---|")
for k, v in sort_desc(st["by_hw"]):
    w(f"| `{k}` | {v} |")
w(f"| **total** | **{st['n_rows']}** |")
w("")
w("`unspecified` means the record names no device at all; `GPU-unspecified` means it says "
  "\"datacenter GPUs\" / \"server GPUs\" / \"GPUs\" without a model. The distinction matters: the "
  "first is a reporting omission, the second is a reporting convention that makes the number "
  "non-reproducible on any specific device.")
w("")
w("### Table B — regime awareness")
w("")
w("| regime_aware | papers | share |")
w("|---|---|---|")
for k in ("yes", "no"):
    n = st["by_regime"].get(k, 0)
    w(f"| `{k}` | {n} | {100 * n / st['n_rows']:.1f}% |")
w("")
w("`yes` requires that prefill and decode are separated *inside the optimization or measurement "
  "design*. Reporting two speedup numbers against FP16 is not enough, and on that strict reading "
  "papers that do exactly that — PrefixQuant (P005), FlatQuant (P004), ZipCache (P038) — are `no`.")
w("")
w("### Table C — regime_aware × budget_aware")
w("")
w("| | budget_aware = yes | budget_aware = no |")
w("|---|---|---|")
w(f"| **regime_aware = yes** | {st['cross'].get('yes/yes', 0)} | {st['cross'].get('yes/no', 0)} |")
w(f"| **regime_aware = no** | {st['cross'].get('no/yes', 0)} | {st['cross'].get('no/no', 0)} |")
w("")
w("The single `yes/yes` cell is AlignQuant (" + pid("AlignQuant") + "), and its budget applies to "
  "weight tiles only — the paper's own gap statement says KV precision is never part of the "
  "allocation. The four other regime-aware papers (" + pids("Disaggregated Quantization",
  "Mix-Quant: Quantized", "Characterizing Prefill", "ELLIE") + ") either exclude KV from the "
  "allocation or do not quantize at all.")
w("")
n_dec = len(st["decisive"])
assert n_dec == 0
w("**Decisive count: 0 papers in this corpus are simultaneously `regime_aware = yes` and evaluated "
  "on 4060-class or Jetson-class hardware.** The single Jetson row in the corpus — "
  f"{pid('Offline AI Modules')} — is `regime_aware = no`; there is **no** RTX 4060 row at all.")
w("")
w("What that 0 licenses:")
w("")
w("- It licenses the statement that **no retrieved work reports a regime-conditional bit "
  "allocation measured on either of our two target devices**. The joint cell of the design space is "
  "empty in the literature we can cite, which is exactly the space C1-C3 claim.")
w("- It licenses a *methodological* claim: because nobody has run the experiment on the class of "
  "device we target, the numbers reported for A100/H100 do not transfer by assumption — the "
  "bandwidth ratio between Orin (~68 GB/s unified) and the 4060 (256 GB/s discrete) is 3.8x, which "
  "is larger than the gap between the datacenter parts those papers use.")
w("")
w("What it does **not** license:")
w("")
w("- It does not license \"no such work exists\". The count is bounded by 108 retrieved rows, "
  f"{len(st['unverified'])} of which have unverified venues and one of which ("
  f"{pid('Characterizing Prefill')}) is the closest prior art and has unresolved hardware. An "
  "absence in this corpus is an absence in our retrieval, not in the world.")
w("- It does not license any claim about our own effect. A zero in the prior-work column says "
  "nothing about whether the reversal exists on real silicon; only the W7 pre-registered "
  "two-device experiment can say that, and a null result demotes the thesis to an empirical "
  "budget-allocation study.")
w("- It does not license the weaker claim that edge devices are unquantized. They are quantized "
  "heavily (AWQ, QServe, T-MAC, llama.cpp-style GGUF stacks); what is missing is the *allocation "
  "decision*, not the compression.")
w("")

# ---------------------------------------------------------------- 3
w("## 3. Per-subarea gap statements")
w("")
w("Verbatim from the retrieval records (A1-A4) and from the recovery pass (A5-A8). Wording is "
  "preserved; nothing here is paraphrased.")
w("")
for m in meta:
    tag = m["tag"]
    title = m["title"]
    w(f"### {title}")
    w("")
    w(f"**Saturation:** {m['saturation']} · **papers in corpus:** "
      f"{st['by_subarea'].get(tag, 0)} · **source:** {m['source']}")
    w("")
    if m["area_gaps"]:
        for g in m["area_gaps"]:
            w(f"- {g}")
    else:
        w("- *No `area_gaps` were captured for this sub-area: the sweep output was truncated "
          "immediately after its fifth paper record, before the sub-area summary fields. The paper "
          "rows themselves are complete and are included in the corpus.*")
    w("")
    if m["open_questions"]:
        w("- **Open questions:**")
        for q in m["open_questions"]:
            w(f"  - {q}")
    else:
        w("- **Open questions:** *not captured (same truncation as above).*")
    w("")

# ---------------------------------------------------------------- 4
w("## 4. Cross-cutting gaps (synthesised)")
w("")
w("These are our own synthesis: a gap counts as cross-cutting only when it is asserted "
  "independently in more than one sub-area's own gap list. Strength: **strong** = 3+ independent "
  "sub-areas, **medium** = 2, **weak** = 1 (kept only where the single sub-area is the decisive "
  "one).")
w("")
w("### CC1 — The budget is never one pool shared by weights and KV")
w("")
w("**Evidence:** A1 (gap 1), A2 (gap 2), A3 (gap 4), A5 (gap 5), A8 (gap 1). **Strength: strong "
  "(5 sub-areas).**")
w("")
w("Each literature allocates inside its own tensor class: rotation and Hessian methods fix a "
  "W/A/KV tuple, KV methods fix the weight precision as an external constant, and the 2026 "
  "allocators (" + pids("GAMMA", "MixQuant: Adaptive", "SGSR-2") + ") constrain a static memory "
  "number that has no KV-growth term. The only paper that touches both sides, KV Pareto ("
  + pid("KV Pareto") + "), indexes a frontier by total memory rather than optimizing under a cap, "
  "so it reports a curve instead of a decision.")
w("")
w("### CC2 — Prefill/decode is reported but never enters the objective")
w("")
w("**Evidence:** A1 (gap 2), A2 (gap 2 and gap 5), A3 (gap 1), A6 (gaps 3-4). **Strength: strong "
  "(4 sub-areas).**")
w("")
w("Three papers do specialize precision by phase — Disaggregated Quantization ("
  + pid("Disaggregated") + "), Mix-Quant (" + pid("Mix-Quant: Quantized") + ") and AlignQuant ("
  + pid("AlignQuant") + ") — and all three explicitly exclude the KV cache from the decision. On "
  "the edge side, the two IEEE papers that do split the regimes (" + pids("Characterizing Prefill",
  "ELLIE") + ") do not quantize. So the exact object we claim (a regime-conditional W-vs-KV split) "
  "has no prior instance, and the nearest neighbours are the threat surface, not the support.")
w("")
w("### CC3 — Edge classes are unmeasured, and the evidence base is structurally mismatched")
w("")
w("**Evidence:** A1 (gap 4), A2 (gap 3), A3 (gap 3), A5 (gap 2), A6 (gaps 1-2). **Strength: "
  "strong (5 sub-areas).**")
w("")
w("The mismatch is twofold. Hardware: "
  f"{st['by_hw'].get('unspecified', 0)} of {st['n_rows']} rows do not name a device at all and "
  f"{st['by_hw'].get('GPU-unspecified', 0)} name only "
  "\"GPUs\"; the named parts are "
  f"{st['by_hw'].get('A100', 0)} A100s, one T4, one M1 Pro, one Raspberry Pi 5, one RTX 4090, one "
  "Jetson, and no 4060. Scale: the accuracy evidence that decides bit-widths is collected on "
  "7B-70B checkpoints that cannot inhabit 8 GB at any bit-width, while the 1-4B models that can "
  "are barely evaluated. Both halves of the mismatch have to be closed by our own experiments, "
  "which is why the contribution is measurement, not method.")
w("")
w("### CC4 — Evaluation practice is unreported, phase-mixed and statistically thin")
w("")
w("**Evidence:** A7 (gaps 1-3), A2 (gap 5), A6 (gap 3). **Strength: strong (3 sub-areas).**")
w("")
w("No paper in the corpus reports CUDA/cuDNN/driver versions, kernel commit or clock policy — the "
  "three things the measurement-bias work (" + pid("Identifying and Mitigating") + ") says make "
  "throughput numbers non-comparable. Accuracy is measured prefill-style while speed is measured "
  "decode-style, and the long-context benchmarks (" + pids("RULER", "Bench: Extending", "LongBench v2")
  + ") carry no precision axis. Since the same corpus reports single-run accuracies without seeds "
  "or paired tests, the deltas our ablations must beat are frequently inside benchmark noise.")
w("")
w("### CC5 — Transforms and low-rank corrections are treated as free")
w("")
w("**Evidence:** A1 (gap 3), A2 (gap 4), A5 (gap 4). **Strength: strong (3 sub-areas).**")
w("")
w("Rotation/FWHT passes, permutations, per-token scaling and low-rank side paths all consume "
  "bandwidth or compute that is never charged against the budget that motivated them. On a "
  "bandwidth-bound decoder this is not a second-order effect: if the transform costs one extra "
  "read of the activation per layer, it can eat the saving that halving the weight bits was "
  "supposed to buy. A 1-3B model on a 68 GB/s device is where that accounting error becomes "
  "visible first.")
w("")
w("### CC6 — Hardware-format assumptions do not exist on Ada SM89 / Ampere SM87")
w("")
w("**Evidence:** A1 (gap 6), A3 (gap 6), A5 (gap 2). **Strength: strong (3 sub-areas).**")
w("")
w("MXFP4 and NVFP4 paths assume Blackwell-class tensor cores; LUT-based tensor cores and GyRot's "
  "rotation unit assume custom silicon. Neither the RTX 4060 (Ada) nor Jetson Orin (Ampere) "
  "provides them in the form these kernels require, so a large fraction of the 2026 speedups are "
  "not executable on the devices the thesis targets. That is a coverage gap we exploit, but it is "
  "also a trap: any method we propose must run through dp4a / INT4 MMA paths that actually exist "
  "on both parts.")
w("")
w("### CC7 — Reasoning damage is measured but never charged against the KV budget")
w("")
w("**Evidence:** A8 (gaps 1, 2, 5), A3 (gap 5). **Strength: medium (2 sub-areas).**")
w("")
w("Long-chain reasoning is the workload where quantization damage accumulates, and it is also the "
  "workload that inflates KV footprint the most (longer chains, longer contexts). The reasoning "
  "literature (" + pids("Quantization Hurts", "Quantization Meets Reasoning") + ") measures the "
  "damage at fixed weight bit-widths, and the token-inflation work (" + pid("Token Inflation")
  + ") measures the extra cost, but nobody connects the two into a KV-pressure term inside the "
  "allocation objective.")
w("")

# ---------------------------------------------------------------- 5
w("## 5. Mapping to our thesis")
w("")
w("| our claim | supporting gap rows | threat | mitigation |")
w("|---|---|---|---|")
w(f"| **C1** — no existing work jointly allocates weight bits and KV bits under one device budget "
  f"| CC1; budget-aware but KV-blind: {pids('GAMMA', 'MixQuant: Adaptive', 'SGSR-2', 'KronQ', 'MixLLM')}; "
  f"KV-only: {pids('SAW-INT4', 'AdaKVQ')}; both sides but frontier-only: {pid('KV Pareto')} "
  f"| AlignQuant ({pid('AlignQuant')}) already runs phase-normalized precision selection over "
  f"weight tiles under a model-wide storage budget, and SGSR-2 ({pid('SGSR-2')}) already publishes a "
  f"measured bits-by-group-size Pareto front on 8 GB consumer hardware. A reviewer can read either "
  f"as \"the allocation problem is solved\" "
  f"| Concede the weight-only case and narrow C1 to the *joint* pool: quote each paper's own "
  f"statement that KV precision is outside its allocation (AlignQuant, GAMMA, MixQuant), and show "
  f"the ablation where KV is removed from our solver and the optimum degenerates to their answer. "
  f"The claim survives only as \"joint + KV growth + rho\", so state it that way |")
w(f"| **C2** — no existing work conditions allocation on the prefill/decode regime "
  f"| CC2; regime-aware but KV-blind: {pids('Disaggregated', 'Mix-Quant: Quantized', 'AlignQuant')}; "
  f"regime-aware but non-quantizing: {pids('Characterizing Prefill', 'ELLIE')} "
  f"| Three 2026 papers already exploit the phase asymmetry (prefill low-precision, decode "
  f"high-precision) and one is a dedicated edge prefill/decode characterization. Saying "
  f"\"no one is regime-aware\" is falsified by our own corpus "
  f"| Never claim regime awareness itself. Claim only that *no paper makes the W-vs-KV split a "
  f"function of the regime*, and prove it by the explicit KV exclusion in those three papers. Cite "
  f"them as the closest prior art in Related Work, and make the regime ablation (alpha swept) the "
  f"load-bearing experiment rather than a rhetorical point |")
w(f"| **C3** — the 8GB laptop-GPU and Jetson Orin classes are unmeasured "
  f"| CC3; {pids('Offline AI Modules', 'PowerInfer', 'SGSR-2', 'Sustainable LLM')} "
  f"| {pid('Offline AI Modules')} benchmarks a Jetson Orin tier today; PowerInfer ("
  f"{pid('PowerInfer')}) serves on a consumer GPU; SGSR-2 runs on an M1 Pro; and \"Jetson\" is a "
  f"family, so a reviewer who has run llama.cpp on an Orin Nano will not accept \"unmeasured\" "
  f"as written "
  f"| Restate C3 as *no controlled joint-allocation measurement on either class*. Cite the Jetson "
  f"row as an engineering report with hand-chosen precisions (no allocation, no seeds, no stack "
  f"report), and the 4090 row as 24 GB, not 8 GB. Then make the 4060-vs-Orin measurement itself the "
  f"contribution — that is exactly what the empty cell buys |")
w(f"| **C4** — evaluation practice is a live methodological gap "
  f"| CC4; {pids('ReproEvalCard', 'Identifying and Mitigating', 'Are LLM Benchmarks', 'Accuracy is Not All')}; "
  f"missing hardware reporting in {pids('QuaRot', 'KVQuant')} "
  f"| A reviewer can answer \"this is well known, and there are reporting standards\" — ReproEvalCard "
  f"({pid('ReproEvalCard')}) exists precisely to fix pipeline reporting, and the measurement-bias "
  f"paper ({pid('Identifying and Mitigating')}) already names the warm-up/caching confounds "
  f"| Accept C4 as a *domain-specific* gap rather than a novel one: the standards cover pipelines "
  f"and serving engines, not quantization experiments, and neither mandates memory cap, achieved "
  f"bandwidth, KV precision or phase-separated throughput. Position our artifact (stack snapshot + "
  f"seeds + paired tests) as the instantiation of an existing standard in a domain that has not "
  f"adopted it |")
w("")

# ---------------------------------------------------------------- 6
w("## 6. Verification status")
w("")
w("Every row below either carries `venue = unverified` or has `year = 2026`. **Any entry listed "
  "here must be resolved before it appears in the submitted manuscript.** Resolution means: the "
  "URL returns the paper, the title matches, and the venue/year on the page match the ledger. If a "
  "URL cannot be resolved, the row is **dropped from the manuscript corpus** — it is never "
  "re-guessed, and no venue is inferred from an arXiv identifier.")
w("")
w("| id | title | url | needs_verification |")
w("|---|---|---|---|")
for r in rows:
    is_unv = "unverified" in r["venue"].lower()
    if not (is_unv or r["year"] == "2026"):
        continue
    flags = []
    if is_unv:
        flags.append("venue unverified")
    if r["year"] == "2026":
        flags.append("year 2026 (not yet in a proceedings record we can cite)")
    if r["year"] == "unknown":
        flags.append("year unresolved")
    w(f"| {r['id']} | {r['title'].replace('|', chr(92) + '|')} | {r['url']} | "
      f"{'; '.join(flags)} |")
w("")
w("Existence is a separate check from venue. Fifteen rows had their arXiv metadata re-queried "
  "directly through `export.arxiv.org/api/query` on 2026-10-20 and returned a title matching this "
  "ledger: "
  + pids("GSQ", "GAMMA", "Disaggregated Quantization", "SpectrumQuant", "KVarN",
         "Mix-Quant: Quantized", "AlignQuant")
  + " (recorded earlier in [CLOSEST_WORK.md](CLOSEST_WORK.md)) and "
  + pids("PagedAttention", "QServe", "T-MAC", "PowerInfer", "Bench: Extending",
         "Outliers and Calibration", "Exploring the Trade-Offs", "RULER")
  + " (checked in this pass). That confirms the paper exists and that the title is not invented; it "
  "does **not** confirm a venue, so the rows above still need their `needs_verification` item "
  "closed.")
w("")
w("Rows that need more than routine verification:")
w("")
w(f"- {pid('Characterizing Prefill')} — IEEE Xplore 11591742. The landing page returns HTTP 200 "
  "but no extractable title/venue/year, so both the year and the evaluated hardware are "
  "unresolved. **This is the single most dangerous row in the corpus**: its title asserts exactly "
  "the regime separation C2 rests on. If it turns out to measure a Jetson Orin Nano, C2 and C3 "
  "must both be reworded before submission.")
w(f"- {pid('ELLIE')} — IEEE Xplore 11113611, same extraction problem, same resolution requirement.")
w(f"- {pid('SGSR-2')} — a Zenodo preprint, single author, non-peer-reviewed, and its own v2 "
  "retracts the v1 entropy-sensitivity claim. It is cited here only as evidence that budget-aware "
  "allocation exists; it must not be used as a headline baseline.")
w(f"- {pid('Offline AI Modules')} — an engineering report with a Jetson Orin tier table. Usable as "
  "evidence that the class is benchmarked in practice, not as a peer-reviewed measurement.")
w(f"- {pid('Deflating the Hessian')} — a W4A4 diffusion-transformer paper whose LLM/edge relevance "
  "rests on a single Qwen3-8B MMLU number; check whether it belongs in the corpus at all.")
w(f"- {pid('EdgeLLM-Nano')} — evaluates an 8.39M-parameter, 33.6 MB model on a Raspberry Pi 5. It "
  "is in scope as an edge measurement but three orders of magnitude away from our deployment scale.")
w("- The A1 gap list records that **\"QuiRot\", named in the original brief, could not be "
  "verified**: arXiv full-text search returns zero hits and no venue page surfaced. If QuiRot "
  "appears in the related-work table, that citation must be re-checked or removed — it may be a "
  "mis-citation in the brief itself.")
w("")

# ---------------------------------------------------------------- 7
w("## 7. Coverage honesty")
w("")
w("The manuscript must describe its own retrieval truthfully, so this section states exactly how "
  "the corpus was produced.")
w("")
w("**A1-A4 (`sweep`).** Produced by the eight-agent `lit-gap-sweep` workflow and captured to a job "
  "output file. The capture is 50,174 bytes and ends with the literal marker "
  "`[truncated: 73177 more characters]`: the original return was roughly 121k characters and only "
  "its first ~50k survived. What survived intact is A1 (18 papers, 7 area gaps, 5 open questions), "
  "A2 (17, 6, 4) and A3 (17, 6, 4). A4 survived only as four complete paper records plus a fifth "
  "(AMQ) cut mid-sentence in its `gap` field, and **no** `area_gaps` or `open_questions` for A4 at "
  "all. AMQ is kept with the truncation marked inside `gap_note` rather than repaired by "
  "inference; A4's summary fields are marked as not captured in §3.")
w("")
w("**A5-A8 (`recovered`).** Not part of the captured sweep. Rederived in this session by web "
  "search on the four sub-area definitions supplied by the project lead, keeping only papers whose "
  "URL resolved in a search hit or fetch, with 13/14/12/12 rows respectively. Venues that could not "
  "be confirmed are `unverified`; years that could not be confirmed are `unknown`. The recovery "
  "pass is therefore *less* verified than the sweep pass, and §6 keeps those rows out of the "
  "manuscript until they are checked.")
w("")
w("**Known incompleteness, stated so it cannot be mistaken for coverage:**")
w("")
w("- The corpus holds 108 rows against the protocol's 120-180 target.")
w("- One paper relevant to A2/A4 surfaced during recovery but was deliberately **not** merged into "
  "the sweep rows: *Budget-Aware LLM Quantization and Low-Rank Correction via Information-Guided "
  "Subspace Matrices* (IJCAI 2026, https://www.ijcai.org/proceedings/2026/475). Merging a single "
  "post-hoc row would have made the A2/A4 counts a splice of two retrieval passes with no "
  "documented union rule. It must be added at the next full refresh, and it is a direct competitor "
  "to C1 that Related Work has to address.")
w("- The A5-A8 pass ran to 2026-10-20, but A1-A4 were captured on the same date from a "
  "single workflow execution; the two passes used different tooling, so a formal PRISMA "
  "hit/screened/included count exists for neither. The manuscript should present the corpus as a "
  "**bounded, reproducible sample** and publish the queries and date, not as an exhaustive census.")
w("- No claim in this document depends on a paper that appears only in the truncated part of the "
  "sweep: every citation above is a row in `corpus.csv`.")
w("")

text = "\n".join(L)
(LIT / "GAP_MATRIX.md").write_text(text, encoding="utf-8")
print("wrote", LIT / "GAP_MATRIX.md", len(text), "chars,", text.count("\n") + 1, "lines")
