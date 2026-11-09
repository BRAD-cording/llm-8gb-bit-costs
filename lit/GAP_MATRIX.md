# Gap Matrix

**Scope.** This matrix is the evidence base for the thesis that, on 8 GB-class edge devices, the optimal split of a single memory budget between weight bits and KV-cache bits is not a static property but reverses with the prefill/decode regime. Every row is a paper retrieved under the PRISMA-style protocol frozen in [SEARCH_PROTOCOL.md](SEARCH_PROTOCOL.md) (window 2023-01-01 to 2026-10-20, the six arXiv/OpenAlex/DBLP/IEEE/ACM queries recorded there, and the inclusion criteria I1-I4). The corpus file is [corpus.csv](corpus.csv); each row's `subarea`, `quant_target`, `hw_evaluated`, `regime_aware`, `budget_aware` and `gap_note` columns are derived from the retrieved record, never from the reviewer's expectations. Where the record is silent the cell says `unknown`/`unspecified` rather than a plausible value. A1-A4 were captured by the original eight-agent sweep; A5-A8 were re-derived in a separate recovery pass because the sweep output was truncated in capture (see §7), so the corpus is a bounded sample, not an exhaustive census, and every claim below is a claim about this corpus.

## 1. Corpus overview

| subarea | #papers | saturation (as reported) | 2025-2026 paper count | source |
|---|---|---|---|---|
| A1 | 18 | high | 13 | sweep |
| A2 | 17 | high | 14 | sweep |
| A3 | 17 | high | 12 | sweep |
| A4 | 5 | high | 2 | sweep |
| A5 | 13 | high | 8 | recovered |
| A6 | 14 | medium | 8 | recovered |
| A7 | 12 | medium | 7 | recovered |
| A8 | 12 | high | 12 | recovered |
| **total** | **108** | — | **76** | — |

Notes on the accounting:

- Saturation for A1-A4 is the value the sweep reported. Saturation for A5-A8 is **our** assessment at recovery time, not a sweep finding, because those sub-areas were not part of the captured run.
- Three rows carry `year = unknown` (P071, P072, P082) because the IEEE/ACM records resolved to a landing page without extractable publication metadata; they are excluded from the 2025-2026 column rather than guessed into it.
- 55 of 108 rows carry `venue = unverified`; §6 lists them and the rule that applies to them before submission.
- `quant_target` is `unknown` for 27 rows. That is a finding about reporting practice, not a coding failure: those records describe an allocation or a measurement without ever naming which tensor class is quantized.
- Enum limitation: the schema has no `W+A+LR` value, so low-rank-compensation papers whose quantized tensors are weights+activations (the LoRC/RILQ/SERQ family) are coded `W+A` and their low-rank path is recorded in `method_one_line`. `W+A+KV+LR` is therefore unused in this corpus: no retrieved paper combines a KV decision variable with low-rank compensation.

## 2. The decisive cross-tabulation

### Table A — hardware actually evaluated

| hw_evaluated | papers |
|---|---|
| `unspecified` | 50 |
| `GPU-unspecified` | 38 |
| `A100` | 7 |
| `mobile-NPU` | 3 |
| `CPU` | 2 |
| `custom-ASIC` | 2 |
| `4090` | 1 |
| `Jetson` | 1 |
| `M1-Pro` | 1 |
| `RaspberryPi5` | 1 |
| `T4` | 1 |
| `simulator` | 1 |
| **total** | **108** |

`unspecified` means the record names no device at all; `GPU-unspecified` means it says "datacenter GPUs" / "server GPUs" / "GPUs" without a model. The distinction matters: the first is a reporting omission, the second is a reporting convention that makes the number non-reproducible on any specific device.

### Table B — regime awareness

| regime_aware | papers | share |
|---|---|---|
| `yes` | 5 | 4.6% |
| `no` | 103 | 95.4% |

`yes` requires that prefill and decode are separated *inside the optimization or measurement design*. Reporting two speedup numbers against FP16 is not enough, and on that strict reading papers that do exactly that — PrefixQuant (P005), FlatQuant (P004), ZipCache (P038) — are `no`.

### Table C — regime_aware × budget_aware

| | budget_aware = yes | budget_aware = no |
|---|---|---|
| **regime_aware = yes** | 1 | 4 |
| **regime_aware = no** | 10 | 93 |

The single `yes/yes` cell is AlignQuant (P048), and its budget applies to weight tiles only — the paper's own gap statement says KV precision is never part of the allocation. The four other regime-aware papers (P047, P049, P071, P072) either exclude KV from the allocation or do not quantize at all.

**Decisive count: 0 papers in this corpus are simultaneously `regime_aware = yes` and evaluated on 4060-class or Jetson-class hardware.** The single Jetson row in the corpus — P083 — is `regime_aware = no`; there is **no** RTX 4060 row at all.

What that 0 licenses:

- It licenses the statement that **no retrieved work reports a regime-conditional bit allocation measured on either of our two target devices**. The joint cell of the design space is empty in the literature we can cite, which is exactly the space C1-C3 claim.
- It licenses a *methodological* claim: because nobody has run the experiment on the class of device we target, the numbers reported for A100/H100 do not transfer by assumption — the bandwidth ratio between Orin (~68 GB/s unified) and the 4060 (256 GB/s discrete) is 3.8x, which is larger than the gap between the datacenter parts those papers use.

What it does **not** license:

- It does not license "no such work exists". The count is bounded by 108 retrieved rows, 55 of which have unverified venues and one of which (P071) is the closest prior art and has unresolved hardware. An absence in this corpus is an absence in our retrieval, not in the world.
- It does not license any claim about our own effect. A zero in the prior-work column says nothing about whether the reversal exists on real silicon; only the W7 pre-registered two-device experiment can say that, and a null result demotes the thesis to an empirical budget-allocation study.
- It does not license the weaker claim that edge devices are unquantized. They are quantized heavily (AWQ, QServe, T-MAC, llama.cpp-style GGUF stacks); what is missing is the *allocation decision*, not the compression.

## 3. Per-subarea gap statements

Verbatim from the retrieval records (A1-A4) and from the recovery pass (A5-A8). Wording is preserved; nothing here is paraphrased.

### A1 — Rotation / equivalent-transformation PTQ (QuaRot, SpinQuant, GSQ, PrefixQuant, OSTQuant, DuQuant, RotateKV and 2025–2026 successors)

**Saturation:** high · **papers in corpus:** 18 · **source:** sweep

- No rotation paper reports a weight-versus-KV bit-allocation curve at a fixed VRAM cap: every method fixes one tuple (typically W4A4KV4, or KV-only INT2) and reports accuracy, so the allocative question is never posed, let alone tested for regime dependence.
- Prefill and decode are reported as two speedup numbers against FP16 on datacenter GPUs (H100/A100, large batch); no paper measures the crossover on a 256 GB/s RTX 4060 or a 68 GB/s Jetson Orin Nano Super where the balance of arithmetic intensity and bandwidth is qualitatively different.
- The online cost of rotation (Hadamard/FWHT multiplies, permutations, per-token scaling) is treated as fusion-overhead-free 'negligible'; nobody quantifies what those extra elementwise passes cost when decode is already bandwidth-bound, which is exactly the regime a unified-memory Jetson sits in.
- Rotation quality is scored by perplexity or zero-shot accuracy on 7B–70B checkpoints; there is no accuracy-vs-bit-width frontier for 1–3B models that actually fit an 8GB budget at usable context length, so the small-model end of the frontier is unmeasured.
- Mixed-precision rotation work (Quant.npu, LIRQuant) allocates by sensitivity or channel importance, never by a cost model; conversely the bit-allocation/waterfilling theory literature assumes allocation-flexible coding and ignores the grouped shared-scale layout that the deployed kernel actually executes.
- Hardware-format assumptions diverge from the target: MXFP4 (DuQuant++, TORQ) needs Blackwell tensor cores, and GyRot/LightRot assume custom ASIC rotation units, so almost none of the 2026 rotation results are executable on Ada or Ampere-class 8GB parts.
- 'QuiRot' as named in the brief could not be verified: arXiv full-text search for the string returns zero hits and no indexed venue page surfaced, so the rotation/query-aware branch of this sub-area may be mis-cited in the paper's related-work table and should be re-checked before submission.

- **Open questions:**
  - If the same 8GB budget must serve both a prefill-heavy and a decode-heavy workload, does the accuracy-optimal split between weight bits and KV bits actually invert, or does it merely flatten — and is the inversion predicted by bytes-per-token arithmetic or by outlier statistics?
  - Is a rotation learned to minimize W4A4KV4 reconstruction loss still the best rotation once weights are pushed to 2–3 bits to leave KV headroom, or do the two optimizations conflict (as GyRot's rotation/group-scale mismatch suggests)?
  - What is the real price of the rotation itself on a bandwidth-bound decoder — do the extra FWHT/permutation passes and the non-fused scale multiplies consume the bandwidth savings that lower bit-width was supposed to buy?
  - Can a single calibration produce a regime-conditional bit plan (per-tensor bit widths selected by measured prefill/decode ratio) without either retraining rotations per plan or storing multiple rotated checkpoints, given that rotated weights cannot be trivially re-grouped after the fact?
  - Do the small-model (1–3B) accuracy frontiers that matter for an 8GB device behave like the 7B–70B curves these methods report, or does reduced over-parameterization make rotation-based outlier smoothing unnecessary and bit reallocation more valuable?

### A2 — Hessian/second-order PTQ and low-rank error compensation, incl. 2026 budget-aware bit allocation

**Saturation:** high · **papers in corpus:** 17 · **source:** sweep

- No paper in this sub-area writes its cost model as bytes moved per token. The 2026 allocators (GAMMA, MixQuant, KronQ) all minimize a reconstruction or sensitivity surrogate under a static memory or average-bit budget, so none can even represent the statement 'a weight bit and a KV bit buy different throughput in prefill than in decode'.
- The reversal question is untested. I found no work, on either the RTX 4060 class or the Orin class, that holds the 8GB budget fixed and reports whether the accuracy-optimal weight/KV split moves when the workload shifts from prefill-dominated to decode-dominated — the closest paper (MixQuant) attributes allocation shift to upstream bit-widths, not to serving regime.
- Second-order methods are evidenced almost entirely at 3-4 bits on 7B-70B models evaluated by WikiText-2 perplexity and MMLU. The 2-bit regime where Hessian versus low-rank compensation actually separates is demonstrated on 70B-class models that cannot inhabit 8GB at any bit-width, so the evidence base is structurally mismatched to the target deployment.
- Low-rank compensation is treated as memory-free. LoRC, RILQ, SERQ, EoRA-style work and Align-Then-Correct all report accuracy at a chosen rank; none solves for the rank that maximizes accuracy per added byte, and none charges adapter bytes against the same ceiling as the weights and KV cache.
- Prefill is largely unmeasured. Papers report accuracy plus at most one end-to-end or decode-only throughput number, so the compute-bound regime — where a lower bit-width can buy nothing because dequantization overhead dominates — is invisible in the reported results.
- Calibration cost is never budgeted for edge. Block-AP requires an A100-80GB and GPTQ-family methods hold a layer's Hessian in memory; no paper reports whether the Hessian or compensation pipeline can run, or be streamed block-by-block, within 8GB of unified memory.

- **Open questions:**
  - Does the accuracy-optimal weight/KV split under a fixed 8GB budget invert between prefill-bound and decode-bound serving, and is the inversion driven by raw bandwidth (Orin, ~68 GB/s) or by quantized-matmul and dequantization throughput (4060, 256 GB/s)?
  - Do Hessian-based sensitivity rankings remain the right allocation signal once cost is bytes-moved-per-token rather than parameter count, or does the optimum collapse toward low uniform weight bits with the remainder spent on KV cache?
  - Can low-rank compensation be made budget-neutral — is the rank that maximizes accuracy per added byte the same rank that maximizes tokens/s on a memory-bound device, given that the extra matmul may help at large prefill batches and hurt at batch-1 decode?
  - How much of the reported 3-4 bit advantage survives at 2-bit on the 1B-4B models that actually fit in 8GB, given that nearly all Hessian and low-rank evidence comes from models an order of magnitude larger?

### A3 — KV cache quantization and compression (KIVI, KVQuant, MixKVQ, KV Pareto, ZipCache, query-aware mixed precision, 2025-2026 phase-aware work)

**Saturation:** high · **papers in corpus:** 17 · **source:** sweep

- No paper in this sub-area indexes KV or weight bit allocation by roofline regime. Every method reports accuracy at a memory budget, so the question of whether the optimal allocation reverses between prefill-bound and decode-bound execution is not merely unanswered — it is never posed.
- KV bit-width is selected per layer, per head, per channel, per token or per chunk, but essentially never per phase. Apart from Disaggregated Quantization and Mix-Quant (which specialize weights and activations, not the KV cache), no method re-allocates the same fixed budget between the prefill and decode portions of a request.
- Edge evidence is thin and disjoint. On-device work uses <=2B models or a 33.6 MB testbed, while the KV-efficiency claims that dominate the literature come from A100/H100-class hardware. The surveyed set contains no measurement on an RTX 4060-class 8GB laptop GPU and none on Jetson Orin Nano, so the two deployment targets are supported only by extrapolation.
- Weight quantization and KV quantization are optimized in separate literatures. Only KV Pareto and AlignQuant touch joint allocation, and neither uses measured latency or achieved bandwidth as the frontier axis, so the coupling that the reversal hypothesis depends on is unmeasured.
- Accuracy metrics do not separate prefill damage from decode error accumulation. KVarN explicitly observes that most baselines are evaluated in prefill-like settings, yet no accepted protocol reports the two phases' contributions independently, so allocation studies cannot attribute a loss to a phase.
- Reported speedups rely on hardware paths that the 8GB targets lack: NVFP4 tensor cores cited by 2026 phase-aware work are Blackwell-class, absent from Ada laptop and Ampere Orin silicon, so prefill-throughput claims do not transfer to either device.

- **Open questions:**
  - Does the accuracy-optimal split of a fixed 8GB budget between weight bits and KV bits actually reverse between a prefill-dominated and a decode-dominated workload, or does the reversal show up only in latency and never in quality?
  - What is the correct control variable for such a study — bits per parameter, bits per token of context, or achieved bytes per second — given that Orin's unified LPDDR5 puts weight reads and KV reads on the same bus while the RTX 4060 has a distinct 256 GB/s path?
  - Can a KV precision policy be made phase-adaptive without a second checkpoint or per-step recalibration, when Orin's ~68 GB/s leaves almost no headroom for per-token precision metadata?
  - Do the query-aware and saliency-based allocations (ZipCache, MixKVQ, Cocktail, TriAxialKV) retain their advantage at batch size 1, where decode is bandwidth-bound rather than compute-bound and their per-token scoring overhead is no longer amortized?

### A4 Mixed-precision / bit-width allocation under a memory budget (HAWQ lineage, RL-based allocation, sensitivity-driven allocation, 2025-2026)

**Saturation:** high · **papers in corpus:** 5 · **source:** sweep

- *No `area_gaps` were captured for this sub-area: the sweep output was truncated immediately after its fifth paper record, before the sub-area summary fields. The paper rows themselves are complete and are included in the corpus.*

- **Open questions:** *not captured (same truncation as above).*

### A5 — Low-bit weight+activation (W4A4) and edge-realizable kernels — LO-BCQ, dp4a, INT4 tensor-core availability on Ampere SM87 / Ada SM89, paged KV kernels

**Saturation:** high · **papers in corpus:** 13 · **source:** recovered

- Every kernel here is built for one fixed precision tuple (W4A4, W4A8KV4, W4A16, 1.58-bit); none exposes weight and KV bit-width as a serving-time or request-time decision, so the kernel layer cannot execute a re-allocation even if a solver produced one.
- INT4/FP4 tensor-core paths are silicon-specific: APEX4, MicroMix and LUT Tensor Core assume datapath support that Ada SM89 and Ampere SM87 (Jetson Orin) do not expose in the same form, and no paper in this set benchmarks an RTX 4060-class 8GB part.
- Paged-KV machinery (PagedAttention, FlashInfer) fixes the cache dtype at allocation time; a plan that moves precision between weights and KV mid-serving has no paged-kernel implementation to run on.
- Dequantization cost is reported as a ratio against FP16 or against a Marlin baseline, never as achieved bytes/s, so a bandwidth-bound edge decode cannot be predicted from these numbers.
- The accuracy-side W4A4 methods (LO-BCQ, MicroMix) and the systems-side kernels (QServe, Atom) are never evaluated together, so nobody reports the accuracy cost of the bit plan that the kernel actually makes fast.

- **Open questions:**
  - On Ada SM89 with 8GB at 256 GB/s, does an INT4 W4A4 path actually beat W4A16 in decode, where the kernel is bandwidth-bound and dequantization is already hidden behind memory latency?
  - Is there a kernel design in which the KV dtype and the weight dtype can be re-chosen per request without repacking weights — i.e. can the weight-vs-KV split be a serving-time decision at all?
  - Do paged-block abstractions permit per-block bit-widths, and what does per-block precision metadata do to the block table's bandwidth cost on a 68 GB/s unified-memory part?

### A6 — Edge / on-device LLM deployment and measurement — Jetson Orin Nano, prefill-decode regime characterization, energy per token, thermal throttling, edge compression surveys, llama.cpp/GGUF deployment studies

**Saturation:** medium · **papers in corpus:** 14 · **source:** recovered

- No paper in this set measures a joint weight-plus-KV bit allocation on an RTX 4060-class 8GB laptop GPU: the 4060 appears only in adjacent engineering write-ups and vendor benchmarking notes, never as the substrate of a controlled study.
- Jetson Orin appears once, as one benchmark tier in an engineering report (Offline AI Modules) with hand-chosen precisions, and never as the platform of an allocation experiment.
- Edge measurement papers report latency and energy at fixed precision levels (Q4_K_M, AWQ, QAT variants) and rarely report achieved bandwidth or the prefill/decode decomposition of cost, so their numbers cannot be turned into the cost model an allocator needs.
- Regime awareness at the edge is still at the characterization stage: the two IEEE papers that do split prefill from decode do not quantize, and the quantization papers do not split the phases.
- Thermal behaviour is measured for phones (MELTing Point, mobile DVFS work) but no paper reports whether a bit plan that is optimal in a cold benchmark stays optimal after an 8GB laptop GPU or Orin throttles under sustained decode.
- Surveys (Computer Science Review 2025) still present weight and KV quantization as separate families, which is why the joint-budget question is missing from the edge deployment literature rather than merely unanswered there.

- **Open questions:**
  - Does the prefill/decode crossover on Orin Nano (~68 GB/s unified LPDDR5) occur at a different context length than on the RTX 4060 (256 GB/s discrete), and does that shift the accuracy-optimal bit split between the two devices?
  - After ten minutes of sustained decoding, how far does thermal throttling move achieved bandwidth, and does the optimal split track throttled or nominal bandwidth?
  - Is the framework/system overhead term rho materially larger on Orin than on the 4060, and does it consume exactly the headroom that reducing KV bits was supposed to create?

### A7 — Evaluation methodology and reproducibility of quantization papers — unreported execution stack, benchmark validity, long-context benchmarks (RULER, InfiniteBench), statistical rigor, calibration-set sensitivity

**Saturation:** medium · **papers in corpus:** 12 · **source:** recovered

- Execution stack is systematically unreported: none of the quantized-LLM papers in this corpus states CUDA/cuDNN/driver versions, kernel commit, or clock policy, which is exactly what the measurement-bias and reproducibility work says makes their throughput numbers non-comparable.
- Phase-mixed benchmarking is the norm: accuracy is measured in prefill-like teacher-forced settings while speed is measured in decode, and the long-context benchmarks (RULER, InfiniteBench, LongBench v2) carry no precision axis at all.
- Statistical rigor is thin: single-run accuracies without seeds, confidence intervals or paired tests, while the contamination literature shows the deltas being reported often sit inside benchmark noise.
- Calibration sensitivity is studied as an accuracy question (diminishing outlier effects, set-cover calibration) but never as a budget question — nobody reports how the accuracy-optimal weight/KV split moves with the calibration set.
- Reporting standards (ReproEvalCard) cover pipelines but not quantization experiments: there is no mandated field for memory cap, achieved bandwidth, KV precision or phase-separated throughput.

- **Open questions:**
  - If every paper were re-run on one documented stack with at least three seeds and paired tests, how many reported quantization gains would survive?
  - What is the minimal reporting checklist that lets a reader reconstruct a weight-vs-KV allocation experiment, and would adopting it have changed any published conclusion?
  - How much of the reported accuracy gap between two bit-widths is calibration-set variance rather than the bit-width effect, on 1-4B models?

### A8 — Degradation of long-chain reasoning under quantization — GSM8K/MATH/CoT under 3-4 bit, error amplification over autoregressive steps, weight-vs-KV error attribution

**Saturation:** high · **papers in corpus:** 12 · **source:** recovered

- Reasoning degradation is measured at fixed bit-widths and almost always on weight-only quantization; no paper asks whether the same byte budget spent on KV precision would shrink the damage.
- Phase attribution is missing: reasoning failures accumulate over decode steps, yet the protocols report one final accuracy, so the decode-side contribution cannot be separated from prefill-side damage.
- The scale mismatch is stark — reasoning studies use 7B-70B models that do not fit an 8GB device at usable context, while the 1-4B models that do fit are barely evaluated on GSM8K/MATH.
- Mitigations are training- or calibration-side (re-tuning, QAT, history calibration) and cost compute the edge cannot pay; none of them is a bit-allocation policy.
- Token inflation is a first-class cost of low-bit reasoning, but no study charges the extra generated tokens against the KV memory budget, even though longer chains raise KV pressure exactly when precision is lowest.

- **Open questions:**
  - Does a fixed 8GB budget produce a different accuracy-optimal split on GSM8K/MATH than on perplexity — i.e. is reasoning the metric that moves the optimum?
  - Is the earliest-failure-step phenomenon attributable to weight error, KV error, or their interaction, and can it be attributed by holding one side at FP16?
  - If quantized models emit materially more reasoning tokens, does the KV-bits term dominate the weight-bits term at 8K+ context on Orin, and by how much?

## 4. Cross-cutting gaps (synthesised)

These are our own synthesis: a gap counts as cross-cutting only when it is asserted independently in more than one sub-area's own gap list. Strength: **strong** = 3+ independent sub-areas, **medium** = 2, **weak** = 1 (kept only where the single sub-area is the decisive one).

### CC1 — The budget is never one pool shared by weights and KV

**Evidence:** A1 (gap 1), A2 (gap 2), A3 (gap 4), A5 (gap 5), A8 (gap 1). **Strength: strong (5 sub-areas).**

Each literature allocates inside its own tensor class: rotation and Hessian methods fix a W/A/KV tuple, KV methods fix the weight precision as an external constant, and the 2026 allocators (P033, P034, P035) constrain a static memory number that has no KV-growth term. The only paper that touches both sides, KV Pareto (P044), indexes a frontier by total memory rather than optimizing under a cap, so it reports a curve instead of a decision.

### CC2 — Prefill/decode is reported but never enters the objective

**Evidence:** A1 (gap 2), A2 (gap 2 and gap 5), A3 (gap 1), A6 (gaps 3-4). **Strength: strong (4 sub-areas).**

Three papers do specialize precision by phase — Disaggregated Quantization (P047), Mix-Quant (P049) and AlignQuant (P048) — and all three explicitly exclude the KV cache from the decision. On the edge side, the two IEEE papers that do split the regimes (P071, P072) do not quantize. So the exact object we claim (a regime-conditional W-vs-KV split) has no prior instance, and the nearest neighbours are the threat surface, not the support.

### CC3 — Edge classes are unmeasured, and the evidence base is structurally mismatched

**Evidence:** A1 (gap 4), A2 (gap 3), A3 (gap 3), A5 (gap 2), A6 (gaps 1-2). **Strength: strong (5 sub-areas).**

The mismatch is twofold. Hardware: 50 of 108 rows do not name a device at all and 38 name only "GPUs"; the named parts are 7 A100s, one T4, one M1 Pro, one Raspberry Pi 5, one RTX 4090, one Jetson, and no 4060. Scale: the accuracy evidence that decides bit-widths is collected on 7B-70B checkpoints that cannot inhabit 8 GB at any bit-width, while the 1-4B models that can are barely evaluated. Both halves of the mismatch have to be closed by our own experiments, which is why the contribution is measurement, not method.

### CC4 — Evaluation practice is unreported, phase-mixed and statistically thin

**Evidence:** A7 (gaps 1-3), A2 (gap 5), A6 (gap 3). **Strength: strong (3 sub-areas).**

No paper in the corpus reports CUDA/cuDNN/driver versions, kernel commit or clock policy — the three things the measurement-bias work (P095) says make throughput numbers non-comparable. Accuracy is measured prefill-style while speed is measured decode-style, and the long-context benchmarks (P085, P086, P087) carry no precision axis. Since the same corpus reports single-run accuracies without seeds or paired tests, the deltas our ablations must beat are frequently inside benchmark noise.

### CC5 — Transforms and low-rank corrections are treated as free

**Evidence:** A1 (gap 3), A2 (gap 4), A5 (gap 4). **Strength: strong (3 sub-areas).**

Rotation/FWHT passes, permutations, per-token scaling and low-rank side paths all consume bandwidth or compute that is never charged against the budget that motivated them. On a bandwidth-bound decoder this is not a second-order effect: if the transform costs one extra read of the activation per layer, it can eat the saving that halving the weight bits was supposed to buy. A 1-3B model on a 68 GB/s device is where that accounting error becomes visible first.

### CC6 — Hardware-format assumptions do not exist on Ada SM89 / Ampere SM87

**Evidence:** A1 (gap 6), A3 (gap 6), A5 (gap 2). **Strength: strong (3 sub-areas).**

MXFP4 and NVFP4 paths assume Blackwell-class tensor cores; LUT-based tensor cores and GyRot's rotation unit assume custom silicon. Neither the RTX 4060 (Ada) nor Jetson Orin (Ampere) provides them in the form these kernels require, so a large fraction of the 2026 speedups are not executable on the devices the thesis targets. That is a coverage gap we exploit, but it is also a trap: any method we propose must run through dp4a / INT4 MMA paths that actually exist on both parts.

### CC7 — Reasoning damage is measured but never charged against the KV budget

**Evidence:** A8 (gaps 1, 2, 5), A3 (gap 5). **Strength: medium (2 sub-areas).**

Long-chain reasoning is the workload where quantization damage accumulates, and it is also the workload that inflates KV footprint the most (longer chains, longer contexts). The reasoning literature (P097, P098) measures the damage at fixed weight bit-widths, and the token-inflation work (P106) measures the extra cost, but nobody connects the two into a KV-pressure term inside the allocation objective.

## 5. Mapping to our thesis

| our claim | supporting gap rows | threat | mitigation |
|---|---|---|---|
| **C1** — no existing work jointly allocates weight bits and KV bits under one device budget | CC1; budget-aware but KV-blind: P033, P034, P035, P027, P056; KV-only: P068, P101; both sides but frontier-only: P044 | AlignQuant (P048) already runs phase-normalized precision selection over weight tiles under a model-wide storage budget, and SGSR-2 (P035) already publishes a measured bits-by-group-size Pareto front on 8 GB consumer hardware. A reviewer can read either as "the allocation problem is solved" | Concede the weight-only case and narrow C1 to the *joint* pool: quote each paper's own statement that KV precision is outside its allocation (AlignQuant, GAMMA, MixQuant), and show the ablation where KV is removed from our solver and the optimum degenerates to their answer. The claim survives only as "joint + KV growth + rho", so state it that way |
| **C2** — no existing work conditions allocation on the prefill/decode regime | CC2; regime-aware but KV-blind: P047, P049, P048; regime-aware but non-quantizing: P071, P072 | Three 2026 papers already exploit the phase asymmetry (prefill low-precision, decode high-precision) and one is a dedicated edge prefill/decode characterization. Saying "no one is regime-aware" is falsified by our own corpus | Never claim regime awareness itself. Claim only that *no paper makes the W-vs-KV split a function of the regime*, and prove it by the explicit KV exclusion in those three papers. Cite them as the closest prior art in Related Work, and make the regime ablation (alpha swept) the load-bearing experiment rather than a rhetorical point |
| **C3** — the 8GB laptop-GPU and Jetson Orin classes are unmeasured | CC3; P083, P080, P035, P084 | P083 benchmarks a Jetson Orin tier today; PowerInfer (P080) serves on a consumer GPU; SGSR-2 runs on an M1 Pro; and "Jetson" is a family, so a reviewer who has run llama.cpp on an Orin Nano will not accept "unmeasured" as written | Restate C3 as *no controlled joint-allocation measurement on either class*. Cite the Jetson row as an engineering report with hand-chosen precisions (no allocation, no seeds, no stack report), and the 4090 row as 24 GB, not 8 GB. Then make the 4060-vs-Orin measurement itself the contribution — that is exactly what the empty cell buys |
| **C4** — evaluation practice is a live methodological gap | CC4; P090, P095, P092, P093; missing hardware reporting in P001, P037 | A reviewer can answer "this is well known, and there are reporting standards" — ReproEvalCard (P090) exists precisely to fix pipeline reporting, and the measurement-bias paper (P095) already names the warm-up/caching confounds | Accept C4 as a *domain-specific* gap rather than a novel one: the standards cover pipelines and serving engines, not quantization experiments, and neither mandates memory cap, achieved bandwidth, KV precision or phase-separated throughput. Position our artifact (stack snapshot + seeds + paired tests) as the instantiation of an existing standard in a domain that has not adopted it |

## 6. Verification status

Every row below either carries `venue = unverified` or has `year = 2026`. **Any entry listed here must be resolved before it appears in the submitted manuscript.** Resolution means: the URL returns the paper, the title matches, and the venue/year on the page match the ledger. If a URL cannot be resolved, the row is **dropped from the manuscript corpus** — it is never re-guessed, and no venue is inferred from an arXiv identifier.

| id | title | url | needs_verification |
|---|---|---|---|
| P005 | PrefixQuant: Eliminating Outliers by Prefixed Tokens for Large Language Models Quantization | https://arxiv.org/abs/2410.05265v2 | venue unverified |
| P008 | GSQ: Highly-Accurate Low-Precision Scalar Quantization for LLMs via Gumbel-Softmax Sampling | https://arxiv.org/abs/2604.18556v2 | venue unverified; year 2026 (not yet in a proceedings record we can cite) |
| P009 | DuQuant++: Fine-grained Rotation Enhances Microscaling FP4 Quantization | https://arxiv.org/abs/2604.17789v2 | venue unverified; year 2026 (not yet in a proceedings record we can cite) |
| P010 | Transforms for LLM Quantization: The Great Inversion and Format Co-Design | https://arxiv.org/abs/2608.25188v1 | venue unverified; year 2026 (not yet in a proceedings record we can cite) |
| P011 | SpectrumQuant: Spectrum Alignment for LLM Quantization | https://arxiv.org/abs/2610.20303v1 | venue unverified; year 2026 (not yet in a proceedings record we can cite) |
| P012 | LIRQuant: An Efficient Locally Isolated Rotation Quantization Method for Large Language Models | https://arxiv.org/abs/2610.25675v1 | year 2026 (not yet in a proceedings record we can cite) |
| P013 | HARP: Hadamard-Preconditioned Adaptive Rotation Processor for Extreme LLM Quantization | https://arxiv.org/abs/2605.29843v2 | venue unverified; year 2026 (not yet in a proceedings record we can cite) |
| P014 | Qift: Shift-Friendly No-Zero W2 Post-Training Quantization for Rotated W2A4/KV4 LLM Inference | https://arxiv.org/abs/2606.02823v1 | venue unverified; year 2026 (not yet in a proceedings record we can cite) |
| P015 | OSCAR: Offline Spectral Covariance-Aware Rotation for 2-bit KV Cache Quantization | https://arxiv.org/abs/2605.17757v1 | venue unverified; year 2026 (not yet in a proceedings record we can cite) |
| P016 | Quant.npu: Enabling Efficient Mobile NPU Inference for on-device LLMs via Fully Static Quantization | https://arxiv.org/abs/2605.20295v1 | venue unverified; year 2026 (not yet in a proceedings record we can cite) |
| P017 | GyRot: Leveraging Hidden Synergy between Rotation and Fine-grained Group Quantization for Low-bit LLM Inference | https://arxiv.org/abs/2607.27694v1 | venue unverified; year 2026 (not yet in a proceedings record we can cite) |
| P018 | QAM-W: Joint 2D Codebook Quantization for LLM Weights via Hadamard Rotation and Activation-Aware Scaling | https://arxiv.org/abs/2605.26339v1 | venue unverified; year 2026 (not yet in a proceedings record we can cite) |
| P021 | ZeroQuant-V2: Exploring Post-training Quantization in LLMs from Comprehensive Study to Low Rank Compensation | https://arxiv.org/abs/2303.08302 | venue unverified |
| P025 | First-Order Error Matters: Accurate Compensation for Quantized Large Language Models | https://arxiv.org/abs/2507.11017 | year 2026 (not yet in a proceedings record we can cite) |
| P026 | ADMM-Q: An Improved Hessian-based Weight Quantizer for Post-Training Quantization of Large Language Models | https://arxiv.org/abs/2605.11222 | venue unverified; year 2026 (not yet in a proceedings record we can cite) |
| P027 | KronQ: LLM Quantization via Kronecker-Factored Hessian | https://arxiv.org/abs/2607.07964 | year 2026 (not yet in a proceedings record we can cite) |
| P028 | Rethinking Residual Errors in Compensation-based LLM Quantization | https://arxiv.org/abs/2604.07955 | year 2026 (not yet in a proceedings record we can cite) |
| P029 | SERQ: Saliency-Aware Low-Rank Error Reconstruction for LLM Quantization | https://arxiv.org/abs/2603.08185 | year 2026 (not yet in a proceedings record we can cite) |
| P030 | Align, Then Correct: Training-Free Two-Stage Low-Rank Compensation for Extremely Quantized Large Language Models | https://arxiv.org/abs/2610.08164 | venue unverified; year 2026 (not yet in a proceedings record we can cite) |
| P031 | GPTQ-intrinsic LoRA: A Near-optimal Algorithm for Low-precision Quantization with Low-rank Adaptation | https://arxiv.org/abs/2606.01412 | venue unverified; year 2026 (not yet in a proceedings record we can cite) |
| P032 | Deflating the Hessian: Rank-4 W4A4 Quantization for Multimodal Diffusion Transformers | https://arxiv.org/abs/2610.11315 | venue unverified; year 2026 (not yet in a proceedings record we can cite) |
| P033 | GAMMA: Global Bit Allocation for Mixed-Precision Models under Arbitrary Budgets | https://arxiv.org/abs/2605.18475 | venue unverified; year 2026 (not yet in a proceedings record we can cite) |
| P034 | MixQuant: Adaptive Mixed-Precision Quantization for Large Language Models | https://arxiv.org/abs/2607.23047 | year 2026 (not yet in a proceedings record we can cite) |
| P035 | SGSR-2: Measured-Cost Pareto Allocation of Bit-Width and Group-Size for On-Device LLM Quantization Language Models | https://zenodo.org/records/21190586 | year 2026 (not yet in a proceedings record we can cite) |
| P039 | SKVQ: Sliding-window Key and Value Cache Quantization for Large Language Models | https://arxiv.org/abs/2405.06219 | venue unverified |
| P040 | GEAR: An Efficient KV Cache Compression Recipe for Near-Lossless Generative Inference of LLM | https://arxiv.org/abs/2403.05527 | venue unverified |
| P043 | KVLinC: KV Cache Quantization with Hadamard Rotation and Linear Correction | https://arxiv.org/abs/2510.05373 | venue unverified |
| P044 | KV Pareto: Systems-Level Optimization of KV Cache and Model Compression for Long Context Inference | https://aclanthology.org/2026.eacl-industry.9/ | year 2026 (not yet in a proceedings record we can cite) |
| P045 | MixKVQ: Query-Aware Mixed-Precision KV Cache Quantization for Long-Context Reasoning | https://aclanthology.org/2026.acl-long.326/ | year 2026 (not yet in a proceedings record we can cite) |
| P046 | Don't Waste Bits! Adaptive KV-Cache Quantization for Lightweight On-Device LLMs | https://arxiv.org/abs/2604.04722 | year 2026 (not yet in a proceedings record we can cite) |
| P047 | Disaggregated Quantization: Specializing LLM Prefill and Decode | https://arxiv.org/abs/2609.26333 | venue unverified; year 2026 (not yet in a proceedings record we can cite) |
| P048 | AlignQuant: Tile-Aligned Mixed-Precision Quantization for Efficient LLM Generation | https://arxiv.org/abs/2610.07457 | venue unverified; year 2026 (not yet in a proceedings record we can cite) |
| P049 | Mix-Quant: Quantized Prefilling, Precise Decoding for Agentic LLMs | https://arxiv.org/abs/2605.20315 | venue unverified; year 2026 (not yet in a proceedings record we can cite) |
| P050 | TriAxialKV: Toward Extreme Low-Precision KV-Cache Quantization for Agentic Inference Tasks | https://arxiv.org/abs/2605.17170 | venue unverified; year 2026 (not yet in a proceedings record we can cite) |
| P051 | KVarN: Variance-Normalized KV-Cache Quantization Mitigates Error Accumulation in Reasoning Tasks | https://arxiv.org/abs/2606.03458 | venue unverified; year 2026 (not yet in a proceedings record we can cite) |
| P052 | EdgeLLM-Nano: Auditing Polar-Guided Key-Cache Compression and Consistency-Anchored Pruning on a 33.6 MB On-Device Language Model | https://arxiv.org/abs/2610.14475 | venue unverified; year 2026 (not yet in a proceedings record we can cite) |
| P056 | MixLLM: LLM Quantization with Global Mixed-precision between Output-features and Highly-efficient System Design | https://arxiv.org/abs/2412.14590 | year 2026 (not yet in a proceedings record we can cite) |
| P065 | MicroMix: Efficient Mixed-Precision Quantization with Microscaling Formats for Large Language Models | https://arxiv.org/abs/2508.02343 | venue unverified |
| P067 | APEX4: Efficient Pure W4A4 LLM Inference via Intra-SM Compute Rebalancing | https://arxiv.org/abs/2606.08761 | venue unverified; year 2026 (not yet in a proceedings record we can cite) |
| P068 | SAW-INT4: System-Aware 4-Bit KV-Cache Quantization for Real-World LLM Serving | https://arxiv.org/abs/2604.19157 | venue unverified; year 2026 (not yet in a proceedings record we can cite) |
| P069 | FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving | https://arxiv.org/abs/2501.01005 | venue unverified |
| P071 | Characterizing Prefill and Decode Regimes for Edge LLM Inference | https://ieeexplore.ieee.org/document/11591742 | venue unverified; year unresolved |
| P072 | ELLIE: Energy-Efficient LLM Inference at the Edge Via Prefill-Decode Splitting | https://ieeexplore.ieee.org/document/11113611 | venue unverified; year unresolved |
| P073 | A Measurement Study of LLM Inference Trade-offs Across Edge Continuum Hardware | https://arxiv.org/abs/2609.08307 | venue unverified; year 2026 (not yet in a proceedings record we can cite) |
| P074 | Sometimes Painful but Certainly Promising: Feasibility and Trade-offs of Language Model Inference at the Edge | https://arxiv.org/abs/2503.09114 | venue unverified |
| P075 | Does Mixture-of-Experts Actually Help Inference on Consumer and Edge Hardware? An Empirical Study | https://arxiv.org/abs/2606.21428 | venue unverified; year 2026 (not yet in a proceedings record we can cite) |
| P076 | Seeing at the Edge: Energy Characterization of Vision-Language Model Inference on Edge Platforms | https://arxiv.org/abs/2610.20136 | venue unverified; year 2026 (not yet in a proceedings record we can cite) |
| P077 | EdgeProfiler: A Fast Profiling Framework for Lightweight LLMs on Edge Using Analytical Model | https://arxiv.org/abs/2506.09061 | venue unverified |
| P081 | LLMCad: Fast and Scalable On-device Large Language Model Inference | https://arxiv.org/abs/2309.04255 | venue unverified |
| P082 | Energy-Efficient Small Language Model Inference for Mobile Agents via DVFS | https://doi.org/10.1145/3812836.3814753 | venue unverified; year unresolved |
| P083 | Offline AI Modules: Voice-First Offline Architecture, Hardware Reference Stack, and Quantization & Benchmarking | https://arxiv.org/abs/2610.07026 | venue unverified; year 2026 (not yet in a proceedings record we can cite) |
| P084 | Sustainable LLM Inference for Edge AI: Evaluating Quantized LLMs for Energy Efficiency, Output Accuracy, and Inference Latency | https://arxiv.org/abs/2504.03360 | venue unverified |
| P086 | ∞Bench: Extending Long Context Evaluation Beyond 100K Tokens | https://arxiv.org/abs/2402.13718 | venue unverified |
| P088 | Outliers and Calibration Sets have Diminishing Effect on Quantization of Modern LLMs | https://arxiv.org/abs/2405.20835 | venue unverified |
| P089 | Coverage-Based Calibration for Post-Training Quantization via Weighted Set Cover over Outlier Channels | https://arxiv.org/abs/2604.24008 | venue unverified; year 2026 (not yet in a proceedings record we can cite) |
| P090 | ReproEvalCard: A Reporting Standard for Reproducible Evaluation of LLM Pipelines | https://aclanthology.org/2026.acl-short.22/ | year 2026 (not yet in a proceedings record we can cite) |
| P091 | An empirical study of LLaMA3 quantization: from LLMs to MLLMs | https://arxiv.org/abs/2404.14047 | venue unverified |
| P092 | Are LLM Benchmarks Already Contaminated? A Systematic Review of Contamination Detection Methods | https://aclanthology.org/2026.gem-main.50/ | year 2026 (not yet in a proceedings record we can cite) |
| P095 | Identifying and Mitigating Systemic Measurement Bias in Production LLM Inference Benchmarks | https://arxiv.org/abs/2605.24217 | venue unverified; year 2026 (not yet in a proceedings record we can cite) |
| P096 | AutoTuneBench: Trustworthy Measurement for Agent Auto-Tuning of LLM Serving Engines | https://arxiv.org/abs/2609.18123 | venue unverified; year 2026 (not yet in a proceedings record we can cite) |
| P097 | Quantization Hurts Reasoning? An Empirical Study on Quantized Reasoning Models | https://arxiv.org/abs/2504.04823 | venue unverified |
| P098 | Quantization Meets Reasoning: Exploring and Mitigating Degradation of Low-Bit LLMs in Mathematical Reasoning | https://arxiv.org/abs/2505.11574 | venue unverified |
| P099 | RECAP: Calibrating Low-Bit Reasoning Models on the History They Read | https://arxiv.org/abs/2610.15259 | venue unverified; year 2026 (not yet in a proceedings record we can cite) |
| P100 | RATIO: Reasoning Analysis and Token-level Inference Optimization for Quantized Reasoning Models | https://arxiv.org/abs/2609.39801 | venue unverified; year 2026 (not yet in a proceedings record we can cite) |
| P101 | AdaKVQ: Adaptive Mixed-Precision KV Cache Quantization For Efficient Reasoning Models | https://neurips.cc/virtual/2026/loc/sydney/poster/153261 | year 2026 (not yet in a proceedings record we can cite) |
| P102 | LFQ: Logit-aware Final-block Quantization for Boosting the Generation Quality of Low-Bit Quantized LLMs | https://icml.cc/virtual/2026/poster/60586 | year 2026 (not yet in a proceedings record we can cite) |
| P103 | ReQAT: Achieving Full-Precision Reasoning Accuracy with 4-bit Floating-Point Quantization-Aware Training | https://icml.cc/virtual/2026/oral/71113 | year 2026 (not yet in a proceedings record we can cite) |
| P104 | Balancing Compute in LLM Inference: Model Selection, Quantization, and Test-Time Scaling | https://dl.acm.org/doi/10.1145/3805621.3807631 | year 2026 (not yet in a proceedings record we can cite) |
| P105 | Not All Bits Are Equal: Scale-Dependent Memory Optimization Strategies for Reasoning Models | https://arxiv.org/abs/2510.10964 | venue unverified |
| P106 | Quantization Inflates Reasoning: Token Inflation as a Hidden Cost of Low-Bit Reasoning Models | https://arxiv.org/abs/2606.25519 | venue unverified; year 2026 (not yet in a proceedings record we can cite) |
| P107 | Integration of Chain-of-Thought and Program-of-Thought on 4-bit Quantized Small Language Models for Financial Question Answering | https://ieeexplore.ieee.org/document/11664865 | venue unverified |
| P108 | BitNet b1.58 2B4T Technical Report | https://arxiv.org/abs/2504.12285 | venue unverified |

Existence is a separate check from venue. Fifteen rows had their arXiv metadata re-queried directly through `export.arxiv.org/api/query` on 2026-10-20 and returned a title matching this ledger: P008, P033, P047, P011, P051, P049, P048 (recorded earlier in [CLOSEST_WORK.md](CLOSEST_WORK.md)) and P058, P062, P063, P080, P086, P088, P094, P085 (checked in this pass). That confirms the paper exists and that the title is not invented; it does **not** confirm a venue, so the rows above still need their `needs_verification` item closed.

Rows that need more than routine verification:

- P071 — IEEE Xplore 11591742. The landing page returns HTTP 200 but no extractable title/venue/year, so both the year and the evaluated hardware are unresolved. **This is the single most dangerous row in the corpus**: its title asserts exactly the regime separation C2 rests on. If it turns out to measure a Jetson Orin Nano, C2 and C3 must both be reworded before submission.
- P072 — IEEE Xplore 11113611, same extraction problem, same resolution requirement.
- P035 — a Zenodo preprint, single author, non-peer-reviewed, and its own v2 retracts the v1 entropy-sensitivity claim. It is cited here only as evidence that budget-aware allocation exists; it must not be used as a headline baseline.
- P083 — an engineering report with a Jetson Orin tier table. Usable as evidence that the class is benchmarked in practice, not as a peer-reviewed measurement.
- P032 — a W4A4 diffusion-transformer paper whose LLM/edge relevance rests on a single Qwen3-8B MMLU number; check whether it belongs in the corpus at all.
- P052 — evaluates an 8.39M-parameter, 33.6 MB model on a Raspberry Pi 5. It is in scope as an edge measurement but three orders of magnitude away from our deployment scale.
- The A1 gap list records that **"QuiRot", named in the original brief, could not be verified**: arXiv full-text search returns zero hits and no venue page surfaced. If QuiRot appears in the related-work table, that citation must be re-checked or removed — it may be a mis-citation in the brief itself.

## 7. Coverage honesty

The manuscript must describe its own retrieval truthfully, so this section states exactly how the corpus was produced.

**A1-A4 (`sweep`).** Produced by the eight-agent `lit-gap-sweep` workflow and captured to a job output file. The capture is 50,174 bytes and ends with the literal marker `[truncated: 73177 more characters]`: the original return was roughly 121k characters and only its first ~50k survived. What survived intact is A1 (18 papers, 7 area gaps, 5 open questions), A2 (17, 6, 4) and A3 (17, 6, 4). A4 survived only as four complete paper records plus a fifth (AMQ) cut mid-sentence in its `gap` field, and **no** `area_gaps` or `open_questions` for A4 at all. AMQ is kept with the truncation marked inside `gap_note` rather than repaired by inference; A4's summary fields are marked as not captured in §3.

**A5-A8 (`recovered`).** Not part of the captured sweep. Rederived in this session by web search on the four sub-area definitions supplied by the project lead, keeping only papers whose URL resolved in a search hit or fetch, with 13/14/12/12 rows respectively. Venues that could not be confirmed are `unverified`; years that could not be confirmed are `unknown`. The recovery pass is therefore *less* verified than the sweep pass, and §6 keeps those rows out of the manuscript until they are checked.

**Known incompleteness, stated so it cannot be mistaken for coverage:**

- The corpus holds 108 rows against the protocol's 120-180 target.
- One paper relevant to A2/A4 surfaced during recovery but was deliberately **not** merged into the sweep rows: *Budget-Aware LLM Quantization and Low-Rank Correction via Information-Guided Subspace Matrices* (IJCAI 2026, https://www.ijcai.org/proceedings/2026/475). Merging a single post-hoc row would have made the A2/A4 counts a splice of two retrieval passes with no documented union rule. It must be added at the next full refresh, and it is a direct competitor to C1 that Related Work has to address.
- The A5-A8 pass ran to 2026-10-20, but A1-A4 were captured on the same date from a single workflow execution; the two passes used different tooling, so a formal PRISMA hit/screened/included count exists for neither. The manuscript should present the corpus as a **bounded, reproducible sample** and publish the queries and date, not as an exhaustive census.
- No claim in this document depends on a paper that appears only in the truncated part of the sweep: every citation above is a row in `corpus.csv`.
