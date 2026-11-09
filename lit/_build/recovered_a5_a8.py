"""Recovered A5-A8 sub-areas (2026-10-20 incremental sweep).

Every row here came from a web search hit or a fetched page during the recovery
in this session; URLs are the ones the sources resolved to. Where a venue could
not be confirmed from a source, the venue is "unverified" (protocol rule: never
invent a venue). Where a year could not be confirmed, the year is "unknown".
"""

A5 = {
    "area": "A5 — Low-bit weight+activation (W4A4) and edge-realizable kernels — LO-BCQ, dp4a, "
            "INT4 tensor-core availability on Ampere SM87 / Ada SM89, paged KV kernels",
    "saturation": "high",
    "papers": [
        dict(
            title="PagedAttention: Efficient Memory Management for Large Language Model Serving with PagedAttention",
            year="2023", venue="SOSP 2023",
            url="https://arxiv.org/abs/2309.06180",
            method="Stores the KV cache in non-contiguous fixed-size blocks addressed through a "
                   "per-sequence block table, so cache memory can be shared, swapped and reclaimed "
                   "without fragmentation.",
            gap="Solves fragmentation for FP16 KV blocks; block size and cache dtype are fixed at "
                "allocation time, so the layout cannot express a bit plan that moves precision "
                "between weights and KV inside one 8GB pool.",
            qt="unknown", hw="A100", ra="no", ba="no"),
        dict(
            title="LUT-GEMM: Quantized Matrix Multiplication based on LUTs for Efficient Inference in Large-Scale Generative Language Models",
            year="2024", venue="ICLR 2024",
            url="https://proceedings.iclr.cc/paper_files/paper/2024/hash/a4f98ce85f440ee269b0df57b4368719-Abstract-Conference.html",
            method="Replaces dequantize-then-GEMM with lookup-table based binary-coding matmul so "
                   "low-bit weights are consumed directly inside the kernel.",
            gap="Weight-only coding on datacenter GPUs: no activation or KV quantization, no memory "
                "budget, and no way to trade a weight bit for a KV bit.",
            qt="W", hw="GPU-unspecified", ra="no", ba="no"),
        dict(
            title="Atom: Low-bit Quantization for Efficient and Accurate LLM Serving",
            year="2024", venue="MLSys 2024",
            url="https://arxiv.org/abs/2310.19102",
            method="Quantizes weights and activations to INT4, rescues outlier channels in FP16 at "
                   "per-channel granularity, and ships custom INT4 tensor-core GEMM kernels.",
            gap="Fixes one W4A4 precision for every layer while the KV cache stays FP16; there is no "
                "byte budget and no regime-dependent allocation to test.",
            qt="W+A", hw="A100", ra="no", ba="no"),
        dict(
            title="VPTQ: Extreme Low-bit Vector Post-Training Quantization for Large Language Models",
            year="2024", venue="EMNLP 2024",
            url="https://aclanthology.org/2024.emnlp-main.467/",
            method="Vector-quantizes weight groups against per-group codebooks with centroid pruning "
                   "and residual correction to reach 2-4 bits per weight.",
            gap="Weight-only compression with decode-oriented kernels; the KV cache and any joint byte "
                "budget are out of scope, so the split question is never posed.",
            qt="W", hw="GPU-unspecified", ra="no", ba="no"),
        dict(
            title="QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving",
            year="2025", venue="MLSys 2025",
            url="https://arxiv.org/abs/2405.04532",
            method="Co-designs progressive group quantization (W4A8KV4) with a register-level "
                   "dequantization kernel family so both GEMM and attention stay on tensor cores.",
            gap="The tuple W4A8KV4 is chosen once by datacenter throughput (A100/L40S); no 8GB cap, "
                "no weight-vs-KV reallocation, and no prefill/decode-conditional bit split.",
            qt="W+A+KV", hw="A100", ra="no", ba="no"),
        dict(
            title="T-MAC: CPU Renaissance via Table Lookup for Low-Bit LLM Deployment on Edge",
            year="2025", venue="EuroSys 2025",
            url="https://arxiv.org/abs/2407.00088",
            method="Implements mixed-precision GEMV on CPUs entirely through lookup tables, so low-bit "
                   "weights need no dequantization arithmetic at all.",
            gap="CPU-only and weight-only: no KV-cache precision, no GPU memory budget, and therefore "
                "nothing about a weight-vs-KV trade inside 8GB of unified memory.",
            qt="W", hw="CPU", ra="no", ba="no"),
        dict(
            title="LUT Tensor Core: A Software-Hardware Co-Design for LUT-Based Low-Bit LLM Inference",
            year="2025", venue="ISCA 2025",
            url="https://dl.acm.org/doi/10.1145/3695053.3731057",
            method="Proposes a LUT-based tensor-core datapath plus ISA support so mixed-precision "
                   "GEMM executes natively at very low bit-widths.",
            gap="A datapath proposal fixed in silicon: bits are not a decision variable across weights "
                "and KV, and its assumptions are absent from Ada SM89 and Ampere SM87.",
            qt="W+A", hw="custom-ASIC", ra="no", ba="no"),
        dict(
            title="MicroMix: Efficient Mixed-Precision Quantization with Microscaling Formats for Large Language Models",
            year="2025", venue="unverified",
            url="https://arxiv.org/abs/2508.02343",
            method="Assigns microscaling (MX) precision levels per channel by average magnitude and "
                   "executes several MX formats inside one mixed-precision GEMM.",
            gap="Mixed precision over weights/activations chosen by quantization error, not by measured "
                "bytes or latency; no KV term and no device budget.",
            qt="W+A", hw="GPU-unspecified", ra="no", ba="no"),
        dict(
            title="LO-BCQ: Block Clustered Quantization for 4-bit (W4A4) LLM Inference",
            year="2025", venue="TMLR 2025",
            url="https://arxiv.org/abs/2502.05376",
            method="Clusters contiguous blocks of each operand by statistics and designs an optimal "
                   "codebook per cluster, reaching W4A4 with reported <1% accuracy loss.",
            gap="Accuracy-only W4A4 codebook design with no kernel timing: it cannot say whether those "
                "bits should move to KV when decode is bandwidth-bound, nor under an 8GB cap.",
            qt="W+A", hw="unspecified", ra="no", ba="no"),
        dict(
            title="APEX4: Efficient Pure W4A4 LLM Inference via Intra-SM Compute Rebalancing",
            year="2026", venue="unverified",
            url="https://arxiv.org/abs/2606.08761",
            method="Rebalances work inside SMs and fuses dequantization so pure W4A4 GEMM approaches "
                   "the throughput of W4A16 Marlin-style kernels.",
            gap="Kernel-level W4A4 speed on server parts; precision is fixed, KV is not considered, and "
                "no 8GB device or prefill/decode split is measured.",
            qt="W+A", hw="GPU-unspecified", ra="no", ba="no"),
        dict(
            title="SAW-INT4: System-Aware 4-Bit KV-Cache Quantization for Real-World LLM Serving",
            year="2026", venue="unverified",
            url="https://arxiv.org/abs/2604.19157",
            method="Co-designs 4-bit KV-cache quantization with the serving scheduler, choosing KV "
                   "precision per request from system state rather than from accuracy alone.",
            gap="KV-side only under a serving objective; weight precision is taken as given, so the "
                "joint weight-vs-KV split under one 8GB budget is still not posed.",
            qt="KV", hw="GPU-unspecified", ra="no", ba="no"),
        dict(
            title="FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving",
            year="2025", venue="unverified",
            url="https://arxiv.org/abs/2501.01005",
            method="Provides composable attention kernels over paged KV layouts, with separate "
                   "specializations for prefill, decode and append phases.",
            gap="Optimizes attention per phase but keeps the KV dtype fixed and never allocates bits; "
                "no weight-side budget and no memory-constrained device study.",
            qt="unknown", hw="GPU-unspecified", ra="no", ba="no"),
        dict(
            title="AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration",
            year="2024", venue="MLSys 2024",
            url="https://arxiv.org/abs/2306.00978",
            method="Scales salient weight channels by activation magnitude before 4-bit rounding and "
                   "keeps activations in FP16, so W4A16 survives without mixed-precision storage.",
            gap="Weight-only W4A16 on-device: the KV cache is untouched and the per-channel scale "
                "search is not tied to a byte or bandwidth budget, so no split is available.",
            qt="W", hw="GPU-unspecified", ra="no", ba="no"),
    ],
    "area_gaps": [
        "Every kernel here is built for one fixed precision tuple (W4A4, W4A8KV4, W4A16, 1.58-bit); none exposes weight and KV bit-width as a serving-time or request-time decision, so the kernel layer cannot execute a re-allocation even if a solver produced one.",
        "INT4/FP4 tensor-core paths are silicon-specific: APEX4, MicroMix and LUT Tensor Core assume datapath support that Ada SM89 and Ampere SM87 (Jetson Orin) do not expose in the same form, and no paper in this set benchmarks an RTX 4060-class 8GB part.",
        "Paged-KV machinery (PagedAttention, FlashInfer) fixes the cache dtype at allocation time; a plan that moves precision between weights and KV mid-serving has no paged-kernel implementation to run on.",
        "Dequantization cost is reported as a ratio against FP16 or against a Marlin baseline, never as achieved bytes/s, so a bandwidth-bound edge decode cannot be predicted from these numbers.",
        "The accuracy-side W4A4 methods (LO-BCQ, MicroMix) and the systems-side kernels (QServe, Atom) are never evaluated together, so nobody reports the accuracy cost of the bit plan that the kernel actually makes fast.",
    ],
    "open_questions": [
        "On Ada SM89 with 8GB at 256 GB/s, does an INT4 W4A4 path actually beat W4A16 in decode, where the kernel is bandwidth-bound and dequantization is already hidden behind memory latency?",
        "Is there a kernel design in which the KV dtype and the weight dtype can be re-chosen per request without repacking weights — i.e. can the weight-vs-KV split be a serving-time decision at all?",
        "Do paged-block abstractions permit per-block bit-widths, and what does per-block precision metadata do to the block table's bandwidth cost on a 68 GB/s unified-memory part?",
    ],
}

A6 = {
    "area": "A6 — Edge / on-device LLM deployment and measurement — Jetson Orin Nano, "
            "prefill-decode regime characterization, energy per token, thermal throttling, "
            "edge compression surveys, llama.cpp/GGUF deployment studies",
    "saturation": "medium",
    "papers": [
        dict(
            title="Characterizing Prefill and Decode Regimes for Edge LLM Inference",
            year="unknown", venue="unverified",
            url="https://ieeexplore.ieee.org/document/11591742",
            method="Separates and characterizes the prefill and decode regimes of LLM inference on "
                   "edge hardware, reporting how each regime's cost behaves rather than assuming a "
                   "single inference cost model.",
            gap="A characterization study: it does not quantize, so it yields no weight-vs-KV "
                "allocation rule, and the device set behind its regime split is not resolvable from "
                "the record captured here.",
            qt="unknown", hw="unspecified", ra="yes", ba="no"),
        dict(
            title="ELLIE: Energy-Efficient LLM Inference at the Edge Via Prefill-Decode Splitting",
            year="unknown", venue="unverified",
            url="https://ieeexplore.ieee.org/document/11113611",
            method="Splits LLM inference into prefill and decode stages and schedules them across edge "
                   "resources to cut energy rather than to maximize throughput.",
            gap="Splits work between phases but not bits between weights and KV: precision is held "
                "fixed, so the regime reversal is never tested at the bit-allocation level.",
            qt="unknown", hw="unspecified", ra="yes", ba="no"),
        dict(
            title="A Measurement Study of LLM Inference Trade-offs Across Edge Continuum Hardware",
            year="2026", venue="unverified",
            url="https://arxiv.org/abs/2609.08307",
            method="Benchmarks identical LLM workloads across cloud, edge-server and device-class "
                   "hardware and reports latency, energy and quality trade-offs per tier.",
            gap="Cross-hardware measurement without an allocator: it compares existing precision "
                "configurations instead of choosing a weight-vs-KV split under a fixed budget.",
            qt="unknown", hw="GPU-unspecified", ra="no", ba="no"),
        dict(
            title="Sometimes Painful but Certainly Promising: Feasibility and Trade-offs of Language Model Inference at the Edge",
            year="2025", venue="unverified",
            url="https://arxiv.org/abs/2503.09114",
            method="Empirically assesses whether language-model inference is feasible on edge platforms, "
                   "quantifying latency, energy and quality trade-offs across model sizes and precisions.",
            gap="Feasibility framing with precision as a fixed configuration: no joint bit allocation "
                "and no prefill/decode-conditional objective.",
            qt="W", hw="unspecified", ra="no", ba="no"),
        dict(
            title="Does Mixture-of-Experts Actually Help Inference on Consumer and Edge Hardware? An Empirical Study",
            year="2026", venue="unverified",
            url="https://arxiv.org/abs/2606.21428",
            method="Measures MoE versus dense inference on consumer and edge devices, isolating memory "
                   "capacity and bandwidth effects from parameter-count effects.",
            gap="Architecture comparison at fixed precision: quantization bits are a control variable, "
                "not a decision variable, so no weight-vs-KV split is produced.",
            qt="unknown", hw="GPU-unspecified", ra="no", ba="no"),
        dict(
            title="Seeing at the Edge: Energy Characterization of Vision-Language Model Inference on Edge Platforms",
            year="2026", venue="unverified",
            url="https://arxiv.org/abs/2610.20136",
            method="Profiles energy and latency of vision-language model inference stage by stage on "
                   "edge platforms.",
            gap="Energy profiling for VLM pipelines; no quantization allocation, no KV budget and no "
                "LLM weight-vs-KV trade-off.",
            qt="unknown", hw="unspecified", ra="no", ba="no"),
        dict(
            title="EdgeProfiler: A Fast Profiling Framework for Lightweight LLMs on Edge Using Analytical Model",
            year="2025", venue="unverified",
            url="https://arxiv.org/abs/2506.09061",
            method="Builds an analytical performance model that predicts edge latency of lightweight "
                   "LLMs from memory and compute parameters instead of exhaustive measurement.",
            gap="Predicts latency but has no bit-width decision variable, so it cannot be used to "
                "search a weight-vs-KV allocation space.",
            qt="unknown", hw="unspecified", ra="no", ba="no"),
        dict(
            title="Empowering large language models to edge intelligence: A survey of edge efficient LLMs and techniques",
            year="2025", venue="Computer Science Review 2025",
            url="https://doi.org/10.1016/j.cosrev.2025.100755",
            method="Surveys edge-efficient LLM techniques spanning compression, quantization, "
                   "architecture adaptation and hardware co-design.",
            gap="Catalogues weight and KV quantization as separate families; the joint-budget question "
                "is absent from the survey's taxonomy, not just from its conclusions.",
            qt="unknown", hw="unspecified", ra="no", ba="no"),
        dict(
            title="MELTing Point: Mobile Evaluation of Language Transformers",
            year="2024", venue="MobiCom 2024",
            url="https://doi.org/10.1145/3636534.3690668",
            method="Measures end-to-end mobile inference of transformer language models, including "
                   "energy, thermal behaviour and latency under realistic phone conditions.",
            gap="Phone-class measurement: quantization is a fixed deployment choice, so no allocation "
                "policy is derived and no 8GB laptop-GPU or Orin comparison exists.",
            qt="unknown", hw="mobile-NPU", ra="no", ba="no"),
        dict(
            title="PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU",
            year="2024", venue="SOSP 2024",
            url="https://arxiv.org/abs/2312.12456",
            method="Exploits activation locality to split hot and cold neurons between a consumer GPU "
                   "and the CPU, so a large model runs on one desktop GPU.",
            gap="Solves placement, not precision: bit-widths are fixed and the KV cache is outside the "
                "budget, so it does not touch the weight-vs-KV split.",
            qt="unknown", hw="4090", ra="no", ba="no"),
        dict(
            title="LLMCad: Fast and Scalable On-device Large Language Model Inference",
            year="2023", venue="unverified",
            url="https://arxiv.org/abs/2309.04255",
            method="Combines a small draft model with on-device speculative verification and "
                   "memory-aware scheduling to accelerate on-device generation.",
            gap="System-level speedup with fixed precision; no quantization budget and no joint "
                "weight/KV allocation.",
            qt="unknown", hw="unspecified", ra="no", ba="no"),
        dict(
            title="Energy-Efficient Small Language Model Inference for Mobile Agents via DVFS",
            year="unknown", venue="unverified",
            url="https://doi.org/10.1145/3812836.3814753",
            method="Tunes dynamic voltage/frequency scaling for small-language-model inference on mobile "
                   "agents so that energy falls under a latency constraint.",
            gap="Frequency-level energy control with fixed bit-widths: it cannot ask whether moving "
                "bits from weights to KV is the cheaper energy trade.",
            qt="unknown", hw="mobile-NPU", ra="no", ba="no"),
        dict(
            title="Offline AI Modules: Voice-First Offline Architecture, Hardware Reference Stack, and Quantization & Benchmarking",
            year="2026", venue="unverified",
            url="https://arxiv.org/abs/2610.07026",
            method="Describes an offline voice-assistant stack and benchmarks quantized LLMs across "
                   "hardware tiers that include a Jetson Orin tier.",
            gap="Engineering report with per-tier benchmarks; precision is chosen per tier by hand and "
                "no joint weight-vs-KV allocation or regime-conditional rule is proposed.",
            qt="unknown", hw="Jetson", ra="no", ba="no"),
        dict(
            title="Sustainable LLM Inference for Edge AI: Evaluating Quantized LLMs for Energy Efficiency, Output Accuracy, and Inference Latency",
            year="2025", venue="unverified",
            url="https://arxiv.org/abs/2504.03360",
            method="Evaluates quantized LLMs on edge hardware for energy efficiency, output accuracy "
                   "and inference latency at several precisions.",
            gap="Compares off-the-shelf quantization levels on edge hardware; the bit budget is never "
                "split between weights and KV and prefill/decode are not separated in the objective.",
            qt="W", hw="unspecified", ra="no", ba="no"),
    ],
    "area_gaps": [
        "No paper in this set measures a joint weight-plus-KV bit allocation on an RTX 4060-class 8GB laptop GPU: the 4060 appears only in adjacent engineering write-ups and vendor benchmarking notes, never as the substrate of a controlled study.",
        "Jetson Orin appears once, as one benchmark tier in an engineering report (Offline AI Modules) with hand-chosen precisions, and never as the platform of an allocation experiment.",
        "Edge measurement papers report latency and energy at fixed precision levels (Q4_K_M, AWQ, QAT variants) and rarely report achieved bandwidth or the prefill/decode decomposition of cost, so their numbers cannot be turned into the cost model an allocator needs.",
        "Regime awareness at the edge is still at the characterization stage: the two IEEE papers that do split prefill from decode do not quantize, and the quantization papers do not split the phases.",
        "Thermal behaviour is measured for phones (MELTing Point, mobile DVFS work) but no paper reports whether a bit plan that is optimal in a cold benchmark stays optimal after an 8GB laptop GPU or Orin throttles under sustained decode.",
        "Surveys (Computer Science Review 2025) still present weight and KV quantization as separate families, which is why the joint-budget question is missing from the edge deployment literature rather than merely unanswered there.",
    ],
    "open_questions": [
        "Does the prefill/decode crossover on Orin Nano (~68 GB/s unified LPDDR5) occur at a different context length than on the RTX 4060 (256 GB/s discrete), and does that shift the accuracy-optimal bit split between the two devices?",
        "After ten minutes of sustained decoding, how far does thermal throttling move achieved bandwidth, and does the optimal split track throttled or nominal bandwidth?",
        "Is the framework/system overhead term rho materially larger on Orin than on the 4060, and does it consume exactly the headroom that reducing KV bits was supposed to create?",
    ],
}

A7 = {
    "area": "A7 — Evaluation methodology and reproducibility of quantization papers — unreported "
            "execution stack, benchmark validity, long-context benchmarks (RULER, InfiniteBench), "
            "statistical rigor, calibration-set sensitivity",
    "saturation": "medium",
    "papers": [
        dict(
            title="RULER: What's the Real Context Size of Your Long-Context Language Models?",
            year="2024", venue="COLM 2024",
            url="https://arxiv.org/abs/2404.06654",
            method="Generates synthetic long-context tasks with configurable length and complexity to "
                   "measure effective context size beyond needle-in-a-haystack retrieval.",
            gap="A benchmark with no precision axis: quantization studies that use it confound context "
                "length with cache error accumulation instead of separating them.",
            qt="unknown", hw="unspecified", ra="no", ba="no"),
        dict(
            title="∞Bench: Extending Long Context Evaluation Beyond 100K Tokens",
            year="2024", venue="unverified",
            url="https://arxiv.org/abs/2402.13718",
            method="Extends long-context evaluation past 100K tokens with tasks requiring reasoning over "
                   "the entire context rather than retrieval from it.",
            gap="Measures capability at one precision; no protocol reports the prefill-side and "
                "decode-side contributions to a quantized failure separately.",
            qt="unknown", hw="unspecified", ra="no", ba="no"),
        dict(
            title="LongBench v2: Towards Deeper Understanding and Reasoning on Realistic Long-context Multitasks",
            year="2025", venue="ACL 2025",
            url="https://aclanthology.org/2025.acl-long.183/",
            method="Builds a long-context benchmark of realistic, reasoning-heavy multitasks with "
                   "controlled input length and difficulty.",
            gap="Validity-focused benchmark with no precision variable; results are reported at one "
                "precision per model, so no KV/weight split can be read off it.",
            qt="unknown", hw="unspecified", ra="no", ba="no"),
        dict(
            title="Outliers and Calibration Sets have Diminishing Effect on Quantization of Modern LLMs",
            year="2024", venue="unverified",
            url="https://arxiv.org/abs/2405.20835",
            method="Ablates calibration-set choice and outlier handling across modern LLMs and finds "
                   "both matter far less than earlier quantization work assumed.",
            gap="Attacks the calibration axis of PTQ, not the allocation axis: no memory budget, no KV "
                "term, and no regime conditioning.",
            qt="W", hw="unspecified", ra="no", ba="no"),
        dict(
            title="Coverage-Based Calibration for Post-Training Quantization via Weighted Set Cover over Outlier Channels",
            year="2026", venue="unverified",
            url="https://arxiv.org/abs/2604.24008",
            method="Selects calibration samples by weighted set cover so that outlier channels are "
                   "covered under a bounded calibration budget.",
            gap="Improves calibration selection only; the quantizer's bit budget stays fixed and KV "
                "precision is untouched.",
            qt="W", hw="unspecified", ra="no", ba="no"),
        dict(
            title="ReproEvalCard: A Reporting Standard for Reproducible Evaluation of LLM Pipelines",
            year="2026", venue="ACL 2026 (Short)",
            url="https://aclanthology.org/2026.acl-short.22/",
            method="Proposes a reporting card that records the execution stack, data and evaluation "
                   "protocol needed to reproduce an LLM pipeline result.",
            gap="A pipeline reporting standard with no field for bit allocation, memory cap, achieved "
                "bandwidth or phase-separated throughput — exactly the fields quantization needs.",
            qt="unknown", hw="unspecified", ra="no", ba="no"),
        dict(
            title="An empirical study of LLaMA3 quantization: from LLMs to MLLMs",
            year="2024", venue="unverified",
            url="https://arxiv.org/abs/2404.14047",
            method="Evaluates ten PTQ and LoRA-finetuning methods on LLaMA3 from 1 to 8 bits, plus a "
                   "LLaVA-Next-8B multimodal model at 2-4 bits, and reports non-negligible "
                   "degradation in both linguistic and visual tasks at ultra-low bit-widths.",
            gap="An accuracy-only sweep at fixed bit-widths with no allocation policy, no KV precision "
                "control and no constrained-hardware measurement.",
            qt="W", hw="unspecified", ra="no", ba="no"),
        dict(
            title="Are LLM Benchmarks Already Contaminated? A Systematic Review of Contamination Detection Methods",
            year="2026", venue="GEM 2026",
            url="https://aclanthology.org/2026.gem-main.50/",
            method="Systematically reviews contamination-detection methods for LLM benchmarks and "
                   "assesses their validity.",
            gap="Explains why small benchmark deltas are unreliable, but never connects that to the "
                "size of the accuracy deltas that bit-allocation studies report.",
            qt="unknown", hw="unspecified", ra="no", ba="no"),
        dict(
            title="Accuracy is Not All You Need",
            year="2024", venue="NeurIPS 2024",
            url="https://proceedings.neurips.cc/paper_files/paper/2024/hash/e0e956681b04ac126679e8c7dd706b2e-Abstract.html",
            method="Shows that compression perturbs model behaviour such as safety and robustness even "
                   "when task accuracy is preserved, and argues for behaviour-level evaluation.",
            gap="Behaviour-level evaluation without a systems axis: no latency/bandwidth objective and "
                "no KV-versus-weight allocation question.",
            qt="unknown", hw="unspecified", ra="no", ba="no"),
        dict(
            title="Exploring the Trade-Offs: Quantization Methods, Task Difficulty, and Model Size in Large Language Models From Edge to Giant",
            year="2025", venue="IJCAI 2025",
            url="https://arxiv.org/abs/2409.11055",
            method="Evaluates instruction-tuned models from 1B to 405B under four quantization methods "
                   "across 13 datasets and finds that smaller models degrade sharply at 4-bit while "
                   "70B-scale models stay stable.",
            gap="Evaluation breadth without a systems axis: hardware is a fixed substrate and no "
                "allocation is optimized or even varied.",
            qt="W", hw="GPU-unspecified", ra="no", ba="no"),
        dict(
            title="Identifying and Mitigating Systemic Measurement Bias in Production LLM Inference Benchmarks",
            year="2026", venue="unverified",
            url="https://arxiv.org/abs/2605.24217",
            method="Identifies systematic biases in production inference benchmarking (warm-up, "
                   "caching, load mixing) and proposes controls for them.",
            gap="Serving-benchmark hygiene; it says nothing about quantization levels, memory caps or "
                "phase-separated measurement.",
            qt="unknown", hw="GPU-unspecified", ra="no", ba="no"),
        dict(
            title="AutoTuneBench: Trustworthy Measurement for Agent Auto-Tuning of LLM Serving Engines",
            year="2026", venue="unverified",
            url="https://arxiv.org/abs/2609.18123",
            method="Defines a measurement protocol for auto-tuning LLM serving engines that separates "
                   "run-to-run noise from real configuration gains.",
            gap="Engine-tuning measurement rather than precision allocation; no weight/KV bit decision "
                "and no edge device class.",
            qt="unknown", hw="GPU-unspecified", ra="no", ba="no"),
    ],
    "area_gaps": [
        "Execution stack is systematically unreported: none of the quantized-LLM papers in this corpus states CUDA/cuDNN/driver versions, kernel commit, or clock policy, which is exactly what the measurement-bias and reproducibility work says makes their throughput numbers non-comparable.",
        "Phase-mixed benchmarking is the norm: accuracy is measured in prefill-like teacher-forced settings while speed is measured in decode, and the long-context benchmarks (RULER, InfiniteBench, LongBench v2) carry no precision axis at all.",
        "Statistical rigor is thin: single-run accuracies without seeds, confidence intervals or paired tests, while the contamination literature shows the deltas being reported often sit inside benchmark noise.",
        "Calibration sensitivity is studied as an accuracy question (diminishing outlier effects, set-cover calibration) but never as a budget question — nobody reports how the accuracy-optimal weight/KV split moves with the calibration set.",
        "Reporting standards (ReproEvalCard) cover pipelines but not quantization experiments: there is no mandated field for memory cap, achieved bandwidth, KV precision or phase-separated throughput.",
    ],
    "open_questions": [
        "If every paper were re-run on one documented stack with at least three seeds and paired tests, how many reported quantization gains would survive?",
        "What is the minimal reporting checklist that lets a reader reconstruct a weight-vs-KV allocation experiment, and would adopting it have changed any published conclusion?",
        "How much of the reported accuracy gap between two bit-widths is calibration-set variance rather than the bit-width effect, on 1-4B models?",
    ],
}

A8 = {
    "area": "A8 — Degradation of long-chain reasoning under quantization — GSM8K/MATH/CoT "
            "under 3-4 bit, error amplification over autoregressive steps, weight-vs-KV error "
            "attribution",
    "saturation": "high",
    "papers": [
        dict(
            title="Quantization Hurts Reasoning? An Empirical Study on Quantized Reasoning Models",
            year="2025", venue="unverified",
            url="https://arxiv.org/abs/2504.04823",
            method="Large-scale empirical study of quantized reasoning models showing drops that grow "
                   "with reasoning depth and are hidden by aggregate benchmark averages.",
            gap="Documents degradation but proposes no allocation rule: it never asks whether moving "
                "bits from weights to KV would shrink the drop, and never measures 8GB hardware.",
            qt="W", hw="GPU-unspecified", ra="no", ba="no"),
        dict(
            title="Quantization Meets Reasoning: Exploring and Mitigating Degradation of Low-Bit LLMs in Mathematical Reasoning",
            year="2025", venue="unverified",
            url="https://arxiv.org/abs/2505.11574",
            method="Aligns chain-of-thought steps to locate the first faulty step caused by PTQ, then "
                   "restores it with a small curated supervised/preference fine-tune.",
            gap="Repairs reasoning damage by fine-tuning at a fixed low-bit weight setting: KV precision "
                "is untouched and the repair is not budget-aware.",
            qt="W", hw="GPU-unspecified", ra="no", ba="no"),
        dict(
            title="RECAP: Calibrating Low-Bit Reasoning Models on the History They Read",
            year="2026", venue="unverified",
            url="https://arxiv.org/abs/2610.15259",
            method="Calibrates low-bit reasoning models on their own generated reasoning histories to "
                   "reduce drift accumulated during long chains.",
            gap="Calibration-side repair for weight quantization; no KV bit budget and no "
                "regime-conditional allocation.",
            qt="W", hw="GPU-unspecified", ra="no", ba="no"),
        dict(
            title="RATIO: Reasoning Analysis and Token-level Inference Optimization for Quantized Reasoning Models",
            year="2026", venue="unverified",
            url="https://arxiv.org/abs/2609.39801",
            method="Analyzes quantized reasoning traces and optimizes token-level inference so accuracy "
                   "is recovered within a bounded generation budget.",
            gap="Optimizes tokens under a compute budget, not bits under a memory budget: no "
                "weight-vs-KV split is available.",
            qt="unknown", hw="GPU-unspecified", ra="no", ba="no"),
        dict(
            title="AdaKVQ: Adaptive Mixed-Precision KV Cache Quantization For Efficient Reasoning Models",
            year="2026", venue="NeurIPS 2026",
            url="https://neurips.cc/virtual/2026/loc/sydney/poster/153261",
            method="Adapts KV-cache precision per head and per token for reasoning models so long "
                   "chains keep critical state at higher precision.",
            gap="KV-side adaptive precision for reasoning, but weights are fixed and the trade-off is "
                "never expressed against one 8GB budget or a prefill/decode objective.",
            qt="KV", hw="GPU-unspecified", ra="no", ba="no"),
        dict(
            title="LFQ: Logit-aware Final-block Quantization for Boosting the Generation Quality of Low-Bit Quantized LLMs",
            year="2026", venue="ICML 2026",
            url="https://icml.cc/virtual/2026/poster/60586",
            method="Quantizes only the final block with a logit-aware objective and leaves earlier "
                   "blocks at higher precision.",
            gap="Places precision by output sensitivity without any byte or latency model, and does not "
                "allocate across KV at all.",
            qt="W", hw="GPU-unspecified", ra="no", ba="no"),
        dict(
            title="ReQAT: Achieving Full-Precision Reasoning Accuracy with 4-bit Floating-Point Quantization-Aware Training",
            year="2026", venue="ICML 2026",
            url="https://icml.cc/virtual/2026/oral/71113",
            method="Uses 4-bit floating-point quantization-aware training to close the reasoning gap to "
                   "full precision.",
            gap="Requires QAT and fixes one 4-bit FP format at training time; no KV allocation and no "
                "inference-time budget adaptation.",
            qt="W+A", hw="GPU-unspecified", ra="no", ba="no"),
        dict(
            title="Balancing Compute in LLM Inference: Model Selection, Quantization, and Test-Time Scaling",
            year="2026", venue="EuroMLSys 2026",
            url="https://dl.acm.org/doi/10.1145/3805621.3807631",
            method="Formulates a joint choice of model size, quantization level and test-time compute "
                   "under a fixed compute budget.",
            gap="Optimizes a compute budget over model/quantization/test-time choices, not a memory-bit "
                "budget split between weights and KV, and does not separate prefill from decode.",
            qt="W", hw="GPU-unspecified", ra="no", ba="no"),
        dict(
            title="Not All Bits Are Equal: Scale-Dependent Memory Optimization Strategies for Reasoning Models",
            year="2025", venue="unverified",
            url="https://arxiv.org/abs/2510.10964",
            method="Shows that memory-optimal precision placement for reasoning models depends on model "
                   "scale and derives scale-dependent placement strategies.",
            gap="Closest budget-aware allocation work in this sub-area, but the budget is per-model "
                "memory without a prefill/decode term and without an 8GB edge measurement.",
            qt="W", hw="GPU-unspecified", ra="no", ba="yes"),
        dict(
            title="Quantization Inflates Reasoning: Token Inflation as a Hidden Cost of Low-Bit Reasoning Models",
            year="2026", venue="unverified",
            url="https://arxiv.org/abs/2606.25519",
            method="Shows low-bit models emit longer reasoning traces, so accuracy per token and "
                   "end-to-end cost both degrade even when final accuracy holds.",
            gap="Identifies a cost-side effect of quantization but never allocates bits, and the inflation"
                "is not charged against a KV memory budget.",
            qt="W", hw="GPU-unspecified", ra="no", ba="no"),
        dict(
            title="Integration of Chain-of-Thought and Program-of-Thought on 4-bit Quantized Small Language Models for Financial Question Answering",
            year="2025", venue="unverified",
            url="https://ieeexplore.ieee.org/document/11664865",
            method="Combines chain-of-thought and program-of-thought prompting for 4-bit small language "
                   "models on financial question answering.",
            gap="Prompting strategy at a fixed 4-bit setting: no allocation, no KV precision control and "
                "no edge-hardware measurement.",
            qt="W", hw="unspecified", ra="no", ba="no"),
        dict(
            title="BitNet b1.58 2B4T Technical Report",
            year="2025", venue="unverified",
            url="https://arxiv.org/abs/2504.12285",
            method="Trains a 2B ternary-weight, 4-bit-activation model end to end and releases weights "
                   "with integer-only inference kernels.",
            gap="Weights are 1.58-bit by construction, so there is no budget left to reallocate and no "
                "KV-cache precision policy is offered.",
            qt="W+A", hw="CPU", ra="no", ba="no"),
    ],
    "area_gaps": [
        "Reasoning degradation is measured at fixed bit-widths and almost always on weight-only quantization; no paper asks whether the same byte budget spent on KV precision would shrink the damage.",
        "Phase attribution is missing: reasoning failures accumulate over decode steps, yet the protocols report one final accuracy, so the decode-side contribution cannot be separated from prefill-side damage.",
        "The scale mismatch is stark — reasoning studies use 7B-70B models that do not fit an 8GB device at usable context, while the 1-4B models that do fit are barely evaluated on GSM8K/MATH.",
        "Mitigations are training- or calibration-side (re-tuning, QAT, history calibration) and cost compute the edge cannot pay; none of them is a bit-allocation policy.",
        "Token inflation is a first-class cost of low-bit reasoning, but no study charges the extra generated tokens against the KV memory budget, even though longer chains raise KV pressure exactly when precision is lowest.",
    ],
    "open_questions": [
        "Does a fixed 8GB budget produce a different accuracy-optimal split on GSM8K/MATH than on perplexity — i.e. is reasoning the metric that moves the optimum?",
        "Is the earliest-failure-step phenomenon attributable to weight error, KV error, or their interaction, and can it be attributed by holding one side at FP16?",
        "If quantized models emit materially more reasoning tokens, does the KV-bits term dominate the weight-bits term at 8K+ context on Orin, and by how much?",
    ],
}
