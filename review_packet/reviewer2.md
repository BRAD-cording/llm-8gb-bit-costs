# Reviewer 2 report

## Manuscript
"The Cache Is Cheaper Than the Weights: Measured Bit Costs and Device-Dependent Capacity for LLM Inference on 8 GB Devices"
Pre-submission self-review, IEEE Internet of Things Journal.
Reviewer emphasis: ORIGINALITY AND SIGNIFICANCE.

## Overall estimate
Marginal to modest. The measured per-bit exchange rate and the treatment of the weight/cache split as a decision variable under a measured device budget are genuinely new against the closest cited work (KV Pareto, EACL 2026 Industry, arXiv:2512.01953), and the executed-policy result (32k context fits a 4B model only when the cache is packed) is concrete and useful for an edge systems reader. However the highest-significance headline, the device-dependent capacity ratio between the two devices, is carried by analytic capacity whose top number is an admitted upper bound contradicted by direct measurement on the very "edge" device that motivates the claim, and the Related Work does not cite the on-device unified-memory and NPU serving work where the device-dependence story differs the most. I would not expect acceptance without meaningful revision to the significance story.
Estimate: low-acceptance borderline (on an accept-neutral scale, sits between major revision and reject for the stated significance claims). Confidence: moderate. The novelty of the exchange-rate measurement and the rho mechanism is well supported, but the significance claims it props up are only partly backed by numbers a device can actually reach.

---

## Major Concerns

### R2-M1
**Claim pointer**: The abstract asserts "Both are wrong" about a literature that "optimises for speed and spends bits on the weights", presenting "the cache is cheaper than the weights" as the central reversal.
**Evidence pointer**: Abstract (manuscript lines 20-28); Related Work Section II-B, which itself describes a "second line compresses the cache [20]-[26]".
**Why it matters**: The KV cache quantisation line the paper itself cites (KVQuant, KIVI, ZipCache, SKVQ, RotateKV, MixKVQ) exists precisely because low-bit or compressed caches are already cheaper and are the compressible side. A reviewer who knows that line will read "cache is cheaper than weights" as the premise of the whole KV-compression literature, not a reversal. The paper later narrows correctly to "our claim is narrower and different" (Section I contribution 4 and end of Section II), namely a measured per-bit exchange rate, the split as a decision variable, and rho as a decision input. The originality and significance case should be led by that narrower, defensible claim and not by a headline that the cited KV work already established.
**Blocking**: No.

### R2-M2
**Claim pointer**: Contribution 3 and Table IV state that capacity is device-dependent through the usable-memory fraction, with the laptop/Jetson max-context ratio 1.63-1.82x across accuracy budgets and rho 0.133 vs 0.419 moving the price of a weight bit by 3.7x.
**Evidence pointer**: Section VI-C (Table IV); Section VIII limitation (4), which admits the Jetson column is an upper bound and that direct measurement puts the 0.6B ceiling near 3k tokens with a 4k prefill aborting the allocator.
**Why it matters**: This is the central significance claim for an edge audience, yet the edge-module side of the headline ratio is capacity the device cannot reach in direct measurement. The device-dependence mechanism through rho is itself measured and credible, but the significance headline is carried by analytic numbers contradicted by the hardware. An IoT/edge reader cannot act on an upper bound that the device aborts at, so the practical significance on the very platform that motivates the work is unproven and needs either a demonstrated working configuration or a re-framing of how much of the device-ratio claim is real.
**Blocking**: No (the discrepancy is disclosed in Limitations and the core rho mechanism stands), but this must be fixed in revision or the significance pitch materially weakens.

### R2-M3
**Claim pointer**: Related Work Section II asserts "None of these treats the weight/KV split as a decision variable under a measured device budget, and none reports the usable-memory fraction of the target device".
**Evidence pointer**: Related Work Section II-C; retrieved competing lines absent from the citations, including Fast On-device LLM Inference with NPUs (arXiv:2407.05858), P3-LLM NPU-PIM (ISCA 2026, 10.1109/isca66397.2026.00055), and Rethinking Unified Memory for NPU-PIM (arXiv:2608.06989).
**Why it matters**: The paper's entire device-dependence argument turns on the difference between a discrete-GPU laptop and a unified-memory edge module (rho 0.133 vs 0.419). Unified-memory NPU and mobile on-device LLM serving is precisely where the capacity economics and the weight/cache trade differ most, yet the Related Work cites none of that line. A reviewer working on mobile or unified-memory on-device LLM inference will expect these and may read the survey as incomplete precisely where the thesis is strongest.
**Blocking**: No.

---

## Minor Comments

### R2-m1
**Claim pointer**: The two targets are chosen because "they bracket the two memory architectures an edge deployment realistically uses" (Section V).
**Evidence pointer**: Section V device paragraph (manuscript line 231).
**Why it matters**: The devices are framed as edge platforms, not as IoT systems, and no concrete IoT workload, power, battery, or latency constraint is connected to the sizing decision. For an IEEE IoT-J readership the significance would be more concrete with one worked IoT scenario showing how rho and the exchange rate change a config. Not blocking.

### R2-m2
**Claim pointer**: The paper ends with the recommendation "compress the cache and leave the weights alone" on the studied stack (Section I contribution 4).
**Evidence pointer**: Section I; Section VIII limitation (6) (no fused quantised-attention kernel, conclusions could change).
**Why it matters**: The actionability a practitioner takes away depends on the stack. A one-paragraph worked example translating rho and the per-bit exchange rate into a concrete device-sizing or allocation recommendation on an off-the-shelf 8 GB edge board would sharpen the significance for a non-specialist IoT reader. Not blocking.

### R2-m3
**Claim pointer**: The Related Work framing and the abstract contributions read as a single interleaved column in the reviewed frozen text.
**Evidence pointer**: manuscript.txt lines 22-36 and 101-126 show two passage streams (abstract contributions and Related Work B) interleaved on the same lines.
**Why it matters**: This is a readability defect in the reviewed source that makes it hard to attribute claims to abstract versus Related Work. If it persists in the compiled PDF it will obscure the careful originality positioning the paper otherwise does. Not blocking.

---

## Reviewer 2 summary statement
Originality: the measured per-bit exchange rate, the weight/cache split as a decision variable under a measured device budget, and rho as a measurable decision input are new against the closest cited work. Significance: real and concrete for capacity sizing (32k context fitting a 4B model only with packed cache), but diluted by (a) an edge-module capacity column that is an upper bound the device cannot reach, and (b) a Related Work that omits the unified-memory and on-device NPU serving line where the device-dependence claim is most relevant. The paper's self-positioning is honest and mostly accurate; lead with the narrower measurable-claim framing rather than the "cache cheaper than weights" headline.