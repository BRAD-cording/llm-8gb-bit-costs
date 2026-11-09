# Reviewer 1 Report. Technical Soundness

Manuscript. "The Cache Is Cheaper Than the Weights: Measured Bit Costs and Device-Dependent Capacity for LLM Inference on 8 GB Devices" (Yibo Zhang, Muhammad Syafiq bin Mohd Pozi, Osman B Ghazali, Universiti Utara Malaysia). Target IEEE Internet of Things Journal.

Reviewer emphasis. TECHNICAL SOUNDNESS. Every numeric claim in the tables, figures and prose was checked against the frozen manuscript, the anchors file, and the local source and results directory. No numeric claim was invented for this review. All numbers quoted below are verbatim from one of those sources.

## Estimate and Confidence

Recommendation. Major Revision on the technical-soundness axis.

Confidence. High. Every headline figure (exchange rate 9.0x and 13.3x, rho 0.133 against 0.419 with 3.15x, 1.6x budget, 3.7x weight-bit price, GQA 32x and 5.8x, passkey 5/5 against 0/5, the 4B/32k executed 9033 MiB against 6873 MiB row) traces exactly to a local result file or JSON, and the internal arithmetic is consistent between the abstract, tables and prose. No example of a fabricated or untraceable headline number was found. The manuscript is unusually transparent about its measurement boundaries, and that transparency is the reason the two Major concerns are non-blocking.

## Strengths on Technical Soundness

S1. Every headline number is traceable to a raw artifact.
The 9.0x and 13.3x exchange rate and the 11.39% against 1.27% and 3.20% against 0.24% figures match KIVI_OFFICIAL.md exactly (fp16 NLL 0.6B 3.851023, GPTQ +11.39%, KIVI official +1.27%, ratio 9.0x. 1.7B fp16 3.643825, GPTQ +3.20%, KIVI +0.24%, ratio 13.3x). The pair is a symmetric published-implementation comparison, as the body states, not a self-authored kernel.

S2. The rho story is measured, not fitted, and the arithmetic is correct.
RHO_FINDINGS.md measures rho 0.133 (RTX 4060 usable 6.93 GiB of nominal 8188 MiB) and 0.419 (Jetson usable 4.32 GiB of nominal 7619 MiB) on 2026-10-20. 0.419 divided by 0.133 is 3.15. The usable budget ratio (1 minus rho) times nominal memory gives 7099 MiB against 4427 MiB, which is 1.60x. The 3.7x weight-bit price is the corrected feasible-region comparison x5.76 against x1.57 (21575 to 33895 against 2587 to 14907), which the results file explicitly records as corrected from an earlier 3.5x.

S3. The honesty caveats required by the anchors file are honored in the frozen text.
The executed-policy claim is worded as "a shipped runtime that stores the cache packed" over llama.cpp per-32-block KV quantisation, and it never asserts that KIVI's kernel ran. The Jetson Table VII caption states that prefill on that device varies up to 20% between runs with decode inside 5% and that no claim rests on prefill, which matches C2_JETSON.md (run-internal stddev up to 315 for a 1208 mean, cross-run 1208.8 against 1009.2, decode stable within 5%).

S4. The GQA finding and its numbers are verified.
Table I gives 2693 MB for the math backend against 84 MB for the fused memory-efficient path, ratio 32.06x. Table II gives 2062.5 ms against 355.9 ms for the 4096-token 0.6B prefill, ratio 5.80x. The abstract's 32x and 5.8x match the body.

S5. The passkey and executed-policy claims are verified.
PASSKEY_EVAL.md records the 2-bit followup as fp16 5/5, naive RTN 4-bit 5/5, KIVI 4-bit 5/5, KIVI 2-bit 5/5, naive RTN 2-bit 0/5, matching Table X. C2_LLAMACPP.md records the 4B/32k row as fp16 allocation 9033 MiB not completing (terminated at 421 s after 20 min pinned near 7840 MiB) against q8_0 6873 MiB completing in 67 s with decode 52.6 tokens/s, and a 4-bit cache perplexity of +0.99% for 4B against +26.15% for 1.7B, supporting the 26x family claim.

S6. The capacity table is traceable.
Table IV cells (111,184 over 67,324 at 1.65x, 122,404 over 73,444 at 1.67x, 176,464 over 108,124 at 1.63x, 221,344 over 121,384 at 1.82x, 243,784 over 143,824 at 1.70x) appear byte-for-byte in results/capacity_table.json, which also carries the weight and KV bit means and the budgets and a ws_token of 0.142 matching the manuscript workspace coefficient. The ratio column is the exact quotient of the two max-L cells.

## Major Concerns

### R1-M1  Abstract exchange-rate magnitude rests on the favorable calibration point.
Claim pointer. The abstract reports the specific 9.0x and 13.3x ratios and the 11.39% against 1.27% and 3.20% against 0.24% perplexity deltas as the headline result.
Evidence pointer. Section VI-E and Limitations 2 report that increasing GPTQ calibration from 2 to 16 texts moves the weight cost to 13.0% on 0.6B and 5.96% on 1.7B, which changes the 0.6B ratio to about 10.2x and the 1.7B ratio to about 24.8x. The cache values (1.27% and 0.24%) are stable.
Why it matters. The headline magnitude is a single calibration setting whose value moves by a factor of around two on one model, and only the favourable point appears in the abstract while the range lives in the limitations section. The ordering direction is robust, so this does not overturn the thesis, but a journal reader who reads only the abstract may take a calibration-dependent magnitude as a fixed constant.
Blocking. No.

### R1-M2  The device-dependence thesis is supported by an analytic upper bound that direct measurement contradicts.
Claim pointer. The abstract and Section III assert the 3.7x price of a weight bit and the device-dependent capacity driven by rho, and Table IV presents the Jetson context ladder.
Evidence pointer. Table IV and Sections VI-C and Limitations 4 state that the Jetson column is an upper bound derived from measured rho plus a workspace curve, and that direct measurement puts the 0.6B ceiling near 3000 tokens with a 4096-token abort (NVML error 12) where the model predicts 67k. QWEN3_4B_8GB_LIMIT.md confirms the analytic 34k figure is itself stale by a factor of about 3.4 against the measured workspace.
Why it matters. The central device-difference ratio and the price of a weight bit are computed in a regime that the Jetson cannot actually reach, so the quantitative strength of the capacity-difference claim is not demonstrated on deployable hardware even though the qualitative claim is. The manuscript is fully transparent about this, which is why it is not blocking, but a technical-soundness reviewer should ask the authors to state in the abstract or body how much of the device-difference claim is extrapolation rather than measurement.
Blocking. No.

## Minor Comments

### R1-m1  Auxiliary capacity artifacts disagree with the backing JSON and with each other.
Claim pointer. Section VII states that every table is generated from a measurement file rather than transcribed.
Evidence pointer. results/CAPACITY_TABLE_V4.md and CAPACITY_TABLE_V3.md report eps 0.01 max L of 105,064 over 55,594 at 1.89x and 107,104 over 54,064 at 1.98x respectively, both differ from the manuscript and from the backing capacity_table.json (111,184 over 67,324 at 1.65x), even where V4 cites the same items file and bucket count.
Why it matters. A reproducer following the .md files will not reach the manuscript's numbers. Only capacity_table.json matches. The .md files look like earlier runs and should be superseded or annotated so the results directory does not contradict the paper.
Blocking. No.

### R1-m2  The cache-only 3.2x byte reduction and the realized total-footprint reductions are not reconciled in one place.
Claim pointer. Section VI-C reports a 3.2x reduction for packed cache bytes (902 to 282 MB at 8192, 3611 to 1129 MB at 32768).
Evidence pointer. Table VI realized allocations for the same configurations are much smaller (9033 to 6873 MiB at roughly 1.31x for q8_0, 9033 to 5721 MiB at roughly 1.58x for q4_0) because weights dominate the footprint.
Why it matters. A reader can substitute the larger cache-only reduction for the total memory saving. The manuscript's accounting is honest, but stating both magnitudes in one sentence would prevent the over-read.
Blocking. No.

### R1-m3  One-per-model cells underpin the 26x family claim.
Claim pointer. Contribution 4 reports that the same 4-bit quantiser costs 26x more accuracy on the 1.7B model than on the 4B model.
Evidence pointer. C2_LLAMACPP.md carries +26.15% for 1.7B against +0.99% for 4B at 4096 tokens, and its own boundary note flags the q4_0 1.7B anomaly as worth reproducing from a single measurement.
Why it matters. The family claim rests on one measurement per model. A small reproducibility note in the text acknowledging the single-cell basis would preempt a reviewer challenge.
Blocking. No.

## No Further Concerns

All other numeric claims checked (Table III latency ordering, Table IV ratios, Table VII Jetson decode, Table VIII and XI perplexity with bootstrap CIs, the 26x, the workspace coefficients 0.142 plus 9 and 0.217 plus 161, the noise floor 0.12) are internally consistent and traceable. No blocking numeric defect was found on the technical-soundness axis.