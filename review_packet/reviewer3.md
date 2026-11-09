# Reviewer 3 report (Framing, Reproducibility, Readability)

Manuscript: The Cache Is Cheaper Than the Weights. Measured Bit Costs and Device-Dependent Capacity for LLM Inference on 8 GB Devices.
Authors: Yibo Zhang, Muhammad Syafiq bin Mohd Pozi, Osman B Ghazali (Universiti Utara Malaysia).
Target: IEEE Internet of Things Journal. Frozen text revision e67c6366.

## Estimate and confidence

Estimate. The two devices and the measurements are real and the argument is internally careful, but for this journal the framing and the reproducibility story are the two make-or-break items, and both currently fall short of what an IoT-J reviewer or a deploying engineer needs. The manuscript never names the Internet of Things and never grounds either device in a concrete on-device deployment, and the much-repeated one command rebuild is asserted but never actually described in the paper text. On readability the paper is dense and honest, and the executed-policy caveat is present and correct in the body, though the abstract and index terms do not carry that honesty into the summary. Confidence is high. Every finding below is grounded in the frozen manuscript text, the anchors file, and the local source and artifact directory; no number or fact is invented.

## Major Concerns

### R3-M1 (Blocking Yes)
Claim pointer. The introduction and device section present the RTX 4060 Laptop GPU and the Jetson Orin Nano Super as bracketing the two memory architectures an edge deployment realistically uses, and the index terms advertise edge inference and energy efficiency. Evidence pointer. A search of the full frozen text finds no occurrence of IoT or Internet of Things anywhere, the index terms list only large language models, post-training quantisation, KV cache, edge inference, memory budget, bit allocation, energy efficiency, and the only concrete deployment context named anywhere is on-device assistants (line 46) and a deployment engineer (line 205). Why it matters. Journal criterion 1 requires the work to connect to real IoT systems, and no reviewer or reader can map either platform to an on-device role such as a gateway, an embedded controller, or a power- and memory-constrained endpoint, so the device choice reads as two compute platforms rather than as an IoT deployment story.

### R3-M2 (Blocking Yes)
Claim pointer. Section VII states that a single command rebuilds the figures, the summary CSV, the allocator inputs, and the manuscript, and runs the checks below, and the Data Availability statement repeats that a single command rebuilds the results and runs the checks described in Section VII. Evidence pointer. A search of the frozen manuscript finds only the phrase single command with no command string, no entry point, and no pinned snapshot anywhere in the text, while the actual command python experiments/run_all.py and the v0.1-submission tag live only in the repository README and docs/ARTIFACT.md, which are not part of the paper. Why it matters. The reproducibility criterion asks whether there is a deterministic path from the claimed repository to the numbers and whether a one command rebuild is described, and the paper asserts that path without describing it, so a reviewer or practitioner acting on the paper text alone cannot execute the rebuild.

## Minor Comments

### R3-m1 (never Blocking)
Claim pointer. The executed-policy section is presented with an honesty boundary that the cache scheme is llama.cpp per-32-block packing and not the grouped quantiser of reference [20]. Evidence pointer. Section VI-D, Four boundaries, states this plainly and separately says no claim rests on the packed accuracy. Why it matters. The body is exemplary here, but the abstract and contribution 4 attach the headline makes a 32k context fit at all to the executed policy without carrying the quantiser-family mismatch into the summary, so a reader who scans only the abstract can conflate the executed 4B packing numbers with the exchange rate measured under different quantisers. The fix is to propagate the caveat, not to weaken the claim.

### R3-m2 (never Blocking)
Claim pointer. The single-command rebuild is claimed to reproduce the results in full. Evidence pointer. Section VII discloses only in prose that the package indices are those reachable from the authors network, and the supporting docs record ModelScope and a Chinese mirror because Hugging Face is not reachable. Why it matters. A practitioner whose network reaches Hugging Face but not the used mirror or ModelScope cannot deterministically reproduce, and neither the manuscript text nor the Data Availability statement tells them that constraint, which is exactly the environment dependence the reproducibility section elsewhere handles so well.

### R3-m3 (never Blocking)
Claim pointer. The repository is the authoritative source behind the single-command reproducibility claim. Evidence pointer. The root README describes the rebuild as a 17 stage pipeline while docs/ARTIFACT.md says a single command runs sixteen stages, so the authoritative supporting documentation disagrees with itself about the build. Why it matters. The close the loop on the one command story, the supporting artifact that must back it should agree internally on how many stages run, and it currently does not.

### R3-m4 (never Blocking)
Claim pointer. The paper states the recommended policy is executed on a shipped runtime. Evidence pointer. The executed-policy table (Table VI on the laptop) reports only the 1.7B and 4B rows and no 0.6B row, while the headline dominated-choice claim on 0.6B at 4k (10.84% worse perplexity, 19% slower per decode step, 644 MB) is read from Table V, whose cache-compressed rows are simulated. Why it matters. A systems engineer who wants to act on the answer for the smallest deployable model cannot see an actually executed packed configuration for the model on which the exchange-rate thesis is strongest, and the text does not state that the 0.6B executed row is absent.

## Verdict on the preassigned emphasis

No further concerns beyond the two majors and four minors above. On balance the honesty of the manuscript is a genuine strength, especially the executed-policy caveat and the falsified-hypothesis appendix, and the weaknesses are concentrated in the journal match (IoT framing) and in the reproducibility description staying declarative rather than concrete. Both are fixable in revision and neither reflects a defect in the measurements themselves.

I state no editorial decision and make no comparison with other reviewers.