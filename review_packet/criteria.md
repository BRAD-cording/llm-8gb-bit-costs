# Common journal criteria (IEEE IoT-J), applied by every reviewer

All three reviewers evaluate the same immutable manuscript (review_packet/manuscript.txt) against the SAME journal-level criteria below, plus their own preassigned emphasis. Each returns ONE freeze-dried report before any comparison.

## IEEE Internet of Things Journal editorial criteria
1. Relevance to IoT: the work must connect to real Internet-of-Things systems. Edge/embedded/on-device LLM inference on power-constrained 8 GB hardware is squarely in scope IF framed against deployable edge controllers, gateways, or embedded devices (not solely a desktop GPU laptop). Note whether the manuscript frames its two devices (discrete-GPU laptop, Jetson Orin Nano Super) as IoT-relevant edge platforms.
2. Novelty / contribution: what is new versus the cited allocation and KV-cache literature, stated precisely.
3. Technical soundness: are the measurements, quantities (perplexity, memory, exchange rate rho), statistics, and reproducibility claims correct and internally consistent.
4. Significance / usefulness: does the result change how someone sizes or configures an edge LLM, or how a practitioner allocates bits.
5. Clarity and completeness: is the argument legible to an IoT/systems audience; is every claim backed by a table/figure/appendix pointer.
6. Reproducibility and data availability: is there a deterministic path from the repo to the numbers.

## Review-report contract (each reviewer)
- Estimate and confidence first.
- Major Concerns: each with a stable ID (R<x>-M<n>), a one-sentence claim pointer (what the paper asserts), a one-sentence evidence pointer (the table/figure/section it cites), why it matters, and Blocking Yes/No. No concern quota: state 'no further concerns' if none.
- Minor Comments: same shape but never Blocking.
- Grounded ONLY in the immutable manuscript, the anchors file, and (where noted) the local source directory. Do not invent new facts.
- Punctuation: avoid em/en dashes and colons as routine prose punctuation.
- Do not present as an editorial decision. Do not compare with other reviewers. Do not claim acceptance/rejection.

## Files each reviewer may read
- review_packet/manuscript.txt (the frozen 13-page text)
- review_packet/anchors.md (stable anchors and key claims)
- Main LaTeX source C:/Users/HP/OneDrive/Desktop/Dev/harness/llm_8gb_bit_costs/paper/main.tex (immutable at commit 79ee170)
- Local data/results in C:/Users/HP/OneDrive/Desktop/Dev/harness/llm_8gb_bit_costs/results/ and experiments/ for checking numbers (read-only).
- The publication is NOT yet made there is no editor decision this is a pre-submission self-review.

## Emphasis per reviewer (blind, disjoint)
- Reviewer 1: Technical soundness. Verify every numeric claim traces to a table/figure; check statistics, the rho measurement, perplexity deltas, and the executed-policy honesty caveat.
- Reviewer 2: Originality and significance. Judge whether the 'allocation equals f(objective, exchange rate, budget), each measured, each reverses prior assumptions' thesis is genuinely novel and useful for edge/IoT practitioners; probe the Related Work for gaps.
- Reviewer 3: Framing, reproducibility, and readability for an IoT/systems audience. Check the IoT relevance framing, reproducibility path, Limitations honesty, and whether a systems engineer can act on the paper.