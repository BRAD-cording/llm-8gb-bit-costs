# Manuscript anchors (stable, from v3.1-jetson)

Title (3-line):
The Cache Is Cheaper Than the Weights:
Measured Bit Costs and Device-Dependent Capacity
for LLM Inference on 8 GB Devices

Authors: Yibo Zhang (corresponding), Muhammad Syafiq bin Mohd Pozi, Osman B Ghazali.
Affiliation: School of Computing, Universiti Utara Malaysia (UUM), Sintok, Kedah, Malaysia.
Target venue: IEEE Internet of Things Journal (IoT-J). Document class IEEEtran V1.8b journal, double column, 13 pages.

## Sections (line anchor -> label)
- Introduction: sec:intro
- Related Work: sec:related
- Usable Memory and the Weight-Bit Exchange Rate: sec:rho
- Method: sec:method
- Experimental Setup: sec:setup
- Results: sec:results
  - Attention dispatch stack: sec:results-stack  (Table tab:gqa, Table tab:stack, Figure fig:stack)
  - No speed win from 4-bit weights: sec:results-nospeed (Table tab:nospeed)
  - Capacity: sec:results-capacity (Table tab:capacity)
  - Head-to-head: Table tab:headtohead
  - Executed policy (llama.cpp laptop): sec:results-exec (Table tab:exec)
  - Executed policy on unified-memory device (Jetson Orin Nano Super): Table tab:exec-orin
  - Bit-cost: sec:results-bitcost (Table tab:bitcost, Table tab:strength, Figure fig:bitcost)
  - Negative results: sec:results-negative
- Reproducibility: sec:repro (Table tab:passkey, Table tab:wikitext)
- Limitations: sec:limitations
- Appendix Pre-registration and Two Falsified Hypotheses: app:prereg
- Appendix Three Measurement Traps: app:trap (timing, projecting-head, silent-spill-WDDM)

## Equations
- eq:budget (memory budget constraint)
- eq:obj (allocation objective)

## Key claims and their evidence pointers (numbers)
- 9.0x / 13.3x: one cache bit costs less accuracy than one weight bit (0.6B / 1.7B). Source tables: tab:bitcost, tab:strength.
- 11.39% vs 1.27%, 3.20% vs 0.24%: perplexity deltas backing the above.
- Ordering reproduces on WikiText-2: tab:wikitext.
- usable-memory fraction rho = 0.133 (discrete-GPU laptop) vs 0.419 (unified-memory edge module); 3.15x difference; moves price of a weight bit by 3.7x. sec:rho.
- Laptop vs Jetson: rho 0.133 vs 0.419; 3.15x fraction; 1.6x budget.
- Executed policy (llama.cpp b12063): compress cache, weights 8-bit; makes 32k context fit at all. Tables tab:exec (laptop), tab:exec-orin (Jetson).
- GQA fallback: grouped-query attention silently disables every fused attention kernel; materialising KV heads cuts attention peak memory 32x and 4k prefill time 5.8x. Table tab:stack.
- 2-bit: cache quantisation scheme decides long-context retrieval where at 4 bits it does not; 5/5 vs 0/5 passkey at 2 bits. Table tab:passkey.
- Four configs deployed at 4096 on Device A: Table tab:headtohead.

## Honesty caveat (fixed in manuscript)
- llama.cpp KV quantisation is per-32-block, NOT KIVI's grouping. The executed-policy claim is 'a shipped runtime that stores the cache packed', never 'KIVI's kernel ran'.
- Prefill timing on the Jetson is not usable as evidence (stddev up to 315/large drift); decode within 5%.