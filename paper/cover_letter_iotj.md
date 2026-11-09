# Cover letter — IEEE Internet of Things Journal

**To:** The Editor-in-Chief, *IEEE Internet of Things Journal*
**Re:** Submission of original research manuscript

**Title:** Measured Bit Costs and Device-Dependent Capacity for Weight–KV Allocation on 8 GB Edge Devices

**Authors:** Yibo Zhang (corresponding), Muhammad Syafiq bin Mohd Pozi, Osman B Ghazali — School of Computing, Universiti Utara Malaysia
**Correspondence:** yibo@ahsgs.uum.edu.my

---

Dear Editor,

We submit the enclosed manuscript for consideration as a regular paper in *IEEE Internet of Things Journal*.

**The problem.** An 8 GB edge node must hold the weights, the KV cache and the runtime in one address space, and no host memory to fall back on. The quantization literature decides the weight width and the cache width separately: one line optimizes weight precision at a fixed width, another compresses the cache at a fixed weight width. On a device where the two compete for the same bytes, neither choice is well posed alone, and the usual input to either choice — how much memory is actually available — is assumed rather than measured.

**What we did.** We measured the three inputs that the allocation question needs, on three nominally 8 GB devices with three different memory architectures (a discrete GPU laptop, a unified-memory Jetson Orin Nano Super, and a CPU-only Raspberry Pi 5):

1. *The objective is capacity, not latency.* Four-bit weights do not buy time on the stacks an edge device runs out of the box; dequantization is not fused into the matmul. Quantization therefore purchases capacity, and an allocator written against latency is degenerate.
2. *The exchange rate is a property of the operating point, not a constant.* Measured on a weight × cache grid at 4,096 tokens, a weight bit costs 4.8–10.2× a cache bit in the aggressive-weight-width regime, and the ratio collapses to below one above six bits, where the weight curve saturates. We report the point ratio between shipped quantizers (9.0× and 13.3×), the matched-strength ratio (1.3×, 95% CI excluding zero only just), and the slope ratio (9.4–15.5× over 48 windows), and state which quantity each claim belongs to.
3. *Capacity is device-dependent through a measurable fraction.* The unusable-memory fraction ρ is 0.047, 0.133 and 0.419 on the three devices — a 3.15× difference in the fraction lost and a 1.6× difference in the usable budget, although all three are sold as 8 GB. The measured module ceiling is 3,072 tokens, against 67k predicted by the same model that fits the laptop to within 2%, which is itself a result about where such models stop being valid.

**Why we believe this belongs in IoT-J.** The paper's central quantity is a property of the device class, not of a model: it is the fraction of nominal memory that firmware, the OS and the driver reserve before a single weight or cache byte is allocated, and it varies by more than 3× across three devices of the same nominal size. That heterogeneity is precisely what a fixed quantization recipe cannot absorb, and it is why we report a per-device budget rather than a recommended bit width. We also report a negative result honestly: on a 108-question task built from public LogHub logs the three configurations are statistically indistinguishable, so we do not claim a task-level gain.

**Reproducibility.** All measurement scripts, the vendored third-party quantizers, the files behind every table and the figure-generation scripts are public at https://github.com/BRAD-cording/llm-8gb-bit-costs, pinned at commit 79ee170 (code MIT, figures CC BY 4.0). The manuscript states which numbers are measured and which are derived from the fitted cost model, and the single command that rebuilds the derived artifacts is documented separately from the path that re-runs the GPU measurements.

**Declarations.** The manuscript is original, has not been published elsewhere and is not under consideration by another journal. All authors have approved the submission and declare no conflict of interest. No human or animal subjects were involved. We have no suggested reviewers to propose.

We would be glad to provide any further material the reviewers require.

Yours sincerely,
Yibo Zhang, on behalf of all authors
School of Computing, Universiti Utara Malaysia
yibo@ahsgs.uum.edu.my
