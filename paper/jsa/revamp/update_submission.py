"""Rewrite SUBMISSION.md for the post-JSA-rejection resubmission to IoT-J.

Adds the status section, replaces the cover letter with the repositioned spine
(three inputs + executed policy), adds R8/R9 pre-emptions, and updates the
checklist counts (13 pages, 17 stages, 10 tables, executed row).
"""
from pathlib import Path

P = Path(r"C:\Users\HP\OneDrive\Desktop\Dev\harness\llm_8gb_bit_costs\SUBMISSION.md")
t = P.read_text(encoding="utf-8")

# ---------------------------------------------------------------- 1. header
old = "最后更新：2026-10-22（对应 `v2.0-submission`）"
new = "最后更新：2026-11-04（JSA 拒稿后重定位；对应 `v3.0-reposition`）"
assert t.count(old) == 1
t = t.replace(old, new, 1)

# ------------------------------------------------------- 2. new section 0
SECTION0 = """## 0. 状态：JSA 拒稿与重定位（2026-11-04）

`JSA-D-26-02451` 于 2026-11-04 被 *Journal of Systems Architecture* 拒稿，理由只有一句
"lack of sufficient novelty"。Editorial Manager 的 "Submissions with a Decision" 里
**没有附任何 reviewer report**，所以这是编辑层面的 desk reject，不是经过评审的科学判定——
但它也不是随机的：原稿在引言里主动论证自己"不是新优化器、分配器是教科书算法"，而 JSA 是
系统结构期刊，期待一个机制，原稿给的是实测刻画 + 现成求解器。

**重定位做了三件事**（均已进 `paper/main.tex`）：

| 改动 | 原来 | 现在 |
|---|---|---|
| 论证骨架 | "我们测量四处错配"（清单式） | "分配 = f(目标, 汇率, 预算)，文献假设三者，我们测三者且三者都反转" |
| 贡献列表 | 3 条，第 1 条是"我们测量了" | 5 条：汇率 → 目标 → 设备依赖 → **执行策略** → 执行栈 |
| 边界声明 | 开头自证不是新优化器（自我削弱） | 移到贡献之后，改写成 scope 声明；被证伪的预注册假设指向附录 |
| 关键缺口 | 推荐策略（权重 8-bit + 压缓存）只被定价，从未执行 | **新增 §VI-D**：llama.cpp b12063 上真跑，4B/32k 用 fp16 缓存分配 9,033 MiB（> 8,188 物理）跑不完，q8_0 缓存 6,873 MiB 跑得完 |

目标期刊：**IEEE Internet of Things Journal**（订阅路径，$0 APC，IEEEtran 无需转格式）。
备选链：ACM TECS → Journal of Systems Software → FGCS。

---

"""
anchor = "## 1. 期刊：已定 IEEE Internet of Things Journal（订阅路径，$0 APC）"
assert t.count(anchor) == 1
t = t.replace(anchor, SECTION0 + anchor, 1)

# ------------------------------------------------------- 3. cover letter
LETTER = """> Dear Editor,
>
> We submit **"The Cache Is Cheaper Than the Weights: Measured Bit Costs and
> Device-Dependent Capacity for LLM Inference on 8 GB Devices"** to the IEEE
> Internet of Things Journal.
>
> **What the paper is.** Three quantities decide how an 8 GB device should split
> its bits between weights and KV cache: the objective the bits are spent for,
> the exchange rate between the two sides, and the memory budget actually
> available. Deployment recipes fix all three. On this device class all three
> reverse, and we measure each, on two devices, before executing the policy that
> follows on a shipped runtime.
>
> **The objective is capacity, not latency.** 4-bit weights buy little or no time
> on this stack (0.6B: nf4 is 1.21x slower than fp16 at 1k and 1.07x at 16k;
> 1.7B: decode gains 9% while prefill loses 11%), because dequantisation is not
> fused into the matmul. Quantisation here purchases capacity, which is why the
> allocation objective must be capacity.
>
> **The exchange rate runs against practice.** At equal width, with both sides
> taken from published implementations (the official GPTQ for weights, KIVI's own
> quantiser for the cache), weights cost +11.39% perplexity where the cache costs
> +1.27% on Qwen3-0.6B, and +3.20% against +0.24% on Qwen3-1.7B -- 9.0x and 13.3x
> in the cache's favour. The ordering reproduces on the WikiText-2 test set
> (17.52% against 1.61%; 6.33% against 0.82%). A task-level check scopes it: at 4
> bits every scheme we tried recovers an 11k-token passkey, so the differences are
> differences in fluency; at 2 bits the scheme decides, and the same budget that
> KIVI's grouping carries destroys the capability under naive rounding (5/5
> against 0/5).
>
> **The budget is device-dependent.** The usable-memory fraction is 0.133 on a
> discrete-GPU laptop and 0.419 on a unified-memory edge module; the 3.15x
> difference moves the price of a weight bit by 3.7x, so dropping Qwen3-4B from 8
> to 4 bits multiplies attainable context by 1.57x on the laptop and 5.76x on the
> module. A single published configuration cannot be optimal on both.
>
> **We execute the policy the allocator recommends.** The allocation step is a
> multiple-choice knapsack solved by exact dynamic programming -- textbook, and we
> say so; the inputs are not. Its policy, weights at 8 bits and the cache
> compressed, was previously only priced. We run it on llama.cpp b12063 (CUDA
> 12.4) with the cache stored packed. The decisive cell is a 4B model at 32k: with
> the default fp16 cache it allocates 9,033 MiB on an 8,188 MiB device and never
> produces a token; with the cache at q8_0 the same length needs 6,873 MiB and
> completes 32k prefill plus 128 decode steps in 67 s. Packing costs prefill
> (2,514 to 2,391 tokens/s at 16k) and leaves decode within 3%; 8-bit cache costs
> +0.01% perplexity, while 4 bits costs +26% on the smaller model and +0.99% on
> the larger -- the exchange rate is per model, which is the paper's claim.
>
> **Before any bit, the serving stack sets the budget.** Because Qwen-style models
> use grouped-query attention, the framework silently skips every fused attention
> kernel and materialises an O(L^2) score matrix; materialising the KV heads, a
> mathematically transparent rewrite, cuts attention peak memory 32x and 4k
> prefill time 5.8x. We release a model-free reproduction, because the effect is
> invisible in the framework's output.
>
> **What we report against ourselves.** The alternative is strictly dominated on
> 0.6B at 4k (10.84% worse perplexity, 12% slower to first token, 19% slower per
> decode step, for 644 MB the device does not need) but defensible on 1.7B at 4k
> (2 GB and 9% decode speed for 4.68% perplexity). Two pre-registered hypotheses
> are reported as falsified. The cache-compressed rows of the deployment table are
> simulated by our dequantising patch, and we say so; the executed section runs a
> shipped runtime instead, and states its three boundaries -- llama.cpp's
> per-32-block scheme rather than KIVI's kernel, allocation logs rather than peak
> memory (the Windows driver pages instead of raising), and one device rather than
> two.
>
> **About the evidence.** Every table is generated from a measurement file rather
> than transcribed; a single command rebuilds the figures, the tables and the
> manuscript and runs seventeen checks, including a numeric audit that extracts
> every factor and percentage claim from the text and compares it to the results.
>
> The manuscript is not under consideration elsewhere. All code, data and
> measurement scripts are released.
>
> Sincerely,
> Yibo Zhang, on behalf of the authors"""

i = t.index("> Dear Editor,")
j = t.index("> Yibo Zhang, on behalf of the authors") + len("> Yibo Zhang, on behalf of the authors")
t = t[:i] + LETTER + t[j:]

# ------------------------------------------------------- 4. R8 / R9
R89 = """### R8（新增，最高威胁）「你们跑的是 llama.cpp，不算你们自己的方法」
**已在正文写死三条边界**：(1) 该方案是 llama.cpp 的 per-32-block 打包，**不是** KIVI 的
grouping，也不是我们的量化器；主张只到"一个已发布运行时装得下、跑得动"，不声称 kernel 级
性能；(2) Windows WDDM 超预算时换页而非 OOM，所以峰值饱和（≈7,800 MiB）本身不能区分
"装得下"与"溢出"，决定性证据是**运行时自己打印的分配量 vs 物理显存** + 完成时间对比；
(3) 单设备、单运行时、单权重格式。
**若被要求"跑你们自己的打包 kernel"**：那需要分块反量化 attention kernel，我们没写，
正文已列为未做项；不软化这条。

### R9（新增）「32k 不完成，是不是你们超时设短了？」
**不是**：同一 cell 曾在无行级超时的情况下跑满 20 分钟，钉在 7,840 MiB / 100% 利用率 /
40 W，仍未产出任何 token；而 q8_0 在 67 s 内完成同样长度。加上"分配 9,033 MiB > 物理
8,188 MiB"这条来自运行时自身日志的独立证据，三条互相印证。

## 4. 投稿前检查清单"""
anchor4 = "## 4. 投稿前检查清单"
assert t.count(anchor4) == 1
t = t.replace(anchor4, R89, 1)

# ------------------------------------------------------- 5. checklist rows
pairs = [
    ("| 编译 | ✅ **12 页**，0 overfull box，0 未定义引用 |",
     "| 编译 | ✅ **13 页**，0 overfull box，0 未定义引用 |"),
    ("| 自动化校验 | ✅ **16 阶段全绿**（`experiments/run_all.py`） |",
     "| 自动化校验 | ✅ **17 阶段全绿**（`experiments/run_all.py`） |"),
    ("| 表/图 | ✅ **10 表 3 图**，逐页看过渲染 |",
     "| 表/图 | ✅ **11 表 3 图**（新增 executed-policy 表），逐页看过渲染 |"),
    ("| 打包缓存实测 | ✅ 3.2×（真实分配，非建模） |",
     "| 打包缓存实测 | ✅ 3.2×（真实分配，非建模） |\n"
     "| **推荐策略实跑** | ✅ llama.cpp b12063：4B/32k fp16 缓存 9,033 MiB 跑不完，q8_0 6,873 MiB 67 s 完成 |"),
    ("| 目标期刊 | ✅ IEEE Internet of Things Journal（订阅路径，$0 APC） |",
     "| 目标期刊 | ✅ IEEE Internet of Things Journal（订阅路径，$0 APC）；JSA 已拒稿 |"),
]
for old, new in pairs:
    assert t.count(old) == 1, old[:40]
    t = t.replace(old, new, 1)

P.write_text(t, encoding="utf-8")
print("SUBMISSION.md rewritten:", len(t.splitlines()), "lines")
