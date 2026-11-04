# 方案 B + C2：三个月路径（2026-11-04 起）

对应文件：
- `jsa/revamp/INTRO_v3.tex` — 替换 `jsa/main.tex` 第 107–195 行整段 Introduction
- `jsa/revamp/FRONT_v3.tex` — 替换第 49–68 行 abstract 与第 79–86 行 Highlights

数字一个没改，全部沿用原文（audit_numbers.py 仍然能对上）。改的是**主语**：
老版主语是"我们测到四个不一致"，新版主语是"分配是一个决策，这里是它的三个测量输入"。

---

## 1. 要删/要搬的原句（按杀伤力排序）

| 行 | 原文 | 处理 |
|---|---|---|
| 114–123 | "*What kind of paper this is.* It is a measurement paper... **Nothing here should be read as a new optimiser**... the allocation step is a multiple-choice knapsack solved by exact dynamic programming, **which is textbook**." | **删掉整段**。这段话把"我没什么新东西"印在第一页。移进 §Method，只在讨论分配器实现时用一句"the optimiser is textbook; the inputs are not"。 |
| 189–194 | "**What this paper does not claim** — We do not claim a new quantiser... we do not claim that the optimal allocation reverses... we do not claim exhaustive literature coverage." | **删标题、搬到最后**，改成 `\paragraph*{Scope}`，并把三条否定句改写成边界句（见 INTRO_v3 末尾）。位置从"贡献之前"挪到"贡献之后"。 |
| 160–161 | 贡献 1 开头："**What sets the budget is measurable, and it is not the bit-width.**" | 降级：这条从"贡献 1"变成"贡献 5"。它是**使能观察**（必须测量再分配），不是主贡献。 |
| 183–184 | "so the optimisation is textbook --- the contribution is the inputs it runs on" | **保留**，但只在 §Method 出现一次，不要在引言重复三遍。重复三次等于喊"我没新东西"。 |
| 225–229 | "Our claim is narrower and different: that..." | 把 "narrower" 删掉，只留 "different"。跟 kvpareto 的精确对比要留（这是加分项），但不要自贬。 |
| 50–51 | abstract： "We measure four mismatches between that literature and the device." | 换成"three inputs decide the split, and none is reported"。 |

---

## 2. 新的论证骨架（为什么这样改就"有 novelty"）

老版：四个测量 → 都指向同一个结论 → 顺手解了个分配。
问题：审稿人问"所以呢？这是个 bug report 合集。"

新版：**一个被所有人假设却没被测量的前提，被你测了，而它反转了三件事。**

> 分配问题 = f(目标函数, 兑换率, 预算)。
> 文献假设：目标是延迟、兑换率偏向权重、预算是标称值。
> 你测出来：目标是容量、兑换率反向 9–13×、预算差 3.15×。

这就不再是"我们测了四个不相干的东西"，而是"**同一个决策的三个输入，全部被测反了**"。
这个框架的好处是完全真实——你的数据本来就长这样，只是原来按"测量"排列，现在按"决策变量"排列。

标题不用动。"The cache is cheaper than the weights" 正好是新骨架的第一句结论。

---

## 3. C2：把推荐配置真跑起来（1–2 周）

### 为什么必须做
`main.tex` 754–769 行自己写着：cache-compressed 的部署行是 **simulated**（patch 量化后立刻反量化），
"their latency and memory therefore **bound what a packed implementation would cost, not what it would achieve**"。
而全文结论是"压缓存、权重留 8bit"。审稿人一句"你推荐的配置你自己跑过吗"就能打死，
而且这句话是**你自己写在正文里的**，不算恶意审稿。

### 怎么做（最小可行）
用现成 runtime，不要自己写 kernel。

| 项 | 选择 | 备注 |
|---|---|---|
| runtime | `llama.cpp`（CUDA backend） | Jetson 与笔记本同栈，数字可比 |
| KV 量化 | `--cache-type-k q8_0 --cache-type-v q8_0`（和 `q4_0` 各跑一次） | q8_0 对应"8 bit cache"，q4_0 对应激进档 |
| 权重 | `Q8_0` 与 `Q4_0` 各一份 | 用来在**第二个 runtime** 上复核"4-bit 权重不省时间" |
| 模型 | Qwen3-1.7B 为主，0.6B 补一个点 | 1.7B 才有"3.4 GB 权重值得压"的故事 |
| 上下文 | 4k / 8k / 16k | 16k 是关键：fp16 KV 在这里放不下，q4 KV 放得下 |
| 指标 | prefill tok/s、decode tok/s、峰值内存（`tegrastats` / `nvidia-smi`）、wikitext PPL、8k passkey | 五项就够，不要扩 |

### 判定成功
至少一行能写进 deployment 表：**推荐配置、真的跑起来、在 16k 上放得下而对照放不下**。
若 llama.cpp 在 Jetson 上构建失败，退路是 LMDeploy（笔记本）或 vLLM + fp8 KV，
但要**在正文里写明换了 runtime**，不能悄悄换。

### 诚实边界（必须写）
llama.cpp 的 KV 量化是 per-32-block，与 KIVI 的分组方案不同族。
所以只能说"a shipped runtime that stores the cache packed"，**不能说"KIVI 的 kernel 跑起来了"**。
这一句写清楚，反而是加分项。

### 已执行（2026-11-04，DONE）

跑法与本文档原计划的差异：**用笔记本的 RTX 4060 先跑**（Jetson 上 llama.cpp 未构建），
runtime 为 llama.cpp **b12063** CUDA 12.4 Windows x64，权重为 HF 官方 `Qwen3-*-Q8_0.gguf`
（8-bit 权重正是推荐策略的权重侧），缓存 `f16 / q8_0 / q4_0` 三种。

| 量 | 结果 |
|---|---|
| 决定性 cell | 4B @ 32k：fp16 缓存分配 **9,033 MiB > 8,188 MiB 物理**，20 分钟无 token（钉在 7,840 MiB / 100% util）；q8_0 缓存 **6,873 MiB**，32k prefill + 128 decode 步 **67 s** |
| 代价落点 | prefill 掉 6–10%（4B 16k: 2,514 → 2,391 / 2,425），decode 三条缓存格式相差 <3% |
| 精度 | q8_0 免费（1.7B 15.6078→15.6257；4B 11.8091→11.8099）；q4_0 **+26.15%（1.7B）vs +0.99%（4B）** |
| 单位成本 | 运行时日志复现论文的 112/144 KiB per token，q8_0≈53%，q4_0≈28% |

产物：`results/c2_llamacpp_17b.json`、`results/c2_llamacpp_4b.json`、`results/c2_alloc.txt`、
`results/c2_summary.json`、`results/c2_logs/`、说明文档 `results/C2_LLAMACPP.md`；
驱动 `src/eval/c2_llamacpp.py`，合并脚本 `experiments/make_c2_summary.py`（已进 `run_all.py`）。
稿件落点：新增 §VI-D 与 Table VI（`\label{tab:exec}`），Limitations 第 2 条改写为
"two cache-compression measurements, neither of them our kernel"，摘要补一句 executed policy。

### 已扩展到 Jetson Orin Nano Super（2026-11-04 夜，DONE）

同一 runtime 在设备上编译（b12063，`-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87`，39 min），
同一份 `Qwen3-1.7B-Q8_0.gguf`，25 W 模式、桌面在跑。与笔记本的关键差别：这台设备的权重是**映射**的
（`model buffer size = 0.00 MiB`），所以缓存直接和整个系统的可用内存竞争。

| 量 | 结果 |
|---|---|
| 阶梯 | 8k @ fp16（896 MiB）→ 16k @ q8_0（952 MiB）→ 32k @ q4_0（1,008 MiB）；每一级都被上一级的三次尝试全部拒绝（1,792 / 1,904 MiB） |
| 不是内存不够 | 峰值只有 **5,235 / 7,620 MB**；runtime 报 4,482 MiB free 却拒绝 1,792 MiB 的分配 |
| 是运行时自己的分配次序 | 同一次序的纯 CUDA 程序能拿到 1,793 + 238 + 952 MiB；llama.cpp 先映射权重再要 KV，2,752 MiB 之后连 1,536 MiB 都拿不到（`NvMapMemAllocInternalTagged ... error 12`） |
| 精度复现 | q8_0 **+0.10%**（15.1822→15.1968）、q4_0 **+25.06%**（→18.9865），与笔记本的 +0.11% / +26.15% 一致 |
| 性能 | decode 各配置差 ≤5%（4k 22.9–24.6 t/s）；prefill 单次抖动可达 25%，**不用于论证** |

产物：`results/C2_JETSON.md`、`results/c2_jetson/`（60+ 原始日志）、`results/c2_jetson_summary.json`、
`tools/jetson_c2/*`（probe/clone/build/grid/ram/repeat + memtest1-3）、`experiments/make_c2_jetson_summary.py`。
稿件落点：§VI-D 新增 `\emph{A second device.}` 段落、Table VII（`\label{tab:exec-orin}`，三次尝试制）、
阶梯段与"精度跨设备复现"段；`Three boundaries` → `Four boundaries`，第四条讲"实例化由环境内存决定"。

---

## 4. 三个月排期

| 周 | 日期 | 事 |
|---|---|---|
| W1 | 11/04–11/10 | 套用 INTRO_v3 + FRONT_v3，重编译，跑 `audit_numbers.py` 确认没有数字漂移 |
| W2–W3 | 11/11–11/24 | C2：Jetson + 笔记本各跑一遍，填表 |
| W4 | 11/25–12/01 | 整合：改 deployment 节、改 limitations（"partly simulated" 降级为只剩个别行）、改结论 |
| W5 | 12/02–12/08 | 通读 + cover letter + 提交 |
| 12 月中–1 月 | — | 等审 / 准备 response |

**关键**：W4 结束时无论 C2 成没成，都要投。
C2 成功 → 贡献 4 保留；C2 失败 → 直接删掉贡献 4，其余四条照样成立，
不能为了保住一条把稿子拖到二月。

---

## 5. 投哪里

**首选 IEEE IoT-J**，理由：
- 稿件已是 IEEEtran，零格式转换（JSA 版是 elsarticle，等于白做一遍）
- 订阅路径 $0 APC，符合你的预算约束
- 主题正对：8 GB 边缘设备 + 双机实测 + 部署
- C2 加上后，"measurement only" 的软肋被补掉，IoT-J 的审稿人吃这一套

**备选 TMLR**：明文"novelty 不是评审标准"，理论上正治你的病。
但它是 ML 刊物，这篇是系统测量，有被判 out-of-scope 的风险，而且没有格式优势。
**不要把两篇同时投**。

如果 IoT-J 再次 desk reject，下一站顺序：`ACM TECS` → `Journal of Systems and Software` → `FGCS`。

---

## 6. 一个月后该有的样子

一句话能说清，且不需要"但是"：

> 我们把权重/KV 的比特分配当成一个决策，测出了它一直被假设的三个输入
> （目标应是容量、缓存比特便宜 9–13 倍、可用内存让预算依赖设备），
> 解出最优分配，并在一个现成 runtime 上把推荐配置真跑了起来。
