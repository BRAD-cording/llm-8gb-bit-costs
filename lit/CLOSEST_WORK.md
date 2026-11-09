# 最近的竞争工作与验证日志

## 为什么这个文件存在

文献扫描返回了 2026 年的高危近邻。我抽查了其中 6 篇的 arXiv 元数据（`export.arxiv.org/api/query`，2026-10-20 核实），**全部为真**，且其中两篇直接威胁本文的新颖性声称。

**结论：不能再用"没人做 regime-aware 量化"这句话。必须换成可辩护的更强表述。**

## 威胁 T1：AlignQuant（arXiv:2610.07457，2026-10-05）

> "Joint prefill/decode calibration scores precision reductions using projection-output perturbations ... **Phase-normalized scores prioritize higher precision for tiles important to either phase under a model-wide weight-storage budget.**"

- 它**已经**做了 phase-aware（prefill/decode 双相打分）的混合精度分配
- 它的预算是 **model-wide weight-storage budget**（只含权重）
- 它覆盖 3B–14B、3 块 GPU、上下文至 64K，最快 2.50× 生成加速

**它没有做的**：KV cache 精度完全不在分配变量里。所以我们的差异点必须精确表述为：

> AlignQuant 在**权重内部**做相感知分配，把 KV 当常量；我们把**权重与 KV 放进同一个预算**做相感知分配。这带来一个它们无法表达的效应：给定 8GB，多给 KV 一个 bit 就必须从权重拿走一个 bit，而这两者的"相边际收益"方向相反。

## 威胁 T2：Disaggregated Quantization（arXiv:2609.26333，2026-09-22）

> "Prefill and decode reward different approaches to quantization ... trains separate compute-native prefill weights ... ODP streams its weights from SSD ... 1.78x TTFT speedup at 8K prompt in llama.cpp"

- 它把 prefill 与 decode 的**格式**分开（NVFP4 prefill 权重 + 权重-only decode）
- 它甚至用了 llama.cpp + GGUF，与我们的工具链重叠
- 但它靠**额外 checkpoint + SSD 流式加载**换取收益

**它没有做的**：没有 KV 预算分配；且 SSD 流式在 Jetson 的统一内存 + eMMC 上代价完全不同（这一点是我们的实验能直接打的）。

**同时它给了我们一个必须警惕的反例**：如果"两份权重"策略在 8GB 上比"一份权重 + 精细 KV 分配"更好，我们的方法就被压制。**这个对照必须进实验矩阵**（记为 baseline `DQ-ODP`）。

## 由此收紧后的三句话（写进 Related Work 的定稿表述）

| 原表述（不可辩护） | 收紧后（可辩护） |
|---|---|
| 没人做 regime-aware 量化 | 相感知分配已被用于**权重内部**（AlignQuant）与**格式层面**（DQ），但**从未跨越权重–KV 边界** |
| 没人做权重+KV 联合量化 | KV Pareto 与 AlignQuant 各自触及一侧，但**都不用实测带宽/时延作为前沿轴**，因此无法给出反转条件 |
| 8GB 设备没人测 | 已有工作用 3 块 GPU、64K 上下文，但**没有一块是 8GB 消费级笔记本 GPU 或 Orin 级统一内存**，两类的可用带宽相差 2.6–3×，roofline 交叉点因此不同 |

## 验证日志（方法学，必须可复现）

- 工具：`https://export.arxiv.org/api/query?id_list=...`（批量，返回标题+摘要+提交日期）
- 本轮核实：`2604.18556`(GSQ)、`2605.18475`(GAMMA)、`2609.26333`(DQ)、`2610.20303`(SpectrumQuant)、`2606.03458`(KVarN)、`2605.20315`(Mix-Quant)、`2610.07457`(AlignQuant) — **7/7 为真**
- 规则：**任何 arXiv ID 若在 API 中查不到，一律从稿件删除，不允许"大概是这个号"**。venue 字段为 `unverified` 的条目在投稿前必须替换为正式出版信息或标注为 preprint。

## 待办

- [ ] 用同一 API 批量核实 corpus 中**全部** arXiv 条目（待子代理产出 `lit/corpus.csv` 后执行）
- [ ] 把 `DQ-ODP` 与 `AlignQuant` 加入 `PLAN.md` §3 的 baseline 列表
- [ ] 在 `PLAN.md` §2 差异化表中新增这两行
