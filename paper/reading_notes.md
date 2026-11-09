# 阅读笔记：The Cache Is Cheaper Than the Weights

> 目标期刊：IEEE Internet of Things Journal（IoT-J，IEEEtran 双栏模板）
> 作者：Yibo Zhang, Muhammad Syafiq bin Mohd Pozi, Osman B Ghazali（Universiti Utara Malaysia, School of Computing）
> 状态：已通读 main.tex 全文 + refs.bib 全量核验

## In one sentence
在一台名义 8 GB 的笔记本 GPU 与一台 Unified-memory 边缘模块上实测发现：**量化 KV cache 的精度代价比量化权重要便宜约一个量级（cache bit 比 weight bit 便宜 9.0×/13.3× PPL）**，且两设备"可用内存份额"相差 3.15×，因此**最优分配方案是设备相关的、不可能同时最优**。

## Problem
量化到底该花在**权重**还是 **KV cache** 上？现有 allocation 文献假设：目标是速度、两端兑换率按实践花在权重、budget 取 datasheet 标称值。作者逐条测量、逐条反转这些假设。

## Approach
1. **目标**：实测本机 GPU 栈上 4-bit 权重几乎不带来时延（dequantisation 未 fuse 进 matmul），故量化买的是**容量**而非吞吐。
2. **兑换率**：同栈同权值（GPTQ vs KIVI），0.6B 上 cache bit 比 weight bit 便宜 9.0×（11.39% vs 1.27% PPL），1.7B 上 13.3×（3.20% vs 0.24%）。
3. **budget**：ρ（可用内存份额）在 RTX 4060 Laptop（ρ=0.133）与 Jetson Orin Nano Super（ρ=0.419）间差 3.15×，把 weight bit 价格抬 3.7×；Qwen3-4B 从 8→4bit 让 attainable context 在笔记本乘 1.57×、在模块乘 5.76×。
4. 三个输入 + allocator 即可导出设备专属分配方案。

## Results（关键数字）
- **物话 KV heads**（禁用 fused kernel 后 KV heads 显式物话）把 attention 峰值内存降 **32×**（L508）、4k prefill 降 **5.8×**（L530）。
- **容量表**（ε=0.01）笔记本 111,184 vs 模块 67,324 token（**1.65×**），比例全程平坦在 1.63–1.82×。
- **passkey**(~11k)：2-bit KIVI + grouping 5/5，ungrouped 0/5 —— 界定了 claim 停止之处（4-bit 无明显差异，2-bit 决定性差异）。
- **设备 B（Jetson）derived 列**：context 长度曲线由实测 ρ + 在 A 上拟合的 workspace curve（0.142L+9MB）转移而来，预测 vs 实测峰值误差 <2%（83/158/307 vs 82/154/300 MB）。
- **repro 门槛**：单命令 `python experiments/run_all.py`（16 stages, ALL OK），仓库 github.com/BRAD-cording/llm-8gb-bit-costs。

## How far the evidence goes
- **measured**：ρ、copy/FP16 带宽、0.6B 512–2048 tok 端到端 sweep（TTFT 165.1/307.6/638.0ms）、每 linear layer 2/3/4bit ΔPPL、cache 全局 ΔPPL。
- **derived / assumption（已明说）**：B 的 context 长度与 exchange curve 是 derived（用 A 上 workspace curve 转移）；per-unit 4bit 噪声受限故保留 ranking + rescale 使 sum=端到端；**cache cost 跨层均匀铺开是 assumption**（无 per-layer cache 测量，L380-388）。
- **局限**（6 条）：只覆盖 0.6B/1.7B；third-party 跨设备；task-level 局限；capacity boundary；simulated cache compression；无 kernel 工程。

## Relation to other work
- 对 KIVI/MixKVQ/CocktailKV/KVTuner（cache）与 GPTQ/RTN（权重）做诚实 novelty 对比——审稿阶段独立复核。
- 与 allocation 文献（假设速度/权重/标称 budget）直接对照，本文的反转点是"实测兑换率 + 实测 ρ"。

## Worth taking away
"同一 8 GB 设备，内置核定分配不可跨设备迁移"这一**实测论点**有工程价值；但必须修复 **AUD-01 ρ 措辞矛盾**，并在跨语料 PPL 可比性、derived 曲线属性上对读者给出明确警示。

## Terms
- ρ（rhoSym）：名义内存中**不可用（lost）份额**（数学式 L336 用 M(1−ρ)）——注意摘要/引言三处误作"usable"，见 AUD-01。
- 物话（materialise）：on-device 实现中从 optimizer 显式物化成显式张量。