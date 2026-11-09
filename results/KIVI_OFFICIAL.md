# 官方 GPTQ + 官方 KIVI：两侧都来自已发表实现

> 2026-10-21。`gptqmodel 4.2.5`（权重）与 KIVI `utils_quant`（缓存）。
> 数据：`results/gptq_official_0{6,17}b.json`、`results/kivi_official_0{6,17}b.json`

## 1. 最终对照（14 个评估窗口，同一批文本）

| 模型 | fp16 NLL | 权重 GPTQ | 缓存 KIVI 官方 | 缓存便宜 |
|---|---|---|---|---|
| Qwen3-0.6B | 3.851023 | **+11.39%** | **+1.27%** | **9.0×** |
| Qwen3-1.7B | 3.643825 | **+3.20%** | **+0.24%** | **13.3×** |

对照我们自己的实现：缓存侧 +1.65%（0.6B）与 +0.95%（1.7B）——
**官方分组比我们的更好，方向符合预期**，这也说明我们的适配没有把 KIVI 做坏。

## 2. KIVI 集成的关键事实

| 项 | 情况 |
|---|---|
| 官方支持的模型 | 只有 `llama_kivi.py` 与 `mistral_kivi.py`，**无 Qwen3** |
| 我们用了什么 | `third_party/KIVI/utils_quant.py`，从 commit `876b4d2` 原样 vendor（保留 Apache-2.0 许可） |
| 我们写了什么 | 适配层：把其 simulate 模式的输出还原成反量化张量，并套上残差窗口 |
| 没有用什么 | 它的 Triton 内核（`quant/new_pack.py`）——Windows 无 Triton，且其内核面向 2024 年的 Triton |

### 两个必须记住的坑

1. **同一文件里 K 与 V 的 scale 约定相反**：
   - K（`quantize_by_channel_and_pack_cache`）：`scale=(mx-mn)/B`，量化是除法 → 反量化要**乘**
   - V（`quantize_and_pack`）：`scale=B/(mx-mn)`，量化是乘法 → 反量化要**除**
   搞反会得到"看起来合理"的错误结果。

2. **`quantize_and_pack` 内部把 batch 轴转置了**：返回的是 `(groups, tokens, g)`，
   不是调用处暗示的 `(tokens, groups, g)`。

## 3. 适配层的自测

| 项 | 值 |
|---|---|
| key MSE | 9.29e-02（方差 15.18，即 0.6%） |
| value MSE | 6.12e-03（方差 1.00，即 0.6%） |
| 合计 vs 朴素 RTN | 0.099 vs 0.209 —— **分组使其好 2.1×** |

最后一行是关键：如果 scale 约定写反，误差会远大于朴素 RTN。

## 4. 论文表述

§6.4 现在写的是"两侧都来自已发表实现"，并明确：
- KIVI 无 Qwen3 模型文件，因此 attention 的接线是我们写的；
- 用的是它 simulate 路径而非 Triton 内核，因此比较的是**量化方案**而非 kernel 速度。

Limitations 第 3 条相应改写：网格表用的是我们自己的实现，第三方核验则两侧都用官方代码。
