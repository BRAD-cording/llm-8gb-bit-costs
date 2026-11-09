# 位宽与时延：排序不稳定（实测补记）

> 数据：`results/synth_qwen3_4b_int8_gqa.json`、`synth_qwen3_4b_nf4_gqa.json`、
> `sweep_qwen06_int8_real.json`、`sweep_qwen06_{nf4,fp16}_real.json`
> 2026-10-20

## 1. 两处"排序反常"

### 0.6B（真实权重，融合注意力）

| L | fp16 | nf4 | int8 |
|---|---|---|---|
| TTFT @1k | **62.2** | 79.4 | 159.2 |
| TTFT @4k | **355.9** | 404.2 | 520.9 |
| TPOT @4k | **28.8** | 37.2 | 129.2 |

排序：fp16 < nf4 < int8（位宽越高越慢，完全非单调）

### 4B（随机权重，融合注意力；时延与权重数值无关）

| L | nf4 (2.60 GB) | int8 (4.41 GB) |
|---|---|---|
| TTFT @1k | 393.7 | **288.6** |
| TTFT @4k | 1744.3 | **1612.8** |
| TPOT @1k | **41.1** | 96.8 |
| TPOT @4k | **54.7** | 128.5 |

排序：**prefill 是 int8 更快，decode 是 nf4 更快**——同一对配置在相位之间反转。

## 2. 机制

bnb 的两条路径完全不同：

- **nf4**：4-bit 存储 → dequant 到 fp16 → fp16 tensor-core matmul。FLOPs 不变，
  额外逐元素流量；dequant 未融合。
- **int8**：LLM.int8()，逐 matmul 做离群值分解（mixed-precision decomposition），
  并对大 batch 走 int8 tensor core。

于是：
- 小模型 + 大 batch（prefill）：int8 的离群值分解开销被摊薄，且 int8 matmul 更快 → 有时反超
- 小 batch（decode）：每步都要做分解，开销无法摊薄 → 显著更慢
- 4B 的权重占比更高，dequant 的绝对成本更大 → int8 在 prefill 上反超 nf4

## 3. 对论文的意义

这不是"哪个更快"的问题，而是**"量化加速"这个说法在缺少相位与模型规模时没有意义**。

论文 §6.2 原来只写了 0.6B 的 int8 反例，现在补了 4B 的相位反转，主张变为：

> A single "quantisation speedup" number is not meaningful without naming the
> phase and the model size.

这直接支撑了"目标函数必须是容量而非时延"这一方法选择。

## 4. 一个必须记下的实现 bug

`synth_sweep.py` 在量化分支里**把精度硬编码成 `"nf4"`**：

```python
model, _ = build_model(str(save_dir), "nf4", "cuda")   # ← 无论传什么都是 nf4
```

第一次跑 int8 时得到 2595 MB 占用、390.7 ms —— 与 nf4 完全一致。
**数字"看起来合理"，所以没有报错、没有警告**。是因为我预期 4B int8 应占 4.4 GB
才注意到。修正后得到 4411.6 MB。

教训：**参数透传的失误不会触发异常，只会安静地给出另一个配置的结果。**
凡是"两个配置结果几乎相同"的情况，都值得怀疑一次。
