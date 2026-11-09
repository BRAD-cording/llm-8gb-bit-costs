# Roofline 与 regime 反转的定量基础（真机实测，非规格书）

数据来源：`results/probe_4060.json`，`src/bench/device_probe.py`，实测于 RTX 4060 Laptop 8GB。

## 1. 实测值 vs 规格书值（第一个必须如实报告的差异）

| 量 | 规格书 | **实测** | 有效率 |
|---|---|---|---|
| 显存带宽 | 256 GB/s | **161.9 GB/s**（copy）/ 179.8 GB/s（GEMV） | 63–70% |
| FP16 算力 | ~60 TFLOPS | **22.4 TFLOP/s** @ bs=128 | ~37% |
| 显存 | 8188 MiB | 8.0 GB 可用 | — |

> **这一条本身就是论文素材**：所有"边缘量化"论文若用规格书数字算 roofline，会把 crossover 点算错 1.5–2.7 倍。我们用实测值。

## 2. prefill/decode 的算力–带宽交叉点

权重 GEMV 的算术强度（FP16）：

$$ I(b) = \frac{2 \cdot b \cdot d_{model} \cdot d_{ff}}{2 \cdot d_{model} \cdot d_{ff}} = b \quad [\text{FLOP/byte}] $$

即**算术强度 ≈ batch/prefix 长度**。设备交叉点：

$$ b^{*} = \frac{P_{eff}}{B_{eff}} $$

**4060**：$b^* = 22400 / 180 \approx \mathbf{124}$

实测佐证：bs=1 时 0.21 TFLOP/s 但 179.8 GB/s（纯带宽受限）；bs=128 时 22.4 TFLOP/s（算力受限）。单调上升的 TFLOPs 曲线正是从带宽受限爬向算力受限的过程。

**Jetson Orin Nano Super 的预测**（待 W1 实测验证）：
- $B_{eff} \approx 55\text{–}65$ GB/s（LPDDR5 统一内存，实测通常 80–90% 标称）
- FP16 有效算力 $\approx 8\text{–}16$ TFLOP/s
- $\Rightarrow b^* \approx 150\text{–}280$

**结论（本文的核心可证伪命题）**：

> 同一台设备在 $b < b^*$ 时是带宽受限，在 $b > b^*$ 时是算力受限；而 $b^*_{Jetson} \approx 2\times b^*_{4060}$。
> 因此在**相同的 $(\text{batch}, L)$ 工作点上，两台机器处在不同的 regime**，最优位宽分配必然不同——这就是"反转"的来源，且它是**由实测 roofline 推出来的、不是编的故事**。

## 3. 由此导出的分配偏好（W7 要被实验证实或推翻）

| 情形 | 受限于 | 应压什么 | 应保什么 |
|---|---|---|---|
| 4060, 中长 prefix ($b>b^*$) | 算力 | **KV cache**（省内存→可跑更长上下文，不影响算力） | 权重精度（避免 dequant 开销 + 无 INT4 tensor core） |
| Jetson, 全场景 ($b<b^*$ 为主) | 带宽 | **权重**（每 token 都要读一遍） | KV 可更激进（2-bit） |
| 两者, 超长上下文 $L$ 大 | 容量 | 必须压 KV，否则 OOM | — |

**反转点**：当 $\alpha$（prefill 权重）高且 $L$ 中等时，4060 最优解 $(b^W{=}4, b^{KV}{=}2)$；Jetson 最优解 $(b^W{=}3, b^{KV}{=}2\text{–}4)$。若 W7 测出相反结果，本文降级路线见 `PLAN.md` §6。

## 4. 待补的测量（W1–W4）

- [ ] Jetson 上跑同一脚本，得到 $B_{eff}, P_{eff}, b^*_{Jetson}$
- [ ] $\rho$ 的真实值：当前 `rho_alloc=0.0` 只测了单次大分配；真实 $\rho$ 来自 **CUDA context + cuBLAS workspace + KV pool 预分配 + 碎片**，需加载真模型后测（W4）
- [ ] 能耗：4060 侧用 `nvidia-smi --query-gpu=power.draw` 积分；Jetson 用 `tegrastats`（`VDD_IN` 功率）
- [ ] 温度/降频：两机都在持续负载下记录
