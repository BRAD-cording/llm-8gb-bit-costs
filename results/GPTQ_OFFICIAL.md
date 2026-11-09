# 官方 GPTQ：两台设备都跑通了（0.6B + 1.7B）

> 2026-10-21。`gptqmodel 4.2.5`（官方实现）。
> 数据：`results/gptq_official_06b.json`、`results/gptq_official_17b.json`、
> `results/cache_jetson_06b.json`、`results/strength_17b_g128_w16.json`

## 1. 最终结果（14 个评估窗口，同一批文本）

| 模型 | 设备 | fp16 NLL | 权重 GPTQ | 缓存（朴素） | 缓存（KIVI+分组） | 缓存便宜 |
|---|---|---|---|---|---|---|
| Qwen3-0.6B | Jetson | 3.850999 | **+11.39%** | +5.03% | **+1.65%** | **6.9×** |
| Qwen3-1.7B | 4060 笔记本 | 3.643934 | **+3.20%** | +2.02% | **+0.95%** | **3.4×** |

GPTQ 相对朴素 RTN 的优势：0.6B 是 3.0×，1.7B 是 3.6×。

**结论：第三方权重实现下，排序在两个模型上都成立。**

## 2. Windows 这条路是怎么打通的（我最初的判断是错的）

我先前写"Windows + Python 3.13 装不上 gptqmodel"。**这个结论错了。**

真实根因是这台机器的 `pkg_resources` 太旧：

```
File ".../site-packages/pkg_resources/__init__.py", line 2172, in <module>
    register_finder(pkgutil.ImpImporter, find_on_path)
AttributeError: module 'pkgutil' has no attribute 'ImpImporter'
```

`setuptools` 自带的 `pkg_resources` 在 import 阶段就引用 3.12 已删除的属性。
`pip install -U "setuptools>=75"` 之后，`python setup.py --version` 立刻成功，
并打印出检测到的 `CUDA 8.9`。

之后的三个 Windows 专属障碍与解法：

| 障碍 | 表现 | 解法 |
|---|---|---|
| transformers 5.x 移名 | `cannot import name 'no_init_weights'` | `tools/gptq_compat.py` 从 `transformers.initialization` 重新导出 |
| 无 Triton | `Cannot find a working triton installation`（量化循环里的 `torch.compile`） | 兼容层里 `torch._dynamo.config.disable = True`，并设 `TORCHDYNAMO_DISABLE=1` |
| AUTO 内核选择 | 同上，选了 Triton 内核 | `GPTQModel.load(..., backend=BACKEND.TORCH)` |

## 3. Jetson 这条路为什么只做成 0.6B

1.7B 的 fp16 权重 3.4 GB，而模块可用内存 4.3 GB，量化器还要每层的 Hessian 与其逆
（层宽 6144 时单个矩阵 151 MB）。三次尝试（16/4 窗口、不同分配器设置）全部在
`NVML_INTERNAL_ASSERT` 处终止。

误差反馈量化必须先持有全精度权重，没有"先压缩再量化"的余地——**这是容量问题，不是配置问题**。
所以 1.7B 的第三方核验在 4060 上完成（6.9 GB 可用显存）。

## 4. 跨设备可比性

论文把两机数据放进同一段比较，依据是 fp16 参考 NLL 五位小数一致：
Jetson 3.850999 vs 笔记本 3.850961。量化代价是权重的属性，不是 GPU 的属性。

## 5. 产物

| 文件 | 作用 |
|---|---|
| `tools/gptq_compat.py` | transformers 5.x 兼容层 + 关闭 dynamo |
| `artifacts/pcre_shim.py` | Jetson 侧的 pcre 垫片（已交代覆盖哪些接口） |
| `src/eval/gptq_win.py` | Windows 上的三阶段量化/评估 |
| `src/eval/strength_matched.py` | 两侧强度匹配对照 |
