from pathlib import Path

p = Path(r"C:\Users\HP\OneDrive\Desktop\Dev\harness\llm_8gb_bit_costs\paper\jsa\revamp\REVAMP_NOTES.md")
t = p.read_text(encoding="utf-8")

anchor = "---\n\n## 4. 三个月排期"
assert t.count(anchor) == 1

block = """### 已扩展到 Jetson Orin Nano Super（2026-11-04 夜，DONE）

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
稿件落点：§VI-D 新增 `\\emph{A second device.}` 段落、Table VII（`\\label{tab:exec-orin}`，三次尝试制）、
阶梯段与"精度跨设备复现"段；`Three boundaries` → `Four boundaries`，第四条讲"实例化由环境内存决定"。

"""

p.write_text(t.replace(anchor, block + anchor), encoding="utf-8")
print("notes updated")
