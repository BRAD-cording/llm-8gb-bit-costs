# P1 项 7 实验准备：同预算 SOTA 端到端基线

状态：**准备（未跑）**。目的：回答 R1/R2 共同指认的最大杠杆——"在相同内存预算下，
你的策略是否真的赢过已发表的 KV 量化方案，而不是只赢过一个弱配置"。

## 1. 为什么现在跑不了"我们的策略 vs 已发表 SOTA"的干净对照

| 候选基线 | 代码可得性 | 同栈可跑 | 判断 |
|---|---|---|---|
| KIVI（官方 `utils_quant`） | 已 vendored（`third_party/KIVI`） | 是（`enable_gqa_sdpa(kv_official=True)`） | **可作 cache 侧 SOTA** |
| KVQuant | 无（需 CUDA 扩展 + 校准管线） | 否 | 不可 |
| MultiAlloc / RDKV | 无公开实现（arXiv，无代码包） | 否 | 不可 |
| llama.cpp Q8_0 / Q4_0 packed | 已装（`§results-exec` 已用） | 是 | **可作 shipped-runtime 端到端基线** |

结论：可执行的"已发表 SOTA 同栈对照"只有两条腿——KIVI 的量化器（质量轴）+ llama.cpp
的 packed cache（运行时轴）。这正好对应稿内已有的两组测量，缺的是把二者放进**同一个
字节预算**里比较。

## 2. 已可从 committed 数据算出的预算表（0.6B，笔记本 B = 7,096 MB）

权重/缓存的字节模型：`weight_MB(w)` 取 `results/cross_06b_rtn.json`；`kv_MB(k,L) = kv_MB(k,1k) × L/1024`
（k=8 → 29.4 MB@1k，k=4 → 14.7，k=2 → 7.3）。已发表惯例 = **4-bit 权重 + fp16 缓存**
（fp16 缓存 @1k = 58.8 MB；质量取单轴 `w_bits_qwen06_v2.json` 的 w=4 = **+32.33%**）。

| L | 已发表惯例（4bit W + fp16 KV） | 我们的策略（W8/K8） | 谁占优 |
|---|---|---|---|
| 1,024 | 590 MB / +32.33% | 781 MB / +2.31% | 我们更准但**多花 32% 字节**；按字节不占优（同字节下 W4K8 = 561 MB / +35.66%，反而更差） |
| 4,096 | 767 MB / +32.33% | 869 MB / +2.31% | 同上，字节差收窄到 13% |
| 8,192 | 1,002 MB / +32.33% | 987 MB / +2.31% | **同字节占优**（少 15 MB，准 30 个点） |
| 32,768 | 2,413 MB / +32.33% | 1,692 MB / +2.31% | **全面占优**（省 30% 字节，准 30 个点） |

**这就是项 7 的核心结论形状**：比特分配的价值随缓存占比上升，8k 是交叉点；1k 上
"分配"不产生收益，这点必须自己先说，否则审稿人会用 1k 的数字反打。

## 3. 待跑的两条实测（决定摘要能写多强）

**(a) 长上下文质量点**：在 L = 8,192 与 32,768 上测 4-bit 权重 + fp16 缓存 vs W8/K8 的
perplexity。现有 PPL 全是 1k 窗，这是唯一缺口。

```
python ../src/eval/bit_cost_cross.py --model C:/models/Qwen3-0.6B \
  --text ../data/calib_arxiv.txt --seq-len 32768 --windows 2 \
  --w-bits 4,8 --k-bits 8 --group 128 --out ../results/cross_06b_32k.json
```
注意：32k 的 PPL 窗只有 2 个，方差会很大 → 必须同时报 1k 网格的 4-fold scatter 作为
参照，并声明 32k 点只用于"是否仍在同一侧"的定性判断，不用于拟合斜率。

**(b) 端到端同预算运行时**：复现 `§results-exec` 的 4B/32k 对照，补上 0.6B/32k 一行
（fp16 缓存 vs packed 8-bit 缓存，同 batch=1）：
```
python ../src/bench/regime_sweep.py --model /home/brad/models/Qwen3-0.6B \
  --length 32768 --precision fp16 --device cuda   # 与 paper 的 exec 表同口径
```
Jetson 侧须带 `export LD_LIBRARY_PATH=.../nvidia/cusparselt/lib`，且 L≥4096 且 batch≥4
会触发 `CUDACachingAllocator:838`（已知，不作为失败）。

## 4. 落稿位置与口径（无论结果好坏）

- 若 (a) 在 8k/32k 上仍显示 W8/K8 占优 → 摘要与贡献改写为
  **"同字节预算下，8k 以上缓存压缩胜出；1k 以下不产生收益"**，把 9.0×/13.3× 降为
  这一主张的支持证据之一。
- 若 (a) 出现反转（4-bit 权重 + fp16 缓存同字节更好）→ **如实写反转**，摘要改为
  "分配的价值只在缓存主导区，且在该区我们的量测确认了这一点/不成立"，并给出交叉点。
- 任何情况下都不改已有数字，只改口径与措辞；落稿后重编并保持 ≤14 页。

## 5. 检查清单（跑完必做）

1. 新 JSON 写入 `results/`，并在 `experiments/run_all.py --with-measurements` 增一步。
2. `check_table_vs_data.py` / `check_stale_numbers.py` / `check_exchange_rate.py` 全过。
3. 重编 `pdflatex → bibtex → pdflatex×2`，确认 14 页、0 Overfull、0 undefined。
4. 若引用新数字进摘要，必须同步改 `abstract` 与 `Contributions` 两处，并用
   `check_exchange_rate.py` 复核一致性。
