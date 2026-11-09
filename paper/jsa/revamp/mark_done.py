"""Mark C2 as executed inside the revamp notes, and add a dated STATUS entry."""
from pathlib import Path

ROOT = Path(r"C:\Users\HP\OneDrive\Desktop\Dev\harness\llm_8gb_bit_costs")

# ------------------------------------------------------------ revamp notes
notes = ROOT / "paper" / "jsa" / "revamp" / "REVAMP_NOTES.md"
t = notes.read_text(encoding="utf-8")

DONE = """### 已执行（2026-11-04，DONE）

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

---

## 4. 三个月排期"""
anchor = "---\n\n## 4. 三个月排期"
assert t.count(anchor) == 1
notes.write_text(t.replace(anchor, DONE, 1), encoding="utf-8")
print("revamp notes updated")

# ----------------------------------------------------------------- STATUS
status = ROOT / "STATUS.md"
s = status.read_text(encoding="utf-8")
head = "# 项目状态\n"
entry = """# 项目状态

## 2026-11-04：JSA 拒稿 → 重定位 + C2 已执行

- **拒稿**：JSA-D-26-02451，理由 "lack of sufficient novelty"，**无 reviewer report**（desk reject）。
- **重定位**（B）：论证骨架改为"分配 = f(目标, 汇率, 预算)"三输入，贡献改 5 条，边界声明后移。
  文件：`paper/main.tex`（IEEEtran 母版，13 页）与由 `paper/make_jsa.py` 生成的 `paper/jsa/main.tex`（15 页）。
- **C2 已执行**：llama.cpp b12063 上真跑推荐策略（Q8_0 权重 + 打包缓存）。
  决定性结果：4B @ 32k，fp16 缓存 9,033 MiB 跑不完，q8_0 缓存 6,873 MiB、67 s 完成。
  详见 `results/C2_LLAMACPP.md` 与稿件 §VI-D。
- **校验**：`experiments/run_all.py` 17 阶段 **ALL OK**；数字审计、表-数据一致、路径检查、
  过期数字守卫全绿；两份 PDF 0 error 0 overfull。
- **下一步**：IoT-J 投稿包（cover letter 已更新见 `SUBMISSION.md`），git tag `v3.0-reposition`。

"""
assert s.startswith(head)
s = entry + s[len(head):].lstrip("\n")
status.write_text(s, encoding="utf-8")
print("STATUS.md updated")
