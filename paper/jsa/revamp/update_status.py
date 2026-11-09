from pathlib import Path

p = Path(r"C:\Users\HP\OneDrive\Desktop\Dev\harness\llm_8gb_bit_costs\STATUS.md")
t = p.read_text(encoding="utf-8")

start = t.index("## 2026-11-04")
stop = t.index("以下为 2026-10-20")

new = """## 2026-11-04：JSA 拒稿 → 重定位 + C2 在两台设备上执行

- **拒稿**：JSA-D-26-02451，理由 "lack of sufficient novelty"，**无 reviewer report**（desk reject）。
- **重定位**（B）：论证骨架改为"分配 = f(目标, 汇率, 预算)"三输入，贡献改 5 条，边界声明后移。
  文件：`paper/main.tex`（IEEEtran 母版，13 页）与由 `paper/make_jsa.py` 生成的 `paper/jsa/main.tex`（15 页）。
- **C2 已在笔记本执行**（RTX 4060 Laptop）：llama.cpp b12063 上真跑推荐策略（Q8_0 权重 + 打包缓存）。
  决定性结果：4B @ 32k，fp16 缓存 9,033 MiB 跑不完；q8_0 缓存 6,873 MiB、67 s 完成。
  详见 `results/C2_LLAMACPP.md` 与稿件 §VI-D。
- **C2 已扩展到 Jetson Orin Nano Super**（8 GB 统一内存，sm_87，同一 runtime 在设备上编译）：
  形成 8k(fp16) → 16k(q8_0) → 32k(q4_0) 的阶梯，每减半一次每 token 缓存就换来一次上下文翻倍；
  16k fp16 与 32k q8_0 三次尝试全部被拒（1,792 / 1,904 MiB），而设备峰值只有 5,235 / 7,620 MB。
  PPL 复现笔记本的 +0.10% / +25.06%。详见 `results/C2_JETSON.md` 与稿件 §VI-D 表 VII。
- **校验**：`experiments/run_all.py` 全阶段 **ALL OK**；数字审计（含 Jetson 四项交叉核对）、表-数据一致、
  路径检查、过期数字守卫全绿；两份 PDF 0 error 0 overfull。
- **下一步**：IoT-J 投稿包（cover letter 见 `SUBMISSION.md`），git tag `v3.0-reposition` / `v3.1-jetson`。

"""

p.write_text(t[:start] + new + t[stop:], encoding="utf-8")
print("status updated")
