"""Append the 2-bit passkey result to the results doc."""
from pathlib import Path

p = Path(__file__).resolve().parent.parent / "results" / "PASSKEY_EVAL.md"
s = p.read_text(encoding="utf-8")

s += """

## 5. 补测：2-bit 缓存把配置分开了

| 缓存方案 | 位宽 | passkey |
|---|---|---|
| 全精度 | 16 | 5/5 |
| 朴素 RTN | 4 | 5/5 |
| KIVI（分组+残差） | 4 | 5/5 |
| **KIVI（分组+残差）** | **2** | **5/5** |
| **朴素 RTN** | **2** | **0/5** |

**结论：4-bit 时方案无关紧要；2-bit 时方案决定一切。**

朴素 2-bit 的输出是填充文本碎片（"The grass is green..."），完全丢失 passkey；
KIVI 的 2-bit 因为保留 128 token 的残差窗口，仍能精确检索。

这给论文提供了一个真正的任务级判据：

- 4-bit 下的困惑度差异是**流畅度**差异，不是检索能力差异；
- 2-bit 下选错方案会**彻底失去**长上下文能力，而这一点在短上下文困惑度上看不出来。

数据：`results/passkey_06b_2bit.json`
"""

p.write_text(s, encoding="utf-8")
print("results doc updated with the 2-bit table")
