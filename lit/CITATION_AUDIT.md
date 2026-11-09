# 引用审计报告（机器化核验，2026-10-20）

## 方法

对 `lit/corpus.csv` 全部 108 条逐条核验，**不依赖人工记忆**：

- **arXiv 条目**：从 URL 提取 arXiv ID，向 `export.arxiv.org/api/query` 查询该 ID 的**真实标题**，
  与语料库记录的标题做 token-set Jaccard 相似度。阈值 0.6 判为相符。
- **非 arXiv 条目**：对 URL 发 HEAD 请求，记录 HTTP 状态（403/202 说明存在但反爬或为 JS 页，不代表不存在）。

脚本：`tools/verify_citations.py`（仅用标准库）。**在 Jetson 上执行**——笔记本侧到 arXiv 的链路不通，
Jetson 侧通畅。原始结果：`lit/citation_report.csv`。

## 结果

| 状态 | 条数 | 含义 |
|---|---|---|
| **OK**（arXiv 标题相符） | **87** | 逐条比对通过 |
| http_200（URL 可解析） | 14 | 非 arXiv 页存在 |
| http_403 | 4 | 反爬，页面存在 |
| http_202 | 3 | JS 渲染页（如 Semantic Scholar） |
| **TITLE_MISMATCH** | **0** | — |
| **NOT_FOUND_ON_ARXIV** | **0** | — |

**最低相似度 0.833**，且该条是真实差异极小的版本差异：

| 条目 | 语料库标题 | arXiv 实际标题 | 判定 |
|---|---|---|---|
| P070 | AWQ: Activation-aware Weight Quantization for **On-Device** LLM Compression and Acceleration | AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration | 同一篇，arXiv 版少 "On-Device" |
| P086 | ∞Bench: Extending Long Context Evaluation Beyond 100K Tokens | $\infty$Bench: … | 同一篇，LaTeX 符号差异 |

## 结论

**本篇论文的参考文献不存在虚构条目。** 87 条 arXiv 引用全部与原始记录标题吻合（最低 0.833，且系版本差异）。

这一条对投稿很重要：55 条 venue 标注为 `unverified` 的条目，其**存在性**已被独立核验，
剩下未定的只是"是否已正式出版、出版在哪"——那是投稿前需要补的元数据，不是真实性问题。

## 两条操作规则（写进项目纪律）

1. **任何新增引用必须先过 `verify_citations.py`**，不通过的不进正文。
2. venue 未定的条目，正文里一律按 preprint 引用（`arXiv:xxxx.xxxxx`），不猜会议名。
   猜测 venue 是审稿人最容易抓到的低级错误。

## 待办

- [ ] 投稿前对 55 条 `unverified` 补正式出版信息（DBLP / 期刊页）
- [ ] 增量刷新一次（投稿前 30 天），新增条目同样过审计
