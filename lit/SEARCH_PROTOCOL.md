# 检索协议（PRISMA 式，可被审稿人复核）

> 目的：把"收集所有最新论文"这句不可兑现的话，换成**可审计的检索边界**。
> 冻结时间窗：**2023-01-01 → 2026-10-20**（截止日 = 项目启动日，投稿前 30 天做一次增量刷新）

## 1. 数据库与检索式

| 库 | 用途 | 检索式（布尔） |
|---|---|---|
| arXiv (cs.LG/cs.CL/cs.AR) | 最新预印本，覆盖率最高 | `(LLM OR "language model") AND (quantiz* OR compression) AND (edge OR device OR KV OR "mixed precision" OR W4A4 OR Jetson)` |
| OpenAlex | 元数据去重、被引网络 | `quantization large language model` + filter `from_publication_date:2023-01-01` |
| DBLP | 会议正式版核对 | `quantization` × `LLM/Language Model` |
| IEEE Xplore | 期刊侧（IoT-J/TII/TCAD/TACO） | `("large language model") AND (quantization) AND (edge OR embedded)` |
| ACM DL | TACO/TECS/MICRO 系 | 同上 |
| Scopus 导出 | **投稿时用于证明检索覆盖面** | 同 arXiv 式，限 `SUBJAREA(COMP)` |

## 2. 纳入 / 排除标准

**纳入**
- I1：提出或系统评测 LLM 量化/压缩方法
- I2：涉及权重、激活、KV cache 之一
- I3：有实验数据（非纯观点）
- I4：2023-01-01 之后首次公开

**排除**
- E1：只做 BERT 级小模型（除非作为基线来源）
- E2：纯硬件电路设计、无 LLM 实验
- E3：无公开可查版本（找不到 URL/DOI）
- E4：与量化无关的通用压缩（如纯剪枝且不含低比特）

## 3. 去重与筛选流程

```
检索命中 N0
  → 按 DOI/arXiv ID 去重          → N1
  → 标题+摘要筛（双准则 I1–I4）    → N2
  → 全文可得性筛                  → N3
  → 纳入矩阵（人工/半自动）        → N4（目标 120–180）
```

每次筛选记录：`excluded_reason`，最终 PRISMA 图放论文 Fig.1 或附录。

## 4. 台账字段（`corpus.csv`）

```
id,title,year,venue,url,doi,type(pub/preprint),
subarea(A1..A8),method_one_line,quant_target(W/A/KV/W+KV),
hw_evaluated(A100/4090/4060/jetson/cpu/other),
metrics_reported(ppl,acc,latency,energy,memory,thermal),
long_context(bool),budget_aware(bool),regime_aware(bool),code_url,
gap_note,included(bool),excluded_reason
```

**关键三列**：`hw_evaluated`、`regime_aware`、`budget_aware`。
这三列的交叉表就是本文的 gap 证明：**同时满足"在 4060/Jetson 级设备上评测"且"regime_aware"的论文数，预期为 0**。这个"0"要能被 Scopus 导出复核。

## 5. 覆盖率论证写法（审稿人质疑"你没读全"时的标准回应）

> We do not claim exhaustive coverage. We report, for each database, the query string, the retrieval date, the number of hits, and the number of records surviving each screening stage (Fig. 1). A post-hoc saturation check shows that adding the 2026 incremental sweep changed no gap statement.

**饱和检验**：把 2026 新增论文单独跑一遍，若没有新增 gap 类别，即声明"结论已饱和"。
