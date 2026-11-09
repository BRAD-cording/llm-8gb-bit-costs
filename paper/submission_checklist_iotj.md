# IoT-J 投稿检查清单（提交前逐项勾）

生成时间：2026-11-09　·　稿件 commit：`a6ee375`（与 `main.tex` 的 Data Availability 引用一致）

## 一、已核实通过

| 项 | 要求 | 实测 | 状态 |
|---|---|---|---|
| 页数 | IoT-J 无硬上限，超 8 页每页 $175 | 14 页（= $1,050 超页费） | ✅ 知悉 |
| 纸张 | US Letter | 612 × 792 pt | ✅ |
| 字体嵌入 | 全部嵌入，**不得有 Type 3** | 全部嵌入，**Type 3 = 0** | ✅ 本轮修复 |
| PDF | 未加密、PDF 1.5 | 未加密，463 KB | ✅ |
| 摘要 | 150–250 词 | **253 词**（超 3 词） | ⚠️ 见下 |
| 关键词 | IEEEkeywords 块 | 已含（bit allocation / edge inference / KV cache / LLM / memory budget / PTQ） | ✅ |
| 参考文献 | 完整 | 41 条，bbl 41 个 bibitem | ✅ |
| 图表 | 矢量 PDF | fig_stack.pdf、fig_bitcost.pdf | ✅ |
| 数据可用性 | 需声明 | Data Availability 节（GitHub + commit pin + MIT/CC BY 4.0） | ✅ |
| 结论节 | 需有 | IX. Conclusion | ✅ |

## 二、本轮修复记录

**Type 3 字体（会直接导致 PDF eXpress 退稿）**：`pdffonts` 查出 3 处 matplotlib 默认的 `DejaVuSans` Type 3（来自 `fig_stack.pdf`、`fig_bitcost.pdf`）。修法：在 `experiments/make_figures.py` 的 `rcParams` 加 `"pdf.fonttype": 42, "ps.fonttype": 42`，重新生成图并重编。复核 `pdffonts | grep -c "Type 3"` → **0**。

## 三、仍需人工处理

1. **摘要 253 词 → 压到 250 以内**：删 "and it is the question this paper answers" 即可（-8 词），不影响任何数字。
2. **作者信息**：ManuscriptCentral 里需填每位作者的姓名、单位、邮箱、ORCID；通讯作者 Yibo Zhang。
3. **IEEE 电子版权表（eCF）**：提交时会自动弹出，需全体作者确认。
4. **超页费确认**：14 页产生 $1,050 超页费，投稿系统中需勾选接受（或在 cover letter 说明经费已落实）。
5. **Cover letter**：`paper/cover_letter_iotj.md` 已写好，粘进 submission 的 "Cover Letter" 字段即可。
6. **建议审稿人**：可不填（信中已声明）。若填，只能填真实存在、无利益冲突的同行。
7. **投稿系统**：IEEE Author Portal（ManuscriptCentral），需选 *IEEE Internet of Things Journal* → Regular Paper。

## 四、投稿后时间线（按官网均值）

| 节点 | 预计 |
|---|---|
| 首轮决定 | 6.9 周（约 12 月底） |
| 若小修 → 接收 | +4–6 周（约 2 月中） |
| 接收 → ePublication | 合计约 14.5 周（约 2 月中下旬） |

若首轮为大修（需补实验），顺延 2–3 个月；若 desk-reject（通常 1–2 周），立即转 IEEE Access，其首轮 4–6 周仍在期限内的余地。
