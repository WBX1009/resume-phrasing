# FactPack 与 LLM 边界参考

本文是当前施工可参考文档，用于记录跨报告形态通用的后端事实包、证据引用和 LLM 介入边界。它不规定最终 case 报告必须采用五段式或其他固定模板。

> 2026-06-30 更新：本文仍作为 FactPack 与 LLM 边界参考，但当前 demo 已推进到最小 Phase 2 Planner-Executor：LLM 已参与假设与白名单工具选择，补证结果写入 `SupplementFact` / `EvidenceChain`。关于当前真实实现状态、上下文工程差距和下一步策略，以 `README.md` 第 13 节“Phase 2 复盘与下一步上下文工程策略”为准。

当前主线仍以 `README.md` 为准：

```text
模型主线
→ 模型内实体级聚焦
→ 模型级 FactPack / 报告片段
→ case 汇编
```

旧版前端样例报告分析已归档到：

```text
docs/archive/case_report_template_analysis_legacy.md
```

该归档文件只用于理解历史样例报告和表达偏好，不作为当前 demo 施工依据。

## 1. 后端必须产出的通用事实

无论最终报告偏“事实罗列”还是偏“结论研判”，后端都需要先产出可追溯事实包。

模型级 `ModelFactPack` 建议至少包含：

```text
case_id
model_id
model_name
model_tree_path
result_count
field_profile_summary
entity_ranking
key_metrics
anomaly_flags
evidence_refs
frontend_query_params
```

其中：

- `field_profile_summary`：字段覆盖率、字段角色、可解析类型、金额/时间/实体/关系字段识别结果。
- `entity_ranking`：该模型下最值得关注的人员、账户、手机号、机构、商户或关系边。
- `key_metrics`：金额、次数、时间、地点、方向、关系等确定性统计结果。
- `anomaly_flags`：压线、快进快出、整进散出、高频短间隔、异常时间段等规则标签。
- `evidence_refs`：可追溯证据引用，必须能回到模型结果记录或前端查询条件。
- `frontend_query_params`：支持从报告跳转到前端模型结果表格的参数。

## 2. LLM 介入边界

LLM 可以做：

- 将结构化事实组织成自然语言；
- 在 FactPack 范围内提炼重点发现；
- 对复杂关系模型做受控解释；
- 将后端风险标签组织成报告表达；
- 基于事实提出审慎的核查建议。

LLM 不可以做：

- 重新计算金额、次数、均值、TopK；
- 创造 FactPack 外的主体、金额、时间、关系；
- 直接生成 SQL 或 Python 并执行；
- 读取全量原始数据；
- 把低证据线索写成确定性结论。

推荐表达边界：

```text
可说：该线索形成交叉印证，建议重点核查。
慎说：该行为提示存在异常风险。
少说：该主体一定存在违法行为。
不用：坐实、确定、必然等过强表述。
```

## 3. 风险标签与策略库

风险标签应尽量由确定性规则或受控策略产生，而不是由 LLM 自由创造。

可先沉淀的标签包括：

```text
压线交易
快进快出
整进散出
清零过账
高频短间隔
异常时间段
大额集中流入
大额集中流出
多对手分散转出
共同交易对手
关系边高密度
```

每个风险标签建议包含：

```text
risk_tag
trigger_rule
applicable_models
required_fields
metrics
evidence_refs
report_expression
```

## 4. 当前 demo 的建议边界

2026-06-30 更新：本文初版写于 Phase 1，当时 `hypothesis_id`、`evidence_chain`、`allowed_followup_models` 只是预留字段。当前 Phase 2.0 最小闭环已经实现这些能力的一部分。

已实现：

```text
hypothesis_id
evidence_chain
allowed_followup_models
PlannerBrief
SupplementFact
白名单补证工具
```

仍不允许：

```text
LLM 自由生成 SQL/Python 并执行
LLM 读取全量原始明细
LLM 自由选择任意模型
无限轮工具调用
把未验证线索写成确定性结论
```

当前边界：

- 模型级 FactPack 和事实数字仍由确定性算子生成。
- LLM 可以在压缩 BriefPack 中选择假设、工具名和参数。
- 工具调用必须通过白名单和参数校验。
- 补证结果必须写入 `SupplementFact` / `EvidenceChain`。
- 最终 Markdown 报告目前仍主要是模板渲染，不等于完整 LLM 报告生成。
- 下一步以 README 第 13.7 节路线为准，重点是 `ModelClassPack` 和第二轮 LLM 判断。
