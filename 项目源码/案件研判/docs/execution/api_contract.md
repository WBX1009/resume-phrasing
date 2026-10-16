# Model Insight Agent 后端接口契约（v1）

> 目标：冻结核心研判链路的边界，使数据、Agent、工具、报告和评测可以独立替换。本文是实现约束，不代表已经接入真实 HTTP 服务；当前公开入口仍是 CLI。

## 1. 运行请求

~~~json
{
  "contract_version": "1.0",
  "run_id": "client-generated-id",
  "case_ids": [13],
  "model_ids": null,
  "snapshot_sha256": "64-hex-digest",
  "online": false,
  "max_steps": 3,
  "budget": {
    "max_calls": 200,
    "max_tokens": 2000000
  }
}
~~~

### 请求约束

- run_id 是幂等键；同一 scope 重试不得覆盖另一运行。
- case_ids 和 model_ids 为空分别表示全量快照和目录内全部模型；显式过滤必须非空且属于目录。
- snapshot_sha256 必须与加载到的快照一致；不一致在 Fact 计算前失败。
- max_steps 为正整数；online=false 时不得产生 provider 调用。
- 预算上限由服务端硬编码为 200 次/2,000,000 Token，客户端不能通过请求体提高。

## 2. 核心领域对象

### Fact

~~~json
{
  "fact_id": "stable-id",
  "model_id": 86,
  "case_id": 13,
  "kind": "numeric_summary",
  "subject": "交易金额",
  "value": {"count": 10, "max": 90000.0},
  "source_row_ids": [101, 102],
  "source_fields": ["交易金额"],
  "complete": true,
  "provenance": "deterministic_operator"
}
~~~

Fact 只能由确定性算子或已验证工具产生；fact_id、模型/案件范围和来源行不能由 LLM 改写。标识符/日期字段不能被当作金额指标。

### ToolRequest / Observation

~~~json
{
  "step_id": "m86-step-1",
  "tool_name": "transfer_structure_analyzer",
  "params": {"entity": "masked-value"},
  "expected_fact_kinds": ["relation_summary"]
}
~~~

Observation 必须包含 observation_id、status（ok/empty/blocked/error）、规范化输入、输出、Fact 引用和原因。工具名、参数、实体来源、重复指纹、步数和预算全部在执行前校验。

### Claim

~~~json
{
  "claim_id": "stable-id",
  "level": "fact",
  "text": "可读结论",
  "model_ids": [86],
  "fact_ids": ["stable-id"],
  "observation_ids": [],
  "status": "accepted",
  "reason": null
}
~~~

当前 v1 的 Claim 字段以实现和 `report.schema.json` 为准：`level`、`text`、
`model_ids`、`fact_ids`、`observation_ids`、`status` 和 `reason`。报告第 9 节
统一承载“仅限当前快照”等限制；未来 HTTP 适配层若使用 `statement`、
`severity` 或 `basis_fact_ids` 等外部命名，必须在边界处显式映射，不能把未实现
的字段写成当前 CLI 的事实。

每条 Claim 必须引用当前 scope 内存在的 Fact/Model；没有足够证据时只能
downgraded 或 rejected，不能通过改写文字绕过校验。

### Markdown 报告

report.md 必须按固定顺序包含：执行范围、研判摘要、模型覆盖、证据结论、证据索引、引用事实摘要、策略与运行轨迹、跨模型复核入口、限制与待核验、审计摘要。机器回放仍读取 JSON/trace/checkpoint 来验证 Markdown 内容。

## 3. 状态机

~~~text
created -> running -> success
                 \-> failed
running --interrupt--> checkpointed -> resuming -> success|failed
unsupported_no_sample 为模型级终态，不进入 provider 调用
~~~

- running 只能提交一次同一模型的增量状态。
- resuming 必须先验证 case/model scope、snapshot hash、committed model IDs 和 contract version。
- success 只在所有 selected model 状态已提交、Claim/报告校验通过后发布。
- failed 必须保留结构化错误和可重试提示；不能发布“成功” Markdown。

## 4. 错误码与重试策略

| 错误码 | 层级 | 是否自动重试 | 处理 |
| --- | --- | --- | --- |
| INPUT_NOT_FOUND | loader | 否 | 返回结构化诊断，修复路径后重新运行 |
| INPUT_SCHEMA_INVALID | loader | 否 | 给出文件/行号，不生成结论 |
| SCOPE_MISMATCH | resume | 否 | 拒绝恢复，要求使用原 scope |
| UNKNOWN_MODEL_FILTER | API/CLI | 否 | 拒绝空/未知过滤器 |
| TOOL_NOT_ALLOWED | tool | 否 | blocked Observation，继续安全路径 |
| TOOL_DUPLICATE_ACTION | tool | 否 | stop 或降级，记录 fingerprint |
| LLM_BUDGET_EXHAUSTED | provider | 否 | 回退确定性策略 |
| LLM_PROVIDER_UNAVAILABLE | provider | 最多一次有界重试 | 记录原因并降级 |
| CACHE_CORRUPT | cache | 一次清理后重取 | 失败则不发布成功缓存 |
| CLAIM_ORPHAN_REFERENCE | evidence | 否 | fail closed，报告不发布 |
| ARTIFACT_DRIFT | replay | 否 | 返回 invalid，保留原始证据 |

外部依赖阻塞遵守项目级“两次尝试”规则；Provider 的一次内部重试不等同于再次尝试整个阻塞分支。

## 5. CLI/未来 HTTP 映射

| 能力 | 当前 CLI | 未来 HTTP 语义 |
| --- | --- | --- |
| 目录校验 | catalog validate | GET /v1/catalog/coverage |
| 启动运行 | run | POST /v1/runs |
| 运行状态 | checkpoint.json | GET /v1/runs/{run_id} |
| Markdown 读取 | report.md | GET /v1/runs/{run_id}/report |
| 回放验收 | replay | POST /v1/runs/{run_id}/replay |
| 质量评测 | evaluate | POST /v1/evaluations |

未来 HTTP 层只能调用同一 Application Service；不能在 Controller 中重算 Fact 或直接拼接 Claim。

## 6. 契约验收矩阵

| 场景 | 必须证明 |
| --- | --- |
| 正常全量运行 | 104/61/43、Markdown 十章节、Claim 引用闭环 |
| focused run | selected model 集合写入 manifest，未选模型不冒充 processed |
| 空/坏输入 | 稳定错误码、无成功报告、无 traceback |
| 四种 policy | 路由理由、工具/计划/Reflection trace 与停止条件存在 |
| provider/cache/预算 | 失败关闭、预算可见、无伪造成功缓存 |
| 中断恢复 | 不重复模型增量，fingerprint 与干净运行一致 |
| 篡改 replay | manifest、Fact、Claim、Markdown、trace 任一漂移均 invalid |
| split 评测 | dev/validation/backtest/bad-case 版本和证据不混用 |

## 7. 版本演进

- 只向后兼容新增可选字段；改变 Fact/Claim 语义必须提升 contract_version。
- 每次版本升级同时更新 schema、fixture、replay 和 resume 测试。
- 报告正文可以改措辞，但必须由同一结构化事实源重新投影并通过内容哈希/回放。
