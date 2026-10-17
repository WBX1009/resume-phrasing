# model_insight_agent：证据优先的模型研判 Agent

这是从实习项目整理出的公开学习版，保留核心源码、Agent 设计与质量工程思路，去除了真实数据库快照、运行产物、账号密码、API Key、内网地址和真实案例报告。

质量规划器通过 OpenAI 兼容接口调用模型。`LLM_PRIMARY_PROVIDER` 描述服务协议，
`LLM_PRIMARY_MODEL` 描述实际模型。若本地部署千问，请将 provider 设为
`openai-compatible`，并把 `LLM_PRIMARY_BASE_URL` 指向服务端的 `/v1` 端点。
千问是模型，OpenAI-compatible 是推理服务暴露的 HTTP 接口约定。

## 项目解决什么问题

系统把“模型结果快照”转成可追溯的研判报告：

```text
受控快照
  → 目录与字段契约
  → 确定性 Fact 计算
  → 受控策略路由
  → 白名单工具补证
  → Evidence Chain / Claim 校验
  → Markdown 报告与机器回放证据
```

核心边界是：金额、次数、时间、TopK 和关系指标由确定性代码计算；大模型只在受控上下文中参与假设生成、工具选择、证据判断和报告叙事，不能直接读取全量原始数据、生成 SQL 或凭空创造事实。

## 目录

- `src/`：事实包、策略路由、Agent Loop、工具注册、证据链、报告投影和质量流水线。
- `tests/`：单元测试、恢复/回放、断言校验和异常输入测试。部分质量测试依赖原项目的私有快照，公开版没有附带该数据。
- `docs/design/`：模型级 Investigation Agent、专属工具、上下文压缩和整体方案设计。
- `docs/execution/`：接口契约、验收边界和生命周期约束。
- `docs/quality_pipeline.md`：离线质量流水线说明。
- `docs/schemas/`：报告结构契约。
- `src/case_api.py`：异步提交、状态、取消、报告读取和 SSE 事件接口；默认处理脱敏/合成输入，并可从结构化事实与模型线索生成候选风险疑点。
- `configs/example_strategy.yaml`：脱敏后的策略示例，不包含真实记录。
- `configs/example_evaluation_protocol.yaml`：用于说明 baseline、完整策略和单项 ablation 如何冻结评测控制条件。
- `AI_CONTEXT.md`：供 ChatGPT 或其他编程助手快速理解项目的上下文入口。

## 公开版的边界

公开版刻意不包含：

- `.env`、秘钥文件和任何真实凭据；
- 数据库/前端平台地址、真实账号和密码；
- 原始 JSONL/数据库快照、真实身份证号、卡号、手机号、姓名和交易记录；
- `output/`、`artifacts/` 等运行产物；
- 真实案例报告、现场排期和个人工作记录。

本仓库中的案例编号、模型编号和字段名只用于说明代码结构，不应被解释为生产数据或业务效果证明。

## Case run API

`src.case_api:app` 提供 `POST /v1/case-runs`、状态/取消/报告端点和
`GET /v1/case-runs/{run_id}/events` SSE。任务台账写入 SQLite，LangGraph
检查点默认写入独立 SQLite 文件；真实 Planner、Worker、Verifier、GaussDB
连接和报告投影通过 `CaseRunService` 注入。公开版的默认服务不包含生产数据库、
专家标注或线上指标。任务可以显式提供候选风险疑点，也可以把脱敏 fact pack 和
结构化 model clues 交给 `src.candidate_events.py` 生成带来源、事实引用和风险
标签的候选风险疑点；不会把疑点当成已经证实的风险事件。集中核验前统一执行疑点校验与
`candidate_limit` 疑点数量限额。服务入口对每个 Task 复用 `src.case_worker_subgraph.py`
的隔离补证子图；验证缺口可由 `src.directed_supplement.py` 转成范围受限的
recheck Task；外部写操作可通过
`TaskExecutionContext.idempotent_tool_call()` 走 SQLite 幂等账本。
为了让公开版能够离线演示这条闭环，合成任务可以在 payload 中提供结构化
`verification.gaps`；补证任务执行后没有新的缺口即视为该缺口已处理。这个输入边界
只用于脱敏测试，生产环境仍由授权的 Verifier 和补证工具返回结果。

## 本地准备

```powershell
python -m venv .venv
\.venv\Scripts\python -m pip install -r requirements.txt
\.venv\Scripts\python -m compileall src
```

需要运行质量流水线时，请先准备经授权的、脱敏的快照，并通过 `--data-dir` 显式传入；不要把快照提交到 GitHub。

## 推荐阅读顺序

1. `AI_CONTEXT.md`
2. `docs/design/核心Agent架构决策与实施路线.md`
3. `docs/design/模型级Investigation-Agent详细设计.md`
4. `docs/design/专属工具详细设计.md`
5. `docs/design/上下文压缩策略详细设计.md`
6. `src/agent_loop.py`、`src/evidence_chain.py`、`src/final_agent_report.py`
7. `docs/execution/api_contract.md` 与 `docs/schemas/report.schema.json`

## 给 ChatGPT 的使用方式

先让模型阅读 `AI_CONTEXT.md`，再指定要分析的模块或设计问题。涉及代码修改时，要求它先说明事实边界、输入输出契约、测试影响和是否需要真实数据；不要让它猜测私有数据库内容或凭空补齐业务指标。

