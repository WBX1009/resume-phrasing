# 案件研判：LangGraph 原生业务闭环

公开学习版以服务端授权的模型结果为输入，确定性代码计算事实，模型提出候选、核查计划与证据判断。**业务策略自定义，任务执行、重试、超时和恢复复用 LangGraph。** API、CLI 使用同一条完整案件图，不再提供旧 Agent Loop、QualityEngine 或独立任务执行器入口。

```text
服务端案件仓库 → 结构化事实
                    ↓
             规则通道 ∥ LLM 通道
                    ↓
          校验 / 去重 / 有界候选选择
                    ↓
      核查目标与信息需求 → TaskPlan
                    ↓
       依赖就绪判断 → Send(Worker)
                    ↓
       Worker: Planner → Tool → Judge
                    ↑        条件边 ↓
                    └──── 继续查询
                    ↓
             集中核验每项信息需求
                    ↓
      独立证据缺口 → 有限轮定向补证 → 再核验
                    ↓
         同一报告 → JSON / Markdown / HTML / DOCX
```

## 运行

Python 3.11+，在本目录执行。运行数据库、下载数据和本地凭据不提交。

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -B -m pytest -q -p no:cacheprovider
.venv\Scripts\python -m src.case_cli --case-id synthetic --repository examples/cases --storage runtime/demo --offline --report output/report.json
```

该示例数据明确为合成数据。`--offline` 运行真实的规则、事实和工具节点，但不提供语义发现与核验：语义通道标记 unavailable_offline，候选保留证据缺口，报告为 partial，绝不把候选直接升级为已验证事件。

在线模式配置服务端 `CASE_LLM_MODEL`、`CASE_LLM_API_KEY`，可选 `CASE_LLM_BASE_URL`；然后去掉 `--offline`。这是实际异步 OpenAI 兼容适配器，不是返回 initial_tasks 的回调示例。SDK 重试关闭，各节点通过 LangGraph RetryPolicy 重试。模型必须支持文本 JSON 输出，字段仍由 Pydantic 和业务规则校验。模型判断是辅助推断，不能替代人工复核。

API 只在 lifespan 中创建服务、数据库与恢复线程，模块导入无这些副作用：

```powershell
$env:CASE_REPOSITORY=(Resolve-Path examples/cases).Path
$env:CASE_RUNTIME_DIR='runtime/api'
$env:CASE_REASONER='offline'  # 在线模式删除该变量并配置模型
.venv\Scripts\python -m uvicorn src.case_api:app --host 127.0.0.1 --port 8000
```

提交 JSON 为 `{"case_id":"synthetic"}`，不接受客户端自授的 case_scope、tasks 或 verification。完整 API 见 [接口契约](docs/execution/api_contract.md)。CLI 的 `--report` 根据后缀支持 .json/.md/.html/.docx；API 通过 `report?format=...` 导出。

## 实现入口

- [架构与恢复语义](docs/execution/native_task_runtime.md)：唯一执行链、原生子图与 @task 的边界。
- [契约](src/investigation_contracts.py)：Fact、Candidate、Goal、TaskPlan、ToolRequest、Assessment、Gap、Report。
- [业务函数](src/investigation_domain.py)：接入已有字段、统计、实体画像、异常规则；检查事实引用与集中核验。
- [数据仓库](src/case_repository.py)：FileCaseRepository；RowSourceCaseRepository 可接现有参数化 DbRowSource，并按服务端策略筛选、限行，固化本次快照。
- [受控工具](src/investigation_tools.py)：统一校验任务/案件白名单、实体、时间、模型、参数 schema，再扣预算并执行；保留原专属补证工具。
- [案件图](src/case_workflow_graph.py)、[任务图](src/case_task_graph.py)、[Worker 图](src/case_worker_subgraph.py)。
- [验收范围](docs/execution/acceptance_criteria.md)、[简历指标边界](docs/execution/resume_alignment_acceptance.md)。

## 数据与部署边界

默认仓库读取管理员准备的规范化 JSON；数据库入口由宿主注入 RowSourceCaseRepository 和授权策略。任务运行始终使用准入时固化的快照，重启不重新读取可能变更的输入。新增事实只追加到本次运行的事实库，不扩大授权范围。State 保存事实 ID、紧凑候选与结果引用，大对象保存在证据存储中。

本服务是本地、单进程 SQLite 部署，不能直接作为公开多租户接口；身份认证与逐案件访问授权应由宿主系统绑定 repository 和 API 完成。每个运行库只允许一个服务持有恢复锁。并发额度在同一事件循环内共享，不承诺跨机器限流或零重复外部副作用。部署要求和迁移说明见运行文档。

真实数据库、专家 Gold、生产凭据和真实案件不在公开仓库。合成端到端测试证明机制，不证明生产准确率、耗时或简历指标。
