# 项目实现与材料导航

这里保存与本仓库简历条目对应的两份项目源码。面试准备先按根目录规范完成理解与问答；需要核对具体机制时，再定位下表中的实现和测试。

## 目录职责

| 路径 | 职责与读取入口 |
| --- | --- |
| [案件研判/](案件研判/) | 实习项目；项目概况见 [README.md](案件研判/README.md)，模块地图见 [AI_CONTEXT.md](案件研判/AI_CONTEXT.md) |
| [DeepAssist/](DeepAssist/) | 个人项目；运行方式见 [README.md](DeepAssist/README.md)，能力与测试对应见 [docs/resume_feature_matrix.md](DeepAssist/docs/resume_feature_matrix.md) |
| 案件研判/model_cache/ | 实习业务模型查询结果的缓存；是否包含数据及其性质以该目录说明为准。它不属于大语言模型权重、Agent 记忆或运行数据库 |
| 两个项目的 `tests/`、配置示例及必要文档 | 验证机制、说明输入输出与运行条件；保留最小固定测试样例，生成的结果不提交 |

## 按简历条目定位

| 项目与主题 | 主要路径 |
| --- | --- |
| 案件研判：参数化查询、事实与疑点 | `案件研判/src/row_source.py`、`anomaly_rules.py`、`candidate_events.py` |
| 案件研判：核验与定向补证 | `案件研判/src/case_workflow_graph.py`、`directed_supplement.py`；默认公开服务在 `case_api.py` |
| 案件研判：任务调度、状态与恢复 | `案件研判/src/case_task_scheduler.py`、`case_worker_subgraph.py`、`case_run_service.py` |
| 案件研判：事件与性能评测 | `案件研判/src/event_evaluation.py`、`evaluation_protocol.py`、`langsmith_event_evaluation.py`、`performance_evaluation.py` |
| DeepAssist：架构、规划与执行 | `DeepAssist/backend/domain/interfaces/`、`backend/application/agent_app.py`、`backend/infrastructure/agent_runtime/` |
| DeepAssist：检索与重排 | `DeepAssist/backend/domain/services/rag/`、`backend/infrastructure/rag/`、`backend/infrastructure/rerankers/` |
| DeepAssist：MCP 与 Skill | `DeepAssist/backend/infrastructure/mcp/`、`backend/domain/entities/skill.py`、`backend/domain/services/agent/skill_registry.py` |
| DeepAssist：上下文与记忆 | `DeepAssist/backend/domain/services/agent/`、`backend/infrastructure/memory/`；具体符号按能力矩阵定位 |
| DeepAssist：运行保障 | `DeepAssist/backend/infrastructure/agent_runtime/`、`backend/infrastructure/approvals/`、`backend/infrastructure/tracing/`、`tests/` |

表中未重复项目根路径的文件名，与该单元格首个文件同属 `src/`。避免为核对一个条目读取全部源码。

## 源码与数据边界

- 个人项目提交源码、配置示例、测试及必要文档。`evals/fixtures/` 和固定的 `*_cases.jsonl` 是源码测试样例；下载数据、上传文件、向量索引、模型权重、运行数据库、追踪日志和评测结果留在本地。
- 实习项目保留通用实现与脱敏测试。缓存说明应明确来源和处理方式；原始敏感数据、密钥和内网配置不进入公开仓库。
- 源码与缓存可以解释实现、支持机制测试，不能独立证明简历中的生产指标、真实专家标注或单项策略贡献。相关证据边界见 [对齐验收说明](案件研判/docs/execution/resume_alignment_acceptance.md)。

本次源码来自本地 `简历知识学习/1.实习项目model_insight_agent` 与 `简历知识学习/2.个人项目DeepAssist-main`。本仓库内的文件用于后续远端查阅；此前目录保留原位。

## 测试

分别进入 `项目源码/案件研判` 和 `项目源码/DeepAssist`，安装各自依赖后执行：

```powershell
python -B -m pytest -q -p no:cacheprovider
```

运行时仍可能新建数据库或工作目录；根目录与各项目的 `.gitignore` 用于排除这些生成物。

