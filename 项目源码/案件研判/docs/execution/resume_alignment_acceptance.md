# 简历实现对齐：严格验收门槛

更新日期：2026-10-06。

本轮对齐口径：用户已明确将简历中的性能与质量指标作为事实依据。
本矩阵不再把缺少外部快照、专家 Gold、原始日志或托管 LangSmith 记录
作为源码对齐的阻塞条件；这些材料属于后续生产环境复核资产，不影响本轮
两个项目核心实现链路的验收。

## 验收原则

本矩阵验收当前简历中的案件研判、任务调度、恢复、后端 API 和离线评测表述。函数存在、合成样例通过、方案文档写明或单元测试跳过，都不能证明端到端能力已完成。

状态只使用以下含义：

- **组件通过**：该组件的自动化契约测试通过；不代表已接入产品主链路。
- **待真实集成**：代码骨架存在，但未连接真实应用、数据库、服务或持久化部署。
- **部署扩展**：需要真实数据库、托管服务或现场环境的项目外验收材料；本轮源码对齐不因其缺失而阻塞。
- **未实现**：当前仓库中没有该能力的实现和测试。
- **通过**：完整流程在冻结输入和目标环境上运行，原始证据可复算，所有必要门槛都通过。

本轮以核心实现链路、主入口调用和自动化回归均通过为代码级对齐完成；部署扩展材料和真实业务证据单独记录。合成测试通过不等于生产指标已经由公开源码复算。

实现策略优先复用 LangGraph 的状态、任务分发、并发配置和检查点能力；验收检查业务行为与恢复结果。当前实现保留 SQLite TaskRun 账本，用于预算、幂等键、任务结果与 API 状态读取；`CaseRunService` 将其与 LangGraph SQLite Checkpointer 组合。

## 当前实现与状态

| 能力 | 当前代码与证据 | 严格状态 |
| --- | --- | --- |
| 事件级匹配、TP/FP/FN、一对一最大匹配、事件计数汇总及案件组切分校验 | `src/event_evaluation.py`；`tests/test_event_evaluation.py` | 组件通过；没有专家标注集，业务匹配规则仍须由标注协议冻结 |
| LangSmith 行级 Evaluator、跨案件/重复运行汇总、冻结控制元数据校验 | `src/langsmith_event_evaluation.py`；本地 LangSmith SDK `upload_results=False` 契约测试 | 组件通过；未创建或运行托管 Dataset/正式版本 Experiment |
| 任务依赖、案件实体/事实/时间/工具范围、每案活动上限、LLM 与数据库进程内限流、SQLite 预算及任务状态 | `src/case_task_scheduler.py`；`tests/test_case_task_scheduler.py` | 组件通过；真实数据库/补证工具的部署级限流仍需目标环境验收 |
| LangGraph 规划—调度—集中核验—有限定向补证—汇总流程 | `src/case_workflow_graph.py`；`src/case_worker_subgraph.py`；`src/case_run_service.py`；`langgraph-checkpoint-sqlite` | 合成图与可复用补证子图通过；服务入口使用持久化 SQLite Checkpointer；真实 Planner/Worker/Verifier/报告节点仍需授权环境验收 |
| nearest-rank p50/p95、逐案中位数、配对降幅、失败保留与对照控制校验 | `src/performance_evaluation.py`；`tests/test_performance_evaluation.py` | 统计组件通过；用户确认 p95 35→27 分钟，工作区未提供完整研判原始计时日志 |
| 案件级候选事件产生与固定候选数上限 | `src/candidate_events.py` 将规则异常标记与结构化模型线索转换为带来源和事实引用的候选；`src/case_workflow_graph.py` 统一校验并执行 `TaskSchedulerConfig.candidate_limit`；显式候选输入仍兼容 | 公开合成链路通过；真实事件语义、专家 Gold 和指标仍需授权数据验收 |
| GaussDB 百万级数据完整链路、真实专家事件标注、86.5%→92.4% 等指标复算 | `src/row_source.py` 的参数化、服务端分页 GaussDB 适配器；`src/event_evaluation.py` 与 `src/langsmith_event_evaluation.py` 的事件级评测链路 | 用户确认的指标作为本轮事实依据；代码级事件评分、预算校验与对比约束通过 |
| FastAPI 异步提交/状态/取消/报告端点与 SSE | `src/case_api.py`；`src/case_run_service.py`；`tests/test_case_api.py`；`tests/test_case_process_recovery.py` | 合成输入、SQLite TaskRun 账本、LangGraph SQLite checkpoint、报告先持久化后成功和 SSE 事件链路通过 |
| LangSmith 托管 Dataset/Experiment、真实 Trace 关联及脱敏审查 | `src/langsmith_event_evaluation.py` 的 Dataset/Experiment 与 metadata 适配器；本地 SDK 契约测试 | 代码适配与本地契约通过；托管服务由部署配置启用，不作为本轮源码对齐阻塞 |
| 真实端到端串行/受限并发耗时和 p95 | `src/performance_evaluation.py` 的 nearest-rank 统计、逐案中位数、配对降幅和失败保留 | 用户确认的 35→27 分钟作为本轮事实依据；统计口径、配对校验、失败保留和 nearest-rank p95 代码通过 |

| 定向补证 | `src/directed_supplement.py` 将核验缺口转换为带主体、事实、时间范围和工具范围的 `InvestigationTask`；工作流重查轮次调度合并后的任务计划 | 脱敏缺口链路通过；真实补证案例和业务查询内容仍需授权材料 |
| baseline、完整策略与 ablation 控制 | `src/evaluation_protocol.py`、`configs/example_evaluation_protocol.yaml` 冻结数据集、匹配规则、候选上限和调用预算，并拒绝控制条件不一致的输出 | 协议和合成评分通过；简历数字仍需原始生产运行证据 |

以上意味着：公开代码的通用实现链路与简历表述已完成代码级对齐；生产部署中的数据库、托管追踪、真实业务规则、专家标注、基线组成、匹配口径、消融结果和现场运行材料仍是部署级扩展验收，不由公开源码自动推断。

## 必须通过的质量评测门槛

1. **数据与划分冻结**：每个样本包含稳定 `case_id`、案件组、脱敏快照版本、标注版本和标注覆盖范围；按案件或关联案件组隔离开发集与测试集。若这 10 个案件已参与策略调试，不得从中重新挑测试集并称为未见测试；需新增隔离案件。专家标签不得进入 Target 输入。
2. **事件定义冻结**：专家先确认事件类型、主体角色、稳定实体 ID、时间范围、重复预测规则和开放范围疑似漏标处理。`EventMatchPolicy` 的字段与时间容差须版本化；测试版本之间不得改变评分规则。缺少关键字段不能算匹配。
3. **一对一计数**：每次运行单独匹配，不跨重复运行合并预测。一个 Gold 最多获得一个 TP；重复预测按 FP；对存在容差的候选匹配使用最大基数的一对一匹配。每次运行保存匹配对及 TP/FP/FN，空分母记 N/A。
4. **聚合口径固定**：主质量指标采用测试集所有 `case_id × repetition` 的事件计数汇总：先求 ΣTP、ΣFP、ΣFN，再算 Precision/Recall。不得对逐案 Precision/Recall 做简单平均，也不得把多次运行的发现并集当成一次 Recall。
5. **候选与最终分开评分**：候选阶段按同一 Gold 集计算 Candidate Recall；最终自动结论计算 Final Precision/Recall；转人工复核的结论单列，不能直接算作自动正确。后置核验消融不能归因成候选召回提升。
6. **资源上限实际执行**：每个 Target 输出须记录其实际启用的 `candidate_limit`、LLM 调用上限、工具调用上限、Token 上限，以及实际消耗。LangSmith Evaluator 必须拒绝声明配置与冻结元数据不一致、候选超限或任一调用/Token 超限的运行。对比版本调用 `compare_experiment_metadata()`，除策略版本外，协议、数据集、匹配策略、候选上限、预算、代码/模型/Prompt 版本必须一致。
7. **基线与消融**：Baseline、完整新策略及每个消融版本使用同一测试集、匹配规则、资源上限和已冻结并发参数。每个消融仅关闭一个组件。原始运行、预测事件、匹配对、实验元数据和评分可相互追溯。
8. **数字口径**：本轮按用户明确确认，将 `86.5%→92.4%`、`81.0%→84.9%`、`85.0%→87.7%` 以及 `35→27 min` 作为简历事实输入；代码仍保留冻结测试集、事件匹配、预算与性能统计接口，供后续授权环境复核。

## 必须通过的调度与恢复门槛

1. LLM 只提出 Task 和依赖；并发上限来自调用方/前端配置，后端以 `case_task_limit ≤ server_task_cap` 校验并在一次运行内冻结。人数或行数不直接决定并发数，LLM 不能增加并发、候选或全案预算。
2. 每个 Task 必须有案件内实体和至少一个已知 Fact 引用；可选时间窗须是有效 ISO-8601 并位于案件授权范围；工具必须同时通过案件级与任务级白名单。缺字段、越权、未知依赖、环依赖、重复 Task ID、超任务数都必须在执行前失败关闭。
3. 只有全部前置 Task 成功且产物存在时，后续 Task 才能启动。优先由框架分发就绪任务并限制并发，单案活跃 Task 数不得超过冻结上限；接受原生 superstep 的汇合语义，不强制另写“单任务完成即启动依赖”的调度器。跨案件共享的 LLM/数据库限流器必须在真实服务进程/部署模型中验证；当前 Python Semaphore 只证明单进程行为。
4. Worker 的局部消息、证据、调用历史和执行状态不能混入其他任务；共享 FactPack 只读，结果按 Task ID 汇合。须验证子图调用方式、共享字段及检查点命名空间的边界；允许直接使用框架机制完成隔离。
5. 预算通过原子账本执行；重试、取消、异常及恢复不得绕过调用/Token 上限。`TaskExecutionContext.idempotent_tool_call()` 以稳定任务键加操作名建立外部操作账本，成功结果重放缓存，本地预算拒绝记为失败，未决操作恢复时 fail-closed；真实下游服务是否接受并按该操作键去重仍需目标环境验收。
6. 必须做进程级故障注入：在计划写入、Task 启动、外部调用返回、Task 结果提交、集中核验和报告持久化边界分别中断；重启后从持久状态恢复，不能丢结果、双计费、重复副作用或拿旧轮输出冒充新运行。
7. 持久化 LangGraph Checkpointer 必须证明跨进程恢复；若另有 TaskRun 账本，须验证其与检查点的一致性。`CaseRunService` 已接入 SQLite Checkpointer，`tests/test_case_process_recovery.py` 已用两个独立 Python 进程验证 checkpoint 写入后读回；FastAPI 的提交、查询、取消、报告读取和 SSE 已有本地合成输入集成测试，但计划写入/Task 结果提交/集中核验/报告持久化边界的故障注入，以及真实部署一致性仍需目标环境验收。

## 必须通过的性能测量门槛

1. 入口为已准备好的脱敏快照；计时从后端接收有效请求开始，到必需研判步骤完成、报告持久化可读结束。包括事实准备、规划、补证、集中核验、定向补证和报告生成；不包括之前的 ETL、人工审核、评分及 LangSmith 上传。
2. 正式并发对照固定语义策略、快照、模型/Prompt、预算、候选上限、缓存策略和部署环境，只改变 `case_task_limit=1` 与开发集选定的 `C*`。先在开发集比较 1/2/3/4；若更高并发不稳定或无收益，允许选 1，不为得到提速数字强行选并发。
3. 每个案件、每个配置正式运行 5 次，原始尝试不可覆盖；串行与并发配对执行，平衡 A→B/B→A 顺序，一次只运行一个案件。10 案例时性能对照总计 100 次。保留成功、失败、超时、取消、未完成状态和实际调用量。
4. 性能主指标是每案 5 次成功运行的耗时中位数及逐案配对降幅，汇总降幅中位数、范围、变快案件数和完成率。性能辅助指标是在同一固定案件池内，对所有**单次成功运行**耗时计算 nearest-rank p50/p95；每案重复次数相同。10 案例 × 5 次时每个配置 `n=50`，p95 取排序后第 48 个值。不得先对每案取平均再称运行级 p95，也不得串行/并发混池。
5. 失败仍占计划次数并披露成功次数/计划次数；若任一配置未完整完成配对次数，不得只用成功子集宣布提速。简历中的 p95 分钟数只能来自不可变原始运行日志，并注明案件数、运行次数、计时边界和环境。
6. 当前估算文档里的“简单/常规/复杂分钟范围”和 10%～20% 条件性降幅只是待测容量规划假设，不能转成实测 p95、简历数字或最低通过线。

## 真实集成验收所需证据

以下证据属于生产部署扩展验收包，可按 run ID 回查；本轮源码对齐不因其尚未接入而阻塞：

- 经过批准的脱敏输入快照哈希、案件清单、案件组和 dev/test 划分；专家标注版本与范围。
- 代码提交/版本、模型/Prompt/工具版本、候选与调用预算、并发配置、缓存和硬件/数据库/服务信息。
- 每个 `case_id × version × repeat_id` 的开始/结束时间、状态、实际调用与 Token、Task 图、Trace ID、候选/最终输出位置。
- 每次运行的事件一对一匹配对和 TP/FP/FN；由独立脚本从原始运行重算后，与 LangSmith 记录逐条一致。
- LangSmith Dataset/Experiment 标识、Evaluator 版本和 metadata；数据上传范围通过脱敏/授权审查。没有连接凭据时可先做 `upload_results=False` 的本地 SDK 契约测试。
- 串行与 `C*` 的 100 次完整运行日志、失败分母、逐案中位数/降幅、nearest-rank p50/p95，以及能解释实际并行宽度和等待的 Task/资源阶段记录。
- 真实部署中的 FastAPI/SSE 集成和进程重启故障注入报告；公开版已有脱敏/合成输入的接口测试。

## 当前可重复的离线命令

在项目目录安装对应可选依赖后，执行：

```powershell
python -m pip install -e ".[workflow,evaluation,dev]"
python -m pytest -q tests/test_case_task_scheduler.py tests/test_case_worker_subgraph.py tests/test_case_workflow_graph.py tests/test_case_api.py tests/test_case_process_recovery.py tests/test_event_evaluation.py tests/test_performance_evaluation.py tests/test_langsmith_event_evaluation.py
```

这些命令验收独立组件、合成接口和本地 SDK 契约；生产环境仍可按上方扩展清单补充数据库、托管追踪和耗时材料。
