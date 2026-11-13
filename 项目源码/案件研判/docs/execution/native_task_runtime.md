# 原生任务运行与恢复

## 执行边界

实现基于 `langgraph==1.2.6`、`langgraph-checkpoint-sqlite==3.1.1`。
案件主图保留规划、候选疑点、集中核验、有限补证和汇总。dispatch 是任务子图的调用边界：

```text
plan → dispatch[resolve_ready → Send(worker) → resolve_ready]
     → candidate_events → central_verify → summarize
                              ↓ 补证
                         plan_rechecks → dispatch
```

`case_execution.py` 定义业务 Task、授权范围、预算、幂等及查询投影。
`case_task_graph.py` 只判断依赖就绪和传播 blocked，然后用 Send 调用同一个 Worker。
`case_worker_subgraph.py` 使用 Runtime.context 注入可重建的资源上下文，取消了内存 invocation 字典。
任务数量运行时变化，图结构不需要重新编译。案件任务只经过这条原生图路径，没有替代执行器或旧版导入别名。

任务子图按 Task ID 合并终态结果，冲突结果拒绝合并；查询表不参与就绪或恢复判断。
前置成功后结果才能传给后继；失败或 blocked 会阻断全部后代，独立任务仍然执行。
成功执行但证据不足属于业务结果，由集中核验处理，不按技术异常重试。

## 并发、异常与预算

单案任务子图通过 max_concurrency 执行冻结的 case_task_limit。
已派发任务可在框架释放额度时补位，新满足依赖的任务等 superstep 汇合后派发。
接受这一执行语义；生产耗时与原实现是否相当必须用冻结输入和资源额度测量。

Worker 默认最多尝试 3 次、每次异步超时 120 秒，可通过 CaseExecutionConfig 配置。
重试分类基于框架默认策略；显式允许超时和 HTTP 408/429/5xx，普通 HTTP 4xx 不重试。每次尝试及退避由 RetryPolicy 执行，没有应用重试循环。
超时、工具/模型服务失败及显式 TaskExecutionError 在 error_handler 中转成 failed 结果；未知 RuntimeError、KeyError、缺失结果等程序错误向上传播。
重试和恢复属于 LangGraph；attempts 是业务库的累计审计记录，不驱动重试。

LLM、数据库请求仍经过共享资源门禁和原子预算预留，包括 Planner、Verifier 和 Summarizer 的模型调用。
真实重试消耗预算；框架复用已持久化结果时不会再次进入调用适配器。同步工具调用被取消后，资源许可保留到实际线程结束，避免重试突破数据库并发上限。
预算是保守的调用/输出 Token 预留，不代表服务端实际计费；进程在预留后退出时不盲目返还额度。

外部写操作使用稳定的 run_id + task_id + operation 幂等键。下游必须实际支持去重。
本地预算拒绝允许修复后重试；一旦请求已派发，异常或进程退出导致的未知结果保持 uncertain，
必须核对下游状态，不将“没有收到结果”解释成“没有执行”。检查点不提供外部 exactly-once。

## 服务恢复与发布

CaseRunService 在一个长期事件循环上承载多个案件，因此共享 asyncio.Semaphore 不跨事件循环。
一个 SQLite 运行库只允许一个服务实例持有文件锁；进程退出由操作系统释放锁。
这不是多副本作业系统，多副本部署需要另行提供作业所有权及分布式资源额度。
关闭服务调用 close()：先通过原生 RunControl 请求在 superstep 边界停止，超时再取消托管协程，保留未完成案件供下次启动恢复。FastAPI lifespan 负责应用关闭时释放后台运行及所有权。

首次提交先持久化 native-v2 admission，包含输入、运行 ID、范围和预算配置，再启动图。
检查点采用 sync durability，thread_id 使用 native-v2:case_id:run_id。
新服务启动时恢复 native-v2 的 running 案件：存在检查点则 ainvoke(None)，
尚未生成检查点则使用持久化输入首次执行。同一服务拒绝并行恢复相同 run。
异常停止的案件可以 POST /v1/case-runs/{run_id}/resume；取消为终态，重新执行须创建新 run。
业务 failed 终态不会因 resume 自动清除结果；新核查应通过补证计划或新 run 明确表达。

case_task_outcomes 保留查询投影与尝试计数，resolve_ready 和最终发布可用图结果修复终态。
图输出报告检查点落盘后，报告和 API 终态在一个业务数据库事务中发布。
若进程在两者之间退出，新服务读取已完成图输出并补发报告，无需重新执行 Worker。
图状态与报告使用 task_outcomes 和 execution_status；结果查询来自 case_task_outcomes 投影。业务 task_id 不等于框架内部 task ID。调用方应使用这套最终字段，未保留旧结果字段别名。

默认业务库为 runtime/case_runtime.sqlite3，检查点为 runtime/langgraph_checkpoints.sqlite3。旧运行库结构会被拒绝，本实现不包含旧版数据迁移器；历史运行数据单独归档，不直接改写检查点。

## 验证

在项目根目录执行：

```powershell
python -B -m pytest -q -p no:cacheprovider tests/test_case_task_graph.py tests/test_case_native_runtime.py tests/test_case_worker_subgraph.py tests/test_case_workflow_graph.py tests/test_case_api.py tests/test_case_process_recovery.py
```

测试覆盖依赖和传递阻断、并发与服务共享额度、任务输入隔离、重试计入预算、超时耗尽、
未知副作用拒绝盲重试、损坏查询记录从检查点修复、取消后的真实资源额度，以及补证复用初始结果。
进程测试在部分任务已保存、集中核验、补证执行和报告发布边界以 os._exit 退出，
新进程自动恢复，断言计划和已保存 Worker 不重跑、未完成 Worker 继续执行、报告可读。
这些是合成输入的机制验证，不证明真实数据库、真实模型效果或生产 p95 指标。

## 复查结论

案件任务的 Send 派发、max_concurrency、RetryPolicy、timeout、error_handler、RunControl
及检查点恢复均交给 LangGraph。业务层只保留任务计划、依赖/阻断规则、范围校验、
预算预留、资源许可、幂等及查询/报告发布；查询表和 attempts 都不决定任务执行或恢复。
生产模型、数据库与业务指标仍需授权环境验收，当前源码与合成测试不能替代这类证据。
