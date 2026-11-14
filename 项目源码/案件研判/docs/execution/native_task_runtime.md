# 原生案件 Runtime 与业务策略

实现版本 native-v3；Python LangGraph 固定 1.2.6、checkpoint 4.1.1、SQLite saver 3.1.1。StateGraph、Send、Reducer、RetryPolicy、@task、AsyncSqliteSaver 与 RunControl 负责通用执行机制，不维护 active dict、asyncio.wait、TaskRun ledger 或独立 restart loop。

## 图与数据

主图：prepare_facts → discover_rules ∥ discover_semantics → select_candidates → plan → dispatch → central_verify → plan_rechecks/ summarize。补证回 dispatch 再核验；轮数由 max_recheck_rounds 配置。语义召回按有界事实批次由 Send 原生并行，覆盖所有已准备事实，不依赖规则先命中。

prepare_facts 复用现有 field_profile、case_fact_pack、entity_profile、anomaly_rules。双路候选统一校验事实引用、实体与时间，按语义身份去重（不信任输入 event_id），在统一候选限额下稳定选择并尽量保留两路覆盖。无效候选记录拒绝原因；超候选限额记录 omitted_count，报告明确 partial。

planner 返回正式 TaskPlan，包含 Goal / Requirement 与候选、Task 的关联；校验覆盖、已知事实、范围、任务数和 DAG。依赖就绪策略只计算谁可运行，Send 调同一个 Worker 子图；同一步遵循 LangGraph superstep 汇合语义，不承诺 FIRST_COMPLETED 的即时补位时序。

Worker 为 plan_step → tool → judge 的 StateGraph。每个操作由原生 @task 使用 RetryPolicy 和 timeout 执行。父节点把耗尽后的已知预算/业务/网络失败转换为 Task outcome；未知程序异常继续抛出，保留修复后的恢复点。重试不会包裹整个 Worker，也不重放已完成的工具节点。HTTP 408/429/5xx、连接故障和超时可重试；权限、schema、预算、程序错误不自动重试。attempt 是 Runtime 的节点执行信息，不维护第二套 Task 尝试计数器。

CaseTaskScope 只表示授权的实体、时间、工具。已存在事实由运行专属 registry 决定，两者分开。每个事实不可变，绑定准入快照、确定性算子/注册工具和行引用；新增事实不能由模型伪造。Worker 输入只有必要事实 ID、前置成功结果的事实引用和对应 Goal；大对象不复制到每个 Send。模型上下文使用有界摘要，需要细节时通过授权工具查询。新事实可被后续 Task 和 Recheck 引用。

## 补证与核验

集中核验按 (goal_id, requirement_id) 生成稳定 gap_id。只有对应需求的 supported/refuted 判断引用已存在、该任务可见且完整的事实，才是有效判断；任务成功、工具成功或没有新增 gap 文本均不能关闭缺口。缺失判断保留 unresolved；相互矛盾判断也转人工。每个补证 Task 引用相关新旧事实、具体 gap_id 和来源 Task；source_task_ids 是来源关系，不强制依赖失败 Task 成功。补证只能回答自己对应的缺口。

全部需求有证据支持才形成 verified 的辅助推断事件；仍明确 requires_human_review。缺少语义模型、数据不完整、候选被拒绝/截断、Task 技术失败或残留缺口时为 partial，报告可读取。未捕获程序/基础设施故障为 failed，尚未完成的报告不发布；cancelled 终止并拒绝恢复。

## 预算、资源与副作用

- case_task_limit 通过每次图调用的 max_concurrency 限制；SharedResourceLimits 的 workers/llm/database 为同进程所有案件的独立资源闸门，默认 4/2/2。它们只限额，不调度任务。没有名不副实的 server_task_cap。
- LLM / Tool 调用和 token 预留以 SQLite BEGIN IMMEDIATE 原子执行；限额来自服务端配置，在准入时冻结。在线适配器关闭 SDK 重试；每次原生重试重新扣预算。文本 BPE 端点使用输入 JSON UTF-8 字节上界及 framing 余量，加输出 token 上限做保守预留，不是事后实际计费统计，也不回收未用预留。特殊 tokenizer 端点应注入相应保守估算适配器。
- 统一 ToolRequest 先校验注册名、参数 schema、Task/Case 实体与时间、模型归属；专属工具的 entity_name/source/target 也必须在 envelope 内。再进入真实资源闸门并扣调用预算。原查询工具在绑定后的记录集合上计算，不让 LLM 生成 SQL。
- 外部写工具必须显式注册 writes=True，稳定 operation key 包含 run/task 与规范化请求；成功结果重放缓存，回执丢失/进程退出后保持 uncertain 并拒绝盲重试，需外部核对。下游仍须接受幂等键；不承诺跨库 exactly-once。外部参数中的实体字段须在 RegisteredTool.entity_params 中声明，schema 禁止未知参数。
- 取消异步等待不能杀死同步线程；数据库许可一直持有至阻塞调用实际结束，避免超时重试冲破资源额度。

## 恢复与部署

准入事务固化 snapshot_ref、完整快照、scope、配置和 graph_version；大对象在业务证据库，图内只存引用。checkpoint 使用独立文件、严格 msgpack allowlist 和 pickle_fallback=False。任务投影和 SSE 不作为恢复依据，投影损坏可由 checkpoint 重建。

thread_id = native-v3:case_id:run_id。重启扫描 running 准入记录，重建相同图，以 ainvoke/astream(None) 从原生 checkpoint 继续；无需重读源数据。完成后先提交 checkpoint，再原子发布报告与终态。若发布前退出，恢复直接发布已完成报告，不重新执行 Worker。发布和取消在同一事务边界裁定。SSE 消费原生 tasks stream 的持久化事件，通过进程通知唤醒并按游标增量查询，保留有界等待以容忍通知丢失。

SQLite 仅支持一个服务进程持有运行库恢复锁，多个案件在同一 asyncio loop 运行。多 worker/多机器部署需要持久数据库、分布式所有权、资源配额及宿主鉴权，不能用进程内 Semaphore 假装分布式限流。关闭时使用原生 drain；未完成调用保留 running 供下次恢复。

native-v2 或旧自定义调度库不自动迁移到 v3。新代码和旧 checkpoint 的图结构不同；请使用独立 runtime/native-v3 目录，旧案件由原版本完成/导出后，再以新 run_id 提交。不得删除已有生产运行库来模拟迁移成功。
