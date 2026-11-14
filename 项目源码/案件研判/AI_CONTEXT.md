# 项目源码阅读入口

唯一端到端入口为 src.case_api:app 或 src.case_cli，两者共享 CaseRunService 与 case_workflow_graph。先读 README 和 docs/execution/native_task_runtime.md，再定位对应模块。

| 职责 | 实现 |
| --- | --- |
| 服务端授权输入、准入快照 | case_repository.py、case_run_service.py |
| 事实准备、规则召回、引用校验、集中核验 | investigation_domain.py；复用 case_fact_pack / entity_profile / anomaly_rules |
| 语义召回、TaskPlan、Worker 决策 | investigation_reasoner.py；在线异步模型，离线明确降级 |
| 案件主图与双通道合流 | case_workflow_graph.py |
| 业务依赖判断、Send、结果 reducer | case_task_graph.py |
| Planner → Tool → Judge 条件循环 | case_worker_subgraph.py；原生 @task 重试/超时/恢复 |
| 参数授权、工具调用、幂等边界 | investigation_tools.py；复用 supplement_tools 与 model_tools |
| 预算、不可变事实存储、查询投影 | case_execution.py |
| 独立缺口与定向补证 | directed_supplement.py |
| 同一报告的多格式投影 | case_report.py |
| 质量与性能评测协议 | event_evaluation / evaluation_protocol / performance_evaluation / langsmith_event_evaluation |

业务 Task 不是 LangGraph PregelTask，也没有自定义 TaskRun 执行器。业务实体、权限、预算、失败阻断、外部幂等是应用策略；可重建任务投影只用于查询。恢复权威仅为原生 checkpoint 与准入快照，不能从投影行重新调度。

本目录已移除旧手写 Agent / QualityEngine 编排与手写模型重试。底层无副作用业务算子可以作为库使用，不能再引入第二套执行/恢复入口。新增行为按上述契约融入节点、条件边、Send 或原生 @task，并覆盖实际业务边界的端到端测试。数据和效果边界见 README。
