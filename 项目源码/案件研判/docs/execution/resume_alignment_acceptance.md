# 简历口径与实现证据边界

当前实现以原生案件图为唯一端到端执行路径，详细职责、状态、失败和恢复语义统一引用 [运行文档](native_task_runtime.md)，验证项目见 [验收表](acceptance_criteria.md)。

可以据此解释：确定性事实 → 双通道候选 → 核查目标/需求 → TaskPlan → Send 动态 Worker → 集中核验 → 定向补证。Task 是业务实体；TaskRun 自定义执行账本已移除。原生 @task、RetryPolicy、checkpointer 承担执行、重试和恢复，应用保留依赖、授权、预算、幂等与共享资源限额。

运行时不会把候选、成功 Task、空 gap 列表直接当最终事实。报告中的 verified 是带证据引用、要求人工复核的辅助推断；未决信息需求保留为 gap。离线模式缺少语义模型，必须明确 partial，而不能用预置 synthetic verifier 伪造成功。

用户已在此前会话确认的简历指标仍属于用户提供的事实材料；本轮代码整改不改写简历，也不以公开合成测试独立证明这些指标。复核质量/性能时仍需冻结案件划分、重复次数、模型/Prompt/代码版本、资源额度、候选与最终事件的统计位置及一对一匹配规则。协议与统计函数保留在 evaluation_protocol、event_evaluation、performance_evaluation 和 langsmith_event_evaluation。

公开仓库不携带生产快照、专家 Gold、现场日志、模型凭据或个人信息，不将其缺失伪装成生产验证已完成。新数据库适配器、真实模型端点和下游幂等承诺，应在授权环境中另外验收。
