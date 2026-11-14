# 验收与复现

在项目目录安装 requirements.txt 后执行：

```powershell
python -B -m pytest -q -p no:cacheprovider
python -B -m compileall -q src tests
```

CI 位于仓库 .github/workflows/case-investigation.yml，在 Linux / Windows 的 Python 3.11 环境运行同一套无密钥测试。完整测试不依赖私有快照、在线模型或真实数据库。2026-11-14 在 Windows / Python 3.12.14 按固定依赖执行：86 passed，2 条来自 LangSmith SDK 的弃用/Beta 警告；pip check 与编译检查通过。线上环境仍需按真实模型与授权数据执行效果验收。

| 业务边界 | 测试 |
| --- | --- |
| 无预置 Task，候选先于计划，原生多步 Worker、依赖新事实 | test_case_end_to_end.py |
| 两路候选、范围/事实锚点、去重与有界选择 | test_case_authorization.py |
| 参数越界在调用预算和实际 I/O 前拒绝，原专属工具复用 | test_case_authorization.py |
| 独立 gap_id、补证引用新证据、只关闭对应需求 | test_case_end_to_end.py |
| RetryPolicy、单节点超时、错误分类、失败阻断、取消 | test_case_runtime_controls.py |
| 每案与共享资源额度、阻塞线程取消不提早归还资源 | test_case_runtime_controls.py |
| 原子预算、真实模型适配器关闭 SDK 重试并预留输入/输出 | test_case_policy.py |
| 外部写幂等、回执不确定拒绝重放 | test_case_runtime_controls.py |
| 硬退出后重启，工具→Judge、核验、补证、报告发布边界 | test_case_process_recovery.py |
| 原生并行 pending writes，成功兄弟任务不重放；不读取内部 writes 表 | test_case_process_recovery.py |
| API / CLI 唯一入口、partial 报告、SSE 投影损坏容错、导入无副作用 | test_case_api.py |
| 同一报告的多格式事实引用一致、SQL 参数化 | test_case_policy.py |
| 数据库仓库准入、连接关闭、有界上下文及失败任务的已完成事实保留 | test_case_repository.py |
| 事件计分、冻结评测条件、LangSmith evaluator、耗时统计 | test_event_evaluation / test_evaluation_protocol / test_langsmith_event_evaluation / test_performance_evaluation |

旧手写编排和私有快照专属测试随入口删除；关键预算、授权、幂等、重试、失败阻断、恢复断言在新端到端路径重新覆盖。测试条数不与旧架构直接比较。当前验收不包含生产数据库连通性、真实模型语义准确率、专家标注或多机器部署能力。
