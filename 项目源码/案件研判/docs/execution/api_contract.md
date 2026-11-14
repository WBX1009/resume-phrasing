# API 与 CLI 契约

服务只在 FastAPI lifespan 中创建，导入 src.case_api 无数据库/线程副作用。默认本地监听 127.0.0.1。宿主系统负责认证和逐案件、逐 run 授权；公开 Demo 不是多租户权限服务。

| 方法与路径 | 行为 |
| --- | --- |
| POST /v1/case-runs | JSON {"case_id":"synthetic","run_id":"optional-id"}，202 返回 run_id。未知字段（包括 tasks、case_scope）422；服务端仓库无此案件 404；重复 run_id 400 |
| GET /v1/case-runs/{run_id} | 状态、调用预算、Task 状态计数、report_available，不返回完整证据 |
| POST /v1/case-runs/{run_id}/cancel | 协作取消；已终态返回 false，不中断外部写到一半 |
| POST /v1/case-runs/{run_id}/resume | 从原生 checkpoint 恢复 failed；活动、success、partial、cancelled 拒绝。未知/无兼容准入为 409 |
| GET /v1/case-runs/{run_id}/report | 完成的 success/partial 报告；尚无报告 409；format=json/md/html/docx |
| GET /v1/case-runs/{run_id}/facts/{fact_id} | 仅返回已发布报告引用的本 run 事实，未知或孤立未提交产物 404 |
| GET /v1/case-runs/{run_id}/events | SSE status / complete / error；含任务简表及增量原生执行事件 |

status：running、success、partial、failed、cancelled。success 表示本次配置的流程完成且无显式缺口，不表示模型推断等同法律定性或专家真值。partial 保留数据/覆盖限制和未决缺口；同样允许导出报告。

CLI：python -m src.case_cli --case-id synthetic --repository examples/cases --storage runtime/demo [--offline] [--run-id id] [--report output/report.json]。--resume --run-id id 读取既有准入；报告后缀可选择 md/html/docx。failed/cancelled 退出码 1，success/partial 退出码 0；调用方仍需检查 report.status 和 verification.gaps。

CASE_REPOSITORY 为服务端批准的 JSON 目录，文件名必须匹配 case_id，不能路径穿越或自动替换成其他案件。CaseSnapshot JSON schema 由 case_repository.CaseSnapshot 定义，示例见 examples/cases/synthetic.json。每条记录带 row_id、规范化实体/时间与 data；这些归属信息必须由授权数据适配器验证，不能接受模型或客户端伪造。权限授予与返回字段脱敏属于宿主数据治理。

生产数据库可由 RowSourceCaseRepository(DbRowSource(config), server_policies) 注入；policy 中 records 为空、strategy 指定 person_fields / time_fields 与允许工具。参数化查询后仍执行范围过滤，超限或坏时间记录标记不完整。模型/工具/原始数据来源在准入时固化，后续源数据变化不改变该 run。

新增工具通过 InvestigationTools 的 RegisteredTool 接入，使用 extra="forbid" Pydantic 参数 schema；回调只由服务端注册，模型不能传 Python callback、SQL 或任意 endpoint。写工具必须显式声明 writes、将传入幂等键交给下游。所有已知越界在扣预算与执行前拒绝。
