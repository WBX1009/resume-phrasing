# Model Insight Agent 完成矩阵

本矩阵把用户目标逐项映射到当前仓库的权威证据。除时间门控外，状态必须
由可执行命令或已渲染产物证明；“未来路线”不计入已完成状态。

本矩阵只对应公开版模型结果快照/质量 CLI，不是简历中的案件级事件研判系统
验收结果。不得把本表 `passed` 直接转述为事件评测、真实案件并发或 API 能力；
这些另按 [`resume_alignment_acceptance.md`](resume_alignment_acceptance.md) 验收。

| ID | 目标要求 | 权威证据 | 当前状态 |
| --- | --- | --- | --- |
| R1 | 104 个模型定义中 61 个有样本模型执行、43 个无样本模型安全降级 | `python -m src.quality_cli catalog validate`；最新 evaluate coverage | passed |
| R2 | 确定性算子与 ReAct/Plan-and-Solve/Reflection 组合 | `tests/test_policy_router_covers_all_reasoning_modes_and_trace_events`；`src/quality_reasoning.py` | passed |
| R3 | Markdown 主交付；JSON、trace、manifest、checkpoint 为内部回放证据（HTML/DOCX 仅兼容投影） | 最新 `output/quality_runs/*/`；`report.json` 为机器 source of truth、`report.md` 为用户 source of truth | passed |
| R4 | 单元、契约、端到端、故障注入和 malformed fuzz | `python -m pytest -q`（82）；`scripts/fuzz_quality_pipeline.py`（10,400 + 独立 2,600 + 扩展 26,000） | passed |
| R5 | 中断恢复、无重复证据、回放指纹稳定、篡改 fail-closed | `test_running_checkpoint_can_resume_without_fingerprint_drift`；原子 checkpoint/投影发布；`replay`；projection/DOCX tamper tests | passed |
| R6 | 性能和上下文预算达标 | 10-run benchmark：独立运行 mean 0.8431s / P95 1.1171s / max 1.1186s，均低于 1.3s；61 events、zero overflow | passed |
| R7 | 已有报告与简历 DOCX 视觉兼容证据（后续不作为核心 Markdown 阻塞） | 历史官方 LibreOffice render；`artifacts/visual_qa_report_v51`、`artifacts/visual_qa41`（resume v61） | passed |
| R8 | 删除神经网络预测项目、保留 DeepAssist、缺失元数据不编造 | `test_resume_generator_preserves_deepassist_and_removes_neural_network_project`；`resume_content` gate | passed |
| R9 | 完整路线、生命周期协作、接口契约、dev/validation/backtest/bad-case 评测、开发故事与面试证据包 | `docs/execution/final_roadmap_and_interview_plan.md`、`docs/execution/product_focus_and_lifecycle_plan.md`、`docs/execution/api_contract.md`、`docs/evaluation/bad_case_register.md`、`docs/evaluation/scorecard_template.md`、`docs/resume_evidence.md`、`docs/resume_interview_qa.md` | passed |
| R10 | 至少 24 小时真实 wall-clock 运行 | `docs/execution/goal_state.json` 的 `started_at`/`minimum_completion_at`；`check_goal_gate.py` | time-gated |
| R11 | 并发运行目录、checkpoint、投影、LLM cache 与预算账本安全 | 同秒/双进程 run-id 回归；缓存 1,280 次线程写与双进程回归；32 线程竞争 4-call ledger；DOCX/JSONL 原子发布回归 | passed |
| R12 | 聚合在线预算超限不能绕过完成门禁 | `test_goal_gate_blocks_failed_acceptance_value` 的调用/Token 超限断言；`check_goal_gate.py`（200 calls / 2,000,000 tokens） | passed |
| R13 | 公共 CLI 在缺失/损坏输入时结构化失败 | `test_cli_catalog_validate_reports_locked_coverage` 的缺失目录断言；`src/quality_cli.py` | passed |

## Release rule

只有 R1–R9、R11–R13 仍为 `passed`、R10 时间到达、严格 gate 返回
`can_complete: true`，并且最终全量巡检无失败时，才允许把 goal state 写为
`complete`。公司、团队和实习日期仍必须由本人补充后再用于投递。
