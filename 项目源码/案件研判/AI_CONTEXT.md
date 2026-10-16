# AI_CONTEXT

## 项目定位

这是一个“证据优先”的模型结果研判 Agent 学习项目。输入不是原始业务库，而是经过授权、脱敏后的模型结果快照；系统先用确定性算子生成 FactPack，再让受控 Agent 在白名单工具和预算约束内补证，最后生成带引用的报告。

## 关键设计原则

1. **事实与推理分离**：数字、统计、排序、时间范围和关系指标由 Python 代码计算，LLM 不负责事实计算。
2. **受控 Agent**：Planner 只能选择注册过的工具，执行器校验参数、模型调用边、重复调用和预算。
3. **追加式证据链**：基础事实不可覆盖；工具结果以 SupplementFact 追加；每条 Claim 必须引用已存在的 Fact/Model。
4. **保守降级**：无数据、字段不完整、工具失败或证据不足时，输出 `unsupported` / `unresolved` / `downgraded`，不猜结论。
5. **可回放**：报告、trace、manifest、checkpoint 和 fingerprint 服务于恢复、篡改检测和重复运行校验。
6. **离线优先**：默认运行不调用在线模型；在线 LLM 必须显式配置，并且不能越过事实和预算边界。

## 代码阅读地图

- `src/quality_registry.py`：模型目录、字段契约、策略族和跨模型依赖。
- `src/quality_operators.py`：确定性事实算子。
- `src/quality_reasoning.py`：策略路由、工具调用和 Claim 构建。
- `src/agent_loop.py`：模型类级 Planner → Tool → Judge 循环。
- `src/supplement_tools.py`：补证工具及参数约束。
- `src/candidate_events.py`：把确定性异常标记和结构化模型线索整理为可追溯候选事件。
- `src/directed_supplement.py`：把核验缺口转换为范围受限的补证任务。
- `src/evaluation_protocol.py`：冻结 baseline、完整策略和 ablation 的评测控制条件。
- `src/evidence_chain.py`：证据步骤、引用和状态转移。
- `src/final_agent_report.py` / `src/report_writer.py`：最终报告和多格式投影。
- `src/quality_engine.py`：质量流水线、checkpoint、resume、replay 和输出物管理。
- `src/config.py` / `src/data_loader.py`：环境配置与数据源适配；不要在其中写入凭据。

## 端到端数据流

```text
Snapshot
  → Registry / Schema validation
  → Deterministic Facts
  → Policy selection
  → Whitelisted ToolRequest
  → Supplement Facts
  → Evidence Chain
  → Claim validation
  → report.json → report.md / replay artifacts
```

## 后续指导约束

当 ChatGPT 修改这个项目时，优先遵循以下顺序：

1. 先定位现有输入、输出和失败关闭路径，再提出改动。
2. 不把真实数据、内网地址、账号、密码、API Key 或个人信息写入代码、测试、日志和文档。
3. 不把抽样快照的结果写成生产精度、召回率或业务收益。
4. 任何新工具都必须有白名单注册、参数 schema、预算/重复调用控制、结构化 Observation 和测试。
5. 任何新 Claim 都必须能回指 Fact/Model，并覆盖无数据、坏数据、工具失败和恢复场景。
6. 先补最小离线测试，再考虑在线模型或数据库适配。

## 公开版已知限制

公开版没有携带原项目的真实快照，因此依赖私有快照的全量质量测试不能直接运行。应使用新的合成或经授权脱敏 fixture 重新建立测试基线，而不是复制原始业务数据。

公开版候选生成只接受确定性 FactPack 和结构化 model clues；它不会从自由文本推断真实风险事件。事件匹配策略、真实风险规则、基线流程、补证案例、专家标注和简历中的生产指标仍需由授权材料确认。
默认公开服务还接受任务 payload 中的结构化 `verification.gaps` 作为合成核验边界，
用来演示缺口转补证任务和再次核验；这不代表公开版包含生产 Verifier 或真实补证结果。
