# 核心 Agent 架构决策与实施路线

> 本文记录一个重点关系模型的纵向样板和早期 Agent 路线；公开版的验收口径请以
> `docs/quality_pipeline.md`、`docs/execution/api_contract.md` 和
> `AI_CONTEXT.md` 为准。

> 本文档是当前仓库关于 Agent 核心能力的顶层施工口径。若与三份详细设计文档存在表达粒度差异，以本文档作为优先级和落地顺序依据；三份详细设计文档作为附录级实现参考。

## 1. 已核验的当前能力

- 当前原型已验证从受控数据源到结构化报告的链路。
- 当前 Agent 主体是模型类级 `Planner -> Tool -> Judge` 循环，已经具备白名单工具、预算约束、补证事实、证据链和结构化断言校验。
- 当前通用工具可证明闭环成立，但对复杂模型的业务洞察偏浅，尤其大规模关系模型不能只停留在“命中多少条记录”。
- 当前 Judge 已消费工具摘要和部分 `key_outputs`，但历史版本没有标准化消费 `structured` 与 `anomaly_flags`。
- 当前后端会拒绝重复工具调用，但历史 Planner 上下文没有显式展示已调用工具指纹。

## 2. 对三份详细设计的采纳判断

| 主题 | 判断 | 落地口径 |
|---|---|---|
| 模型级 Investigation Agent | 采纳 | 先只做一个重点关系模型的纵向样板，不立即铺到所有 L2 模型。 |
| 专属结构化工具 | 采纳 | 先实现 `transfer_structure_analyzer`，通用工具保留为 fallback。 |
| `structured/anomaly_flags` 给 Judge | 采纳 | Judge 仍可读 summary，但优先依据结构化字段和异常标记。 |
| 上下文压缩策略 | 采纳 | 模型级 Planner 使用 Top3 实体、Top2 关系、最近轮次摘要、工具 schema、已调用指纹和预算。 |
| ModelClaim | 采纳 | 作为内部中间产物，服务模型类研判和最终 case 报告，不作为前端独立交付报告。 |
| Judge 版本化与 stale 重判 | 部分采纳 | 当前先预留状态模型，完整版本投影后续实现。 |
| Model Dependency Graph | 暂缓 | 当前只预留概念和接口，待纵向样板稳定后再做少量跨类边。 |
| Case Claim Graph | 暂缓 | 当前最终报告仍消费证据链和断言；Claim 图后续建设。 |
| 引入通用 Agent 框架 | 不采纳 | 当前继续使用手写状态机，便于审计和权限控制。 |

## 3. 当前最高优先级实现路线

```text
示例案例 / 重点关系模型
  ↓
ModelBaselinePack（不可变事实基线）
  ↓
确定性种子假设（来自 focus_entities / relation_profiles）
  ↓
模型级 Planner 压缩上下文
  ↓
transfer_structure_analyzer
  ↓
SupplementFact(structured + anomaly_flags)
  ↓
Judge 结构化判断
  ↓
ModelClaim
  ↓
模型类 Agent 只读消费 ModelClaim 摘要
  ↓
最终六段式 case 报告
```

第一阶段验收标准：

- `--analysis-mode full-agent --model-id 86` 可运行。
- 至少产出 1 条 `ModelClaim`。
- 至少产出 1 条来自 `transfer_structure_analyzer` 的 `SupplementFact`。
- 证据链中包含模型级 Agent 的假设、工具、Judge 判断和最终状态。
- 最终报告中模型 86 的发现应体现自转账、对手方集中度、夜间占比、资金循环等结构化信息。

## 4. 状态与边界

- `ModelBaselinePack` 不可变；补证只追加为 `SupplementFact`。
- `ModelClaim` 是当前证据链最新投影，不覆盖基础事实。
- 模型级 Agent 默认只能调用本模型工具；模型类 Agent 可调用本类成员模型工具；case 层默认不自由调用工具。
- 任何 `truncated=true` 或 `aggregation_complete=false` 的工具结果必须保持 `unresolved`。
- LLM 不计算金额、次数、TopK、时间分布和关系指标；这些只能来自确定性工具。

## 5. 后续扩展顺序

1. 稳定模型 86 纵向样板。
2. 扩展模型 43、70、101 的专属工具。
3. 将更多模型级 Claim 接入模型类 Agent 上下文。
4. 建立最小 `Model Dependency Graph`，只做少量业务含义明确的跨类边。
5. 引入 Judge 版本化、stale 重判和 `Case Claim Graph`。
6. 做 SQL 下推、缓存复用和百万级性能验证。
