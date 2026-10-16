# Evaluation Scorecard Template

这是一份版本化评测记录模板。它不预填业务效果数字；只有有来源的金标准或
可复现命令才能填写 `actual`，否则保持 `unmeasured`。

## Run metadata

| 字段 | 值 |
| --- | --- |
| candidate_version | `unmeasured` |
| snapshot_hash | `unmeasured` |
| split | `dev / validation / backtest / bad-case` |
| run_id | `unmeasured` |
| evaluator | `unmeasured` |
| timestamp | `unmeasured` |

## Layered metrics

| 层级 | 指标 | 计算口径 | 目标/硬门槛 | actual | evidence |
| --- | --- | --- | --- | --- | --- |
| 事实 | schema pass rate | 合法输入中通过字段契约的比例 | dev/回归必须 100% | `unmeasured` | command/test |
| 事实 | Fact correctness | 与参考确定性算子逐项相等 | 有金标准时 100% | `unmeasured` | fixture |
| Agent | route correctness | 任务特征到预期 policy 的比例 | 每个 policy 至少一组 fixture | `unmeasured` | trace |
| Agent | tool request validity | 白名单、参数、实体和预算均通过的比例 | 不得越权 | `unmeasured` | trace/test |
| Agent | duplicate/no-progress rate | 重复动作或无进展轮次占比 | 目标为 0 | `unmeasured` | trace |
| 证据 | Claim grounding | Claim 的 Fact/Model 引用存在且在 scope 内 | 硬门槛 100% | `unmeasured` | replay |
| 证据 | cross-model source count | 候选链接的独立来源数量 | 至少 2，且待核验 | `unmeasured` | report |
| 报告 | section completeness | 必需章节均出现且限制项未丢失 | Markdown 主交付必须满足 | `unmeasured` | report.md |
| 报告 | human groundedness | 领域盲评 rubric | 需金标准后设阈值 | `unmeasured` | review |
| 系统 | replay fingerprint | 同输入重复运行的 fingerprint | 必须稳定 | `unmeasured` | replay |
| 系统 | resume equivalence | 中断恢复与干净运行差异 | 必须为 0 | `unmeasured` | checkpoint |
| 系统 | p95 latency | 离线快照端到端耗时 | 当前门槛 < 1.3s | `unmeasured` | benchmark |
| 成本 | online budget | provider calls/tokens 累计 | ≤200 / ≤2,000,000 | `unmeasured` | manifest |

## Decision

```text
status: unmeasured  # pass / conditional / fail
blocking_metrics:
follow_up_bad_cases:
release_owner:
```

### 记录规则

1. Dev 可以用于调试，不能作为最终业务效果；validation 用于选候选版本；
   backtest 只评估冻结版本；bad-case 修复后必须回归全部相关 split。
2. 没有业务金标准时，只报告结构、证据、稳定性和成本指标，明确写出
   `unmeasured`，不推导 precision/recall。
3. 每次分数变化必须关联代码/策略版本和 bad-case 编号，避免只保留一个
   无法解释的总分。
