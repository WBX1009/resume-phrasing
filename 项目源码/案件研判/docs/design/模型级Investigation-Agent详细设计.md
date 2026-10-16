# 模型级 Investigation Agent 详细设计

> 本文档是 Agent 引擎核心架构的施工级设计说明。Codex 应基于此文档实现模型级 Investigation Agent，然后将其接入现有模型类 Agent。
>
> 阅读前提：先阅读根目录 `AI_CONTEXT.md`，并结合 `src/agent_loop.py` 理解当前的模型类级循环实现。

## 一、定位与边界

### 当前状态（代码事实）

当前 `src/agent_loop.py` 的 `run_agent_loop_demo()` 实现的是**模型类级 Agent 循环**：

```
对每个推理模型类（多维分析/金融画像/人员画像）：
  1. ModelClassPack → 候选假设
  2. LLM Planner 选 1 条假设 + 1 个工具
  3. 执行工具 → SupplementFact
  4. LLM Judge → enhanced / weakened / unresolved
  5. enhanced/weakened → 停止；unresolved → 下一轮（最多 3 轮）
```

**缺失**：每个重点复杂模型（L2）内部的独立调查循环。当前模型类 Agent 在模型类级做假设→工具→判断，但它没有深入到单个模型内部做"逐步收窄"的调查。例如对于模型 86（银行转账关系分析，42 万条结果），当前只调用一次 `relation_neighbor_subgraph` 就停止——这太浅。

### 目标架构

```
Phase 1: Baseline Pipeline（所有模型，确定性）
  → ModelBaselinePack（不可变事实基座）

Phase 2: Model Investigation Agent（仅 L2 primary 模型）
  → 对每个重点模型独立运行循环
  → 多轮：假设 → 工具调用 → Judge → 收窄 → 再假设
  → 产出：ModelClaim + ModelEvidencePack

Phase 3: ModelClass Corroboration Agent（模型类级）
  → 消费 ModelClaim + ModelBaselinePack
  → 识别跨模型交叉印证/矛盾/缺口
  → 产出：ModelClassClaim

Phase 4: Case Claim Graph（案例级）
  → 消费所有 Claim，去重、排序、冲突检测
  → 产出：ranked_claims → 驱动最终报告
```

**本次施工范围：Phase 2（模型级 Investigation Agent），先只做模型 86。**

---

## 二、整体代码骨架

### 2.1 新增模块

```
src/
  model_investigation.py    # 新增：模型级 Investigation Agent
  model_tools/
    __init__.py             # 新增：模型专属工具包
    base.py                 # 工具基类和通用接口
    model_86_tools.py       # 模型 86 专属工具
    model_70_tools.py       # 模型 70 专属工具（后续扩展）
    ...
```

### 2.2 核心数据结构

```python
# src/model_investigation.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import json
import re
import time

from .evidence_chain import (
    build_evidence_chains,
    make_chain_step,
    make_hypothesis as _make_hypothesis,
    make_planner_decision,
    make_supplement_fact,
    is_incomplete_tool_output,
)
from .execution_policy import FallbackPolicy, normalize_fallback_policy
from .judge import build_judge_steps
from .llm_utils import cached_chat_json, model_candidates, retry_delay_seconds, update_audit_from_result
from .supplement_tools import execute_tool, ToolExecutionBudget
from .tool_registry import TOOL_REGISTRY, allowed_tools_for_strategy, validate_tool_request


@dataclass(frozen=True)
class ModelInvestigationConfig:
    """模型级调查循环的配置"""
    model_id: int                              # 目标模型
    max_rounds: int = 4                        # 最多调查轮次
    max_tool_calls: int = 6                    # 最多工具调用总数
    max_invalid_requests: int = 2              # 最多无效请求数
    max_tool_rows_per_call: int = 50_000       # 单次工具调用最大扫描行数
    max_tool_unique_groups: int = 20_000       # 单次工具调用最大分组数
    max_tool_evidence_records: int = 5         # 单次工具调用最大证据记录数
    fallback_policy: str = "strict_demo"

    @property
    def tool_budget(self) -> ToolExecutionBudget:
        return ToolExecutionBudget(
            max_rows_per_call=self.max_tool_rows_per_call,
            max_unique_groups=self.max_tool_unique_groups,
            max_evidence_records=self.max_tool_evidence_records,
        )


@dataclass(frozen=True)
class ModelInvestigationResult:
    """模型级调查的完整产出"""
    model_id: int
    hypotheses: list[dict[str, Any]]           # 所有假设
    supplement_facts: list[dict[str, Any]]     # 所有补证结果
    chain_steps: list[dict[str, Any]]          # 所有证据链步骤
    evidence_chains: list[dict[str, Any]]      # 完整证据链
    claims: list[ModelClaim]                   # 形成的模型级声明
    audit: dict[str, Any]                      # 审计信息
    tool_cache: dict[str, dict[str, Any]]      # 工具缓存

    @property
    def enhanced_claims(self) -> list[ModelClaim]:
        return [c for c in self.claims if c.status == "enhanced"]

    @property
    def unresolved_claims(self) -> list[ModelClaim]:
        return [c for c in self.claims if c.status == "unresolved"]


@dataclass(frozen=True)
class ModelClaim:
    """模型级声明——模型内 Agent 调查完成后形成的可引用结论"""
    claim_id: str                              # 如 "m86_claim_01"
    model_id: int                              # 所属模型
    entity_name: str | None                    # 关注主体
    hypothesis_id: str                         # 关联的假设 ID
    status: str                                # enhanced | weakened | unresolved
    statement: str                             # 中文结论陈述
    basis_fact_ids: list[str]                  # 引用的基础事实 ID
    supplement_fact_ids: list[str]             # 引用的补证事实 ID
    evidence_chain_id: str                     # 关联的证据链 ID
    limitation: str                            # 限制说明
    severity: str = "一般关注"                 # 重点关注 | 一般关注 | 待核查
    version: int = 1                           # 版本号
    superseded_by: str | None = None           # 被哪个新版本取代


@dataclass
class InvestigationRound:
    """单轮调查的记录"""
    round_index: int
    tool_name: str
    params: dict[str, Any]
    tool_output_summary: str
    tool_status_signal: str                    # enhanced | weakened | unresolved
    judge_status: str
    judge_assessment: str
    rows_read: int = 0
    truncated: bool = False
```

### 2.3 主函数骨架

```python
def run_model_investigation(
    config: Any,
    case_id: int,
    model_id: int,
    model_baseline: dict[str, Any],             # 本模型的 ModelBaselinePack（即 model_factpack）
    model_strategy: dict[str, Any],             # 本模型在 YAML 中的策略配置
    entity_dictionary: Any,                     # EntityDictionary 实例
    row_source: Any,                            # RowSource 实例
    investigation_config: ModelInvestigationConfig | None = None,
    external_tool_cache: dict[str, dict[str, Any]] | None = None,
) -> ModelInvestigationResult:
    """
    对单个模型运行 Investigation Agent 循环。

    Args:
        config: AppConfig 实例
        case_id: 案例 ID
        model_id: 目标模型 ID
        model_baseline: 该模型的 ModelBaselinePack（由 Phase 1 生成）
        model_strategy: 该模型的策略配置（从 YAML bundle 中获取）
        entity_dictionary: 实体字典
        row_source: 数据库行源
        investigation_config: 调查配置
        external_tool_cache: 外部工具缓存（用于跨模型复用）

    Returns:
        ModelInvestigationResult: 完整的调查结果
    """
```

## 三、循环机制的详细设计

### 3.1 循环初始化：从 Baseline 生成种子假设

模型级 Agent 的第一步不是让 LLM 自由发挥，而是**从 BaselinePack 中提取结构化的候选假设**。这确保了 Agent 的起点是可追溯的事实，不是 LLM 的想象。

```python
def _seed_hypotheses_from_baseline(
    model_id: int,
    model_baseline: dict[str, Any],
    model_strategy: dict[str, Any],
    max_hypotheses: int = 3,
) -> list[dict[str, Any]]:
    """
    从模型基线事实中确定性生成种子假设。

    每条假设必须：
    - 绑定到一个具体的实体/关系（entity_name 来自 focus_entities 或 relation_profiles）
    - 包含一个可证伪的陈述（不是"可能有异常"，而是"该主体的自转账占比和夜间高频
      是否形成集中控制的证据"）
    - 关联具体的 fact_id（可追溯到 BaselinePack 中的具体事实）
    - 声明期望的证据类型（指导后续工具选择）

    种子假设不是 LLM 生成的——它由代码从 entity_ranking 和 anomaly_flags 中
    确定性构建。LLM Planner 的职责是选择先调查哪条、用什么工具、以及如何在
    调查过程中动态生成新的收窄假设。
    """
    hypotheses = []

    # 来源 1：Top Focus Entities
    for entity in model_baseline.get("focus_entities", [])[:3]:
        entity_name = entity.get("entity_name")
        fact_id = entity.get("fact_id")
        if not entity_name or not fact_id:
            continue

        metrics = entity.get("metrics", {})
        tags = [f.get("risk_tag") for f in entity.get("anomaly_flags", [])]

        # 根据风险标签构造假设方向
        if "high_frequency_window" in tags:
            direction = "存在短时间集中交易模式，需要验证时间窗口内的对手方和金额分布"
        elif "many_counterparties" in tags:
            direction = "对手方数量较多，需要验证关系网络的集中度和关键路径"
        elif "self_transfer_high_density" in tags:
            direction = "自转账占比异常高，需要验证是否存在账户间归集/分散模式"
        elif "near_threshold_amount" in tags:
            direction = "存在临界金额交易，需要验证金额分布和规避阈值的模式"
        else:
            direction = "是该模型的重点主体，需要验证其交易行为是否存在可报告线索"

        statement = (
            f"{entity_name}（记录{metrics.get('record_count', 0)}条，"
            f"金额{_fmt_amount(metrics.get('amount_sum', 0))}元）{direction}"
        )

        hypotheses.append(
            _make_hypothesis(
                len(hypotheses) + 1,
                entity_name,
                statement,
                [fact_id],
                "model_baseline_focus_entity",
                [model_id],
            )
        )

    # 来源 2：Top Relations（仅 relation 类模型）
    for relation in model_baseline.get("relation_profiles", [])[:2]:
        source = relation.get("source")
        fact_id = relation.get("fact_id")
        if not source or not fact_id:
            continue

        statement = (
            f"{source} → {relation.get('target')} 是重点关系边"
            f"（{relation.get('count', 0)}笔，{_fmt_amount(relation.get('amount_sum', 0))}元），"
            f"需要验证邻域结构和资金承接路径"
        )
        hypotheses.append(
            _make_hypothesis(
                len(hypotheses) + 1,
                source,
                statement,
                [fact_id],
                "model_baseline_relation",
                [model_id],
            )
        )

    return hypotheses[:max_hypotheses]
```

### 3.2 主循环逻辑

```python
def _run_investigation_loop(
    config: Any,
    case_id: int,
    model_id: int,
    model_baseline: dict[str, Any],
    model_strategy: dict[str, Any],
    entity_dictionary: Any,
    row_source: Any,
    hypotheses: list[dict[str, Any]],
    available_tools: list[str],
    investigation_config: ModelInvestigationConfig,
    tool_cache: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[ModelClaim], dict[str, Any]]:
    """
    主调查循环：对每条假设运行 Planner → Tool → Judge → 收窄循环。

    返回: (supplement_facts, chain_steps, evidence_chains, claims, audit)
    """
    cfg = investigation_config
    supplement_facts: list[dict[str, Any]] = []
    chain_steps: list[dict[str, Any]] = []
    claims: list[ModelClaim] = []
    all_called_keys: set[str] = set()
    total_invalid_count = 0

    audit = {
        "mode": "model-investigation",
        "model_id": model_id,
        "status": "not_started",
        "hypothesis_count": len(hypotheses),
        "total_tool_calls": 0,
        "total_invalid_requests": 0,
        "enhanced_claims": 0,
        "weakened_claims": 0,
        "unresolved_claims": 0,
        "llm_call_count": 0,
        "fallback_used": False,
        "hypothesis_runs": [],
    }

    for hypothesis in hypotheses:
        hyp_id = str(hypothesis.get("hypothesis_id"))
        entity_name = hypothesis.get("entity_name", "未知主体")
        previous_rounds: list[InvestigationRound] = []
        hyp_supplements: list[dict[str, Any]] = []
        called_keys: set[str] = set()
        tool_call_count = 0
        invalid_count = 0
        final_status = "unresolved"
        stop_reason = ""

        agent_round = 1
        while agent_round <= cfg.max_rounds:
            if tool_call_count >= cfg.max_tool_calls:
                stop_reason = "tool_budget_exhausted"
                break

            # ── Step A: LLM Planner 决策 ──
            planner_context = _build_model_planner_context(
                model_id=model_id,
                model_baseline=model_baseline,
                hypothesis=hypothesis,
                previous_rounds=previous_rounds,
                available_tools=available_tools,
                model_strategy=model_strategy,
                all_called_keys=all_called_keys | called_keys,
                budget_rounds_remaining=cfg.max_rounds - agent_round + 1,
                budget_calls_remaining=cfg.max_tool_calls - tool_call_count,
            )

            # 尝试 LLM Planner
            request = None
            validation_feedback: list[str] = []
            for _ in range(cfg.max_invalid_requests):
                request, planner_audit = _ask_planner_for_next_action(
                    config, planner_context, agent_round, validation_feedback
                )
                audit["llm_call_count"] += planner_audit.get("llm_call_count", 0)

                if request is None:
                    request = _fallback_model_action(
                        hypothesis, model_id, available_tools, agent_round, previous_rounds
                    )
                    audit["fallback_used"] = True

                if request is None:
                    stop_reason = "no_available_action"
                    break

                if request.get("action") == "stop":
                    stop_reason = request.get("stop_reason", "llm_stop")
                    break

                # 校验工具请求
                ok, reason = _validate_model_tool_request(
                    request, model_id, model_strategy, available_tools,
                    called_keys, entity_name
                )
                if ok:
                    break

                invalid_count += 1
                total_invalid_count += 1
                validation_feedback.append(reason)

            if stop_reason or request is None:
                break

            if request.get("action") == "stop":
                break

            # ── Step B: 执行工具 ──
            tool_name = request["tool_name"]
            params = request["params"]
            call_key = _tool_call_key(model_id, tool_name, params)
            called_keys.add(call_key)
            all_called_keys.add(call_key)

            try:
                output = execute_tool(
                    tool_name, config, case_id,
                    {model_id: model_strategy},  # strategies_by_model
                    entity_dictionary,
                    [model_baseline],             # factpacks
                    params,
                    {},                           # thresholds — 从 strategy 中获取
                    row_source=row_source,
                    tool_cache=tool_cache,
                    budget=cfg.tool_budget,
                )
            except Exception as exc:
                output = {
                    "summary": f"工具执行失败：{type(exc).__name__}: {str(exc)[:120]}",
                    "row_count": 0,
                    "status_signal": "unresolved",
                }

            supplement = make_supplement_fact(
                len(supplement_facts) + 1,
                hyp_id,
                make_planner_decision(
                    hyp_id, agent_round, tool_name, params,
                    request.get("reason", ""), "llm_model_planner"
                ),
                output,
            )
            supplement_facts.append(supplement)
            hyp_supplements.append(supplement)
            tool_call_count += 1

            tool_status = _determine_tool_status(output)
            chain_steps.append(
                make_chain_step(
                    hyp_id, agent_round,
                    make_planner_decision(hyp_id, agent_round, tool_name, params,
                                          request.get("reason", ""), "llm_model_planner"),
                    supplement, tool_status,
                    f"{output.get('summary', '')} 工具信号：{tool_status}",
                )
            )

            # ── Step C: LLM Judge 判断 ──
            judge_steps, judge_audit = build_judge_steps(
                config, [hypothesis],
                [s for s in supplement_facts if s.get("hypothesis_id") == hyp_id],
                [s for s in chain_steps if s.get("hypothesis_id") == hyp_id],
                agent_round + 1,
                fallback_policy=cfg.fallback_policy,
            )
            chain_steps.extend(judge_steps)
            audit.setdefault("judge_audits", []).append(judge_audit)

            judge_step = judge_steps[-1] if judge_steps else {}
            judge_status = str(judge_step.get("status", "unresolved"))
            judge_assessment = str(judge_step.get("assessment", ""))

            # ── Step D: 记录本轮 ──
            previous_rounds.append(InvestigationRound(
                round_index=agent_round,
                tool_name=tool_name,
                params=params,
                tool_output_summary=output.get("summary", ""),
                tool_status_signal=tool_status,
                judge_status=judge_status,
                judge_assessment=judge_assessment,
                rows_read=output.get("rows_read", 0),
                truncated=bool(output.get("truncated")),
            ))

            final_status = judge_status

            # ── Step E: 终止判断 ──
            if judge_status in ("enhanced", "weakened"):
                stop_reason = f"judge_{judge_status}"
                break

            if tool_call_count >= cfg.max_tool_calls:
                stop_reason = "tool_budget_exhausted"
                break

            if _no_investigation_progress(previous_rounds):
                stop_reason = "no_progress"
                break

            agent_round += 1

        # ── 为当前假设生成 ModelClaim ──
        claim = _synthesize_claim(
            model_id=model_id,
            hypothesis=hypothesis,
            supplement_facts=hyp_supplements,
            status=final_status,
            stop_reason=stop_reason,
            previous_rounds=previous_rounds,
        )
        claims.append(claim)

        # 更新审计
        audit["hypothesis_runs"].append({
            "hypothesis_id": hyp_id,
            "entity_name": entity_name,
            "final_status": final_status,
            "stop_reason": stop_reason,
            "tool_call_count": tool_call_count,
            "invalid_request_count": invalid_count,
            "rounds": [
                {
                    "round": r.round_index,
                    "tool_called": r.tool_name,
                    "params": r.params,
                    "judge_status": r.judge_status,
                }
                for r in previous_rounds
            ],
        })

    # 组装证据链
    evidence_chains = build_evidence_chains(hypotheses, chain_steps)

    # 更新审计
    status_counts = {"enhanced": 0, "weakened": 0, "unresolved": 0}
    for claim in claims:
        status_counts[claim.status] = status_counts.get(claim.status, 0) + 1
    audit.update({
        "status": "success" if claims else "no_claims",
        "total_tool_calls": sum(
            run.get("tool_call_count", 0) for run in audit["hypothesis_runs"]
        ),
        "total_invalid_requests": total_invalid_count,
        "enhanced_claims": status_counts["enhanced"],
        "weakened_claims": status_counts["weakened"],
        "unresolved_claims": status_counts["unresolved"],
    })

    return supplement_facts, chain_steps, evidence_chains, claims, audit
```

### 3.3 循环终止条件（优先级从高到低）

```
1. LLM Planner 主动 stop（认为证据已足够）
2. Judge 返回 enhanced → 形成正面 Claim，正常退出
3. Judge 返回 weakened → 形成负面 Claim，正常退出
4. 工具调用次数达到上限 → unresolved Claim
5. 调查轮次达到上限 → unresolved Claim
6. 连续两轮无进展（同一工具+同参数返回相同结果，Judge 仍为 unresolved）
7. 所有可用工具均已调用过 → unresolved Claim
8. LLM Planner 连续产生无效请求（超过 max_invalid_requests）
```

### 3.4 关键：LLM Planner 的上下文设计

模型级 Planner 的上下文与模型类级有本质不同——它只关注**一个模型**，但需要更深入的工具选择：

```python
def _build_model_planner_context(
    model_id: int,
    model_baseline: dict[str, Any],
    hypothesis: dict[str, Any],
    previous_rounds: list[InvestigationRound],
    available_tools: list[str],
    model_strategy: dict[str, Any],
    all_called_keys: set[str],
    budget_rounds_remaining: int,
    budget_calls_remaining: int,
) -> dict[str, Any]:
    """构建模型级 Planner 的压缩上下文。总字符数控制在 3,000-5,000。"""

    # Step 1: 压缩基线——只给 Top3 实体 + 关键指标
    ctx = model_baseline.get("model_context", {})
    compressed_baseline = {
        "model_id": model_id,
        "model_name": _clean_name(ctx.get("model_name")),
        "model_kind": model_strategy.get("model_kind"),
        "result_count": ctx.get("result_count"),
        "key_entities": [
            {
                "entity_name": e.get("entity_name"),
                "fact_id": e.get("fact_id"),
                "record_count": e.get("metrics", {}).get("record_count"),
                "amount_sum": e.get("metrics", {}).get("amount_sum"),
                "amount_max": e.get("metrics", {}).get("amount_max"),
                "risk_tags": [f.get("risk_tag") for f in e.get("anomaly_flags", [])],
            }
            for e in model_baseline.get("focus_entities", [])[:3]
        ],
        "key_relations": [
            {
                "fact_id": r.get("fact_id"),
                "source": r.get("source"),
                "target": r.get("target"),
                "count": r.get("count"),
                "amount_sum": r.get("amount_sum"),
            }
            for r in model_baseline.get("relation_profiles", [])[:2]
        ],
        "key_metrics": {
            k: v for k, v in model_baseline.get("key_metrics", {}).items()
            if k in ("aggregation_basis",)
        },
    }

    # Step 2: 当前假设 + 已收集的证据摘要
    current_hypothesis = {
        "hypothesis_id": hypothesis.get("hypothesis_id"),
        "entity_name": hypothesis.get("entity_name"),
        "statement": hypothesis.get("statement"),
        "basis_fact_ids": hypothesis.get("basis_fact_ids", []),
    }

    # Step 3: 前几轮摘要（压缩，不是完整 chain_steps）
    compressed_rounds = []
    for r in previous_rounds[-3:]:  # 只保留最近 3 轮
        compressed_rounds.append({
            "round": r.round_index,
            "tool_used": r.tool_name,
            "params": r.params,
            "result_summary": r.tool_output_summary[:200],  # 截断到 200 字符
            "judge_status": r.judge_status,
            "judge_comment": r.judge_assessment[:200],
        })

    # Step 4: 可用工具（含专属工具）
    tool_schemas = {}
    for name in available_tools:
        if name in TOOL_REGISTRY:
            spec = TOOL_REGISTRY[name]
            tool_schemas[name] = {
                "description": spec.description,
                "required_params": spec.required_params,
                "optional_params": spec.optional_params,
            }
        elif name in _MODEL_SPECIFIC_TOOLS:
            spec = _MODEL_SPECIFIC_TOOLS[name]
            tool_schemas[name] = {
                "description": spec["description"],
                "required_params": spec["required_params"],
                "optional_params": spec.get("optional_params", []),
            }

    # Step 5: 已调用工具指纹（防止重复）
    already_used = sorted(all_called_keys)[-10:]  # 最多 10 个

    return {
        "investigation_scope": {
            "model_id": model_id,
            "model_name": compressed_baseline["model_name"],
            "model_kind": compressed_baseline["model_kind"],
        },
        "baseline_snapshot": compressed_baseline,
        "current_hypothesis": current_hypothesis,
        "previous_rounds": compressed_rounds,
        "available_tools": tool_schemas,
        "already_called_tools": already_used,
        "budget": {
            "rounds_remaining": budget_rounds_remaining,
            "calls_remaining": budget_calls_remaining,
        },
        "planning_hints": [
            "如果上一轮 Judge 为 unresolved，选择与上一轮不同的工具，聚焦到更具体的问题上。",
            "优先选择能产生新维度信息的工具（如时间分布、对手方重合、两跳路径）。",
            "如果已有两轮以上 unresolved 且无新工具可用，输出 stop。",
            "不要在参数中使用 time_range 字段（当前未开放时间范围过滤）。",
            "所有参数值（entity_name, model_id, source, target）必须来自 baseline_snapshot 或 previous_rounds。",
        ],
    }
```

### 3.5 Planner 的 Prompt 模板

```python
_MODEL_PLANNER_SYSTEM_PROMPT = """\
你是模型"{model_name}"（model_id={model_id}）的专属调查 Agent Planner。

## 你的任务
围绕当前假设中的主体，通过调用本模型的调查工具，逐步收窄调查范围，收集支持或削弱假设的证据。

## 可用工具
{available_tools_description}

## 决策规则
1. 只能从 available_tools 中选择工具。参数必须来自 baseline_snapshot 或 previous_rounds。
2. 如果上一轮 Judge 为 unresolved：
   - 选择一个与上一轮不同的工具
   - 聚焦到更具体的子问题上（例如：上一轮查了关系邻域全貌，
     这轮聚焦到特定对手方的时间窗口或金额分布）
3. 如果 Judge 已经增强或削弱，或者你认为证据已经足够：
   输出 {{"action": "stop", "stop_reason": "sufficient_evidence"}}
4. 如果所有可用工具都已被调用过（见 already_called_tools），输出 stop。
5. 不要创造 baseline_snapshot 之外的主体、金额或关系。
6. 不要使用 time_range 参数（当前未开放）。

## 输出格式
只输出以下 JSON，不要输出推理过程：
{{"action":"call_tool|stop","tool_name":"...","params":{{...}},"reason":"为什么选择这个工具","expected_evidence":"期望获得什么证据"}}
如果 stop：{{"action":"stop","stop_reason":"sufficient_evidence|no_more_tools|budget_exhausted"}}
"""


def _ask_planner_for_next_action(
    config: Any,
    planner_context: dict[str, Any],
    agent_round: int,
    validation_feedback: list[str] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """调用 LLM Planner，获取下一步行动。"""
    audit = {
        "status": "not_started",
        "round": agent_round,
        "llm_call_count": 0,
        "fallback_used": False,
    }

    if not config.llm.is_complete:
        audit["status"] = "llm_not_configured"
        audit["fallback_used"] = True
        return None, audit

    # 构造用户消息
    model_name = _clean_name(
        planner_context.get("investigation_scope", {}).get("model_name", "")
    )
    model_id = planner_context.get("investigation_scope", {}).get("model_id", "")

    # 工具描述
    tool_descriptions = []
    for name, spec in planner_context.get("available_tools", {}).items():
        req = ", ".join(spec.get("required_params", []))
        opt = ", ".join(spec.get("optional_params", []))
        desc = f"- {name}: {spec.get('description', '')} (必需参数: {req}"
        if opt:
            desc += f", 可选参数: {opt}"
        desc += ")"
        tool_descriptions.append(desc)

    system_prompt = _MODEL_PLANNER_SYSTEM_PROMPT.format(
        model_name=model_name,
        model_id=model_id,
        available_tools_description="\n".join(tool_descriptions),
    )

    # 如果有上一轮的校验反馈，追加
    if validation_feedback:
        system_prompt += (
            f"\n\n## 上一轮请求被拒绝的原因\n"
            + "\n".join(f"- {fb}" for fb in validation_feedback[-2:])
            + "\n请修正后重新输出。"
        )

    user_content = json.dumps(planner_context, ensure_ascii=False, default=str)

    candidates = model_candidates(config.llm.model, "planner")[:2]
    audit["model_candidates"] = candidates

    for attempt, candidate_model in enumerate(candidates, start=1):
        audit["llm_call_count"] += 1
        try:
            result = cached_chat_json(
                config,
                model_name=candidate_model,
                role=f"model-planner-{model_id}",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=min(config.llm.max_tokens, 2048),
            )
            update_audit_from_result(audit, result, candidate_model, attempt)
            action = _parse_planner_action(result.payload)
            if action is not None:
                action["source"] = "llm_model_planner"
                audit["status"] = "success"
                audit["used_model"] = candidate_model
                return action, audit
        except Exception as exc:
            audit["last_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"

    audit["status"] = "fallback"
    audit["fallback_used"] = True
    return None, audit
```

---

## 四、模型专属工具设计

### 4.1 设计原则

当前 5 个通用工具（`entity_model_slice`, `relation_neighbor_subgraph`, `fund_flow_path_probe`, `same_amount_window_probe`, `counterparty_overlap_probe`）对所有模型提供基线能力。但模型级 Agent 需要**更精细化的专属工具**：

1. **每个专属工具绑定到特定模型**——不在全局 `TOOL_REGISTRY` 中注册，而是在模型策略中声明
2. **返回结构化 schema**——不止是 summary 文本，还包括类型化字段供 Judge 精确判断
3. **知道本模型的字段语义**——例如模型 86 知道"流出对象（起点）"和"流入对象（终点）"的字段名，不需要策略配置告诉它
4. **可以调用通用工具作为 fallback**

### 4.2 模型 86（银行转账关系分析）专属工具

```python
# src/model_tools/model_86_tools.py

"""
模型 86（银行转账关系分析）专属调查工具。

模型 86 的数据特征：
- 字段：流出对象（起点）、流入对象（终点）、流出账户、流入账户、交易金额、交易时间
- 关系语义：每条记录是一笔转账的起点→终点边
- 结果规模：420,476 条
"""

def transfer_structure_analyzer(
    config: Any,
    case_id: int,
    params: dict[str, Any],
    row_source: Any,
    budget: ToolExecutionBudget | None = None,
) -> dict[str, Any]:
    """
    分析指定实体在模型 86 中的转账结构深度画像。

    参数：
        entity_name: str  # 必填，主体名称
        min_amount: float | None  # 可选，最低金额过滤

    返回结构化字段：
        self_transfer_ratio: float       # 自转账笔数占比
        self_transfer_amount_ratio: float # 自转账金额占比
        counterparty_hhi: float          # 对手方集中度（HHI 指数，0-10000）
        top_outgoing: list[dict]         # 主要流出对手方（去自转）
        top_incoming: list[dict]         # 主要流入对手方（去自转）
        hour_concentration: float        # 小时集中度（交易量最大的 3 小时占比）
        day_concentration: float         # 日集中度
        account_count: int               # 涉及账户数
        night_ratio: float               # 夜间交易占比（22:00-06:00）
        counterparty_type_distribution: dict  # 对手方类型分布
        fund_cycle_patterns: list[dict]  # 资金循环模式
        evidence_records: list[dict]     # 代表性记录
    """
    budget = budget or ToolExecutionBudget()
    entity_name = str(params["entity_name"])
    model_id = 86

    # 字段名硬编码（因为这是模型 86 专属工具）
    SOURCE_FIELD = "流出对象（起点）"
    TARGET_FIELD = "流入对象（终点）"
    AMOUNT_FIELD = "交易金额"
    TIME_FIELD = "交易时间"
    SOURCE_ACCOUNT_FIELD = "流出账户"
    TARGET_ACCOUNT_FIELD = "流入账户"

    # Step 1: 读取数据
    min_amount = params.get("min_amount")
    rows = []
    total_scanned = 0
    truncated = False

    for row_id, payload in row_source.iter_model_rows(case_id, model_id):
        total_scanned += 1
        source = clean_value(payload.get(SOURCE_FIELD))
        target = clean_value(payload.get(TARGET_FIELD))
        if entity_name not in {source, target}:
            continue
        amount = safe_float(payload.get(AMOUNT_FIELD))
        if amount is None:
            continue
        if min_amount is not None and amount < min_amount:
            continue
        rows.append((row_id, payload, source, target, amount))
        if len(rows) >= budget.max_rows_per_call:
            truncated = True
            break

    if not rows:
        return _finalize_empty_result(case_id, "transfer_structure_analyzer", params, entity_name)

    # Step 2: 聚合计算
    self_transfer_count = 0
    self_transfer_amount = 0.0
    total_amount = 0.0
    counterparty_amounts: dict[str, float] = defaultdict(float)
    hour_counts: dict[int, int] = defaultdict(int)
    day_counts: dict[str, int] = defaultdict(int)
    day_amounts: dict[str, float] = defaultdict(float)
    night_count = 0
    accounts: set[str] = set()
    outgoing_amounts: dict[str, float] = defaultdict(float)
    incoming_amounts: dict[str, float] = defaultdict(float)
    outgoing_counts: dict[str, int] = defaultdict(int)
    incoming_counts: dict[str, int] = defaultdict(int)

    for row_id, payload, source, target, amount in rows:
        total_amount += amount
        accounts.add(str(payload.get(SOURCE_ACCOUNT_FIELD)))
        accounts.add(str(payload.get(TARGET_ACCOUNT_FIELD)))

        if source == entity_name and target == entity_name:
            self_transfer_count += 1
            self_transfer_amount += amount
        elif source == entity_name:
            outgoing_amounts[target] += amount
            outgoing_counts[target] += 1
            counterparty_amounts[target] += amount
        elif target == entity_name:
            incoming_amounts[source] += amount
            incoming_counts[source] += 1
            counterparty_amounts[source] += amount

        dt = _parse_datetime(payload.get(TIME_FIELD))
        if dt:
            hour_counts[dt.hour] += 1
            day_counts[dt.strftime("%Y-%m-%d")] += 1
            day_amounts[dt.strftime("%Y-%m-%d")] += amount
            if dt.hour >= 22 or dt.hour < 6:
                night_count += 1

    row_count = len(rows)
    self_transfer_ratio = self_transfer_count / row_count if row_count else 0
    self_transfer_amount_ratio = self_transfer_amount / total_amount if total_amount else 0
    night_ratio = night_count / row_count if row_count else 0

    # HHI 指数（对手方集中度）
    counterparty_hhi = _compute_hhi(counterparty_amounts)

    # Top outgoing（去自转）
    top_outgoing = sorted(
        [(t, outgoing_amounts[t], outgoing_counts[t])
         for t in outgoing_amounts if t != entity_name],
        key=lambda x: x[1], reverse=True
    )[:5]

    # Top incoming（去自转）
    top_incoming = sorted(
        [(s, incoming_amounts[s], incoming_counts[s])
         for s in incoming_amounts if s != entity_name],
        key=lambda x: x[1], reverse=True
    )[:5]

    # 小时集中度
    sorted_hours = sorted(hour_counts.values(), reverse=True)
    hour_concentration = sum(sorted_hours[:3]) / row_count if row_count else 0

    # 日集中度
    sorted_days = sorted(day_amounts.values(), reverse=True)
    day_concentration = sorted_days[0] / total_amount if total_amount and sorted_days else 0

    # 对手方类型分布
    counterparty_types = _classify_counterparties(list(counterparty_amounts.keys()))
    type_distribution = {
        k: {"count": v["count"], "amount": v["amount"]}
        for k, v in counterparty_types.items()
    }

    # 资金循环检测
    fund_cycles = _detect_fund_cycles(
        entity_name, rows, SOURCE_FIELD, TARGET_FIELD, AMOUNT_FIELD
    )

    # 代表性证据记录
    evidence = _select_evidence(rows, AMOUNT_FIELD, TIME_FIELD, budget.max_evidence_records)

    # Step 3: 生成结构化 summary
    anomalies = []
    if self_transfer_ratio > 0.5:
        anomalies.append(f"自转账占比{self_transfer_ratio:.1%}（异常高）")
    if night_ratio > 0.3:
        anomalies.append(f"夜间交易占比{night_ratio:.1%}（异常高）")
    if counterparty_hhi > 2500:
        anomalies.append(f"对手方集中度HHI={counterparty_hhi:.0f}（高度集中）")
    if len(accounts) >= 5:
        anomalies.append(f"涉及{len(accounts)}个账户")
    if fund_cycles:
        anomalies.append(f"检测到{len(fund_cycles)}组资金循环模式")

    summary = (
        f"{entity_name}在银行转账关系中共{row_count}条记录（扫描{total_scanned}条），"
        f"金额合计{_fmt_amount(total_amount)}元，涉及{len(accounts)}个账户。"
        + ("；".join(anomalies) if anomalies else "；未触发异常结构特征")
    )

    output = {
        "summary": summary,
        "row_count": row_count,
        "total_scanned": total_scanned,
        "total_amount": total_amount,
        "self_transfer_ratio": round(self_transfer_ratio, 4),
        "self_transfer_amount_ratio": round(self_transfer_amount_ratio, 4),
        "counterparty_hhi": round(counterparty_hhi, 1),
        "top_outgoing": [
            {"target": t, "amount": round(a, 2), "count": c}
            for t, a, c in top_outgoing
        ],
        "top_incoming": [
            {"source": s, "amount": round(a, 2), "count": c}
            for s, a, c in top_incoming
        ],
        "hour_concentration": round(hour_concentration, 4),
        "day_concentration": round(day_concentration, 4),
        "night_ratio": round(night_ratio, 4),
        "account_count": len(accounts),
        "counterparty_type_distribution": type_distribution,
        "fund_cycle_patterns": fund_cycles[:5],
        "evidence_records": evidence,
        "evidence_refs": [],
        "status_signal": "enhanced" if anomalies else "unresolved",
        "rows_read": total_scanned,
        "truncated": truncated,
        "aggregation_complete": not truncated,
        "limit_reason": "max_rows_per_call" if truncated else None,
    }

    return _finalize_tool_result(case_id, "transfer_structure_analyzer", params, output)
```

### 4.3 工具注册扩展

```python
# 在当前 TOOL_REGISTRY 之外，增加模型专属工具注册表
_MODEL_SPECIFIC_TOOLS: dict[str, dict[str, Any]] = {
    "transfer_structure_analyzer": {
        "description": "分析指定实体在银行转账关系模型中的转账结构深度画像：自转比例、对手方集中度、时间模式、资金循环检测",
        "required_params": ["entity_name"],
        "optional_params": ["min_amount"],
        "model_ids": [86],
        "model_kinds": ["relation"],
    },
    "counterparty_identity_analyzer": {
        "description": "分析指定实体的对手方身份特征：个人/企业/金融机构分类、对手方与主体的账户关联",
        "required_params": ["entity_name"],
        "optional_params": [],
        "model_ids": [86],
        "model_kinds": ["relation"],
    },
    "same_amount_pattern_detector": {
        "description": "检测同日同额交易的模式特征：金额分布、对手方分布、时间聚集度、可能的拆分转账模式",
        "required_params": ["entity_name"],
        "optional_params": [],
        "model_ids": [70],
        "model_kinds": ["entity"],
    },
    "cash_withdrawal_time_analyzer": {
        "description": "分析大额现金存取的时间模式：时段集中度、临界金额分布、多卡操作特征",
        "required_params": ["entity_name"],
        "optional_params": [],
        "model_ids": [43],
        "model_kinds": ["entity"],
    },
}


def get_model_tools(model_id: int, model_kind: str) -> list[str]:
    """获取某模型可用的所有工具（通用 + 专属）。"""
    tools = []
    # 通用工具
    for name, spec in TOOL_REGISTRY.items():
        if not spec.model_kinds or model_kind in spec.model_kinds:
            tools.append(name)
    # 专属工具
    for name, spec in _MODEL_SPECIFIC_TOOLS.items():
        if model_id in spec.get("model_ids", []) or model_kind in spec.get("model_kinds", []):
            tools.append(name)
    return tools
```

---

## 五、接入现有系统的集成方案

### 5.1 修改 `src/phase1_demo.py` 的 `build_phase1_demo()`

在现有的 Phase 2 流程中插入模型级 Agent：

```python
# 在 build_phase1_demo() 中，model_class_packs 构建完成后：

# ── 新增：Phase 2a - Model Investigation Agent ──
model_claims: dict[int, list[ModelClaim]] = {}
model_investigation_results: dict[int, ModelInvestigationResult] = {}

# 确定哪些模型需要 Investigation Agent
# 条件：model_kind in ("entity", "relation") AND result_count > 0
#       AND demo_level in ("L1", "L2") AND demo_role 不是 no_data
primary_model_ids = [
    model_id for model_id in model_ids
    if _should_run_investigation(model_id, bundle, loaded)
]

if analysis_mode in ("agent-loop-demo", "full-agent") and primary_model_ids:
    from .model_investigation import (
        run_model_investigation,
        ModelInvestigationConfig,
        ModelInvestigationResult,
    )

    tool_cache: dict[str, dict[str, Any]] = {}

    for model_id in primary_model_ids:
        factpack = _get_factpack_by_model_id(model_factpacks, model_id)
        strategy = strategy_for(bundle, model_id)

        investigation_config = ModelInvestigationConfig(
            model_id=model_id,
            max_rounds=getattr(agent_loop_config, 'max_agent_rounds', 4),
            max_tool_calls=getattr(agent_loop_config, 'max_tool_calls_per_hypothesis', 6),
            fallback_policy=fallback_policy.value,
        )

        result = run_model_investigation(
            config, case_id, model_id,
            model_baseline=factpack,
            model_strategy=strategy,
            entity_dictionary=dictionary,
            row_source=row_source,
            investigation_config=investigation_config,
            external_tool_cache=tool_cache,
        )

        model_investigation_results[model_id] = result
        model_claims[model_id] = result.claims

        # 将模型级 SupplementFact 追加到全局列表
        if planner_result is None:
            planner_result = {
                "supplement_facts": [],
                "evidence_chains": [],
                "planner_audit": {},
                "hypotheses": [],
                "planner_decisions": [],
                "evidence_chain_steps": [],
                "model_report_packs": [],
            }
        planner_result["supplement_facts"].extend(result.supplement_facts)
        planner_result["evidence_chain_steps"].extend(result.chain_steps)
        # ...

# ── 再进入 Phase 3: ModelClass Corroboration Agent ──
# 现有 agent_loop.py 的 run_agent_loop_demo() 需要修改：
#   消费 model_claims（模型级已形成的结论）
#   不再重复做模型内部的基础调查
```

### 5.2 修改 `src/agent_loop.py`

模型类 Agent 需要能消费模型级 Claim：

```python
def run_agent_loop_demo(
    config: Any,
    case_id: int,
    model_factpacks: list[dict[str, Any]],
    model_class_packs: list[dict[str, Any]],
    case_pack: dict[str, Any],
    bundle: Any,
    dictionary: Any,
    row_source: Any,
    loop_config: AgentLoopConfig | None = None,
    model_claims: dict[int, list[ModelClaim]] | None = None,  # ← 新增参数
) -> dict[str, Any]:
    """
    新增 model_claims 参数：
    - 如果提供了模型级 Claim，模型类 Agent 应该：
      1. 优先引用已有的 enhanced Claim（不需要重复调查）
      2. 对于 unresolved Claim，考虑是否需要在模型类级进一步补证
      3. 对于没有 Claim 的模型，使用现有的种子假设逻辑
    """
```

---

## 六、与现有代码的关系

### 不修改的部分
- `src/evidence_chain.py` — 数据结构完全兼容
- `src/judge.py` — Judge 逻辑不变，可复用
- `src/supplement_tools.py` — 通用工具不变，模型专属工具作为扩展
- `src/tool_registry.py` — `TOOL_REGISTRY` 不变，新增 `_MODEL_SPECIFIC_TOOLS` 作为扩展注册表
- `src/entity_profile.py` — Baseline 计算不变

### 需要修改的部分
- `src/phase1_demo.py` — 在 `build_phase1_demo()` 中插入模型级 Agent 调用
- `src/agent_loop.py` — `run_agent_loop_demo()` 接受并消费 `model_claims`
- `src/final_agent_report.py` — 最终报告渲染器可以消费 ModelClaim（可选，当前渲染器已消费 assertions）

### 新增的部分
- `src/model_investigation.py` — 模型级 Agent 主逻辑
- `src/model_tools/__init__.py` — 工具包初始化
- `src/model_tools/base.py` — 工具基类
- `src/model_tools/model_86_tools.py` — 模型 86 专属工具

---

## 七、施工步骤

### Step 1：创建模型级 Agent 骨架（本次）
1. 创建 `src/model_investigation.py`，包含上述骨架代码
2. 创建 `src/model_tools/` 目录
3. 实现 `transfer_structure_analyzer`（模型 86 专属工具）
4. 在 `build_phase1_demo()` 中添加 `--analysis-mode full-agent` 选项，仅在传入时启用模型级 Agent
5. 先用 `--analysis-mode agent-loop-demo` 保持现有行为不变

### Step 2：端到端验证（本次）
6. 对模型 86 单独运行 `--analysis-mode full-agent`，验证：
   - 种子假设能从 Baseline 正确生成
   - Planner 能选择专属工具（而非只选通用工具）
   - Judge 能基于结构化工具输出做更精确判断
   - 至少形成 1-2 条 ModelClaim
7. 对比 full-agent vs agent-loop-demo 的报告质量差异

### Step 3：扩展到其他 L2 模型（后续）
8. 为模型 43（大额存取现）设计专属工具
9. 为模型 70（相近时间多笔相同金额转账）设计专属工具
10. 修改模型类 Agent 消费模型级 Claim

---

## 八、关键设计决策总结

| 决策点 | 选择 | 理由 |
|---|---|---|
| 模型级 Agent 是否可以调用其他模型的工具？ | **不可以** | 模型级 Agent 的职责是深挖本模型。跨模型印证是模型类 Agent 的职责。 |
| 模型级 Agent 是否可以读取其他模型的 Baseline？ | **可以（只读引用）** | Planner 可以引用其他模型的 fact_id 作为上下文，但不能调用其工具。 |
| 模型级 Agent 的产出是否覆盖 Baseline？ | **不覆盖** | SupplementFact 追加，BaselinePack 不可变。 |
| 模型级 Claim 形成后，模型类 Agent 是否可以直接引用？ | **是** | 模型类 Agent 不应重复做模型内部的调查。如果模型级已有 enhanced Claim，可以直接作为模型类级假设的依据。 |
| 同一假设下多轮工具调用的结果如何组织？ | **追加到同一 EvidenceChain** | 同一个 `hypothesis_id` 下的所有 `chain_steps` 聚合为一条 `evidence_chain`。 |
| 新证据改变了之前 Claim 的结论怎么办？ | **生成新 Claim（v2），旧 Claim 标记 superseded** | 不删除旧 Claim。最终报告消费最新非 superseded 版本。 |
| LLM Planner 失败时的 fallback？ | **确定性回退**：从可用工具中选一个未调用过的，用 `entity_name` 作为参数 | 与当前 `_fallback_next_action()` 逻辑一致。 |
