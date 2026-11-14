"""Case-specific policies and deterministic business operators; no execution loop."""
from __future__ import annotations

from collections import defaultdict
import json
from typing import Any

from .case_execution import InvestigationTask, PlanError, validate_task_plan
from .case_fact_pack import build_case_fact_pack
from .field_profile import build_field_profiles
from .entity_profile import build_model_profiles
from .entity_resolver import EntityDictionary
from .anomaly_rules import attach_anomalies
from .investigation_contracts import (
    EvidenceFact, Candidate, TaskPlan, Assessment, EvidenceGap,
    authorize_bounds, validate_fact_references, stable_id,
)


def prepare_facts(snapshot, store, run_id):
    """Reuse the original field, aggregate, entity and anomaly operators."""
    facts = {}
    dictionary = EntityDictionary()
    for entity in snapshot.entity_ids:
        dictionary.add(entity, [], [])
    for model in snapshot.models:
        groups = defaultdict(list)
        for row in model.records:
            groups[tuple(sorted(row.entity_ids))].append(row)
        for subjects, rows in sorted(groups.items()):
            raw = [{"id": r.row_id, "json_data": r.data} for r in rows]
            profiles = build_field_profiles({model.model_id: raw})
            pack = build_case_fact_pack(snapshot.case_id, [{"model_id": model.model_id,
                "model_name": model.name, "model_category": model.category,
                "result_count": len(rows)}], {model.model_id: raw}, profiles, snapshot.data_label, [])
            computed = [(f["fact_type"], f["description"], {"metrics": f["metrics"],
                        "evidence_refs": f["evidence_refs"]}) for f in pack["models"][0]["facts"]]
            entity_pack = build_model_profiles(model.model_id,
                [(r.row_id, r.data) for r in rows], model.strategy, dictionary,
                {"entities": 30, "relations": 30, "evidence": 5}, snapshot.thresholds)
            attach_anomalies(entity_pack, model.strategy, snapshot.thresholds)
            for profile in entity_pack.get("focus_entities", []) + entity_pack.get("relation_profiles", []):
                computed.append(("profile", "实体/关系指标及规则命中", {
                    "metrics": profile.get("metrics", profile),
                    "rule_flags": profile.get("anomaly_flags", []),
                    "evidence_refs": profile.get("evidence_refs", [])}))
            for kind, summary, content in computed:
                content = {"model_id": model.model_id, **content}
                fid = stable_id("fact", [snapshot.reference(), kind, subjects, content])
                fact = EvidenceFact(fact_id=fid, entity_ids=list(subjects), time_range=snapshot.time_range,
                    kind=kind, summary=summary, content=content, complete=model.complete,
                    provenance={"source": "deterministic_operator", "snapshot_ref": snapshot.reference(),
                        "model_id": model.model_id, "row_ids": [r.row_id for r in rows[:100]],
                        "row_count": len(rows), "row_ids_truncated": len(rows) > 100})
                store.put_artifact(run_id, fid, "fact", fact.model_dump(mode="json"))
                facts[fid] = fact.model_dump(mode="json")
    return facts


def fact_context(store, run_id, ids, *, limit, max_chars=32_000):
    ids = list(dict.fromkeys(ids))
    if len(ids) > limit:
        # Do not silently truncate a task's supporting evidence.
        raise PlanError("task evidence context exceeds configured fact limit")
    facts = store.facts(run_id, ids)
    # Full artifacts/rows remain in the evidence store. Every context explicitly
    # says which parts are summaries; a model can request a bounded tool query.
    entries = [{"fact_id": fid, "entity_ids": f["entity_ids"], "kind": f["kind"],
             "summary": f["summary"][:200], "complete": f["complete"], "content_omitted": True,
             "model_id": f["content"].get("model_id", f["content"].get("request", {}).get("params", {}).get("model_id")),
             } for fid, f in facts.items()]
    def size():
        return len(json.dumps(entries, ensure_ascii=False))
    if size() > max_chars:
        raise PlanError("fact reference metadata exceeds the context budget")
    share = max(0, (max_chars-size()) // max(1,len(entries))-100)
    for entry, fact in zip(entries, facts.values()):
        compact = _compact(fact["content"])
        text = json.dumps(compact, ensure_ascii=False)
        entry["content"] = compact if len(text) <= share else {"preview": text[:share//3], "truncated": True}
        entry["content_omitted"] = len(text) > share
        if size() > max_chars:
            entry.pop("content")
            entry["content_omitted"] = True
    return entries


def _compact(value, depth=0):
    if depth >= 5:
        return "[see evidence artifact]"
    if isinstance(value, dict):
        return {k: _compact(v, depth+1) for k, v in list(value.items())[:20]}
    if isinstance(value, list):
        return [_compact(v, depth+1) for v in value[:10]]
    if isinstance(value, str):
        return value[:1000]
    return value


def rule_candidates(facts):
    return [dict(event_type=f"risk_{flag['risk_tag']}", subject_ids=fact["entity_ids"],
            fact_ids=[fid], time_range=fact["time_range"],
            rationale=flag.get("report_expression") or flag["risk_tag"],
            priority=8 if flag.get("severity") == "high" else 5)
        for fid, fact in facts.items() for flag in fact["content"].get("rule_flags", [])]


def validate_plan(raw, state, store, config, scope):
    plan = TaskPlan.model_validate(raw)
    candidates = {c["event_id"]: c for c in state["candidate_events"]}
    goals = {g.goal_id: g for g in plan.goals}
    if len(goals) != len(plan.goals) or {g.candidate_id for g in plan.goals} != candidates.keys():
        raise PlanError("each selected candidate requires uniquely identified investigation goals")
    facts = store.facts(state["run_id"], state["fact_ids"])
    tasks = [InvestigationTask.from_dict(t.model_dump(mode="json")) for t in plan.tasks]
    validate_task_plan(tasks, max_total_tasks=config.max_total_tasks, scope=scope, known_fact_ids=set(facts))
    if {t.goal_id for t in tasks} != goals.keys():
        raise PlanError("every goal must have planned tasks")
    for task in tasks:
        goal = goals[task.goal_id]
        candidate = candidates.get(task.candidate_id)
        if candidate is None or task.candidate_id != goal.candidate_id:
            raise PlanError("task candidate/goal linkage is invalid")
        if not set(task.entity_ids) <= set(candidate["subject_ids"]):
            raise PlanError("task expands candidate subjects")
        authorize_bounds(task.entity_ids, task.time_range, scope)
        validate_fact_references(task.basis_fact_ids, facts, task.entity_ids)
        if task.gap_ids or task.source_task_ids:
            raise PlanError("initial plans cannot invent recheck references")
    return plan, tasks


def validate_assessments(raw, *, goals, facts, allowed_ids, task=None):
    allowed = {(g["goal_id"], r["requirement_id"]) for g in goals for r in g["requirements"]}
    output, seen = [], set()
    for item in raw:
        assessment = Assessment.model_validate(item)
        key = (assessment.goal_id, assessment.requirement_id)
        if key not in allowed or key in seen:
            raise PlanError("unknown or duplicate assessment requirement")
        if task and assessment.goal_id != task.goal_id:
            raise PlanError("worker assessment belongs to another goal")
        if task and task.gap_ids and stable_id("gap", list(key)) not in task.gap_ids:
            raise PlanError("recheck assessment belongs to another evidence gap")
        seen.add(key)
        if assessment.fact_ids:
            validate_fact_references(assessment.fact_ids, facts, task.entity_ids if task else None)
            if not set(assessment.fact_ids) <= set(allowed_ids):
                raise PlanError("assessment cites evidence unavailable to this task")
        if assessment.status != "unresolved":
            if not assessment.fact_ids or not all(facts[fid]["complete"] for fid in assessment.fact_ids):
                raise PlanError("supported/refuted assessments require complete, traceable evidence")
        output.append(assessment.model_dump(mode="json"))
    return output


def central_verify(state, store):
    """Only explicit requirement-level, evidence-bound judgments close gaps."""
    facts = store.facts(state["run_id"], state["fact_ids"])
    valid = defaultdict(list)
    sources = defaultdict(list)
    for tid, outcome in state.get("task_outcomes", {}).items():
        if outcome["status"] != "succeeded":
            continue
        result = outcome["result"]
        task = InvestigationTask.from_dict(next(t for t in state["tasks"] if t["task_id"] == tid))
        assessments = validate_assessments(result.get("assessments", []), goals=state["goals"],
            facts=facts, allowed_ids=result.get("context_fact_ids", []), task=task)
        for item in assessments:
            key = item["goal_id"], item["requirement_id"]
            valid[key].append(item)
            sources[key].append(tid)
    gaps, final, rejected, assessments = [], [], [], []
    for candidate in state["candidate_events"]:
        candidate_goals = [g for g in state["goals"] if g["candidate_id"] == candidate["event_id"]]
        statuses = []
        evidence_ids = set()
        for goal in candidate_goals:
            for req in goal["requirements"]:
                key = goal["goal_id"], req["requirement_id"]
                entries = valid[key]
                decisive = [a for a in entries if a["status"] != "unresolved"]
                kinds = {a["status"] for a in decisive}
                # New evidence may settle an earlier unresolved result; explicit
                # contradictory assessments always remain for human review.
                status = next(iter(kinds)) if len(kinds) == 1 else "unresolved"
                ids = sorted({fid for a in decisive for fid in a["fact_ids"]})
                evidence_ids.update(ids)
                statuses.append(status)
                assessments.append({"goal_id": key[0], "requirement_id": key[1], "status": status,
                    "fact_ids": ids, "rationale": "conflicting judgments" if len(kinds) > 1 else
                        (decisive[-1]["rationale"] if decisive else "缺少完整证据或明确核验结论")})
                if status == "unresolved":
                    gap = EvidenceGap(gap_id=stable_id("gap", list(key)), goal_id=key[0],
                        candidate_id=candidate["event_id"], requirement_id=key[1], question=req["question"],
                        basis_fact_ids=sorted(set(candidate["fact_ids"]) | {fid for a in entries for fid in a["fact_ids"]}),
                        source_task_ids=sorted(set(sources[key])))
                    gaps.append(gap.model_dump(mode="json"))
        if statuses and all(s == "supported" for s in statuses):
            final.append({**candidate, "status": "verified", "verification_fact_ids": sorted(evidence_ids),
                "evidence_level": "inference", "requires_human_review": True})
        elif "refuted" in statuses and "unresolved" not in statuses:
            rejected.append(candidate["event_id"])
    return {"gaps": gaps, "final_events": final, "rejected_candidate_ids": rejected,
            "assessments": assessments, "requires_human_review": bool(gaps or final)}
