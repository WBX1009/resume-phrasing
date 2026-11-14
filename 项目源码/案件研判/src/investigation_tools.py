"""Registered tools with one mandatory authorization and budget boundary."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import Field, BaseModel

from .case_execution import PlanError, ExternalOperationUncertain, _invoke_tool, _parse_time
from .case_repository import CaseSnapshot, load_snapshot
from .investigation_contracts import Contract, ToolRequest, EvidenceFact, authorize_bounds, stable_id
from .case_operators import run_case_operators
from .field_profile import build_field_profiles
from .row_source import RowSource, _matches_row_query
from .tool_registry import TOOL_REGISTRY, validate_tool_request
from .supplement_tools import execute_tool, ToolExecutionBudget
from .entity_resolver import EntityDictionary


class QueryParams(Contract):
    model_id: int = Field(gt=0, strict=True)
    limit: int = Field(default=100, ge=1, le=1000, strict=True)


@dataclass(frozen=True)
class RegisteredTool:
    schema: type[BaseModel]
    handler: Callable
    writes: bool = False
    # Any entity-valued custom parameters must be declared here. Undeclared
    # parameters are rejected by the tool's extra="forbid" JSON contract.
    entity_params: tuple[str, ...] = ()


class _BoundRows(RowSource):
    def __init__(self, models):
        self.models = models

    def iter_model_rows(self, case_id, model_id):
        for row in self.models.get(model_id, []):
            yield row.row_id, dict(row.data)

    def iter_model_rows_query(self, case_id, model_id, query):
        count = 0
        for row_id, payload in self.iter_model_rows(case_id, model_id):
            if _matches_row_query(payload, query):
                yield row_id, payload
                count += 1
                if query.limit is not None and count >= query.limit:
                    break


def bounded_rows(snapshot, request):
    start, end = [_parse_time(x, "tool time")[0] for x in request.time_range]
    entities = set(request.entity_ids)
    # Exclude mixed-scope records instead of leaking a second subject through
    # a correctly bounded envelope. Broadening requires a new authorized task.
    return {model.model_id: [r for r in model.records if set(r.entity_ids) <= entities
                            and start <= _parse_time(r.occurred_at, "row time")[0] <= end]
            for model in snapshot.models}


class InvestigationTools:
    def __init__(self, store, *, extra: dict[str, RegisteredTool] | None = None):
        self.store = store
        self.extra = dict(extra or {})
        for spec in self.extra.values():
            if not issubclass(spec.schema, BaseModel) or spec.schema.model_config.get("extra") != "forbid":
                raise ValueError("registered tools require extra=forbid parameter contracts")
            if not callable(spec.handler) or not set(spec.entity_params) <= spec.schema.model_fields.keys():
                raise ValueError("invalid registered tool handler/entity parameter declaration")
        if set(self.extra) & ({"query_records"} | set(TOOL_REGISTRY)):
            raise ValueError("custom tools cannot replace built-in authorization")

    def names(self):
        return {"query_records"} | set(TOOL_REGISTRY) | set(self.extra)

    def describe(self, names):
        descriptions = {}
        for name in names:
            if name == "query_records":
                descriptions[name] = QueryParams.model_json_schema()
            elif name in self.extra:
                descriptions[name] = self.extra[name].schema.model_json_schema()
            elif name in TOOL_REGISTRY:
                descriptions[name] = {"description": TOOL_REGISTRY[name].description,
                    "params": [{"name": p.name, "kind": p.kind, "required": p.required,
                                "max_items": p.max_items, "max_length": p.max_length}
                               for p in TOOL_REGISTRY[name].params]}
        return descriptions

    def authorize(self, raw_request, context):
        request = ToolRequest.model_validate(raw_request)
        task, scope = context.task, context.scope
        if request.tool_name not in task.allowed_tools or request.tool_name not in scope.allowed_tools:
            raise PlanError("tool is outside the task/case whitelist")
        authorize_bounds(request.entity_ids, request.time_range, scope, task=task)
        snapshot = load_snapshot(self.store, {"run_id": context.run_id,
                **{key: self.store.admission(context.run_id)[key] for key in ("snapshot_ref", "case_id")}})
        model_ids = {model.model_id for model in snapshot.models}
        params = dict(request.params)
        if request.tool_name == "query_records":
            params = QueryParams.model_validate(params).model_dump()
        elif request.tool_name in TOOL_REGISTRY:
            if "model_id" in params and (not isinstance(params["model_id"], int) or isinstance(params["model_id"], bool)):
                raise PlanError("model_id must be an integer")
            if "model_ids" in params and (not isinstance(params["model_ids"], list) or
                    any(not isinstance(v, int) or isinstance(v, bool) for v in params["model_ids"])):
                raise PlanError("model_ids must be an integer list")
            strategies = {m.model_id: m.strategy for m in snapshot.models}
            checked = {"tool_name": request.tool_name, "params": params}
            ok, reason = validate_tool_request(checked, strategies, set(request.entity_ids))
            if not ok:
                raise PlanError(reason)
            params = checked["params"]
            if request.tool_name == "transfer_structure_analyzer" and params["model_id"] != 86:
                raise PlanError("transfer_structure_analyzer requires model_id=86")
            if request.tool_name == "counterparty_overlap_probe" and not params.get("model_ids"):
                if len(model_ids) > 8:
                    raise PlanError("specify at most eight model_ids for a cross-model probe")
                params["model_ids"] = sorted(model_ids)
        elif request.tool_name in self.extra:
            spec = self.extra[request.tool_name]
            params = spec.schema.model_validate(params).model_dump()
            for name in spec.entity_params:
                values = params.get(name)
                values = [values] if isinstance(values, str) else values or []
                if not set(values) <= set(request.entity_ids):
                    raise PlanError("custom tool entity parameter escapes authorized envelope")
        else:
            raise PlanError("unknown registered tool")
        requested_models = list(params.get("model_ids") or [])
        if params.get("model_id") is not None:
            requested_models.append(params["model_id"])
        if not set(requested_models) <= model_ids:
            raise PlanError("tool references a model outside the case")
        return request.model_copy(update={"params": params}), snapshot

    async def execute(self, raw_request, context):
        # All checks happen before any budget reservation, effect claim or I/O.
        request, snapshot = self.authorize(raw_request, context)
        spec = self.extra.get(request.tool_name)
        operation_key = stable_id("operation", [context.idempotency_key, request.model_dump(mode="json")])
        if spec and spec.writes:
            state, cached = self.store.begin_effect(context.run_id, operation_key)
            if state == "cached":
                return cached
            if state == "uncertain":
                raise ExternalOperationUncertain("external operation requires reconciliation")
        dispatched = False

        def reserve():
            nonlocal dispatched
            if self.store.is_cancel_requested(context.run_id):
                raise asyncio.CancelledError
            self.store.reserve(context.run_id, "tool_calls", 1, context.config)
            dispatched = True

        try:
            kwargs = {"idempotency_key": operation_key} if spec and spec.writes else {}
            handler = spec.handler if spec else self._builtin
            rows = bounded_rows(snapshot, request)
            scoped_snapshot = snapshot.model_copy(update={"entity_ids": list(request.entity_ids),
                "time_range": request.time_range, "allowed_tools": list(context.task.allowed_tools),
                "models": [m.model_copy(update={"records": rows[m.model_id]}) for m in snapshot.models]})
            output = await _invoke_tool(context.resources.database, handler, reserve,
                scoped_snapshot, request.model_copy(deep=True), **kwargs)
            if not isinstance(output, dict):
                raise TypeError("a tool must return a JSON object")
            content = {"request": request.model_dump(mode="json"), "output": output}
            fid = stable_id("fact", [snapshot.reference(), operation_key, content])
            fact = EvidenceFact(fact_id=fid, entity_ids=request.entity_ids, time_range=request.time_range,
                kind=f"tool:{request.tool_name}", summary=f"{request.tool_name} 受控查询结果",
                content=content, provenance={"source": "registered_tool", "snapshot_ref": snapshot.reference(),
                    "operation_key": operation_key, "tool": request.tool_name},
                complete=output.get("aggregation_complete", False) is True)
            self.store.put_artifact(context.run_id, fid, "fact", fact.model_dump(mode="json"))
            result = {"fact_ids": [fid], "complete": fact.complete}
            if spec and spec.writes:
                self.store.complete_effect(context.run_id, operation_key, result)
            return result
        except BaseException as exc:
            if spec and spec.writes and not dispatched:
                self.store.fail_effect(context.run_id, operation_key, type(exc).__name__)
            raise

    def _builtin(self, snapshot: CaseSnapshot, request: ToolRequest):
        rows = bounded_rows(snapshot, request)
        by_id = {m.model_id: m for m in snapshot.models}
        if request.tool_name == "query_records":
            model = by_id[request.params["model_id"]]
            selected = rows[model.model_id]
            limit = request.params["limit"]
            records = [{"id": r.row_id, "json_data": r.data} for r in selected[:limit]]
            profiles = build_field_profiles({model.model_id: records})
            computed = run_case_operators(model.model_id, records, profiles[model.model_id],
                {"model_category": model.category})
            return {"rows": [r.model_dump(mode="json") for r in selected[:limit]],
                "operator_results": computed["operator_results"], "rows_read": min(limit, len(selected)),
                "aggregation_complete": model.complete and len(selected) <= limit,
                "truncated": len(selected) > limit}
        dictionary = EntityDictionary()
        for entity in request.entity_ids:
            dictionary.add(entity, [], [])
        from .entity_profile import build_model_profiles
        packs = [{"model_context": {"model_id": m.model_id, "model_name": m.name},
            **build_model_profiles(m.model_id, [(r.row_id, r.data) for r in rows[m.model_id]],
            m.strategy, dictionary, {"entities": 30, "relations": 30, "evidence": 5}, snapshot.thresholds)}
            for m in snapshot.models if m.model_id in request.params.get("model_ids", [])]
        output = execute_tool(request.tool_name, None, snapshot.case_id,
            {m.model_id: m.strategy for m in snapshot.models}, dictionary, packs, request.params,
            snapshot.thresholds, row_source=_BoundRows(rows), budget=ToolExecutionBudget(max_rows_per_call=1000))
        # Legacy operator identifiers are source annotations, not IDs in the
        # canonical evidence registry. Only the outer EvidenceFact is citable.
        for old, new in (("fact_id", "source_operator_id"), ("fact_refs", "source_operator_refs"),
                         ("evidence_refs", "source_evidence_refs")):
            if old in output:
                output[new] = output.pop(old)
        output["aggregation_complete"] = output.get("aggregation_complete", True) and all(m.complete for m in snapshot.models)
        return output
