from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import json

import psycopg2.extras

from .config import AppConfig
from .data_loader import _connect
from .entity_profile import safe_json


@dataclass(frozen=True)
class RowQuery:
    any_field_equals: tuple[tuple[str, Any], ...] = ()
    all_field_equals: tuple[tuple[str, Any], ...] = ()
    field_in: tuple[tuple[str, tuple[Any, ...]], ...] = ()
    limit: int | None = None


class RowSource:
    source_name: str = "unknown"
    warnings: list[str]

    def iter_model_rows(self, case_id: int, model_id: int) -> Iterable[tuple[Any, dict[str, Any]]]:
        raise NotImplementedError

    def iter_model_rows_filtered(
        self,
        case_id: int,
        model_id: int,
        exact_filters: dict[str, Any],
        limit: int | None = None,
    ) -> Iterable[tuple[Any, dict[str, Any]]]:
        query = RowQuery(any_field_equals=_filter_pairs(exact_filters), limit=limit)
        yield from self.iter_model_rows_query(case_id, model_id, query)

    def iter_model_rows_query(
        self,
        case_id: int,
        model_id: int,
        query: RowQuery,
    ) -> Iterable[tuple[Any, dict[str, Any]]]:
        if query.limit is not None and query.limit <= 0:
            raise ValueError("RowQuery.limit must be positive")
        emitted = 0
        for row_id, payload in self.iter_model_rows(case_id, model_id):
            if not _matches_row_query(payload, query):
                continue
            yield row_id, payload
            emitted += 1
            if query.limit is not None and emitted >= query.limit:
                return

    def count_model_rows(self, case_id: int, model_ids: list[int]) -> dict[int, int]:
        raise NotImplementedError

    def load_model_info(self, model_ids: list[int]) -> dict[int, dict[str, Any]]:
        raise NotImplementedError

    def load_orchestration(self, model_ids: list[int]) -> dict[int, list[dict[str, Any]]]:
        raise NotImplementedError

    def load_model_tree_paths(self, model_ids: list[int]) -> dict[int, list[str]]:
        info = self.load_model_info(model_ids)
        return {
            model_id: [str(info.get(model_id, {}).get("model_name") or f"model_{model_id}")]
            for model_id in model_ids
        }

    def load_echarts(self, model_ids: list[int]) -> dict[int, list[dict[str, Any]]]:
        return {}

    def load_case_info(self, case_id: int) -> dict[str, Any]:
        return {"id": case_id}


class DbRowSource(RowSource):
    source_name = "database"

    def __init__(self, config: AppConfig):
        self.config = config
        self.warnings: list[str] = []

    def iter_model_rows(self, case_id: int, model_id: int) -> Iterable[tuple[Any, dict[str, Any]]]:
        with _connect(self.config) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, json_data
                    FROM ch_model_info_data
                    WHERE case_id = %s AND model_id = %s
                    ORDER BY id
                    """,
                    [case_id, model_id],
                )
                while True:
                    rows = cur.fetchmany(5000)
                    if not rows:
                        break
                    for row_id, raw_payload in rows:
                        try:
                            yield row_id, safe_json(raw_payload)
                        except Exception:
                            continue

    def iter_model_rows_query(
        self,
        case_id: int,
        model_id: int,
        query: RowQuery,
    ) -> Iterable[tuple[Any, dict[str, Any]]]:
        sql, params = _build_db_row_query(case_id, model_id, query)
        with _connect(self.config) as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, params)
                while True:
                    rows = cur.fetchmany(5000)
                    if not rows:
                        break
                    for row in rows:
                        try:
                            yield row["id"], safe_json(row["json_data"])
                        except Exception:
                            continue

    def count_model_rows(self, case_id: int, model_ids: list[int]) -> dict[int, int]:
        if not model_ids:
            return {}
        placeholders = ", ".join(["%s"] * len(model_ids))
        with _connect(self.config) as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    f"""
                    SELECT model_id, COUNT(*) AS row_count
                    FROM ch_model_info_data
                    WHERE case_id = %s AND model_id IN ({placeholders})
                    GROUP BY model_id
                    ORDER BY model_id
                    """,
                    [case_id, *model_ids],
                )
                return {int(row["model_id"]): int(row["row_count"]) for row in cur.fetchall()}

    def load_model_info(self, model_ids: list[int]) -> dict[int, dict[str, Any]]:
        if not model_ids:
            return {}
        placeholders = ", ".join(["%s"] * len(model_ids))
        with _connect(self.config) as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    f"""
                    SELECT id, model_name, model_desc, required_data, tree_id
                    FROM ch_model_info
                    WHERE id IN ({placeholders})
                    """,
                    model_ids,
                )
                return {int(row["id"]): dict(row) for row in cur.fetchall()}

    def load_orchestration(self, model_ids: list[int]) -> dict[int, list[dict[str, Any]]]:
        if not model_ids:
            return {}
        placeholders = ", ".join(["%s"] * len(model_ids))
        with _connect(self.config) as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    f"""
                    SELECT id, model_id, name, description
                    FROM ch_model_orchestration
                    WHERE model_id IN ({placeholders})
                    ORDER BY model_id, id
                    """,
                    model_ids,
                )
                result: dict[int, list[dict[str, Any]]] = defaultdict(list)
                for row in cur.fetchall():
                    result[int(row["model_id"])].append(dict(row))
                return dict(result)

    def load_case_info(self, case_id: int) -> dict[str, Any]:
        with _connect(self.config) as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT id, event_code, event_name, batch_id, event_phase
                    FROM ch_analyse_case
                    WHERE id = %s
                    """,
                    [case_id],
                )
                row = cur.fetchone()
                return dict(row) if row else {"id": case_id}

    def load_model_tree_paths(self, model_ids: list[int]) -> dict[int, list[str]]:
        if not model_ids:
            return {}
        placeholders = ", ".join(["%s"] * len(model_ids))
        with _connect(self.config) as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    f"""
                    SELECT mi.id AS model_id, mi.tree_id, mi.model_name
                    FROM ch_model_info mi
                    WHERE mi.id IN ({placeholders})
                    """,
                    model_ids,
                )
                model_tree_rows = [dict(row) for row in cur.fetchall()]
                cur.execute(
                    """
                    SELECT id, parent_id, ancestors, node_name, model_category
                    FROM ch_model_tree
                    WHERE COALESCE(delete_flag, '0') <> '1'
                    """,
                )
                tree_by_id = {int(row["id"]): dict(row) for row in cur.fetchall()}

        def path_for(tree_id: Any, model_name: str) -> list[str]:
            if tree_id is None:
                return [model_name]
            path: list[str] = []
            seen: set[int] = set()
            current = int(tree_id)
            while current and current not in seen and current in tree_by_id:
                seen.add(current)
                row = tree_by_id[current]
                name = row.get("node_name") or row.get("model_category")
                if name:
                    path.append(str(name))
                parent = row.get("parent_id")
                current = int(parent) if parent not in [None, 0, "0", ""] else 0
            path.reverse()
            path.append(model_name)
            return path

        result: dict[int, list[str]] = {}
        for row in model_tree_rows:
            model_id = int(row["model_id"])
            result[model_id] = path_for(row.get("tree_id"), row.get("model_name") or f"model_{model_id}")
        return result


class GaussDBRowSource(DbRowSource):
    """GaussDB/PostgreSQL-compatible source with the production source name.

    GaussDB exposes the PostgreSQL wire protocol used by this adapter.  The
    SQL remains parameterized and server-side paged; credentials and endpoint
    settings continue to come from ``AppConfig``.
    """

    source_name = "gaussdb"


class SnapshotRowSource(RowSource):
    source_name = "snapshot"

    def __init__(self, snapshot_dir: Path | None):
        if snapshot_dir is None:
            raise RuntimeError("snapshot directory was not found")
        self.snapshot_dir = snapshot_dir
        self.warnings = ["snapshot mode uses sampled JSONL rows; counts and evidence are not full database totals"]
        self._data_rows = _load_jsonl(snapshot_dir / "ch_model_info_data_sample.jsonl")
        self._info_rows = _load_jsonl(snapshot_dir / "ch_model_info.jsonl")
        self._orchestration_rows = _load_jsonl(snapshot_dir / "ch_model_orchestration.jsonl")
        self._rows_by_case_model: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for row in self._data_rows:
            try:
                key = (int(row.get("case_id")), int(row.get("model_id")))
            except Exception:
                continue
            self._rows_by_case_model[key].append(row)
        self._available_cases = sorted({case_id for case_id, _ in self._rows_by_case_model})

    def _effective_case_id(self, case_id: int, model_ids: list[int]) -> int:
        if case_id in self._available_cases:
            return case_id
        if self._available_cases:
            fallback = self._available_cases[0]
            warning = f"requested case_id={case_id} was absent in snapshot; using sampled case_id={fallback}"
            if warning not in self.warnings:
                self.warnings.append(warning)
            return fallback
        return case_id

    def iter_model_rows(self, case_id: int, model_id: int) -> Iterable[tuple[Any, dict[str, Any]]]:
        effective_case = self._effective_case_id(case_id, [model_id])
        for row in self._rows_by_case_model.get((effective_case, int(model_id)), []):
            try:
                yield row.get("id"), safe_json(row.get("json_data"))
            except Exception:
                continue

    def count_model_rows(self, case_id: int, model_ids: list[int]) -> dict[int, int]:
        effective_case = self._effective_case_id(case_id, model_ids)
        counts = Counter()
        wanted = set(int(mid) for mid in model_ids)
        for (cid, mid), rows in self._rows_by_case_model.items():
            if cid == effective_case and mid in wanted:
                counts[mid] = len(rows)
        return {model_id: int(counts.get(model_id, 0)) for model_id in model_ids}

    def load_model_info(self, model_ids: list[int]) -> dict[int, dict[str, Any]]:
        wanted = set(int(mid) for mid in model_ids)
        return {
            int(row["id"]): row
            for row in self._info_rows
            if int(row.get("id", -1)) in wanted
        }

    def load_orchestration(self, model_ids: list[int]) -> dict[int, list[dict[str, Any]]]:
        wanted = set(int(mid) for mid in model_ids)
        result: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in self._orchestration_rows:
            raw_mid = row.get("model_id")
            if raw_mid is None:
                continue
            mid = int(raw_mid)
            if mid in wanted:
                result[mid].append(row)
        return dict(result)

    def load_case_info(self, case_id: int) -> dict[str, Any]:
        return {"id": case_id}


class AutoRowSource(RowSource):
    def __init__(self, config: AppConfig):
        try:
            db_source = GaussDBRowSource(config)
            with _connect(config):
                pass
            self._source: RowSource = db_source
        except Exception as exc:
            self._source = SnapshotRowSource(config.snapshot_dir)
            self._source.warnings.append(
                f"database unavailable; auto mode fell back to snapshot: {type(exc).__name__}"
            )

    @property
    def source_name(self) -> str:
        return self._source.source_name

    @property
    def warnings(self) -> list[str]:
        return self._source.warnings

    def iter_model_rows(self, case_id: int, model_id: int) -> Iterable[tuple[Any, dict[str, Any]]]:
        return self._source.iter_model_rows(case_id, model_id)

    def iter_model_rows_filtered(
        self,
        case_id: int,
        model_id: int,
        exact_filters: dict[str, Any],
        limit: int | None = None,
    ) -> Iterable[tuple[Any, dict[str, Any]]]:
        return self._source.iter_model_rows_filtered(case_id, model_id, exact_filters, limit)

    def iter_model_rows_query(
        self,
        case_id: int,
        model_id: int,
        query: RowQuery,
    ) -> Iterable[tuple[Any, dict[str, Any]]]:
        return self._source.iter_model_rows_query(case_id, model_id, query)

    def count_model_rows(self, case_id: int, model_ids: list[int]) -> dict[int, int]:
        return self._source.count_model_rows(case_id, model_ids)

    def load_model_info(self, model_ids: list[int]) -> dict[int, dict[str, Any]]:
        return self._source.load_model_info(model_ids)

    def load_orchestration(self, model_ids: list[int]) -> dict[int, list[dict[str, Any]]]:
        return self._source.load_orchestration(model_ids)

    def load_model_tree_paths(self, model_ids: list[int]) -> dict[int, list[str]]:
        return self._source.load_model_tree_paths(model_ids)

    def load_echarts(self, model_ids: list[int]) -> dict[int, list[dict[str, Any]]]:
        return self._source.load_echarts(model_ids)

    def load_case_info(self, case_id: int) -> dict[str, Any]:
        return self._source.load_case_info(case_id)


def create_row_source(config: AppConfig, data_source: str) -> RowSource:
    if data_source in {"db", "gaussdb"}:
        return GaussDBRowSource(config)
    if data_source == "snapshot":
        return SnapshotRowSource(config.snapshot_dir)
    if data_source == "auto":
        return AutoRowSource(config)
    raise ValueError(f"unsupported data_source: {data_source}")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _matches_exact_filters(payload: dict[str, Any], exact_filters: dict[str, Any]) -> bool:
    return _matches_row_query(payload, RowQuery(any_field_equals=_filter_pairs(exact_filters)))


def _filter_pairs(filters: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    pairs: list[tuple[str, Any]] = []
    for field, raw_values in filters.items():
        values = raw_values if isinstance(raw_values, (list, tuple, set)) else [raw_values]
        for value in values:
            if field and value not in [None, ""]:
                pairs.append((str(field), value))
    return tuple(pairs)


def _matches_row_query(payload: dict[str, Any], query: RowQuery) -> bool:
    if query.any_field_equals and not any(
        str(payload.get(field)) == str(value) for field, value in query.any_field_equals
    ):
        return False
    if any(str(payload.get(field)) != str(value) for field, value in query.all_field_equals):
        return False
    for field, values in query.field_in:
        wanted = {str(value) for value in values}
        if not wanted or str(payload.get(field)) not in wanted:
            return False
    return True


def _build_db_row_query(case_id: int, model_id: int, query: RowQuery) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = [case_id, model_id]
    if query.any_field_equals:
        any_clauses = []
        for field, value in query.any_field_equals:
            any_clauses.append("(json_data ->> %s) = %s")
            params.extend([str(field), str(value)])
        clauses.append("(" + " OR ".join(any_clauses) + ")")
    for field, value in query.all_field_equals:
        clauses.append("(json_data ->> %s) = %s")
        params.extend([str(field), str(value)])
    for field, values in query.field_in:
        normalized_values = [str(value) for value in values]
        if normalized_values:
            clauses.append("(json_data ->> %s) = ANY(%s)")
            params.extend([str(field), normalized_values])
        else:
            clauses.append("FALSE")
    sql = "SELECT id, json_data FROM ch_model_info_data WHERE case_id = %s AND model_id = %s"
    if clauses:
        sql += " AND " + " AND ".join(clauses)
    sql += " ORDER BY id"
    if query.limit is not None:
        if query.limit <= 0:
            raise ValueError("RowQuery.limit must be positive")
        sql += " LIMIT %s"
        params.append(int(query.limit))
    return sql, params
