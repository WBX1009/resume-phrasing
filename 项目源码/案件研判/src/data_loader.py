from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

try:
    import psycopg2
    import psycopg2.extras
except ModuleNotFoundError:  # Keep the demo usable when DB dependencies are not installed.
    psycopg2 = None  # type: ignore[assignment]

from .config import AppConfig


@dataclass(frozen=True)
class LoadedData:
    source: str
    records: list[dict[str, Any]]
    model_info: dict[int, dict[str, Any]]
    orchestration: dict[int, list[dict[str, Any]]]
    row_counts: dict[int, int]
    warnings: list[str]


class DataLoadError(RuntimeError):
    """Raised when the requested data source cannot be loaded safely."""


def _safe_load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def _placeholders(values: list[int]) -> str:
    if not values:
        raise ValueError("model_ids cannot be empty")
    return ", ".join(["%s"] * len(values))


def _connect(config: AppConfig):
    if psycopg2 is None:
        raise RuntimeError("psycopg2 is not installed")
    db = config.db
    if not db.is_complete:
        raise RuntimeError("database configuration is incomplete")
    conn = psycopg2.connect(
        host=db.host,
        port=db.port,
        user=db.user,
        password=db.password,
        dbname=db.database,
        sslmode=db.ssl_mode,
        connect_timeout=5,
    )
    conn.set_client_encoding("UTF8")
    conn.set_session(readonly=True, autocommit=True)
    return conn


def _load_from_db(config: AppConfig, case_id: int, model_ids: list[int]) -> LoadedData:
    model_ph = _placeholders(model_ids)
    with _connect(config) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            row_counts = _count_case_model_rows_with_cursor(cur, case_id, model_ids)

            cur.execute(
                f"""
                SELECT id, model_id, sql_text, select_fields, json_data, create_date, case_id
                FROM ch_model_info_data
                WHERE case_id = %s AND model_id IN ({model_ph})
                ORDER BY model_id, id
                """,
                [case_id, *model_ids],
            )
            records = [dict(row) for row in cur.fetchall()]

            cur.execute(
                f"""
                SELECT id, model_name, model_desc, required_data
                FROM ch_model_info
                WHERE id IN ({model_ph})
                """,
                model_ids,
            )
            model_info = {int(row["id"]): dict(row) for row in cur.fetchall()}

            cur.execute(
                f"""
                SELECT id, model_id, name, description
                FROM ch_model_orchestration
                WHERE model_id IN ({model_ph})
                ORDER BY model_id, id
                """,
                model_ids,
            )
            orchestration: dict[int, list[dict[str, Any]]] = {}
            for row in cur.fetchall():
                mid = int(row["model_id"])
                orchestration.setdefault(mid, []).append(dict(row))

    fetched_counts = Counter(int(row["model_id"]) for row in records)
    warnings = []
    for model_id in model_ids:
        expected = row_counts.get(model_id, 0)
        fetched = fetched_counts.get(model_id, 0)
        if expected != fetched:
            warnings.append(
                f"database count mismatch for model_id={model_id}: expected {expected}, fetched {fetched}"
            )

    return LoadedData(
        source="database",
        records=records,
        model_info=model_info,
        orchestration=orchestration,
        row_counts=row_counts,
        warnings=warnings,
    )


def _count_case_model_rows_with_cursor(cur: Any, case_id: int, model_ids: list[int]) -> dict[int, int]:
    model_ph = _placeholders(model_ids)
    cur.execute(
        f"""
        SELECT model_id, COUNT(*) AS row_count
        FROM ch_model_info_data
        WHERE case_id = %s AND model_id IN ({model_ph})
        GROUP BY model_id
        ORDER BY model_id
        """,
        [case_id, *model_ids],
    )
    return {int(row["model_id"]): int(row["row_count"]) for row in cur.fetchall()}


def list_case_model_counts(config: AppConfig, case_id: int) -> dict[int, int]:
    try:
        with _connect(config) as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT model_id, COUNT(*) AS row_count
                    FROM ch_model_info_data
                    WHERE case_id = %s
                    GROUP BY model_id
                    ORDER BY row_count DESC, model_id
                    """,
                    [case_id],
                )
                return {int(row["model_id"]): int(row["row_count"]) for row in cur.fetchall()}
    except Exception as exc:
        raise DataLoadError(
            "database model count listing failed. Check read-only DB access."
        ) from exc


def count_case_model_rows(config: AppConfig, case_id: int, model_ids: list[int]) -> dict[int, int]:
    try:
        with _connect(config) as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                return _count_case_model_rows_with_cursor(cur, case_id, model_ids)
    except Exception as exc:
        raise DataLoadError(
            "database count check failed. Check the current Python environment, network, and read-only DB access."
        ) from exc


def _load_from_snapshot(config: AppConfig, case_id: int, model_ids: list[int], reason: str) -> LoadedData:
    if config.snapshot_dir is None:
        return LoadedData(
            source="snapshot",
            records=[],
            model_info={},
            orchestration={},
            row_counts={},
            warnings=[reason, "snapshot directory was not found"],
        )

    data_rows = _safe_load_jsonl(config.snapshot_dir / "ch_model_info_data_sample.jsonl")
    info_rows = _safe_load_jsonl(config.snapshot_dir / "ch_model_info.jsonl")
    orchestration_rows = _safe_load_jsonl(config.snapshot_dir / "ch_model_orchestration.jsonl")

    wanted = set(model_ids)
    records = [
        row
        for row in data_rows
        if int(row.get("case_id", -1)) == case_id and int(row.get("model_id", -1)) in wanted
    ]
    model_info = {
        int(row["id"]): row
        for row in info_rows
        if int(row.get("id", -1)) in wanted
    }
    orchestration: dict[int, list[dict[str, Any]]] = {}
    for row in orchestration_rows:
        raw_mid = row.get("model_id")
        if raw_mid is None:
            continue
        mid = int(raw_mid)
        if mid in wanted:
            orchestration.setdefault(mid, []).append(row)
    row_counts = dict(Counter(int(row["model_id"]) for row in records))

    return LoadedData(
        source="snapshot",
        records=records,
        model_info=model_info,
        orchestration=orchestration,
        row_counts=row_counts,
        warnings=[reason],
    )


def load_case_model_data(config: AppConfig, case_id: int, model_ids: list[int]) -> LoadedData:
    return load_case_model_data_from_source(config, case_id, model_ids, data_source="auto")


def load_case_model_data_from_source(
    config: AppConfig,
    case_id: int,
    model_ids: list[int],
    data_source: str,
) -> LoadedData:
    if data_source == "db":
        try:
            return _load_from_db(config, case_id, model_ids)
        except Exception as exc:
            raise DataLoadError(
                "database load failed; no snapshot fallback was used. "
                "Check the current Python environment, network, and read-only DB access."
            ) from exc
    if data_source == "snapshot":
        return _load_from_snapshot(
            config,
            case_id,
            model_ids,
            "snapshot mode was explicitly requested",
        )
    if data_source != "auto":
        raise ValueError(f"unsupported data_source: {data_source}")
    try:
        return _load_from_db(config, case_id, model_ids)
    except Exception:
        return _load_from_snapshot(
            config,
            case_id,
            model_ids,
            "database unavailable; loaded filtered local snapshot instead",
        )
