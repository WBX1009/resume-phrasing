from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import load_config
from src.data_loader import count_case_model_rows
from src.phase1_demo import (
    CASE_ID,
    DEMO_MODEL_IDS,
    OBSERVATION_MODEL_IDS,
    build_phase1_demo,
)
from src.row_source import create_row_source
from src.agent_loop import AgentLoopConfig
from src.execution_policy import FallbackPolicy


def _parse_model_ids(value: str) -> list[int]:
    model_ids = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        model_ids.append(int(item))
    if not model_ids:
        raise argparse.ArgumentTypeError("--model-ids cannot be empty")
    return model_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a fixed-model case report demo.")
    parser.add_argument("--case-id", type=int, default=CASE_ID)
    parser.add_argument("--model-ids", type=_parse_model_ids, default=None)
    parser.add_argument(
        "--data-source",
        choices=["db", "snapshot", "auto"],
        default="db",
        help="Choose read-only database, local JSONL snapshot, or DB with snapshot fallback.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only count matching DB rows for case_id/model_ids; do not load details or write outputs.",
    )
    parser.add_argument(
        "--report-mode",
        choices=["template", "llm", "both", "final"],
        default="both",
        help="Generate template report, LLM-assisted report, or both. LLM falls back safely when unavailable.",
    )
    parser.add_argument("--strict-demo", action="store_true", help="Fail when QwQ roles fall back or final validation fails.")
    parser.add_argument(
        "--fallback-policy",
        choices=[item.value for item in FallbackPolicy],
        default=FallbackPolicy.REVIEW_SAFE.value,
        help="Choose strict failure, conservative Agent fallback, or deterministic facts-only output.",
    )
    parser.add_argument("--max-model-classes", type=int, default=3)
    parser.add_argument("--max-hypotheses-per-class", type=int, default=1)
    parser.add_argument("--max-agent-rounds", type=int, default=3)
    parser.add_argument("--max-tool-calls-per-hypothesis", type=int, default=3)
    parser.add_argument("--max-invalid-requests", type=int, default=2)
    parser.add_argument("--max-tool-rows-per-call", type=int, default=50_000)
    parser.add_argument("--max-tool-unique-groups", type=int, default=20_000)
    parser.add_argument("--max-tool-evidence-records", type=int, default=5)
    parser.add_argument(
        "--analysis-mode",
        choices=["template", "llm-narrative", "planner-demo", "class-planner-demo", "agent-loop-demo", "full-agent"],
        default="template",
        help="template keeps Phase 1.5 behavior; planner-demo uses per-model context; class-planner-demo uses ModelClassPack context; agent-loop-demo runs Planner-Tool-Judge loops; full-agent runs the complete guarded pipeline.",
    )
    return parser.parse_args()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def remove_stale_optional_outputs(paths: list[Path]) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


def main() -> int:
    args = parse_args()
    config = load_config()
    model_ids = args.model_ids or DEMO_MODEL_IDS
    started = time.perf_counter()
    fallback_policy = FallbackPolicy.STRICT_DEMO.value if args.strict_demo else args.fallback_policy

    if args.check_only:
        try:
            if args.data_source == "db":
                counts = count_case_model_rows(config, args.case_id, [*model_ids, *OBSERVATION_MODEL_IDS])
            else:
                row_source = create_row_source(config, args.data_source)
                counts = row_source.count_model_rows(args.case_id, [*model_ids, *OBSERVATION_MODEL_IDS])
        except Exception as exc:
            print(str(exc))
            return 2
        print(f"{args.data_source} row counts for case {args.case_id}:")
        total = 0
        for model_id in model_ids:
            count = counts.get(model_id, 0)
            total += count
            print(f"- {model_id}: {count}")
        print("Observation-only counts:")
        for model_id in OBSERVATION_MODEL_IDS:
            print(f"- {model_id}: {counts.get(model_id, 0)}")
        print(f"demo_total: {total}")
        return 0

    try:
        report_mode = args.report_mode
        if args.analysis_mode == "llm-narrative" and report_mode == "template":
            report_mode = "both"
        result = build_phase1_demo(
            config,
            args.case_id,
            model_ids,
            report_mode=report_mode,
            analysis_mode=args.analysis_mode,
            data_source=args.data_source,
            agent_loop_config=AgentLoopConfig(
                max_model_classes=args.max_model_classes,
                max_hypotheses_per_class=args.max_hypotheses_per_class,
                max_agent_rounds=args.max_agent_rounds,
                max_tool_calls_per_hypothesis=args.max_tool_calls_per_hypothesis,
                max_invalid_requests=args.max_invalid_requests,
                max_tool_rows_per_call=args.max_tool_rows_per_call,
                max_tool_unique_groups=args.max_tool_unique_groups,
                max_tool_evidence_records=args.max_tool_evidence_records,
                fallback_policy=fallback_policy,
            ),
        )
    except Exception as exc:
        print(str(exc))
        return 2

    final_mode = args.report_mode == "final" and args.analysis_mode in {"agent-loop-demo", "full-agent"}
    output_dir = (
        config.output_dir / f"case_{args.case_id}_agent_demo"
        if final_mode
        else config.output_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "" if final_mode else f"phase1_5_demo_case_{args.case_id}_"

    def output_path(name: str) -> Path:
        return output_dir / f"{prefix}{name}"

    outputs = {
        "model_contexts": output_path("model_contexts.json"),
        "field_profiles": output_path("field_profiles.json"),
        "model_factpacks": output_path("model_factpacks.json"),
        "model_class_packs": output_path("model_class_packs.json"),
        "case_pack": output_path("case_pack.json"),
        "llm_brief_pack": output_path("llm_brief_pack.json"),
        "report_manifest": output_path("report_manifest.json"),
        "template_report": output_path("template_report.md"),
    }
    planner_result = result.get("planner_result") or {}
    optional_outputs = {
        "planner_report": output_path("planner_report.md"),
        "llm_narrative_report": output_path("llm_narrative_report.md"),
        "llm_report": output_path("llm_report.md"),
    }
    if planner_result:
        outputs.update(
            {
                "planner_audit": output_path("planner_audit.json"),
                "supplement_facts": output_path("supplement_facts.json"),
                "evidence_chains": output_path("evidence_chains.json"),
                "model_report_packs": output_path("model_report_packs.json"),
                "planner_report": optional_outputs["planner_report"],
            }
        )
    if result.get("llm_narrative_report") is not None:
        outputs["llm_narrative_report"] = optional_outputs["llm_narrative_report"]
    if result.get("llm_report") is not None and not final_mode:
        outputs["llm_report"] = optional_outputs["llm_report"]
    if result.get("final_report") is not None:
        outputs["final_report"] = output_path("final_report.md")
        outputs["report_assertions"] = output_path("report_assertions.json")
        outputs["run_audit"] = output_path("run_audit.json")
    write_json(outputs["model_contexts"], result["contexts"])
    write_json(outputs["field_profiles"], result["field_profiles"])
    write_json(outputs["model_factpacks"], result["model_factpacks"])
    write_json(outputs["model_class_packs"], result["model_class_packs"])
    write_json(outputs["case_pack"], result["case_pack"])
    write_json(outputs["llm_brief_pack"], result["llm_brief_pack"])
    write_json(outputs["report_manifest"], result["report_manifest"])
    if planner_result:
        write_json(outputs["planner_audit"], planner_result["planner_audit"])
        write_json(outputs["supplement_facts"], planner_result["supplement_facts"])
        write_json(outputs["evidence_chains"], planner_result["evidence_chains"])
        write_json(outputs["model_report_packs"], planner_result["model_report_packs"])
        outputs["planner_report"].write_text(result["planner_report"], encoding="utf-8")
    outputs["template_report"].write_text(result["template_report"], encoding="utf-8")
    if result.get("llm_narrative_report") is not None and "llm_narrative_report" in outputs:
        outputs["llm_narrative_report"].write_text(result["llm_narrative_report"], encoding="utf-8")
    if result.get("llm_report") is not None and "llm_report" in outputs:
        outputs["llm_report"].write_text(result["llm_report"], encoding="utf-8")
    strict_errors = (
        _strict_demo_errors(result, config.llm.model)
        if fallback_policy == FallbackPolicy.STRICT_DEMO.value
        else []
    )
    if result.get("final_report") is not None:
        outputs["final_report"].write_text(result["final_report"], encoding="utf-8")
        write_json(outputs["report_assertions"], result.get("report_assertions", []))
        run_audit = {
            "case_id": args.case_id,
            "data_source": result.get("case_pack", {}).get("data_source"),
            "model": config.llm.model,
            "base_url": config.llm.base_url,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "model_counts": {
                str(ctx.get("model_id")): int(ctx.get("result_count") or 0)
                for ctx in result.get("contexts", [])
            },
            "planner": planner_result.get("planner_audit", {}),
            "final_writer": result.get("final_report_audit", {}),
            "strict_demo": fallback_policy == FallbackPolicy.STRICT_DEMO.value,
            "fallback_policy": fallback_policy,
            "tool_truncation_count": int(planner_result.get("planner_audit", {}).get("tool_truncation_count") or 0),
            "rejected_request_count": int(planner_result.get("planner_audit", {}).get("rejected_request_count") or 0),
            "strict_passed": not strict_errors,
            "strict_errors": strict_errors,
        }
        write_json(outputs["run_audit"], run_audit)
    remove_stale_optional_outputs(
        [path for name, path in optional_outputs.items() if name not in outputs]
    )

    print(f"Generated Phase 1.5 demo outputs for case {args.case_id} using {args.data_source}.")
    print("Rows by model_id:")
    for context in result["contexts"]:
        print(
            f"- {context['model_id']}: counted={context['result_count']}, "
            f"sample={context['loaded_sample_count']}, aggregation={context['aggregation_scope']}"
        )
    print(f"LLM audit: {result['case_pack'].get('llm_audit')}")
    if planner_result:
        print(f"Planner audit: {planner_result.get('planner_audit')}")
    for name, path in outputs.items():
        print(f"- {name}: {path}")
    if strict_errors:
        print("Strict demo validation failed:")
        for error in strict_errors:
            print(f"- {error}")
        return 3
    return 0


def _strict_demo_errors(result: dict[str, object], expected_model: str) -> list[str]:
    errors: list[str] = []
    planner_result = result.get("planner_result") or {}
    planner_audit = planner_result.get("planner_audit", {}) if isinstance(planner_result, dict) else {}
    if planner_audit.get("status") != "success":
        errors.append("Agent Planner did not complete successfully")
    if planner_audit.get("fallback_used"):
        errors.append("Agent Planner used fallback")
    class_runs = planner_audit.get("class_runs", [])
    if len(class_runs) < 3:
        errors.append(f"Only {len(class_runs)} reasoning model classes were processed")
    for class_run in class_runs:
        if not class_run.get("hypothesis_count") and not class_run.get("reason"):
            errors.append(f"Model class {class_run.get('class_name')} has no hypothesis or reason")
    judge_audits = planner_audit.get("judge_audits", [])
    if not judge_audits:
        errors.append("No Judge calls were recorded")
    for audit in judge_audits:
        if audit.get("status") != "success" or audit.get("judge_fallback_count"):
            errors.append("A Judge call failed or used fallback")
        if audit.get("used_model") and audit.get("used_model") != expected_model:
            errors.append(f"Judge used unexpected model {audit.get('used_model')}")
    final_audit = result.get("final_report_audit") or {}
    if not isinstance(final_audit, dict) or final_audit.get("status") != "success":
        errors.append("Final Writer did not produce validated assertions")
    if isinstance(final_audit, dict) and final_audit.get("fallback_used"):
        errors.append("Final Writer used fallback")
    if isinstance(final_audit, dict) and final_audit.get("used_model") not in {None, expected_model}:
        errors.append(f"Final Writer used unexpected model {final_audit.get('used_model')}")
    if not result.get("final_report"):
        errors.append("Final report was not generated")
    return errors


if __name__ == "__main__":
    raise SystemExit(main())
