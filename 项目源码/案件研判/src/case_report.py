"""Deterministic projections of one validated report; no second reasoning path."""
from html import escape
from io import BytesIO
import json
from pathlib import Path
import re

from .investigation_contracts import CaseReport


def report_lines(raw):
    report = CaseReport.model_validate(raw).model_dump(mode="json")
    lines = [f"案件 {report['case_id']} · {report['status']}",
        f"运行 {report['run_id']}；数据标记 {report['data_label']}", report["evidence_boundary"],
        f"结构化来源完整：{report['coverage']['complete']}；语义通道：{report['discovery'].get('semantic_status')}",
        "候选线索（尚未证实）"]
    for event in report["candidate_events"]:
        lines.append(f"{event['event_id']} · {event['event_type']}：{event['rationale']}；事实 {', '.join(event['fact_ids'])}")
    lines.append("核验结果（辅助推断，需人工复核）")
    for assessment in report["verification"]["assessments"]:
        lines.append(f"{assessment['goal_id']}/{assessment['requirement_id']} · {assessment['status']}：{assessment['rationale']}；事实 {', '.join(assessment['fact_ids'])}")
    lines.append("未解决的证据缺口")
    for gap in report["verification"]["gaps"]:
        lines.append(f"{gap['gap_id']} · {gap['question']}；依据 {', '.join(gap['basis_fact_ids'])}")
    lines.append("任务执行状态")
    lines.extend(f"{tid} · {value['status']}；{value.get('error') or ''}" for tid,value in report["task_outcomes"].items())
    lines.append(f"预算：{json.dumps(report['budget'], ensure_ascii=False, sort_keys=True)}")
    lines.append("证据索引")
    lines.extend(report["fact_ids"])
    return lines


def render_report(raw, format="json"):
    report = CaseReport.model_validate(raw).model_dump(mode="json")
    if format == "json":
        return (json.dumps(report, ensure_ascii=False, indent=2)+"\n").encode("utf-8"), "application/json"
    lines = report_lines(report)
    if format == "md":
        # Untrusted narrative remains text, including embedded HTML/links.
        escaped = [re.sub(r"([\\`*_{}\[\]()#+.!|>])", r"\\\1", escape(line)) for line in lines]
        return ("# 研判报告\n\n"+"\n\n".join(escaped)+"\n").encode("utf-8"), "text/markdown; charset=utf-8"
    if format == "html":
        body = "".join(f"<p>{escape(line)}</p>" for line in lines)
        return (f"<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>研判报告</title><body>{body}</body></html>").encode("utf-8"), "text/html; charset=utf-8"
    if format == "docx":
        from docx import Document
        document = Document()
        document.add_heading("研判报告", 0)
        for line in lines:
            document.add_paragraph(line)
        output = BytesIO()
        document.save(output)
        return output.getvalue(), "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    raise ValueError("format must be json, md, html or docx")


def write_report(raw, path):
    path = Path(path)
    content, _ = render_report(raw, path.suffix.lstrip(".").lower() or "json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
