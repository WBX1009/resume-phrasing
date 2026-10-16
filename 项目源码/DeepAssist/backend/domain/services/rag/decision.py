import re
from typing import Any


class RAGDecisionPolicy:
    """Shared RAG fallback and grounding policy for chat and agent workers."""

    _EXPLICIT_KB_QUERY_PATTERN = re.compile(
        r"(根据|基于).*(知识库|文档|资料)|"
        r"(知识库|文档|资料).*(回答|作答|引用|依据|检索|查找)|"
        r"(请).*(引用|标注)\s*",
        flags=re.IGNORECASE,
    )

    def decide(self, pipeline_result, is_explicit_kb: bool = False) -> dict:
        diagnostics = (
            pipeline_result.retrieval_result.metadata.get("diagnostics", {})
            if pipeline_result.retrieval_result.metadata
            else {}
        )
        reason_code = diagnostics.get("reason_code", "ok")
        reason_message = diagnostics.get("reason_message", "retrieval_ready")

        if reason_code == "all_channels_failed":
            return {
                "action": "fallback_to_chat",
                "status": "Knowledge-base retrieval is currently degraded; falling back to general chat.",
            }

        if reason_code in {"no_hits", "low_relevance"}:
            if is_explicit_kb:
                return {
                    "action": "direct_kb_miss",
                    "status": "The knowledge-base retrieval result is insufficient for a source-grounded answer.",
                    "answer": self.render_kb_miss_answer(reason_message),
                }
            return {
                "action": "fallback_to_chat",
                "status": "No sufficiently relevant KB context was found; falling back to general chat.",
            }

        if reason_code in {"partial_channel_failure", "single_channel_recall"}:
            return {
                "action": "proceed_with_warning",
                "status": "Retrieval completed with partial signal quality; grounding checks remain enabled.",
            }

        return {"action": "proceed", "status": ""}

    def is_explicit_kb_query(self, query: str) -> bool:
        normalized = " ".join((query or "").split())
        return bool(normalized and self._EXPLICIT_KB_QUERY_PATTERN.search(normalized))

    def render_kb_miss_answer(self, reason_message: str) -> str:
        return (
            "根据当前知识库检索，暂时没有找到足够相关、可直接支撑答案的资料。"
            f"原因：{reason_message}。"
            "你可以换一个更具体的关键词，或者直接追问通用建议；后者不会被标记为知识库依据下的结论。"
        )

    def apply_guard_action(
        self,
        answer: str,
        guard_report: Any,
        pipeline_result: Any,
        is_explicit_kb: bool,
    ) -> str:
        if guard_report.grounded:
            return answer

        if guard_report.recommended_action == "fallback_without_kb_claims":
            if is_explicit_kb:
                return self.render_kb_miss_answer(guard_report.reason)
            return (
                f"{answer}\n\n"
                "注：当前回答没有可靠的知识库检索证据支撑，请将其视为通用回答，而不是知识库结论。"
            ).strip()

        if guard_report.recommended_action in {
            "regenerate_with_citations",
            "regenerate_with_known_citations",
        }:
            refs = ", ".join(
                citation.ref_id for citation in pipeline_result.context_pack.citations[:3]
            ) or "无"
            return (
                "根据当前检索到的知识库片段，系统暂时无法生成带可靠引用的答案，"
                "因此已触发防幻觉降级。"
                f"可参考的检索片段编号：{refs}。"
                "请换一个更具体的问题，或明确要求“仅基于知识库并附引用回答”。"
            )

        return answer
