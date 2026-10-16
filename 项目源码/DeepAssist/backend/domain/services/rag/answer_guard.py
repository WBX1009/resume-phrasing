import re
from typing import List, Set

from backend.domain.entities.answer import AnswerGroundingReport
from backend.domain.entities.retrieval import RAGContextPack


class SourceAwareResponseGuard:
    """Checks whether a RAG answer uses available citations in a conservative way."""

    _citation_pattern = re.compile(r"\[(C\d+)\]")
    _insufficient_patterns = (
        "no retrieved reference",
        "not supported by the retrieved",
        "references are insufficient",
        "没有检索到",
        "资料不足",
        "参考材料不足",
        "无法从参考材料",
    )

    def check(self, answer: str, context_pack: RAGContextPack) -> AnswerGroundingReport:
        answer = answer or ""
        available = {citation.ref_id for citation in context_pack.citations}
        used = self._extract_citations(answer)
        unknown = sorted(used - available)
        used_known = sorted(used & available)
        warnings: List[str] = []
        recommended_action = "accept"
        reason = "answer_grounded"

        if available and not used_known:
            warnings.append("answer_missing_citations")
            recommended_action = "regenerate_with_citations"
            reason = "retrieved_support_was_available_but_not_cited"

        if unknown:
            warnings.append("answer_has_unknown_citations")
            if recommended_action == "accept":
                recommended_action = "regenerate_with_known_citations"
                reason = "answer_used_unknown_citations"

        if not available and not self._states_insufficient_context(answer):
            warnings.append("answer_without_retrieved_support")
            recommended_action = "fallback_without_kb_claims"
            reason = "no_retrieved_support_for_answer"

        if context_pack.truncated and available and not used_known:
            warnings.append("truncated_context_high_risk")
            if recommended_action == "accept":
                recommended_action = "regenerate_with_citations"
                reason = "context_was_truncated_and_answer_is_not_grounded"

        missing = sorted(available - used) if available and used_known else []
        grounded = not warnings

        return AnswerGroundingReport(
            grounded=grounded,
            recommended_action="accept" if grounded else recommended_action,
            reason=reason,
            citation_count=len(available),
            used_citations=used_known,
            missing_citations=missing,
            unknown_citations=unknown,
            warnings=warnings,
            metadata={
                "guard": "source_aware_v1",
                "truncated_context": context_pack.truncated,
                "omitted_context_chunks": context_pack.omitted,
            },
        )

    def _extract_citations(self, answer: str) -> Set[str]:
        return set(self._citation_pattern.findall(answer))

    def _states_insufficient_context(self, answer: str) -> bool:
        lowered = answer.lower()
        return any(pattern in lowered for pattern in self._insufficient_patterns)
