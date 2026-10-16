import json
import re
from typing import Any, Iterable, Sequence

from pydantic import ValidationError

from backend.common.config import settings
from backend.common.logger import get_logger
from backend.domain.entities.route import (
    ConversationRoute,
    RiskLevel,
    RouteDecision,
    SourcePolicy,
)
from backend.domain.interfaces.llm import BaseLLM

logger = get_logger(__name__)


class ConversationRouter:
    """Route a user turn to the appropriate conversation workflow."""

    def __init__(
        self,
        llm: BaseLLM | None = None,
        model_name: str | None = None,
        temperature: float | None = None,
        min_confidence: float | None = None,
    ):
        self.llm = llm
        self.model_name = model_name or settings.ROUTER_MODEL_NAME or settings.LLM_CHAT_MODEL
        self.temperature = settings.ROUTER_TEMPERATURE if temperature is None else temperature
        self.min_confidence = (
            settings.ROUTER_MIN_CONFIDENCE if min_confidence is None else min_confidence
        )

    def route(
        self,
        query: str,
        history: Sequence[dict[str, Any]] | None = None,
        available_capabilities: Iterable[str] | None = None,
        knowledge_base_available: bool = True,
        tools_available: bool = True,
    ) -> RouteDecision:
        normalized_query = " ".join((query or "").split()).strip()
        if not normalized_query:
            return self._clarify(
                query=query,
                reason="The user query is empty.",
                question="What would you like DeepAssist to help with?",
                confidence=0.0,
            )
        deterministic = self._deterministic_route_override(normalized_query)
        if deterministic is not None:
            return deterministic

        if self.llm is None:
            return self._heuristic_route(
                normalized_query,
                reason_suffix="LLM router unavailable; used heuristic fallback.",
            )

        messages = [
            {"role": "system", "content": self._system_prompt()},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "query": normalized_query,
                        "recent_history": self._history_preview(history or []),
                        "available_capabilities": list(
                            available_capabilities
                            or [
                                "direct_chat",
                                "knowledge_base",
                                "file_read",
                                "file_catalog",
                                "runtime_capabilities",
                                "multi_step_planning",
                            ]
                        ),
                        "knowledge_base_available": knowledge_base_available,
                        "tools_available": tools_available,
                    },
                    ensure_ascii=False,
                ),
            },
        ]

        try:
            response = self.llm.chat(
                messages,
                model_name=self.model_name,
                temperature=self.temperature,
            )
            content = str(getattr(response, "content", response)).strip()
            payload = self._parse_json(content)
            payload = self._normalize_payload(payload)
            decision = RouteDecision.model_validate(payload)
            decision = self._post_process(decision, normalized_query)
            logger.info(
                "Conversation route resolved to %s with confidence %.2f",
                decision.route.value,
                decision.confidence,
            )
            return decision
        except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
            logger.error("Conversation router returned invalid output: %s", exc)
            return self._clarify(
                query=normalized_query,
                reason="The router returned invalid structured JSON.",
                question="I could not confidently choose the right workflow. Do you want a general answer, a knowledge-base answer, or a tool-based task?",
                confidence=0.0,
            )
        except Exception as exc:
            logger.error("Conversation routing failed: %s", exc)
            return self._heuristic_route(
                normalized_query,
                reason_suffix=f"LLM router failed; used heuristic fallback: {exc}",
            )

    def _normalize_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(payload)
        route_aliases = {
            "chat": ConversationRoute.DIRECT_CHAT.value,
            "quick": ConversationRoute.DIRECT_CHAT.value,
            "quick_chat": ConversationRoute.DIRECT_CHAT.value,
            "direct": ConversationRoute.DIRECT_CHAT.value,
        }
        route_value = str(normalized.get("route", "")).strip().lower()
        if route_value in route_aliases:
            normalized["route"] = route_aliases[route_value]

        source_aliases = {
            "source_required": SourcePolicy.REQUIRED.value,
            "required_sources": SourcePolicy.REQUIRED.value,
            "source_optional": SourcePolicy.OPTIONAL.value,
            "optional_sources": SourcePolicy.OPTIONAL.value,
            "none": SourcePolicy.NOT_APPLICABLE.value,
            "n/a": SourcePolicy.NOT_APPLICABLE.value,
        }
        source_value = str(normalized.get("source_policy", "")).strip().lower()
        if source_value in source_aliases:
            normalized["source_policy"] = source_aliases[source_value]
        return normalized

    def _deterministic_route_override(self, query: str) -> RouteDecision | None:
        lowered = query.lower()
        agent_terms = ("agent", "assistant", "workflow")
        kb_terms = ("knowledge base", "kb", "rag", "document", "docs")
        if any(term in lowered for term in agent_terms) and any(term in lowered for term in kb_terms):
            return RouteDecision(
                route=ConversationRoute.AGENT,
                confidence=0.88,
                reason=(
                    "User explicitly requested an agent workflow with knowledge-base retrieval; "
                    "route to Agent so internal tools or workers can perform RAG retrieval."
                ),
                rewritten_query=query,
                source_policy=SourcePolicy.NOT_APPLICABLE,
                required_capabilities=["knowledge_base", "multi_step_planning"],
                risk_level=RiskLevel.MEDIUM,
                should_log_for_training=True,
            )
        task_terms = ("execute", "run", "task", "workflow", "plan")
        if any(term in lowered for term in ("agent", "assistant")) and any(
            term in lowered for term in task_terms
        ):
            return RouteDecision(
                route=ConversationRoute.AGENT,
                confidence=0.82,
                reason="User explicitly requested an agent task; route to Agent orchestration.",
                rewritten_query=query,
                source_policy=SourcePolicy.NOT_APPLICABLE,
                required_capabilities=["multi_step_planning"],
                risk_level=RiskLevel.MEDIUM,
                should_log_for_training=True,
            )
        return None

    def _post_process(self, decision: RouteDecision, query: str) -> RouteDecision:
        if not decision.rewritten_query:
            decision = decision.model_copy(update={"rewritten_query": query})

        if decision.confidence < self.min_confidence:
            return self._clarify(
                query=decision.rewritten_query or query,
                reason=(
                    f"Router confidence {decision.confidence:.2f} is below "
                    f"minimum {self.min_confidence:.2f}; original route was {decision.route.value}."
                ),
                question=decision.clarifying_question
                or "Can you clarify whether you want a general answer, knowledge-base grounded answer, or tool-based execution?",
                confidence=decision.confidence,
                risk_level=decision.risk_level,
            )

        if decision.route == ConversationRoute.AGENT and decision.risk_level == RiskLevel.HIGH:
            return self._clarify(
                query=decision.rewritten_query or query,
                reason="The request may require a high-risk tool action and needs confirmation.",
                question=decision.clarifying_question
                or "This may perform an external or high-impact action. What exact action should I take, and do you confirm it?",
                confidence=decision.confidence,
                risk_level=RiskLevel.HIGH,
            )

        return decision

    def _parse_json(self, content: str) -> dict[str, Any]:
        cleaned = content.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:].strip()
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].strip()
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:].strip()
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if not match:
                raise
            return json.loads(match.group(0))

    def _heuristic_route(self, query: str, reason_suffix: str = "") -> RouteDecision:
        lowered = query.lower()
        kb_terms = (
            "knowledge base",
            "document",
            "docs",
            "source",
            "citation",
            "reference",
            "uploaded",
            "kb",
            "rag",
        )
        required_terms = ("only from", "based on", "cite", "citation", "with sources")
        tool_terms = (
            "run",
            "execute",
            "python",
            "sql",
            "database",
            "file",
            "api",
            "tool",
        )
        vague_terms = ("handle this", "process this", "that thing", "this item")

        if any(term in lowered for term in vague_terms) and len(query) < 20:
            return self._clarify(
                query=query,
                reason=f"The request is underspecified. {reason_suffix}".strip(),
                question="What exactly should I process, and what outcome do you want?",
                confidence=0.5,
            )

        if any(term in lowered for term in tool_terms):
            return RouteDecision(
                route=ConversationRoute.AGENT,
                confidence=0.72,
                reason=f"Heuristic detected a tool or execution request. {reason_suffix}".strip(),
                rewritten_query=query,
                source_policy=SourcePolicy.NOT_APPLICABLE,
                required_capabilities=["multi_step_planning"],
                risk_level=RiskLevel.MEDIUM,
            )

        if any(term in lowered for term in kb_terms):
            required = any(term in lowered for term in required_terms)
            return RouteDecision(
                route=ConversationRoute.RAG,
                confidence=0.7,
                reason=f"Heuristic detected a knowledge-base or source-grounded request. {reason_suffix}".strip(),
                rewritten_query=query,
                source_policy=SourcePolicy.REQUIRED if required else SourcePolicy.OPTIONAL,
                required_capabilities=["knowledge_base"],
                risk_level=RiskLevel.LOW,
            )

        return RouteDecision(
            route=ConversationRoute.DIRECT_CHAT,
            confidence=0.68,
            reason=f"Heuristic defaulted to direct chat. {reason_suffix}".strip(),
            rewritten_query=query,
            source_policy=SourcePolicy.NOT_APPLICABLE,
            required_capabilities=[],
            risk_level=RiskLevel.LOW,
        )

    def _clarify(
        self,
        query: str,
        reason: str,
        question: str,
        confidence: float,
        risk_level: RiskLevel = RiskLevel.MEDIUM,
    ) -> RouteDecision:
        return RouteDecision(
            route=ConversationRoute.CLARIFY,
            confidence=max(0.0, min(1.0, confidence)),
            reason=reason,
            rewritten_query=query,
            source_policy=SourcePolicy.NOT_APPLICABLE,
            required_capabilities=[],
            clarifying_question=question,
            risk_level=risk_level,
            should_log_for_training=True,
        )

    def _history_preview(self, history: Sequence[dict[str, Any]], limit: int = 6) -> list[dict[str, str]]:
        preview: list[dict[str, str]] = []
        for message in list(history)[-limit:]:
            role = str(message.get("role", "unknown"))
            content = " ".join(str(message.get("content", "")).split())
            if content:
                preview.append({"role": role, "content": content[:500]})
        return preview

    def _system_prompt(self) -> str:
        return (
            "You are DeepAssist's conversation router. Your job is not to answer the user, "
            "but to choose the backend workflow for this turn. Output JSON only, with no Markdown. "
            "Allowed route values: direct_chat, rag, agent, clarify. "
            "Allowed source_policy values: required, optional, not_applicable. "
            "Allowed risk_level values: low, medium, high.\n\n"
            "Rules:\n"
            "1. General concepts, summaries, rewriting, translation, and basic Q&A -> direct_chat.\n"
            "2. If the user explicitly asks according to a knowledge base, document, uploaded content, source material, references, citations, or only based on materials -> rag with source_policy required.\n"
            "3. If the user continues a knowledge-base topic without requiring sources, route rag with source_policy optional or direct_chat. Do not force a KB-only refusal for ordinary conceptual follow-ups.\n"
            "4. If the user needs external tool actions, controlled file operations, database access, API calls, or multi-step task execution -> agent.\n"
            "5. If the request is incomplete, missing key parameters, ambiguous between RAG and Agent, or high-risk without confirmation -> clarify.\n"
            "6. Do not choose RAG merely because the question is hard. Choose RAG only for external/private/source-grounded knowledge needs.\n"
            "7. Do not choose Agent merely because the question is complex. Choose Agent only when tools, actions, or multi-step execution are needed.\n\n"
            "Return this exact JSON shape: "
            '{"route":"direct_chat|rag|agent|clarify","confidence":0.0,'
            '"reason":"short reason","rewritten_query":"context-aware query",'
            '"source_policy":"required|optional|not_applicable",'
            '"required_capabilities":[],"clarifying_question":"",'
            '"risk_level":"low|medium|high","should_log_for_training":true}'
        )
