import json
from functools import lru_cache
from typing import Iterable, Optional, Set

from backend.common.logger import get_logger
from backend.domain.entities.intent import IntentDecision, IntentType
from backend.domain.interfaces.llm import BaseLLM

logger = get_logger(__name__)

class IntentRouter:
    """LLM-based deterministic intent router for workflow selection."""

    def __init__(self, llm: BaseLLM | None = None):
        self.llm = llm

    @lru_cache(maxsize=32)
    def _analyze_intent(self, query: str) -> dict:
        """Classify one query and cache repeated analysis within the process."""
        default_result = {
            "intent": "chat",
            "confidence": 0.4,
            "reason": "fallback to default chat",
            "is_tool_inventory": False,
            "is_kb_catalog": False
        }
        if not query.strip():
            return default_result

        if self.llm is None:
            return self._heuristic_intent(query)

        system_prompt = (
            "You are DeepAssist intent router. Classify the user input and return JSON only. "
            "Fields: intent must be one of chat, rag, agent; confidence is 0.0-1.0; "
            "reason is a short explanation; is_tool_inventory is true when the user asks "
            "what tools are available; is_kb_catalog is true when the user asks which "
            "knowledge bases, documents, collections, or KB sources are connected. "
            "Use agent for external tool actions or multi-step execution, rag for "
            "knowledge-base grounded questions, and chat otherwise."
        )

        messages =[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

        try:
            # Use deterministic settings for stable routing.
            response = self.llm.chat(messages, temperature=0.0)
            content = response.content.strip()

            # Strip optional Markdown fences from model output.
            if content.startswith("```json"):
                content = content[7:-3].strip()
            elif content.startswith("```"):
                content = content[3:-3].strip()

            result = json.loads(content)
            
            # Coerce unexpected labels to the safest default.
            if result.get("intent") not in {"chat", "rag", "agent"}:
                result["intent"] = "chat"
                
            return result
        except Exception as exc:
            logger.error(f"LLM intent routing failed: {exc}")
            return self._heuristic_intent(query)

    def route(
        self,
        query: str,
        allowed_intents: Optional[Iterable[IntentType]] = None,
    ) -> IntentDecision:
        allowed: Set[IntentType] = set(allowed_intents or IntentType)
        
        analysis = self._analyze_intent(query)
        
        intent_str = analysis.get("intent", "chat")
        intent_val = IntentType.CHAT
        if intent_str == "rag":
            intent_val = IntentType.RAG
        elif intent_str == "agent":
            intent_val = IntentType.AGENT
            
        decision = IntentDecision(
            intent=intent_val,
            confidence=float(analysis.get("confidence", 0.6)),
            reason=str(analysis.get("reason", "LLM based intent routing")),
            signals=[f"llm_routed_{intent_str}"]
        )

        return self._coerce(decision, allowed)

    def is_tool_inventory_query(self, query: str) -> bool:
        analysis = self._analyze_intent(query)
        return bool(analysis.get("is_tool_inventory", False))

    def is_kb_catalog_query(self, query: str) -> bool:
        analysis = self._analyze_intent(query)
        return bool(analysis.get("is_kb_catalog", False))

    def _coerce(self, decision: IntentDecision, allowed: Set[IntentType]) -> IntentDecision:
        if decision.intent in allowed:
            return decision

        fallback = IntentType.CHAT if IntentType.CHAT in allowed else next(iter(allowed))
        return IntentDecision(
            intent=fallback,
            confidence=max(0.3, decision.confidence - 0.3),
            reason=f"{decision.reason}; coerced to {fallback.value}",
            signals=decision.signals,
        )

    def _heuristic_intent(self, query: str) -> dict:
        normalized = " ".join((query or "").split()).lower()
        kb_terms = ("knowledge base", "document", "docs", "collection", "collections", "kb", "rag")
        tool_terms = ("tool", "execute", "run", "python", "sql", "database", "file", "api")
        catalog_terms = ("available", "which", "list", "catalog", "status", "connected")
        orchestration_terms = ("then", "finally", "summarize", "multi-step", "step by step")

        is_kb_catalog = any(term in normalized for term in kb_terms) and any(
            term in normalized for term in catalog_terms
        )
        is_tool_inventory = "tool" in normalized and any(
            term in normalized for term in ("which", "what", "available", "list", "call")
        )
        if is_kb_catalog or is_tool_inventory:
            return {
                "intent": "agent",
                "confidence": 0.85,
                "reason": "heuristic capability/catalog query",
                "is_tool_inventory": is_tool_inventory,
                "is_kb_catalog": is_kb_catalog,
            }

        if any(term in normalized for term in tool_terms) or any(
            term in normalized for term in orchestration_terms
        ):
            return {
                "intent": "agent",
                "confidence": 0.75,
                "reason": "heuristic tool or multi-step task",
                "is_tool_inventory": False,
                "is_kb_catalog": False,
            }

        if any(term in normalized for term in kb_terms):
            return {
                "intent": "rag",
                "confidence": 0.7,
                "reason": "heuristic knowledge-base query",
                "is_tool_inventory": False,
                "is_kb_catalog": False,
            }

        return {
            "intent": "chat",
            "confidence": 0.55,
            "reason": "heuristic default chat",
            "is_tool_inventory": False,
            "is_kb_catalog": False,
        }
