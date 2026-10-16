from typing import Any, Dict, Iterator, List, Optional

from backend.domain.entities.agent_worker import (
    AgentWorkerType,
    SupervisorDecision,
)
from backend.domain.entities.intent import IntentDecision, IntentType
from backend.domain.services.agent.task_decomposer import TaskDecomposer
from backend.domain.services.agent.intent_router import IntentRouter
from backend.domain.services.agent.workers import BaseAgentWorker


class AgentSupervisor:
    """Lightweight supervisor that routes a task to a focused worker."""

    def __init__(
        self,
        intent_router: IntentRouter,
        chat_worker: BaseAgentWorker,
        rag_worker: Optional[BaseAgentWorker],
        tool_worker: BaseAgentWorker,
        orchestrator_worker: Optional[BaseAgentWorker] = None,
        task_decomposer: Optional[TaskDecomposer] = None,
    ):
        self.intent_router = intent_router
        self.chat_worker = chat_worker
        self.rag_worker = rag_worker
        self.tool_worker = tool_worker
        self.orchestrator_worker = orchestrator_worker
        self.task_decomposer = task_decomposer or TaskDecomposer()

    def decide(self, query: str) -> SupervisorDecision:
        if self.intent_router.is_tool_inventory_query(query):
            intent = IntentDecision(
                intent=IntentType.AGENT,
                confidence=0.95,
                reason="tool inventory query detected",
                signals=["tool_inventory"],
            )
            return SupervisorDecision(
                worker=AgentWorkerType.TOOL,
                intent=intent,
                reason="tool inventory query detected; routed to tool_agent_worker",
                signals=intent.signals,
            )

        if self.intent_router.is_kb_catalog_query(query):
            intent = IntentDecision(
                intent=IntentType.AGENT,
                confidence=0.93,
                reason="knowledge-base catalog or capability query detected",
                signals=["kb_catalog"],
            )
            return SupervisorDecision(
                worker=AgentWorkerType.TOOL,
                intent=intent,
                reason="knowledge-base catalog query detected; routed to tool_agent_worker",
                signals=intent.signals,
            )

        if self.orchestrator_worker is not None and self.task_decomposer.should_orchestrate(
            query,
            rag_available=self.rag_worker is not None,
        ):
            intent = IntentDecision(
                intent=IntentType.AGENT,
                confidence=0.9,
                reason="complex multi-capability task detected",
                signals=self.task_decomposer.preview(
                    query,
                    rag_available=self.rag_worker is not None,
                ).signals,
            )
            return SupervisorDecision(
                worker=AgentWorkerType.ORCHESTRATOR,
                intent=intent,
                reason="complex task detected; routed to orchestrator_worker",
                signals=intent.signals,
            )

        allowed = {IntentType.CHAT, IntentType.AGENT}
        if self.rag_worker is not None:
            allowed.add(IntentType.RAG)

        intent = self.intent_router.route(query, allowed_intents=allowed)
        if intent.intent == IntentType.RAG and self.rag_worker is not None:
            worker = AgentWorkerType.RAG
        elif intent.intent == IntentType.AGENT:
            worker = AgentWorkerType.TOOL
        else:
            worker = AgentWorkerType.CHAT

        return SupervisorDecision(
            worker=worker,
            intent=intent,
            reason=f"{intent.reason}; routed to {worker.value}",
            signals=intent.signals,
        )

    def stream(
        self,
        query: str,
        history: List[Dict[str, Any]],
        user_profile: Optional[str] = None,
        model_options: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Dict[str, Any]]:
        decision = self.decide(query)
        yield {
            "type": "supervisor_route",
            "worker": decision.worker.value,
            "worker_kind": self._worker_kind(decision.worker),
            "intent": decision.intent.intent.value,
            "confidence": decision.intent.confidence,
            "reason": decision.reason,
            "signals": decision.signals,
        }

        worker = self._worker_for(decision.worker)
        yield from worker.stream(
            query=query,
            history=history,
            user_profile=user_profile,
            model_options=model_options,
        )

    def stream_recovery(
        self,
        worker_kind: str,
        query: str,
        history: List[Dict[str, Any]],
        user_profile: Optional[str] = None,
        model_options: Optional[Dict[str, Any]] = None,
        recovery_state: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Dict[str, Any]]:
        worker = self._worker_for_kind(worker_kind)
        yield {
            "type": "supervisor_route",
            "worker": worker.worker_type.value,
            "worker_kind": self._worker_kind(worker.worker_type),
            "intent": IntentType.AGENT.value,
            "confidence": 0.99,
            "reason": f"resuming previously interrupted task on {worker_kind}",
            "signals": ["task_recovery"],
        }
        yield from worker.stream(
            query=query,
            history=history,
            user_profile=user_profile,
            model_options=model_options,
            recovery_state=recovery_state,
        )

    def _worker_for(self, worker_type: AgentWorkerType) -> BaseAgentWorker:
        if (
            worker_type == AgentWorkerType.ORCHESTRATOR
            and self.orchestrator_worker is not None
        ):
            return self.orchestrator_worker
        if worker_type == AgentWorkerType.RAG and self.rag_worker is not None:
            return self.rag_worker
        if worker_type == AgentWorkerType.TOOL:
            return self.tool_worker
        return self.chat_worker

    def _worker_for_kind(self, worker_kind: str) -> BaseAgentWorker:
        normalized = (worker_kind or "").strip()
        for worker_type in AgentWorkerType:
            if worker_type.value == normalized:
                return self._worker_for(worker_type)
        return self.tool_worker

    def _worker_kind(self, worker_type: AgentWorkerType) -> str:
        if worker_type == AgentWorkerType.ORCHESTRATOR:
            return "orchestrator"
        if worker_type == AgentWorkerType.RAG:
            return "rag"
        if worker_type == AgentWorkerType.TOOL:
            return "tool"
        return "chat"
