import time
from typing import Any, Dict, Iterator, List, Optional

from backend.common.logger import get_logger
from backend.domain.entities.agent_plan import CollaboratorTask, MultiAgentPlan
from backend.domain.entities.agent_worker import AgentWorkerType
from backend.domain.interfaces.llm import BaseLLM
from backend.domain.services.agent.engine import AgentEngine
from backend.domain.services.agent.task_decomposer import TaskDecomposer
from backend.domain.services.session.context_engine import ContextEngine
from backend.domain.services.rag.decision import RAGDecisionPolicy
from backend.domain.services.rag.pipeline import RAGPipeline

logger = get_logger(__name__)

class BaseAgentWorker:
    worker_type: AgentWorkerType

    def stream(
        self,
        query: str,
        history: List[Dict[str, Any]],
        user_profile: Optional[str] = None,
        model_options: Optional[Dict[str, Any]] = None,
        recovery_state: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Dict[str, Any]]:
        raise NotImplementedError

    def _model_options(self, model_options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not model_options:
            return {}
        return {key: value for key, value in model_options.items() if value is not None}

    def _model_call_event(
        self,
        *,
        messages: List[Dict[str, Any]],
        model_options: Dict[str, Any],
        started_at: float,
        phase: str,
        output_chars: int = 0,
        error: Optional[Exception] = None,
    ) -> Dict[str, Any]:
        model_name = (
            model_options.get("model_name")
            or getattr(getattr(self, "llm", None), "model_name", None)
            or type(getattr(self, "llm", self)).__name__
        )
        data: Dict[str, Any] = {
            "phase": phase,
            "model_name": str(model_name),
            "provider": type(getattr(self, "llm", self)).__name__,
            "message_count": len(messages),
            "latency_ms": round((time.perf_counter() - started_at) * 1000, 3),
            "output_chars": output_chars,
            "success": phase == "completed",
        }
        if error is not None:
            data.update({"error_type": type(error).__name__, "error": str(error)})
        return {"type": "model_call", "data": data}

class ChatWorker(BaseAgentWorker):
    worker_type = AgentWorkerType.CHAT

    def __init__(self, llm: BaseLLM, context_engine: ContextEngine):
        self.llm = llm
        self.context_engine = context_engine

    def stream(
        self,
        query: str,
        history: List[Dict[str, Any]],
        user_profile: Optional[str] = None,
        model_options: Optional[Dict[str, Any]] = None,
        recovery_state: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        ChatWorker 不支持恢复中断任务，因此即便 recovery_state 被传入，也会被忽略。
        """
        yield {"type": "status", "content": "Supervisor selected ChatWorker"}
        context = self.context_engine.build_quick_context(
            query=query,
            history=history,
            user_profile=user_profile,
        )
        final_answer = ""
        effective_options = self._model_options(model_options)
        started_at = time.perf_counter()
        try:
            for chunk in self.llm.chat_stream(context.messages, **effective_options):
                if isinstance(chunk, dict):
                    if chunk.get("type") == "reasoning":
                        yield {"type": "reasoning", "content": chunk["content"]}
                    else:
                        final_answer += chunk["content"]
                        yield {"type": "message_delta", "content": chunk["content"]}
                else:
                    final_answer += chunk
                    yield {"type": "message_delta", "content": chunk}
        except Exception as exc:
            yield self._model_call_event(
                messages=context.messages,
                model_options=effective_options,
                started_at=started_at,
                phase="error",
                output_chars=len(final_answer),
                error=exc,
            )
            raise
        yield self._model_call_event(
            messages=context.messages,
            model_options=effective_options,
            started_at=started_at,
            phase="completed",
            output_chars=len(final_answer),
        )
        yield {
            "type": "finish",
            "new_messages": [
                {"role": "assistant", "content": final_answer},
            ],
            "worker": self.worker_type.value,
        }

class RAGWorker(BaseAgentWorker):
    worker_type = AgentWorkerType.RAG

    def __init__(
        self,
        llm: BaseLLM,
        context_engine: ContextEngine,
        rag_pipeline: RAGPipeline,
        collection_name: str = "__all__",
        rag_decision_policy: Optional[RAGDecisionPolicy] = None,
    ):
        self.llm = llm
        self.context_engine = context_engine
        self.rag_pipeline = rag_pipeline
        self.collection_name = collection_name
        self.rag_policy = rag_decision_policy or RAGDecisionPolicy()

    def stream(
        self,
        query: str,
        history: List[Dict[str, Any]],
        user_profile: Optional[str] = None,
        model_options: Optional[Dict[str, Any]] = None,
        recovery_state: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Dict[str, Any]]:
        yield {"type": "status", "content": "Supervisor selected RAGWorker"}
        pipeline_result = self.rag_pipeline.build_context(query, self.collection_name)
        context = self.context_engine.build_rag_context_from_pipeline(
            history=history,
            pipeline_result=pipeline_result,
            user_profile=user_profile,
        )

        yield {
            "type": "retrieval_trace",
            "data": pipeline_result.retrieval_result.to_stream_data(),
        }
        yield {
            "type": "citation_trace",
            "data": pipeline_result.context_pack.to_stream_data(),
        }

        explicit_kb_query = self.rag_policy.is_explicit_kb_query(query)
        rag_decision = self.rag_policy.decide(
            pipeline_result,
            is_explicit_kb=explicit_kb_query,
        )
        if rag_decision["action"] == "direct_kb_miss":
            yield {
                "type": "status",
                "content": rag_decision["status"],
            }
            answer = rag_decision["answer"]
            yield {"type": "message_delta", "content": answer}
            yield {
                "type": "answer_guard",
                "data": {
                    "grounded": False,
                    "recommended_action": "fallback_without_kb_claims",
                    "reason": "explicit_kb_query_without_reliable_support",
                    "warnings": ["retrieval_insufficient_for_kb_answer"],
                },
            }
            yield {
                "type": "finish",
                "new_messages": [
                    {"role": "assistant", "content": answer},
                ],
                "worker": self.worker_type.value,
            }
            return

        if rag_decision["action"] == "fallback_to_chat":
            yield {
                "type": "status",
                "content": rag_decision["status"],
            }
            context = self.context_engine.build_quick_context(
                query=query,
                history=history,
                user_profile=user_profile,
            )
        elif rag_decision["action"] == "proceed_with_warning":
            yield {
                "type": "status",
                "content": rag_decision["status"],
            }

        final_answer_chunks = []
        effective_options = self._model_options(model_options)
        started_at = time.perf_counter()
        try:
            for chunk in self.llm.chat_stream(context.messages, **effective_options):
                if isinstance(chunk, dict):
                    if chunk.get("type") == "reasoning":
                        yield {"type": "reasoning", "content": chunk["content"]}
                    else:
                        final_answer_chunks.append(chunk["content"])
                else:
                    final_answer_chunks.append(chunk)
        except Exception as exc:
            yield self._model_call_event(
                messages=context.messages,
                model_options=effective_options,
                started_at=started_at,
                phase="error",
                output_chars=sum(len(str(item)) for item in final_answer_chunks),
                error=exc,
            )
            raise

        final_answer = "".join(final_answer_chunks)
        yield self._model_call_event(
            messages=context.messages,
            model_options=effective_options,
            started_at=started_at,
            phase="completed",
            output_chars=len(final_answer),
        )

        guard_report = self.rag_pipeline.check_answer(final_answer, pipeline_result)
        final_answer = self.rag_policy.apply_guard_action(
            answer=final_answer,
            guard_report=guard_report,
            pipeline_result=pipeline_result,
            is_explicit_kb=explicit_kb_query,
        )
        for chunk in self._chunk_text(final_answer):
            yield {"type": "message_delta", "content": chunk}
        yield {"type": "answer_guard", "data": guard_report.to_stream_data()}
        yield {
            "type": "finish",
            "new_messages": [
                {"role": "assistant", "content": final_answer},
            ],
            "worker": self.worker_type.value,
        }

    def _chunk_text(self, text: str, chunk_size: int = 160) -> List[str]:
        if not text:
            return []
        return [text[index : index + chunk_size] for index in range(0, len(text), chunk_size)]

class ToolAgentWorker(BaseAgentWorker):
    worker_type = AgentWorkerType.TOOL

    def __init__(self, agent_engine: AgentEngine, context_engine: ContextEngine):
        self.agent_engine = agent_engine
        self.context_engine = context_engine

    def stream(
        self,
        query: str,
        history: List[Dict[str, Any]],
        user_profile: Optional[str] = None,
        model_options: Optional[Dict[str, Any]] = None,
        recovery_state: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Dict[str, Any]]:
        yield {"type": "status", "content": "Supervisor selected ToolAgentWorker"}
        if recovery_state:
            yield {"type": "status", "content": "Resuming interrupted tool-agent task."}
            messages = list(recovery_state.get("messages", []))
            resume_state = recovery_state.get("lifecycle_state", {})
        else:
            context = self.context_engine.build_agent_context(
                query=query,
                history=history,
                use_user_memory=bool(user_profile),
                user_profile=user_profile,
            )
            messages = list(context.messages)
            tool_inventory = self.agent_engine.tool_registry.describe_tools()
            inventory_message = {
                "role": "system",
                "content": (
                    "Registered tools are listed below. Do not invent tools that are not in this list.\n"
                    "For questions about which knowledge bases are connected, use "
                    "`list_knowledge_base_collections` or `list_knowledge_base_files`.\n"
                    "For questions that need evidence from indexed content, use "
                    "`search_knowledge_base`.\n\n"
                    f"{tool_inventory}"
                ),
            }
            if messages and messages[0].get("role") == "system":
                messages.insert(1, inventory_message)
            else:
                messages.insert(0, inventory_message)
            resume_state = None
        yield from self.agent_engine.stream_run(
            messages,
            model_options=model_options,
            resume_state=resume_state,
        )

class OrchestratorWorker(BaseAgentWorker):
    worker_type = AgentWorkerType.ORCHESTRATOR

    def __init__(
        self,
        llm: BaseLLM,
        chat_worker: BaseAgentWorker,
        rag_worker: Optional[BaseAgentWorker],
        tool_worker: BaseAgentWorker,
        task_decomposer: Optional[TaskDecomposer] = None,
    ):
        self.llm = llm
        self.chat_worker = chat_worker
        self.rag_worker = rag_worker
        self.tool_worker = tool_worker
        self.task_decomposer = task_decomposer or TaskDecomposer()

    def stream(
        self,
        query: str,
        history: List[Dict[str, Any]],
        user_profile: Optional[str] = None,
        model_options: Optional[Dict[str, Any]] = None,
        recovery_state: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Dict[str, Any]]:
        yield {"type": "status", "content": "Supervisor selected OrchestratorWorker"}
        if recovery_state:
            yield {"type": "status", "content": "Resuming interrupted orchestration task."}
            plan = MultiAgentPlan.model_validate(recovery_state.get("plan", {}))
            collaborator_results: List[Dict[str, Any]] = list(
                recovery_state.get("collaborator_results", [])
            )
            start_index = int(recovery_state.get("next_task_index", 0))
        else:
            plan = self.task_decomposer.preview(
                query,
                rag_available=self.rag_worker is not None,
            )
            collaborator_results = []
            start_index = 0
        yield {"type": "multi_agent_plan", "data": plan.to_trace_data()}
        yield {
            "type": "task_snapshot",
            "data": {
                "route_worker": self.worker_type.value,
                "status": "running",
                "plan": plan.model_dump(exclude_none=True),
                "collaborator_results": collaborator_results,
                "next_task_index": start_index,
            },
        }

        for index, task in enumerate(plan.tasks[start_index:], start=start_index):
            worker = self._worker_for(task.worker)
            task_query = self._build_task_query(query, task.query, collaborator_results)
            yield {
                "type": "collaborator_trace",
                "data": {
                    "phase": "start",
                    "task_id": task.task_id,
                    "title": task.title,
                    "worker": task.worker.value,
                    "rationale": task.rationale,
                },
            }
            answer_text = ""
            for event in worker.stream(
                query=task_query,
                history=history,
                user_profile=user_profile,
                model_options=model_options,
                recovery_state=None,
            ):
                event_type = event.get("type")
                if event_type == "message_delta":
                    answer_text += event.get("content", "")
                    continue
                if event_type == "final_answer":
                    answer_text = event.get("content", "") or answer_text
                    continue
                if event_type == "finish":
                    finish_messages = event.get("new_messages", [])
                    if not answer_text:
                        answer_text = self._extract_finish_answer(finish_messages)
                    break
                yield event

            collaborator_results.append(
                {
                    "task_id": task.task_id,
                    "title": task.title,
                    "worker": task.worker.value,
                    "answer": answer_text.strip(),
                }
            )
            yield {
                "type": "collaborator_trace",
                "data": {
                    "phase": "finish",
                    "task_id": task.task_id,
                    "title": task.title,
                    "worker": task.worker.value,
                    "output_preview": (answer_text or "")[:240],
                },
            }
            yield {
                "type": "task_snapshot",
                "data": {
                    "route_worker": self.worker_type.value,
                    "status": "running",
                    "plan": plan.model_dump(exclude_none=True),
                    "collaborator_results": collaborator_results,
                    "next_task_index": index + 1,
                },
            }

        reflection = self._reflect(plan, collaborator_results)
        plan.reflection_required = not reflection["complete"]
        yield {"type": "reflection", "data": reflection}

        # A missing collaborator result gets one targeted retry.  The retry
        # is represented as another plan task and therefore is persisted by
        # the same task snapshot/recovery path as the original plan.
        if reflection["should_replan"]:
            missing_worker = reflection["missing_workers"][0]
            recovery_task = CollaboratorTask(
                task_id=f"replan-{plan.replan_count + 1}",
                title="Close the evidence gap identified by reflection",
                worker=AgentWorkerType(missing_worker),
                query=(
                    f"Re-evaluate the original task and fill the missing result for: {query}"
                ),
                rationale=(
                    "Reflection found an empty collaborator result; retry once within the "
                    "bounded re-planning budget."
                ),
            )
            plan.replan_count += 1
            plan.reflection_required = False
            plan.tasks.append(recovery_task)
            yield {"type": "multi_agent_plan", "data": plan.to_trace_data()}
            yield {
                "type": "collaborator_trace",
                "data": {
                    "phase": "start",
                    "task_id": recovery_task.task_id,
                    "title": recovery_task.title,
                    "worker": recovery_task.worker.value,
                    "rationale": recovery_task.rationale,
                    "reason": "reflection_replan",
                },
            }
            retry_worker = self._worker_for(recovery_task.worker)
            retry_query = self._build_task_query(
                query,
                recovery_task.query,
                collaborator_results,
            )
            retry_answer = ""
            for event in retry_worker.stream(
                query=retry_query,
                history=history,
                user_profile=user_profile,
                model_options=model_options,
                recovery_state=None,
            ):
                event_type = event.get("type")
                if event_type == "message_delta":
                    retry_answer += event.get("content", "")
                    continue
                if event_type == "final_answer":
                    retry_answer = event.get("content", "") or retry_answer
                    continue
                if event_type == "finish":
                    finish_messages = event.get("new_messages", [])
                    if not retry_answer:
                        retry_answer = self._extract_finish_answer(finish_messages)
                    break
                yield event
            collaborator_results.append(
                {
                    "task_id": recovery_task.task_id,
                    "title": recovery_task.title,
                    "worker": recovery_task.worker.value,
                    "answer": retry_answer.strip(),
                }
            )
            yield {
                "type": "collaborator_trace",
                "data": {
                    "phase": "finish",
                    "task_id": recovery_task.task_id,
                    "title": recovery_task.title,
                    "worker": recovery_task.worker.value,
                    "output_preview": (retry_answer or "")[:240],
                    "reason": "reflection_replan",
                },
            }
            yield {
                "type": "task_snapshot",
                "data": {
                    "route_worker": self.worker_type.value,
                    "status": "running",
                    "plan": plan.model_dump(exclude_none=True),
                    "collaborator_results": collaborator_results,
                    "next_task_index": len(plan.tasks),
                    "reflection": self._reflect(plan, collaborator_results),
                },
            }

        yield {
            "type": "task_snapshot",
            "data": {
                "route_worker": self.worker_type.value,
                "status": "running",
                "plan": plan.model_dump(exclude_none=True),
                "collaborator_results": collaborator_results,
                "next_task_index": len(plan.tasks),
            },
        }
        synthesis_messages = self._build_synthesis_messages(
            query=query,
            collaborator_results=collaborator_results,
            user_profile=user_profile,
        )
        final_answer = ""
        effective_options = self._model_options(model_options)
        started_at = time.perf_counter()
        try:
            for chunk in self.llm.chat_stream(synthesis_messages, **effective_options):
                if isinstance(chunk, dict):
                    if chunk.get("type") == "reasoning":
                        yield {"type": "reasoning", "content": chunk["content"]}
                    else:
                        final_answer += chunk["content"]
                        yield {"type": "message_delta", "content": chunk["content"]}
                else:
                    final_answer += chunk
                    yield {"type": "message_delta", "content": chunk}
        except Exception as exc:
            yield self._model_call_event(
                messages=synthesis_messages,
                model_options=effective_options,
                started_at=started_at,
                phase="error",
                output_chars=len(final_answer),
                error=exc,
            )
            raise
        yield self._model_call_event(
            messages=synthesis_messages,
            model_options=effective_options,
            started_at=started_at,
            phase="completed",
            output_chars=len(final_answer),
        )

        yield {
            "type": "finish",
            "new_messages": [
                {"role": "assistant", "content": final_answer},
            ],
            "worker": self.worker_type.value,
        }

    def _worker_for(self, worker_type: AgentWorkerType) -> BaseAgentWorker:
        if worker_type == AgentWorkerType.RAG and self.rag_worker is not None:
            return self.rag_worker
        if worker_type == AgentWorkerType.TOOL:
            return self.tool_worker
        return self.chat_worker

    def _build_task_query(
        self,
        original_query: str,
        task_query: str,
        collaborator_results: List[Dict[str, Any]],
    ) -> str:
        if not collaborator_results:
            return task_query or original_query

        findings = "\n".join(
            f"- {item['title']} ({item['worker']}): {item['answer']}"
            for item in collaborator_results
            if item.get("answer")
        )
        if not findings:
            return task_query or original_query

        return (
            f"Original task:\n{original_query}\n\n"
            f"Current subtask:\n{task_query or original_query}\n\n"
            f"Previous collaborator findings:\n{findings}"
        )

    def _reflect(
        self,
        plan: MultiAgentPlan,
        collaborator_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Check collaborator coverage and decide whether one retry is useful."""
        result_by_task = {
            str(item.get("task_id")): item for item in collaborator_results
        }
        missing_tasks = [
            task
            for task in plan.tasks
            if not str(result_by_task.get(task.task_id, {}).get("answer", "")).strip()
        ]
        missing_workers = self._dedupe_workers(task.worker.value for task in missing_tasks)
        complete = not missing_tasks
        return {
            "complete": complete,
            "missing_task_ids": [task.task_id for task in missing_tasks],
            "missing_workers": missing_workers,
            "replan_count": plan.replan_count,
            "max_replans": plan.max_replans,
            "should_replan": bool(missing_tasks) and plan.replan_count < plan.max_replans,
        }

    def _dedupe_workers(self, workers: Iterator[str]) -> List[str]:
        seen = set()
        result: List[str] = []
        for worker in workers:
            if worker in seen:
                continue
            seen.add(worker)
            result.append(worker)
        return result

    def _extract_finish_answer(self, messages: List[Dict[str, Any]]) -> str:
        for message in reversed(messages or []):
            if message.get("role") == "assistant" and message.get("content"):
                return str(message.get("content"))
        return ""

    def _build_synthesis_messages(
        self,
        query: str,
        collaborator_results: List[Dict[str, Any]],
        user_profile: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        system_prompt = (
            "You are the final synthesis coordinator for DeepAssist. "
            "Merge collaborator findings into one clear, direct answer. "
            "If collaborators disagree or some evidence is weak, say so explicitly."
        )
        if user_profile:
            system_prompt = f"{system_prompt}\n\n[User Profile]\n{user_profile}"

        findings = "\n\n".join(
            (
                f"[{item['task_id']}] {item['title']} | worker={item['worker']}\n"
                f"{item['answer'] or '(no useful result)'}"
            )
            for item in collaborator_results
        )
        user_prompt = (
            f"Original user task:\n{query}\n\n"
            f"Collaborator findings:\n{findings}\n\n"
            "Produce the final answer for the user. "
            "Do not mention internal worker names unless it improves clarity."
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
