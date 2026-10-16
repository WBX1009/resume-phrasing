import asyncio
import json
import time
from typing import Iterator, Optional

from backend.common.logger import get_logger
from backend.domain.entities.route import SourcePolicy
from backend.domain.entities.stream_event import StreamEvent
from backend.domain.interfaces.llm import BaseLLM
from backend.domain.services.agent.intent_router import IntentRouter
from backend.domain.services.session.context_engine import ContextEngine
from backend.domain.services.session.profile_extractor import ProfileExtractor
from backend.domain.services.rag.fusion import HybridRetriever
from backend.domain.services.rag.pipeline import RAGPipeline
from backend.domain.services.rag.decision import RAGDecisionPolicy
from backend.domain.services.session.manager import SessionManager
from backend.infrastructure.streaming.sse_manager import SSEManager

logger = get_logger(__name__)

class ChatApplication:
    """Chat workflow orchestrator for quick chat and RAG chat modes."""

    # NOTE: Legacy constant retained for backward-compatibility. 动态诊断已取代该阈值。
    _RAG_RELEVANCE_THRESHOLD = 0.1

    def __init__(
        self,
        llm: BaseLLM,
        session_manager: SessionManager,
        context_engine: ContextEngine,
        intent_router: IntentRouter,
        profile_extractor: Optional[ProfileExtractor] = None,
        retriever: Optional[HybridRetriever] = None,
        rag_pipeline: Optional[RAGPipeline] = None,
        rag_decision_policy: Optional[RAGDecisionPolicy] = None,
    ):
        self.llm = llm
        self.session_mgr = session_manager
        self.context_engine = context_engine
        self.intent_router = intent_router
        self.profile_extractor = profile_extractor
        self.retriever = retriever
        self.rag_pipeline = rag_pipeline or (RAGPipeline(retriever) if retriever else None)
        self.rag_policy = rag_decision_policy or RAGDecisionPolicy()

    def list_sessions(self) -> list[dict]:
        return self.session_mgr.list_sessions()

    def get_history(self, session_id: str, max_rounds: int = 50) -> list[dict]:
        return self.session_mgr.get_session_history(session_id, limit=max_rounds)

    def delete_session(self, session_id: str) -> bool:
        return self.session_mgr.delete_session(session_id)

    def _analyze_query_intent(self, query: str) -> dict:
        default_intent = {"is_small_talk": False, "is_explicit_kb": False, "is_negative_kb": False}
        if not query.strip():
            return default_intent
        heuristic_intent = self._heuristic_query_intent(query)
        if heuristic_intent.get("is_explicit_kb") or heuristic_intent.get("is_negative_kb"):
            return heuristic_intent

        system_prompt = (
            "你是一个自然语言理解引擎。你的任务是分析用户的输入，并输出纯JSON对象，包含以下三个布尔值字段：\n"
            "1. is_small_talk: 用户是否仅仅在进行日常问候、告别或简单闲聊（如“你好”、“在吗”、“拜拜”、“谢谢”）。如果包含实质性提问，设为 false。\n"
            "2. is_explicit_kb: 用户是否明确要求根据“知识库”、“文档”、“资料”或“参考”来回答。\n"
            "3. is_negative_kb: 用户是否明确要求【不要】查知识库、不查资料或不查数据库。\n"
            "请注意：只返回JSON，不要包含任何Markdown标记（如```json）或其他说明文字。"
        )
        messages =[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        try:
            if not hasattr(self.llm, "chat"):
                return heuristic_intent
            response = self.llm.chat(messages, temperature=0.0)
            content = response.content.strip()
            
            if content.startswith("```json"):
                content = content[7:-3].strip()
            elif content.startswith("```"):
                content = content[3:-3].strip()
            
            result = json.loads(content)
            is_explicit_kb = bool(result.get("is_explicit_kb", False)) or heuristic_intent.get(
                "is_explicit_kb", False
            )
            is_negative_kb = bool(result.get("is_negative_kb", False)) or heuristic_intent.get(
                "is_negative_kb", False
            )
            return {
                "is_small_talk": bool(result.get("is_small_talk", False))
                and not is_explicit_kb
                and not is_negative_kb,
                "is_explicit_kb": is_explicit_kb,
                "is_negative_kb": is_negative_kb,
            }
        except Exception as exc:
            logger.error(f"LLM query intent analysis failed: {exc}")
            return heuristic_intent

    def _heuristic_query_intent(self, query: str) -> dict:
        normalized = " ".join((query or "").split()).strip().lower()
        small_talk_terms = {
            "hi",
            "hello",
            "thanks",
            "thank you",
            "bye",
            "你好",
            "您好",
            "在吗",
            "谢谢",
            "拜拜",
        }
        explicit_kb_terms = ("知识库", "文档", "资料", "参考", "引用")
        negative_kb_patterns = ("不要查", "不查", "别查", "无需查", "不要用知识库")
        small_talk_terms.update({"你好", "您好", "在吗", "谢谢", "拜拜"})
        explicit_kb_terms = explicit_kb_terms + (
            "知识库",
            "文档",
            "资料",
            "材料",
            "上传内容",
            "引用",
            "依据",
            "根据",
            "基于",
            "查找知识库",
            "查看知识库",
            "reference",
            "citation",
            "kb",
        )
        negative_kb_patterns = negative_kb_patterns + (
            "不要查",
            "不查",
            "别查",
            "无需查",
            "不要用知识库",
            "不用知识库",
            "without kb",
        )
        return {
            "is_small_talk": normalized in small_talk_terms,
            "is_explicit_kb": any(term in normalized for term in explicit_kb_terms),
            "is_negative_kb": any(term in normalized for term in negative_kb_patterns),
        }

    def stream_chat(
        self,
        session_id: str,
        query: str,
        mode: str,
        collection_name: str = "tech_docs_kb",
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        history_rounds: int = 10,
        use_user_memory: bool = False,
        source_policy: str | SourcePolicy | None = None,
        original_query: Optional[str] = None,
    ) -> Iterator[str]:
        full_response = ""
        response_chunks: list[str] =[]
        should_persist = False
        history_budget = max(1, history_rounds)
        try:
            query_intent = self._analyze_query_intent(query)
            source_policy_provided = source_policy is not None
            normalized_source_policy = self._normalize_source_policy(source_policy, mode)

            if mode == "rag" and query_intent.get("is_negative_kb") and not source_policy_provided:
                logger.info("User explicitly requested to bypass KB in session %s", session_id)
                yield SSEManager.format_event(
                    StreamEvent.status("已识别到不查阅知识库的指令；自动切换至快速对话模式。")
                )
                mode = "quick"

            if mode == "quick":
                logger.info("Entering quick chat mode")
                context_plan = self.session_mgr.plan_chat_context(
                    session_id,
                    max_rounds=history_budget,
                    query=query,
                    use_long_term_memory=use_user_memory,
                )
                history = context_plan.flattened_messages()
                yield SSEManager.format_event(
                    StreamEvent.context_window_trace(context_plan.to_trace_data())
                )
                if context_plan.recalled_memories:
                    yield SSEManager.format_event(
                        StreamEvent.status(
                            f"Recalled {len(context_plan.recalled_memories)} long-term memory item(s)."
                        )
                    )
                if context_plan.summary is not None:
                    yield SSEManager.format_event(
                        StreamEvent.status(
                            f"Compressed {context_plan.summary.dropped_turn_count} earlier turns into a continuity summary."
                        )
                    )
                user_profile = self._render_user_profile(use_user_memory)
                context = self.context_engine.build_quick_context(
                    query=query,
                    history=history,
                    user_profile=user_profile,
                )
                should_persist = True
                yield from self._stream_llm_response(
                    context.messages,
                    response_chunks,
                    model_name=model_name,
                    temperature=temperature,
                    top_p=top_p,
                )
                full_response = "".join(response_chunks)
                yield SSEManager.format_end()
                return

            if mode == "rag":
                logger.info("Entering RAG chat mode for session %s", session_id)
                
                if query_intent.get("is_small_talk") and not source_policy_provided:
                    logger.info("Bypassing RAG retrieval for small-talk query in session %s", session_id)
                    yield SSEManager.format_event(
                        StreamEvent.status(
                            "Detected a casual chat query; switching to general chat."
                        )
                    )
                    context_plan = self.session_mgr.plan_chat_context(
                        session_id,
                        max_rounds=history_budget,
                        query=query,
                        use_long_term_memory=use_user_memory,
                    )
                    history = context_plan.flattened_messages()
                    yield SSEManager.format_event(
                        StreamEvent.context_window_trace(context_plan.to_trace_data())
                    )
                    if context_plan.recalled_memories:
                        yield SSEManager.format_event(
                            StreamEvent.status(
                                f"Recalled {len(context_plan.recalled_memories)} long-term memory item(s)."
                            )
                        )
                    if context_plan.summary is not None:
                        yield SSEManager.format_event(
                            StreamEvent.status(
                                f"Compressed {context_plan.summary.dropped_turn_count} earlier turns into a continuity summary."
                            )
                        )
                    user_profile = self._render_user_profile(use_user_memory)
                    fallback_context = self.context_engine.build_quick_context(
                        query=query,
                        history=history,
                        user_profile=user_profile,
                    )
                    should_persist = True
                    yield from self._stream_llm_response(
                        fallback_context.messages,
                        response_chunks,
                        model_name=model_name,
                        temperature=temperature,
                        top_p=top_p,
                    )
                    full_response = "".join(response_chunks)
                    yield SSEManager.format_end()
                    return

                if self.rag_pipeline is None:
                    data = {
                        "available": False,
                        "component": "rag_pipeline",
                        "reason": (
                            "RAG pipeline is not initialized. Check embedding model, "
                            "vector database, keyword database, and retriever diagnostics."
                        ),
                        "next_action": "Open /api/runtime/diagnostics?refresh=true or the UI capability panel for details.",
                    }
                    yield SSEManager.format_event(StreamEvent.rag_unavailable(data))
                    yield SSEManager.format_error(
                        "知识库问答服务当前不可用：RAG pipeline 未初始化。请查看运行时诊断中的 embedding/vector/keyword/retriever 状态。"
                    )
                    yield SSEManager.format_end()
                    return

                context_plan = self.session_mgr.plan_chat_context(
                    session_id,
                    max_rounds=history_budget,
                    query=query,
                    use_long_term_memory=use_user_memory,
                )
                history = context_plan.flattened_messages()
                yield SSEManager.format_event(
                    StreamEvent.context_window_trace(context_plan.to_trace_data())
                )
                if context_plan.recalled_memories:
                    yield SSEManager.format_event(
                        StreamEvent.status(
                            f"Recalled {len(context_plan.recalled_memories)} long-term memory item(s)."
                        )
                    )
                if context_plan.summary is not None:
                    yield SSEManager.format_event(
                        StreamEvent.status(
                            f"Compressed {context_plan.summary.dropped_turn_count} earlier turns into a continuity summary."
                        )
                    )
                user_profile = self._render_user_profile(use_user_memory)
                pipeline_result = self.rag_pipeline.build_context(
                    query=query,
                    collection_name=collection_name,
                )
                is_source_required = normalized_source_policy == SourcePolicy.REQUIRED or (
                    not source_policy_provided and query_intent.get("is_explicit_kb", False)
                )
                rag_decision = self.rag_policy.decide(
                    pipeline_result,
                    is_explicit_kb=is_source_required,
                )
                context = self.context_engine.build_rag_context_from_pipeline(
                    history=history,
                    pipeline_result=pipeline_result,
                    user_profile=user_profile,
                )
                if context.retrieval_result:
                    trace_data = context.retrieval_result.to_stream_data()
                    if is_source_required and rag_decision["action"] == "direct_kb_miss":
                        metadata = trace_data.setdefault("metadata", {})
                        diagnostics = metadata.setdefault("diagnostics", {})
                        diagnostics["source_policy"] = SourcePolicy.REQUIRED.value
                        diagnostics["final_action"] = "direct_kb_miss"
                        if diagnostics.get("suggested_action") == "fallback_to_chat":
                            diagnostics["suggested_action"] = "direct_kb_miss"
                    yield SSEManager.format_event(
                        StreamEvent.retrieval_trace(trace_data)
                    )
                if context.rag_context_pack:
                    yield SSEManager.format_event(
                        StreamEvent.citation_trace(
                            context.rag_context_pack.to_stream_data()
                        )
                    )

                if rag_decision["action"] == "direct_kb_miss":
                    logger.info(
                        "RAG retrieval is insufficient for explicit KB query in session %s",
                        session_id,
                    )
                    yield SSEManager.format_event(
                        StreamEvent.status(rag_decision["status"])
                    )
                    should_persist = True
                    full_response = rag_decision["answer"]
                    yield from self._emit_buffered_text(full_response, response_chunks)
                    yield SSEManager.format_end()
                    return

                if rag_decision["action"] == "fallback_to_chat":
                    logger.info(
                        "RAG retrieval is insufficient; falling back to quick chat for session %s",
                        session_id,
                    )
                    yield SSEManager.format_event(StreamEvent.status(rag_decision["status"]))
                    fallback_context = self.context_engine.build_quick_context(
                        query=query,
                        history=history,
                        user_profile=user_profile,
                    )
                    should_persist = True
                    yield from self._stream_llm_response(
                        fallback_context.messages,
                        response_chunks,
                        model_name=model_name,
                        temperature=temperature,
                        top_p=top_p,
                    )
                    full_response = "".join(response_chunks)
                    yield SSEManager.format_end()
                    return

                if rag_decision["action"] == "proceed_with_warning":
                    yield SSEManager.format_event(StreamEvent.status(rag_decision["status"]))

                should_persist = True
                
                rag_answer_chunks = []
                model_started_at = time.perf_counter()
                try:
                    for chunk in self.llm.chat_stream(
                        context.messages,
                        model_name=model_name,
                        temperature=temperature,
                        top_p=top_p,
                    ):
                        if isinstance(chunk, dict):
                            if chunk.get("type") == "reasoning":
                                yield SSEManager.format_event(
                                    StreamEvent.reasoning(chunk["content"])
                                )
                            else:
                                rag_answer_chunks.append(chunk["content"])
                        else:
                            rag_answer_chunks.append(chunk)
                except Exception as exc:
                    yield SSEManager.format_event(
                        StreamEvent.model_call(
                            self._model_call_data(
                                messages=context.messages,
                                model_name=model_name,
                                started_at=model_started_at,
                                phase="error",
                                output_chars=sum(len(str(item)) for item in rag_answer_chunks),
                                error=exc,
                            )
                        )
                    )
                    raise

                rag_answer = "".join(rag_answer_chunks)
                yield SSEManager.format_event(
                    StreamEvent.model_call(
                        self._model_call_data(
                            messages=context.messages,
                            model_name=model_name,
                            started_at=model_started_at,
                            phase="completed",
                            output_chars=len(rag_answer),
                        )
                    )
                )

                if context.rag_context_pack:
                    guard_report = self.rag_pipeline.check_answer(
                        rag_answer,
                        pipeline_result,
                    )
                    yield SSEManager.format_event(
                        StreamEvent.answer_guard(guard_report.to_stream_data())
                    )
                    rag_answer = self.rag_policy.apply_guard_action(
                        answer=rag_answer,
                        guard_report=guard_report,
                        pipeline_result=pipeline_result,
                        is_explicit_kb=is_source_required
                    )
                full_response = rag_answer
                yield from self._emit_buffered_text(full_response, response_chunks)

                yield SSEManager.format_end()
                return

            yield SSEManager.format_error(f"Unknown chat mode: {mode}")
            yield SSEManager.format_end()

        except (GeneratorExit, asyncio.CancelledError):
            logger.warning("Chat stream closed early for session %s", session_id)
            raise
        except Exception as exc:
            logger.error("Streaming chat failed: %s", exc)
            yield SSEManager.format_error(str(exc))
            yield SSEManager.format_end()
        finally:
            if response_chunks and not full_response:
                full_response = "".join(response_chunks)
            if should_persist and (query.strip() or full_response.strip()):
                self.session_mgr.save_interaction(
                    session_id,
                    original_query if original_query is not None else query,
                    full_response,
                )

    def stream_direct_chat(
        self,
        session_id: str,
        query: str,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        history_rounds: int = 10,
        use_user_memory: bool = False,
        original_query: Optional[str] = None,
    ) -> Iterator[str]:
        yield from self.stream_chat(
            session_id=session_id,
            query=query,
            mode="quick",
            model_name=model_name,
            temperature=temperature,
            top_p=top_p,
            history_rounds=history_rounds,
            use_user_memory=use_user_memory,
            source_policy=SourcePolicy.NOT_APPLICABLE,
            original_query=original_query,
        )

    def stream_rag_chat(
        self,
        session_id: str,
        query: str,
        collection_name: str = "tech_docs_kb",
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        history_rounds: int = 10,
        use_user_memory: bool = False,
        source_policy: str | SourcePolicy = SourcePolicy.OPTIONAL,
        original_query: Optional[str] = None,
    ) -> Iterator[str]:
        yield from self.stream_chat(
            session_id=session_id,
            query=query,
            mode="rag",
            collection_name=collection_name,
            model_name=model_name,
            temperature=temperature,
            top_p=top_p,
            history_rounds=history_rounds,
            use_user_memory=use_user_memory,
            source_policy=source_policy,
            original_query=original_query,
        )

    def _normalize_source_policy(
        self,
        source_policy: str | SourcePolicy | None,
        mode: str,
    ) -> SourcePolicy:
        if mode != "rag":
            return SourcePolicy.NOT_APPLICABLE
        if isinstance(source_policy, SourcePolicy):
            return source_policy
        normalized = (source_policy or SourcePolicy.OPTIONAL.value).strip().lower()
        try:
            return SourcePolicy(normalized)
        except ValueError:
            return SourcePolicy.OPTIONAL

    def _render_user_profile(self, enabled: bool) -> Optional[str]:
        if not enabled or self.profile_extractor is None:
            return None
        profile = self.profile_extractor.render_profile()
        return profile or None

    def _stream_llm_response(
        self,
        messages,
        response_chunks: list[str],
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> Iterator[str]:
        model_started_at = time.perf_counter()
        try:
            for chunk in self.llm.chat_stream(
                messages,
                model_name=model_name,
                temperature=temperature,
                top_p=top_p,
            ):
                if isinstance(chunk, dict):
                    if chunk.get("type") == "reasoning":
                        yield SSEManager.format_event(StreamEvent.reasoning(chunk["content"]))
                    else:
                        response_chunks.append(chunk["content"])
                        yield SSEManager.format_event(StreamEvent.message_delta(chunk["content"]))
                else:
                    response_chunks.append(chunk)
                    yield SSEManager.format_event(StreamEvent.message_delta(chunk))
        except Exception as exc:
            yield SSEManager.format_event(
                StreamEvent.model_call(
                    self._model_call_data(
                        messages=messages,
                        model_name=model_name,
                        started_at=model_started_at,
                        phase="error",
                        output_chars=sum(len(str(item)) for item in response_chunks),
                        error=exc,
                    )
                )
            )
            raise
        yield SSEManager.format_event(
            StreamEvent.model_call(
                self._model_call_data(
                    messages=messages,
                    model_name=model_name,
                    started_at=model_started_at,
                    phase="completed",
                    output_chars=sum(len(str(item)) for item in response_chunks),
                )
            )
        )

    def _model_call_data(
        self,
        *,
        messages,
        model_name: Optional[str],
        started_at: float,
        phase: str,
        output_chars: int,
        error: Optional[Exception] = None,
    ) -> dict:
        data = {
            "phase": phase,
            "model_name": str(model_name or getattr(self.llm, "model_name", None) or type(self.llm).__name__),
            "provider": type(self.llm).__name__,
            "message_count": len(messages),
            "latency_ms": round((time.perf_counter() - started_at) * 1000, 3),
            "output_chars": output_chars,
            "success": phase == "completed",
        }
        if error is not None:
            data.update({"error_type": type(error).__name__, "error": str(error)})
        return data

    def _emit_buffered_text(
        self,
        text: str,
        response_chunks: list[str],
        chunk_size: int = 160,
    ) -> Iterator[str]:
        if not text:
            return
        for start in range(0, len(text), chunk_size):
            chunk = text[start : start + chunk_size]
            response_chunks.append(chunk)
            yield SSEManager.format_event(StreamEvent.message_delta(chunk))

    def _should_fallback_from_rag(self, pipeline_result) -> bool:
        """
        🌟 Legacy fallback predicate based on a static relevance threshold.
        It remains for backward-compatibility but is now practically unused.
        Fallback decisions rely on dynamic diagnostics such as 'reason_code' in retrieval metadata.
        """
        diagnostics = (
            pipeline_result.retrieval_result.metadata.get("diagnostics", {})
            if pipeline_result.retrieval_result.metadata
            else {}
        )
        reason_code = diagnostics.get("reason_code")
        return reason_code in {"no_hits", "low_relevance", "all_channels_failed"}
