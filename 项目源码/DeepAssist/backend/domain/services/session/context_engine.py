from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.domain.entities.document import DocumentChunk
from backend.domain.entities.rag_pipeline import RAGPipelineResult
from backend.domain.entities.retrieval import RAGContextPack, RetrievalResult
from backend.domain.services.agent.prompt import PromptManager
from backend.domain.services.agent.skill_registry import SkillRegistry
from backend.domain.services.rag.context_packer import ContextPacker
from backend.domain.services.rag.fusion import HybridRetriever
from backend.common.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ContextBundle:
    messages: List[Dict[str, Any]]
    retrieved_docs: List[DocumentChunk] = field(default_factory=list)
    retrieval_result: Optional[RetrievalResult] = None
    rag_context_pack: Optional[RAGContextPack] = None


class ContextEngine:
    """Pure context and prompt assembly for chat, RAG, and agent workflows."""

    def __init__(
        self,
        max_history_messages: int = 20,
        context_packer: Optional[ContextPacker] = None,
        skill_registry: Optional[SkillRegistry] = None,
    ):
        self.max_history_messages = max_history_messages
        self.context_packer = context_packer or ContextPacker()
        self.skill_registry = skill_registry

    def build_quick_context(
        self,
        query: str,
        history: Optional[List[Dict[str, Any]]] = None,
        user_profile: Optional[str] = None,
    ) -> ContextBundle:
        system_prompt = PromptManager.CHAT_SYSTEM_PROMPT
        if user_profile:
            system_prompt = f"{system_prompt}\n\n[User Profile]\n{user_profile}"

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.trim_chat_history(history or []))
        messages.append({"role": "user", "content": query})
        return ContextBundle(
            messages=messages
        )

    def build_rag_context(
        self,
        query: str,
        history: List[Dict[str, Any]],
        retriever: Optional[HybridRetriever],
        collection_name: str,
        user_profile: Optional[str] = None,
    ) -> ContextBundle:
        docs: List[DocumentChunk] = []
        user_content = query
        rag_context_pack: Optional[RAGContextPack] = None

        if retriever:
            retrieval_result = retriever.retrieve_with_trace(collection_name, query)
            docs = retrieval_result.documents
            rag_context_pack = self.context_packer.pack(query, retrieval_result)
            user_content = rag_context_pack.to_prompt()
            logger.info(
                "RAG context packed with %s citations, used_chars=%s",
                len(rag_context_pack.citations),
                rag_context_pack.used_chars,
            )
        else:
            logger.warning("RAG context requested without a retriever")

        system_prompt = PromptManager.RAG_SYSTEM_PROMPT
        if user_profile:
            system_prompt = f"{system_prompt}\n\n[User Profile]\n{user_profile}"

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.trim_chat_history(history))
        messages.append({"role": "user", "content": user_content})
        return ContextBundle(
            messages=messages,
            retrieved_docs=docs,
            retrieval_result=retrieval_result if retriever else None,
            rag_context_pack=rag_context_pack,
        )

    def build_rag_context_from_pipeline(
        self,
        history: List[Dict[str, Any]],
        pipeline_result: RAGPipelineResult,
        user_profile: Optional[str] = None,
    ) -> ContextBundle:
        system_prompt = PromptManager.RAG_SYSTEM_PROMPT
        if user_profile:
            system_prompt = f"{system_prompt}\n\n[User Profile]\n{user_profile}"

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.trim_chat_history(history))
        messages.append(
            {
                "role": "user",
                "content": pipeline_result.context_pack.to_prompt(),
            }
        )
        return ContextBundle(
            messages=messages,
            retrieved_docs=pipeline_result.retrieval_result.documents,
            retrieval_result=pipeline_result.retrieval_result,
            rag_context_pack=pipeline_result.context_pack,
        )

    def build_agent_context(
        self,
        query: str,
        history: List[Dict[str, Any]],
        use_user_memory: bool = False,
        user_profile: Optional[str] = None,
        available_tool_names: Optional[List[str]] = None,
    ) -> ContextBundle:
        system_prompt = PromptManager.AGENT_SYSTEM_PROMPT
        if self.skill_registry:
            skill_prompt = self.skill_registry.render_for_task(
                query,
                tool_names=available_tool_names,
            )
            if skill_prompt:
                system_prompt = f"{system_prompt}\n\n[Loaded Skills]\n{skill_prompt}"
        if use_user_memory and user_profile:
            system_prompt = f"{system_prompt}\n\n[User Profile]\n{user_profile}"

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.trim_history(history))
        messages.append({"role": "user", "content": query})
        return ContextBundle(messages=messages)

    def trim_history(self, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(history) <= self.max_history_messages:
            return list(history)
        return list(history[-self.max_history_messages :])

    def trim_chat_history(self, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        cleaned: List[Dict[str, Any]] = []
        for message in history:
            role = message.get("role")
            if role == "tool" or message.get("tool_calls"):
                continue
            if role not in {"user", "assistant", "system"}:
                continue
            content = str(message.get("content") or "").strip()
            if not content:
                continue
            cleaned.append({"role": role, "content": content})
        return self.trim_history(cleaned)
