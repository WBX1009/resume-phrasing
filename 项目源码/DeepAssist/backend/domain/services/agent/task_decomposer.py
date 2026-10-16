import re
from typing import List

from backend.domain.entities.agent_plan import CollaboratorTask, MultiAgentPlan
from backend.domain.entities.agent_worker import AgentWorkerType


class TaskDecomposer:
    """Deterministic complex-task decomposition for multi-worker orchestration."""

    _complex_patterns = (
        r"复杂",
        r"多跳",
        r"多步",
        r"分步骤?",
        r"先.*再",
        r"然后",
        r"并且",
        r"同时",
        r"分别",
        r"最后.*总结",
        r"综合",
        r"对比",
        r"汇总",
    )
    _rag_patterns = (
        r"知识库",
        r"文档",
        r"资料",
        r"引用",
        r"根据.*内容",
        r"rag",
        r"检索",
    )
    _tool_patterns = (
        r"工具",
        r"调用",
        r"执行",
        r"运行",
        r"计算",
        r"生成",
        r"分析",
        r"天气",
        r"sql",
        r"数据库",
        r"文件",
        r"python",
    )

    def preview(self, query: str, rag_available: bool = True) -> MultiAgentPlan:
        normalized = " ".join((query or "").split()).strip()
        complex_markers = self._match_patterns(normalized, self._complex_patterns)
        rag_markers = self._match_patterns(normalized, self._rag_patterns) if rag_available else []
        tool_markers = self._match_patterns(normalized, self._tool_patterns)

        tasks: List[CollaboratorTask] = []
        if rag_markers:
            tasks.append(
                CollaboratorTask(
                    task_id="task-rag-1",
                    title="Collect grounded knowledge-base evidence",
                    worker=AgentWorkerType.RAG,
                    query=normalized,
                    rationale="The request references knowledge-base material or cited evidence.",
                )
            )
        if tool_markers:
            tasks.append(
                CollaboratorTask(
                    task_id="task-tool-1",
                    title="Execute tools and gather operational results",
                    worker=AgentWorkerType.TOOL,
                    query=normalized,
                    rationale="The request contains execution, calculation, or tool-use signals.",
                )
            )

        distinct_workers = {task.worker for task in tasks}
        if not tasks:
            tasks.append(
                CollaboratorTask(
                    task_id="task-chat-1",
                    title="Handle the task as a focused reasoning dialogue",
                    worker=AgentWorkerType.CHAT,
                    query=normalized,
                    rationale="No knowledge-base or tool-specific signals were detected.",
                )
            )

        if len(distinct_workers) == 1 and complex_markers and AgentWorkerType.TOOL in distinct_workers:
            tasks.append(
                CollaboratorTask(
                    task_id="task-chat-2",
                    title="Review and structure the tool findings for the user",
                    worker=AgentWorkerType.CHAT,
                    query="Summarize the findings for the original task clearly and concisely.",
                    rationale="The task is complex enough to benefit from a separate synthesis collaborator.",
                )
            )

        complexity = "high" if len(tasks) >= 2 or len(complex_markers) >= 2 else "medium"
        signals = [*complex_markers, *rag_markers, *tool_markers]
        return MultiAgentPlan(
            complexity=complexity,
            signals=self._dedupe(signals),
            tasks=tasks,
        )

    def should_orchestrate(self, query: str, rag_available: bool = True) -> bool:
        plan = self.preview(query, rag_available=rag_available)
        if len(plan.tasks) < 2:
            return False
        return True

    def _match_patterns(self, text: str, patterns) -> List[str]:
        lowered = text.lower()
        return [pattern for pattern in patterns if re.search(pattern, lowered, flags=re.IGNORECASE)]

    def _dedupe(self, values: List[str]) -> List[str]:
        seen = set()
        deduped: List[str] = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            deduped.append(value)
        return deduped
