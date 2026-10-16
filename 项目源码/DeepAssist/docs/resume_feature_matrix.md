# DeepAssist 简历条目验收矩阵

本表把 `resume.tex` 中 **Agent 后端与知识库问答系统｜个人项目** 的四组描述拆成可验收的功能点。每一行只有在“源码实现存在、对应单元测试存在且本轮通过”后才能标记为 `✅`；任一行不是 `✅` 都不能把本轮重构视为完成。测试名称由 `tests/test_resume_feature_matrix.py` 自动核对。

| 编号 | 简历条目中的能力 | 可核查的源码证据 | 单元测试 / 回归测试 | 状态 |
|---|---|---|---|---|
| 1.1 | Clean Architecture：领域、应用、基础设施分层，Port 由 Adapter 实现 | `backend/domain/interfaces/`；`backend/infrastructure/`；`tests/test_acceptance_contract.py` | `test_domain_and_application_keep_vendor_sdks_out_of_business_layers` | ✅ |
| 1.2 | 最小 Agent 控制循环、迭代上限与终止状态 | `backend/domain/services/agent/engine.py`；`lifecycle.py` | `test_agent_engine_records_completed_model_call` | ✅ |
| 1.3 | 记录模型调用（模型、耗时、消息数、工具数、成功/失败） | `StreamEvent.model_call`；`engine.py`；`langgraph_runtime.py` | `test_agent_engine_records_completed_model_call`；`test_custom_runtime_maps_model_call_to_transport_event` | ✅ |
| 1.4 | 工具调用与 ToolResult 统一流转 | `backend/domain/entities/tooling.py`；`ToolRegistry.execute` | `test_agent_engine_emits_tool_result_and_state_snapshot` | ✅ |
| 1.5 | 状态变化以 task snapshot 持久化并可恢复 | `backend/domain/entities/task_snapshot.py`；`AgentApplication` | `test_agent_engine_emits_tool_result_and_state_snapshot`；`test_sqlite_snapshot_survives_store_reconstruction` | ✅ |
| 1.6 | LangGraph 规划—审批—执行—结束主流程 | `backend/infrastructure/agent_runtime/langgraph_runtime.py` | `test_langgraph_checkpoint_gate_surfaces_pending_approval` | ✅ |
| 2.1 | 文档解析、分块、索引与在线查询职责分离 | `DocumentChunker`；`LangChainIngestionAdapter`；`KnowledgeBaseApp.process_document` | `test_document_chunker_keeps_heading_source_and_overlap_metadata`；`test_kb_ingestion_separates_chunk_embedding_and_dual_index_writes`；`test_langchain_loader_splitter_and_kb_ingestion_boundary_are_connected`；`test_default_kb_upload_uses_langchain_ingestion_boundary` | ✅ |
| 2.2 | BGE-M3 向量编码与批量推理 | `backend/infrastructure/embeddings/bge_m3_local.py` | `test_bge_m3_adapter_normalizes_single_and_batch_embeddings` | ✅ |
| 2.3 | Chroma / LangChain 向量写入与查询适配 | `backend/infrastructure/databases/chroma_store.py`；`LangChainVectorStoreAdapter` | `test_chroma_adapter_adds_and_reads_document_chunks`；`test_langchain_vector_store_adapter_uses_project_document_contract`；`test_langchain_vector_store_adapter_preserves_project_embeddings` | ✅ |
| 2.4 | Elasticsearch 关键词检索与 Whoosh 离线回退 | `backend/infrastructure/search/elasticsearch_store.py`；`resilient_keyword_store.py` | `test_elasticsearch_keyword_adapter_indexes_and_searches_chunks`；`test_resilient_keyword_store_falls_back_to_local_index` | ✅ |
| 2.5 | Weighted RRF 融合并保留各通道贡献 | `backend/domain/services/rag/fusion.py` | `test_hybrid_retrieval_records_weighted_rrf_contributions` | ✅ |
| 2.6 | 重排及失败时的融合顺序回退 | `backend/infrastructure/rerankers/`；`backend/domain/services/rag/reranker.py` | `test_cross_encoder_reranker_orders_documents_by_model_score`；`test_cross_encoder_reranker_falls_back_to_lexical_when_model_load_fails` | ✅ |
| 2.7 | 引用打包、引用校验与无依据回答拦截 | `ContextPacker`；`SourceAwareResponseGuard` | `test_context_packing_and_citation_guard_validate_grounding`；`test_answer_guard_recommends_regeneration_when_citations_are_missing` | ✅ |
| 2.8 | 召回、排序、引用和生成阶段的结构化诊断 | `RetrievalResult.to_stream_data`；`TraceEvent` | `test_hybrid_retriever_attaches_failure_attribution_metadata`；`test_conversation_trace_preserves_model_call_payload` | ✅ |
| 2.9 | LangChain Loader/Splitter/VectorStore/Retriever/Runnable 与领域文档/链路的边界适配 | `backend/infrastructure/rag/langchain_adapters.py` | `test_langchain_retriever_and_runnable_adapters_normalize_framework_values`；`test_langchain_loader_splitter_and_kb_ingestion_boundary_are_connected`；`test_langchain_vector_store_adapter_uses_project_document_contract`；`test_langchain_vector_store_adapter_preserves_project_embeddings` | ✅ |
| 3.1 | MCP 知识库 / 文件工具发现与执行 | `backend/infrastructure/mcp/`；`backend/infrastructure/tools/`；默认 `ToolRegistry` | `test_mcp_server_exposes_governed_knowledge_and_file_tools`；`test_mcp_adapter_supports_async_discovery_and_execution`；`test_mcp_tool_client_adapter_normalizes_and_executes_tools`；`test_default_tool_registry_routes_read_only_kb_and_file_tools_through_mcp` | ✅ |
| 3.2 | 工具参数校验与修复诊断 | `ToolRegistry._validate_arguments` | `test_tool_policy_validates_arguments_and_permission_boundaries`；`test_tool_registry_returns_structured_argument_repair_metadata` | ✅ |
| 3.3 | 工具白名单、分类权限和高风险审批 | `ToolPolicy`；`LocalApprovalStore` | `test_tool_policy_validates_arguments_and_permission_boundaries`；`test_high_risk_tool_requires_hitl_before_execution` | ✅ |
| 3.4 | 工具调用预算 / 超时边界 | `ToolPolicy.tool_timeout_seconds`；`ToolRegistry._execute_spec` | `test_tool_policy_enforces_execution_timeout` | ✅ |
| 3.5 | 工具异常统一为可诊断 ToolResult | `ToolRegistry._execution_failure_metadata`；`ToolResult` | `test_tool_registry_returns_structured_argument_repair_metadata`；`test_tool_policy_enforces_execution_timeout` | ✅ |
| 3.6 | 操作步骤、约束、检查标准、工具和资源的版本化 Skill | `backend/domain/entities/skill.py`；`skill_registry.py` | `test_versioned_skills_are_selected_and_injected_into_agent_context` | ✅ |
| 3.7 | 会话消息、历史轮次与用户画像分别组装 | `ContextEngine`；`SessionManager` | `test_rag_prompt_includes_user_profile`；`test_session_history_remains_full_without_summary_injection` | ✅ |
| 3.8 | 任务状态、长期偏好与 Token 上下文预算分离 | `PriorityContextWindowManager`；`LongTermMemoryRecallService` | `test_long_term_memory_consumes_budget_but_keeps_recent_turns`；`test_context_window_trace_data_contains_summary_and_memory` | ✅ |
| 4.1 | SQLite Checkpoint、Interrupt、Resume 与人工审批 | `LangGraphAgentRuntimeAdapter`；`SqliteSaver`；`SQLiteMemoryStore` | `test_langgraph_checkpoint_gate_surfaces_pending_approval`；`test_agent_application_resumes_interrupted_task_from_snapshot` | ✅ |
| 4.2 | SSE 推送状态、节点、工具和模型进度 | `StreamEvent`；`AgentApplication`；`SSEManager` | `test_agent_application_can_stream_with_runtime_adapter`；`test_conversation_agent_route_preserves_outer_and_inner_route_events` | ✅ |
| 4.3 | Trace 关联模型调用、工具执行和状态事件；可选 LangSmith 上报 | `ConversationApplication._trace_sse_chunk`；`LocalTraceStore`；`LangSmithTraceStore` | `test_conversation_trace_preserves_model_call_payload`；`test_conversation_stream_records_local_trace_events`；`test_langsmith_trace_store_mirrors_local_event` | ✅ |
| 4.4 | 异常诊断、自纠错、失败恢复与部分结果收敛 | `AgentEngine`；`AgentLifecycle` | `test_agent_engine_injects_self_correction_instruction_after_tool_failure`；`test_agent_engine_finalizes_partial_answer_at_step_limit_after_successful_tools` | ✅ |

## 验收命令

说明：4.3 默认验收本地 trace；LangSmith 远端上报只有在显式设置
`LANGSMITH_TRACING=true` 和 `LANGSMITH_API_KEY` 后才启用。

简历中的 Plan-and-Solve 是对外层任务规划、执行和 Reflection 重规划流程的概念归纳，源码没有声称实现一个同名论文算法。Cross-Encoder 是应用默认重排路径，模型加载失败时会记录诊断并退回词法重排。

```powershell
& C:\Users\Neuracle\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m pytest tests -q
& C:\Users\Neuracle\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m compileall backend tests evals
```
