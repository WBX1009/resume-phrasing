from types import SimpleNamespace
from unittest.mock import patch
import asyncio

from backend.api import dependencies
from backend.common.config import settings
from backend.infrastructure.llms.openai_compatible import OpenAICompatibleClient
from backend.infrastructure.search.resilient_keyword_store import ResilientKeywordDB
from backend.domain.entities.document import DocumentChunk
from backend.domain.entities.skill import SkillSpec
from backend.domain.services.agent.skill_registry import SkillRegistry
from backend.domain.services.session.context_engine import ContextEngine
from backend.infrastructure.mcp.client_adapter import MCPToolClientAdapter
from backend.domain.entities.tooling import ToolCall


class _Completions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return [
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(reasoning_content="think", content="")
                        )
                    ]
                ),
                SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content="answer"))]
                ),
            ]
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="answer", tool_calls=[])
                )
            ]
        )


class _FakeClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=_Completions())


def test_openai_compatible_client_uses_chat_completions_contract():
    fake = _FakeClient()
    client = OpenAICompatibleClient(
        model_name="deepseek-chat",
        base_url="http://127.0.0.1:8000/v1",
        api_key="EMPTY",
        client=fake,
    )

    message = client.chat(
        [{"role": "user", "content": "hello"}],
        tools=[{"type": "function", "function": {"name": "lookup"}}],
        temperature=0,
    )
    assert message.content == "answer"
    assert fake.chat.completions.calls[0]["model"] == "deepseek-chat"
    assert fake.chat.completions.calls[0]["tools"]

    stream = list(client.chat_stream([{"role": "user", "content": "hello"}]))
    assert stream == [
        {"type": "reasoning", "content": "think"},
        {"type": "content", "content": "answer"},
    ]


def test_openai_provider_selects_optional_adapter():
    dependencies.get_llm.cache_clear()
    with patch.object(settings, "LLM_PROVIDER", "openai_compatible"):
        llm = dependencies.get_llm()
    try:
        assert isinstance(llm, OpenAICompatibleClient)
    finally:
        dependencies.get_llm.cache_clear()


def test_resilient_keyword_store_falls_back_to_local_index():
    class FakeIndex:
        def __init__(self, results=None, build=True):
            self.results = results or []
            self.build = build

        def build_index(self, collection_name, chunks):
            return self.build

        def search(self, collection_name, query_text, top_k):
            return self.results

        def delete_by_source(self, collection_name, source_file):
            return True

        def list_sources(self, collection_name):
            return []

        def list_collections(self):
            return []

    chunk = DocumentChunk(id="c1", content="local result", metadata={})
    store = ResilientKeywordDB(FakeIndex(), FakeIndex([chunk]))
    assert store.search("kb", "local", 5) == [chunk]
    assert store.build_index("kb", [chunk]) is True


def test_versioned_skills_are_selected_and_injected_into_agent_context():
    registry = SkillRegistry(
        [
            SkillSpec(
                name="core",
                version="1.0.0",
                instructions="Use the supplied tool schema.",
            ),
            SkillSpec(
                name="rag",
                version="1.2.0",
                trigger_keywords=["知识库"],
                tool_names=["search_knowledge_base"],
                constraints=["Use retrieved evidence for factual claims."],
                check_standards=["Every citation must refer to a retrieved source."],
                resources={"citation_format": "[C#]"},
                instructions="Cite retrieved evidence.",
            ),
        ]
    )
    assert registry.get("rag").version == "1.2.0"
    rendered = registry.render_for_task("查询知识库")
    assert "[core@1.0.0]" in rendered
    assert "[rag@1.2.0]" in rendered

    context = ContextEngine(skill_registry=registry).build_agent_context(
        query="查询知识库",
        history=[],
    )
    assert "Loaded Skills" in context.messages[0]["content"]
    assert "Cite retrieved evidence." in context.messages[0]["content"]
    assert "search_knowledge_base" in context.messages[0]["content"]
    assert "Use retrieved evidence for factual claims." in context.messages[0]["content"]
    assert "Every citation must refer to a retrieved source." in context.messages[0]["content"]
    assert '"citation_format": "[C#]"' in context.messages[0]["content"]


def test_mcp_adapter_supports_async_discovery_and_execution():
    class AsyncClient:
        async def list_tools(self):
            return [{"name": "lookup", "description": "Lookup", "inputSchema": {"type": "object"}}]

        async def call_tool(self, name, arguments):
            return {"name": name, "query": arguments["query"]}

    async def run():
        adapter = MCPToolClientAdapter(AsyncClient(), tool_prefix="mcp__")
        tools = await adapter.alist_tools()
        result = await adapter.aexecute(
            ToolCall(id="call-async", name="mcp__lookup", args={"query": "DeepAssist"})
        )
        return tools, result

    tools, result = asyncio.run(run())
    assert tools[0]["name"] == "mcp__lookup"
    assert result.success is True
    assert "DeepAssist" in result.content
