import importlib
import unittest

from backend.infrastructure.tools import file_ops


class SecurityHardeningTests(unittest.TestCase):
    def test_file_tool_blocks_prefix_path_escape(self):
        result = file_ops.write_local_file("../agent_sandbox_evil/escape.txt", "nope")

        self.assertIn("Security blocked", result)

    def test_stage1_enterprise_skeleton_modules_are_importable(self):
        modules = [
            "backend.domain.interfaces.agent_runtime",
            "backend.domain.interfaces.approval_store",
            "backend.domain.interfaces.trace_store",
            "backend.domain.interfaces.tool_provider",
            "backend.domain.entities.agent_state",
            "backend.domain.entities.approval",
            "backend.domain.entities.trace_event",
            "backend.domain.entities.capability",
            "backend.infrastructure.approvals.local_approval_store",
            "backend.infrastructure.agent_runtime.custom_runtime",
            "backend.infrastructure.agent_runtime.langgraph_runtime",
            "backend.infrastructure.rerankers.cross_encoder_reranker",
            "backend.infrastructure.rerankers.noop_reranker",
            "backend.infrastructure.mcp.server",
            "backend.infrastructure.mcp.client_adapter",
            "backend.infrastructure.tool_adapters.langchain_adapter",
            "backend.infrastructure.tool_adapters.mcp_adapter",
            "backend.infrastructure.search.elasticsearch_store",
            "backend.infrastructure.memory.mysql_memory",
            "backend.infrastructure.tracing.local_trace_store",
            "backend.infrastructure.tools.python_sandbox",
        ]

        for module_name in modules:
            with self.subTest(module=module_name):
                self.assertIsNotNone(importlib.import_module(module_name))


if __name__ == "__main__":
    unittest.main()
