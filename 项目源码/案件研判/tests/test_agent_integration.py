from __future__ import annotations

from dataclasses import replace
import unittest

from src.agent_loop import AgentLoopConfig
from src.config import load_config
from src.phase1_demo import build_phase1_demo


class AgentIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = load_config()
        if config.snapshot_dir is None:
            raise unittest.SkipTest(
                "Public clean copy excludes the private snapshot; provide an authorized fixture to run integration tests."
            )
        cls.config = replace(
            config,
            primary_llm=replace(config.primary_llm, api_key=None, base_url=None),
            fallback_llm=None,
        )

    def test_facts_only_snapshot_skips_agent_roles(self):
        result = build_phase1_demo(
            self.config,
            case_id=130021,
            report_mode="final",
            analysis_mode="agent-loop-demo",
            data_source="snapshot",
            agent_loop_config=AgentLoopConfig(fallback_policy="facts_only"),
        )
        self.assertIsNone(result["planner_result"])
        self.assertEqual(result["final_report_audit"]["mode"], "facts-only")
        self.assertEqual(result["report_assertions"], [])
        self.assertTrue(result["final_report"])

    def test_review_safe_snapshot_keeps_fallback_conclusions_unresolved(self):
        result = build_phase1_demo(
            self.config,
            case_id=130022,
            report_mode="final",
            analysis_mode="agent-loop-demo",
            data_source="snapshot",
            agent_loop_config=AgentLoopConfig(
                max_model_classes=1,
                max_hypotheses_per_class=1,
                max_agent_rounds=2,
                max_tool_calls_per_hypothesis=2,
                fallback_policy="review_safe",
            ),
        )
        planner_result = result["planner_result"]
        self.assertTrue(planner_result["planner_audit"]["fallback_used"])
        self.assertGreater(len(planner_result["supplement_facts"]), 0)
        self.assertTrue(all(chain["final_status"] == "unresolved" for chain in planner_result["evidence_chains"]))
        self.assertTrue(all(item["status"] == "unresolved" for item in result["report_assertions"]))
        self.assertTrue(result["final_report_audit"]["fallback_used"])

    def test_full_agent_snapshot_emits_model_investigation_evidence(self):
        """The model-level 86 investigation must remain executable without an LLM key."""
        result = build_phase1_demo(
            self.config,
            case_id=130021,
            report_mode="final",
            analysis_mode="full-agent",
            data_source="snapshot",
            agent_loop_config=AgentLoopConfig(
                max_model_classes=1,
                max_hypotheses_per_class=1,
                max_agent_rounds=2,
                max_tool_calls_per_hypothesis=2,
                fallback_policy="review_safe",
            ),
        )
        planner_result = result["planner_result"]
        self.assertTrue(planner_result)
        self.assertGreaterEqual(planner_result["planner_audit"]["model_claim_count"], 1)
        model_claim = next(item for item in planner_result["model_claims"] if item["model_id"] == 86)
        self.assertIn(model_claim["status"], {"enhanced", "weakened", "unresolved"})
        investigation = planner_result["planner_audit"]["model_investigations"][0]
        self.assertEqual(investigation["model_id"], 86)
        self.assertIn("transfer_structure_analyzer", investigation["allowed_tools"])
        self.assertGreaterEqual(len(planner_result["supplement_facts"]), 1)
        self.assertTrue(
            any(item["tool_name"] == "transfer_structure_analyzer" for item in planner_result["supplement_facts"])
        )
        # No provider response is available in this fixture, so the review-safe policy
        # must not promote fallback evidence into a definitive final assertion.
        self.assertTrue(all(item["status"] == "unresolved" for item in result["report_assertions"]))


if __name__ == "__main__":
    unittest.main()
