"""Model-specific investigation tools.

These tools complement the generic supplement tools. They remain deterministic:
LLM planners can select a registered tool, but all numeric facts are computed in
Python from the read-only row source.
"""

from .model_86_tools import transfer_structure_analyzer

__all__ = ["transfer_structure_analyzer"]
