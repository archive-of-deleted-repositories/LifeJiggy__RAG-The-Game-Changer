"""
Generation Components - RAG-The-Game-Changer

Advanced components for answer generation and formatting.
"""

from .answer_generation import GroundedGenerator
from .prompt_engineering import PromptEngineer, PromptConfig
from .hallucination_control import HallucinationController
from .output_formatting import OutputFormatter, OutputFormat

__all__ = [
    "GroundedGenerator",
    "PromptEngineer",
    "PromptConfig",
    "HallucinationController",
    "OutputFormatter",
    "OutputFormat",
]
