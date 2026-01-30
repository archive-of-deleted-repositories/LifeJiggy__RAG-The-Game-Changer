"""
Evaluation Framework - RAG-The-Game-Changer

Comprehensive evaluation system for RAG pipelines.
"""

from .metrics import MetricsCalculator
from .hallucination_detection import HallucinationDetector
from .benchmarks import BenchmarkRunner, Benchmark, BenchmarkResult
from .evaluator import Evaluator, EvaluationConfig, EvaluationResult

__all__ = [
    "MetricsCalculator",
    "HallucinationDetector",
    "BenchmarkRunner",
    "Benchmark",
    "BenchmarkResult",
    "Evaluator",
    "EvaluationConfig",
    "EvaluationResult",
]
