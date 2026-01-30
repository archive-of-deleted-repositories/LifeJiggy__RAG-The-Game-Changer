"""
Evaluator - RAG-The-Game-Changer

Comprehensive evaluation orchestrator for RAG systems.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from .metrics import MetricsCalculator
from .hallucination_detection import HallucinationDetector
from .benchmarks import BenchmarkRunner

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation runs."""

    datasets: Dict[str, List[Dict]] = field(default_factory=dict)
    metrics: List[str] = field(
        default_factory=lambda: ["precision", "recall", "ndcg", "rouge", "bertscore"]
    )
    benchmarks: List[str] = field(default_factory=list)
    top_k_values: List[int] = field(default_factory=lambda: [5, 10, 20])
    enable_hallucination_check: bool = True
    enable_quality_assessment: bool = True


@dataclass
class EvaluationResult:
    """Result from evaluation run."""

    rag_pipeline_id: str
    overall_score: float
    metric_scores: Dict[str, float]
    benchmark_results: List[Dict[str, Any]]
    hallucination_stats: Dict[str, Any]
    quality_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    evaluation_time_ms: float


class Evaluator:
    """Main evaluation orchestrator for RAG systems."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.eval_config = EvaluationConfig(**self.config)

        self.metrics_calculator = MetricsCalculator()
        self.hallucination_detector = (
            HallucinationDetector() if self.eval_config.enable_hallucination_check else None
        )
        self.benchmark_runner = BenchmarkRunner(self.config.get("benchmark_config"))

    async def evaluate(self, rag_pipeline, test_data: Dict[str, List[Dict]]) -> EvaluationResult:
        """Run comprehensive evaluation of RAG pipeline."""
        start_time = asyncio.get_event_loop().time()

        logger.info(f"Starting evaluation for {self.eval_config.metrics} metrics")

        # Initialize results
        metric_scores = {}
        benchmark_results = []
        hallucination_stats = {}
        quality_score = 0.0

        # 1. Run metrics-based evaluation
        metric_scores = await self._evaluate_metrics(rag_pipeline, test_data)

        # 2. Run benchmarks
        if self.eval_config.benchmarks:
            benchmark_results = await self.benchmark_runner.run_all(rag_pipeline, test_data)

        # 3. Check for hallucinations
        if self.hallucination_detector:
            hallucination_stats = await self._evaluate_hallucinations(rag_pipeline, test_data)

        # 4. Quality assessment
        if self.eval_config.enable_quality_assessment:
            quality_score = await self._assess_quality(rag_pipeline, test_data)

        # Calculate overall score
        overall_score = self._calculate_overall_score(
            metric_scores, benchmark_results, hallucination_stats, quality_score
        )

        evaluation_time = (asyncio.get_event_loop().time() - start_time) * 1000

        result = EvaluationResult(
            rag_pipeline_id=str(id(rag_pipeline)),
            overall_score=overall_score,
            metric_scores=metric_scores,
            benchmark_results=[
                {"name": r.get("name"), "score": r.get("score"), "details": r.get("details")}
                for r in benchmark_results
            ],
            hallucination_stats=hallucination_stats,
            quality_score=quality_score,
            metadata={
                "config": self.eval_config.metrics,
                "top_k_values": self.eval_config.top_k_values,
            },
            evaluation_time_ms=evaluation_time,
        )

        logger.info(f"Evaluation complete. Overall score: {overall_score:.4f}")
        return result

    async def _evaluate_metrics(
        self, rag_pipeline, test_data: Dict[str, List[Dict]]
    ) -> Dict[str, float]:
        """Evaluate RAG pipeline using configured metrics."""
        scores = {}

        for metric in self.eval_config.metrics:
            try:
                score = await self.metrics_calculator.calculate_metric(
                    metric=metric,
                    rag_pipeline=rag_pipeline,
                    test_data=test_data,
                    top_k_values=self.eval_config.top_k_values,
                )
                scores[metric] = score
                logger.info(f"Metric {metric}: {score:.4f}")
            except Exception as e:
                logger.error(f"Error calculating metric {metric}: {e}")
                scores[metric] = 0.0

        return scores

    async def _evaluate_hallucinations(
        self, rag_pipeline, test_data: Dict[str, List[Dict]]
    ) -> Dict[str, Any]:
        """Evaluate hallucination rate of RAG pipeline."""
        if not self.hallucination_detector:
            return {}

        all_queries = []
        for dataset_queries in test_data.values():
            all_queries.extend(dataset_queries[:50])  # Sample 50 queries per dataset

        hallucinated = 0
        total = 0
        detailed_results = []

        for item in all_queries:
            try:
                query = item.get("query", "")
                result = await rag_pipeline.query(query=query, top_k=5)
                answer = result.answer
                retrieved_contexts = [chunk.get("content") for chunk in result.retrieved_chunks]

                # Check for hallucination
                is_hallucinated = await self.hallucination_detector.detect_hallucination(
                    query=query, answer=answer, contexts=retrieved_contexts
                )

                if is_hallucinated:
                    hallucinated += 1

                total += 1

                detailed_results.append(
                    {
                        "query": query,
                        "answer": answer,
                        "hallucinated": is_hallucinated,
                        "confidence": result.confidence,
                    }
                )

            except Exception as e:
                logger.error(f"Error checking hallucination: {e}")
                continue

        hallucination_rate = hallucinated / total if total > 0 else 0

        stats = {
            "total_queries": total,
            "hallucinated_count": hallucinated,
            "hallucination_rate": hallucination_rate,
            "results": detailed_results,
        }

        logger.info(f"Hallucination rate: {hallucination_rate:.2%}")
        return stats

    async def _assess_quality(self, rag_pipeline, test_data: Dict[str, List[Dict]]) -> float:
        """Assess overall quality of RAG responses."""
        all_queries = []
        for dataset_queries in test_data.values():
            all_queries.extend(dataset_queries[:50])

        quality_scores = []

        for item in all_queries:
            try:
                query = item.get("query", "")
                result = await rag_pipeline.query(query=query, top_k=5)
                answer = result.answer
                retrieved_chunks = result.retrieved_chunks

                # Assess quality
                relevance_score = self._assess_relevance(query, answer, retrieved_chunks)
                coherence_score = self._assess_coherence(answer)
                completeness_score = self._assess_completeness(query, answer)

                quality = (relevance_score + coherence_score + completeness_score) / 3
                quality_scores.append(quality)

            except Exception as e:
                logger.error(f"Error assessing quality: {e}")
                quality_scores.append(0.0)

        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

        logger.info(f"Average quality score: {avg_quality:.4f}")
        return avg_quality

    def _assess_relevance(self, query: str, answer: str, contexts: List) -> float:
        """Assess relevance of answer to query."""
        query_lower = query.lower()
        answer_lower = answer.lower()

        # Simple keyword overlap
        query_words = set(query_lower.split())
        answer_words = set(answer_lower.split())
        context_words = set(" ".join([c.get("content", "") for c in contexts]).lower().split())

        if len(query_words) == 0:
            return 0.5

        query_overlap = len(answer_words & query_words) / len(query_words)
        context_overlap = (
            len(answer_words & context_words) / len(context_words) if context_words else 0
        )

        return (query_overlap + context_overlap) / 2

    def _assess_coherence(self, answer: str) -> float:
        """Assess coherence of generated answer."""
        sentences = answer.split(".")

        if len(sentences) <= 1:
            return 1.0

        # Check for contradictions
        score = 1.0

        for i in range(len(sentences) - 1):
            s1_words = set(sentences[i].lower().split())
            s2_words = set(sentences[i + 1].lower().split())

            # If sentences share no words, might be incoherent
            if len(s1_words & s2_words) == 0:
                score -= 0.2

        return max(0.0, score)

    def _assess_completeness(self, query: str, answer: str) -> float:
        """Assess completeness of answer relative to query."""
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())

        if len(query_words) == 0:
            return 1.0

        # How much of query is addressed
        addressed = len(query_words & answer_words) / len(query_words)

        return min(1.0, addressed + 0.2)  # Bonus for covering all query aspects

    def _calculate_overall_score(
        self,
        metric_scores: Dict[str, float],
        benchmark_results: List[Dict],
        hallucination_stats: Dict,
        quality_score: float,
    ) -> float:
        """Calculate weighted overall evaluation score."""
        weights = {"metrics": 0.4, "benchmarks": 0.3, "hallucination": 0.2, "quality": 0.1}

        # Metric score (average of all metrics)
        if metric_scores:
            metric_avg = sum(metric_scores.values()) / len(metric_scores)
        else:
            metric_avg = 0.0

        # Benchmark score (average of all benchmarks)
        if benchmark_results:
            benchmark_avg = sum(r.get("score", 0) for r in benchmark_results) / len(
                benchmark_results
            )
        else:
            benchmark_avg = 0.0

        # Hallucination score (1 - hallucination_rate)
        hallucination_rate = hallucination_stats.get("hallucination_rate", 0)
        hallucination_score = 1.0 - hallucination_rate

        # Weighted average
        overall = (
            weights["metrics"] * metric_avg
            + weights["benchmarks"] * benchmark_avg
            + weights["hallucination"] * hallucination_score
            + weights["quality"] * quality_score
        )

        return overall

    def generate_report(self, result: EvaluationResult) -> str:
        """Generate human-readable evaluation report."""
        lines = [
            "=" * 80,
            "RAG PIPELINE EVALUATION REPORT",
            "=" * 80,
            "",
            f"Pipeline ID: {result.rag_pipeline_id}",
            f"Overall Score: {result.overall_score:.4f}",
            f"Quality Score: {result.quality_score:.4f}",
            f"Evaluation Time: {result.evaluation_time_ms:.2f}ms",
            "",
            "-" * 80,
            "METRIC SCORES",
            "-" * 80,
        ]

        for metric, score in result.metric_scores.items():
            lines.append(f"  {metric.upper()}: {score:.4f}")

        lines.extend(
            [
                "",
                "-" * 80,
                "HALLUCINATION STATS",
                "-" * 80,
                f"  Total Queries: {result.hallucination_stats.get('total_queries', 0)}",
                f"  Hallucinated: {result.hallucination_stats.get('hallucinated_count', 0)}",
                f"  Hallucination Rate: {result.hallucination_stats.get('hallucination_rate', 0):.2%}",
                "",
                "-" * 80,
                "BENCHMARK RESULTS",
                "-" * 80,
            ]
        )

        for bench in result.benchmark_results:
            lines.append(f"  {bench['name']}: {bench['score']:.4f}")

        lines.extend(
            [
                "",
                "=" * 80,
                "END OF REPORT",
                "=" * 80,
            ]
        )

        return "\n".join(lines)
