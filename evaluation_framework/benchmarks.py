"""
Benchmarks - RAG-The-Game-Changer

Standard benchmark implementations for evaluating RAG systems.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result from running a benchmark."""

    name: str
    score: float
    details: Dict[str, Any]
    metadata: Dict[str, Any]
    execution_time_ms: float


class Benchmark(ABC):
    """Abstract base class for RAG benchmarks."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    @abstractmethod
    async def run(self, rag_pipeline, test_data: List[Dict]) -> BenchmarkResult:
        """Run the benchmark."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get benchmark name."""
        pass


class SQuADBenchmark(Benchmark):
    """Stanford Question Answering Dataset benchmark."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.dataset_path = self.config.get("dataset_path")
        self.sample_size = self.config.get("sample_size", 100)

    def get_name(self) -> str:
        return "SQuAD"

    async def run(self, rag_pipeline, test_data: List[Dict]) -> BenchmarkResult:
        """Run SQuAD benchmark evaluating EM and F1."""
        start_time = time.time()

        correct_exact = 0
        correct_f1 = 0
        total = len(test_data)
        predictions = []

        for item in test_data[: self.sample_size]:
            try:
                context = item.get("context", "")
                question = item.get("question", "")
                answers = item.get("answers", [])

                result = await rag_pipeline.query(query=question, top_k=5, include_sources=True)

                answer = result.answer
                predictions.append({"id": item.get("id"), "prediction": answer, "answers": answers})

                # Calculate exact match score
                for correct_answer in answers:
                    if self._exact_match(answer, correct_answer):
                        correct_exact += 1
                        break

                # Calculate F1 score
                for correct_answer in answers:
                    f1 = self._calculate_f1(answer, correct_answer)
                    correct_f1 += f1

            except Exception as e:
                logger.error(f"Error processing item {item.get('id')}: {e}")
                continue

        execution_time = (time.time() - start_time) * 1000

        em_score = correct_exact / total if total > 0 else 0
        f1_score = correct_f1 / total if total > 0 else 0

        return BenchmarkResult(
            name=self.get_name(),
            score=(em_score + f1_score) / 2,
            details={
                "exact_match": em_score,
                "f1_score": f1_score,
                "total_questions": total,
                "sample_size": self.sample_size,
            },
            metadata={"predictions": predictions},
            execution_time_ms=execution_time,
        )

    def _exact_match(self, prediction: str, reference: str) -> bool:
        """Check if prediction exactly matches reference."""
        prediction_clean = prediction.strip().lower()
        reference_clean = reference.strip().lower()
        return prediction_clean == reference_clean

    def _calculate_f1(self, prediction: str, reference: str) -> float:
        """Calculate F1 score between prediction and reference."""
        pred_tokens = prediction.lower().split()
        ref_tokens = reference.lower().split()

        common = set(pred_tokens) & set(ref_tokens)

        if len(pred_tokens) == 0:
            return 0.0

        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(ref_tokens)

        if precision + recall == 0:
            return 0.0

        f1 = 2 * (precision * recall) / (precision + recall)
        return f1


class MSMARCOBenchmark(Benchmark):
    """MS MARCO passage ranking benchmark."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.dataset_path = self.config.get("dataset_path")
        self.sample_size = self.config.get("sample_size", 100)

    def get_name(self) -> str:
        return "MS-MARCO"

    async def run(self, rag_pipeline, test_data: List[Dict]) -> BenchmarkResult:
        """Run MS MARCO benchmark evaluating MRR@10."""
        start_time = time.time()

        mrr_sum = 0
        total = len(test_data)
        predictions = []

        for item in test_data[: self.sample_size]:
            try:
                query = item.get("query", "")
                relevant_passages = item.get("passages", [])
                relevant_ids = {p.get("id") for p in relevant_passages}

                result = await rag_pipeline.query(query=query, top_k=10, include_sources=True)

                retrieved_ids = {chunk.get("document_id") for chunk in result.retrieved_chunks}

                # Calculate MRR
                mrr = self._calculate_mrr(retrieved_ids, relevant_ids)
                mrr_sum += mrr

                predictions.append(
                    {
                        "query": query,
                        "mrr": mrr,
                        "retrieved": len(retrieved_ids),
                        "relevant": len(relevant_ids),
                    }
                )

            except Exception as e:
                logger.error(f"Error processing query: {e}")
                continue

        execution_time = (time.time() - start_time) * 1000

        mrr_score = mrr_sum / total if total > 0 else 0

        return BenchmarkResult(
            name=self.get_name(),
            score=mrr_score,
            details={"mrr@10": mrr_score, "total_queries": total, "sample_size": self.sample_size},
            metadata={"predictions": predictions},
            execution_time_ms=execution_time,
        )

    def _calculate_mrr(self, retrieved: set, relevant: set) -> float:
        """Calculate Mean Reciprocal Rank."""
        for i, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant:
                return 1.0 / i
        return 0.0


class NaturalQuestionsBenchmark(Benchmark):
    """Natural Questions benchmark for open-domain QA."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.dataset_path = self.config.get("dataset_path")
        self.sample_size = self.config.get("sample_size", 100)

    def get_name(self) -> str:
        return "NaturalQuestions"

    async def run(self, rag_pipeline, test_data: List[Dict]) -> BenchmarkResult:
        """Run Natural Questions benchmark."""
        start_time = time.time()

        correct_count = 0
        total = len(test_data)
        predictions = []

        for item in test_data[: self.sample_size]:
            try:
                question = item.get("question", "")
                answer = item.get("answer", "")

                result = await rag_pipeline.query(query=question, top_k=5)

                is_correct = self._fuzzy_match(result.answer, answer)
                if is_correct:
                    correct_count += 1

                predictions.append(
                    {
                        "question": question,
                        "prediction": result.answer,
                        "answer": answer,
                        "correct": is_correct,
                    }
                )

            except Exception as e:
                logger.error(f"Error processing question: {e}")
                continue

        execution_time = (time.time() - start_time) * 1000

        accuracy = correct_count / total if total > 0 else 0

        return BenchmarkResult(
            name=self.get_name(),
            score=accuracy,
            details={
                "accuracy": accuracy,
                "correct": correct_count,
                "total": total,
                "sample_size": self.sample_size,
            },
            metadata={"predictions": predictions},
            execution_time_ms=execution_time,
        )

    def _fuzzy_match(self, prediction: str, reference: str) -> bool:
        """Fuzzy match for Natural Questions."""
        pred_lower = prediction.strip().lower()
        ref_lower = reference.strip().lower()
        return pred_lower == ref_lower


class RetrievalBenchmark(Benchmark):
    """Pure retrieval evaluation benchmark."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.top_k = self.config.get("top_k", 10)

    def get_name(self) -> str:
        return "Retrieval"

    async def run(self, rag_pipeline, test_data: List[Dict]) -> BenchmarkResult:
        """Evaluate pure retrieval performance (Precision@k, Recall@k)."""
        start_time = time.time()

        total_relevant = 0
        total_retrieved = 0
        predictions = []

        for item in test_data:
            try:
                query = item.get("query", "")
                relevant_ids = set(item.get("relevant_doc_ids", []))

                # Direct retrieval without generation
                from retrieval_systems.base import RetrievalResult

                if hasattr(rag_pipeline, "retriever"):
                    retrieval_result = await rag_pipeline.retriever.retrieve(
                        query=query, top_k=self.top_k
                    )
                else:
                    # Fallback to query method
                    result = await rag_pipeline.query(query=query, top_k=self.top_k)
                    retrieval_result = RetrievalResult(
                        query=query,
                        chunks=result.retrieved_chunks,
                        strategy=rag_pipeline.retrieval_strategy,
                        total_chunks=len(result.retrieved_chunks),
                        retrieval_time_ms=result.retrieval_time_ms,
                    )

                retrieved_ids = {chunk.get("document_id") for chunk in retrieval_result.chunks}
                retrieved_relevant = retrieved_ids & relevant_ids

                total_relevant += len(retrieved_relevant)
                total_retrieved += self.top_k

                predictions.append(
                    {
                        "query": query,
                        "retrieved": list(retrieved_ids),
                        "relevant": len(relevant_ids),
                        "precision": len(retrieved_relevant) / self.top_k,
                        "recall": len(retrieved_relevant) / len(relevant_ids)
                        if relevant_ids
                        else 0,
                    }
                )

            except Exception as e:
                logger.error(f"Error processing retrieval: {e}")
                continue

        execution_time = (time.time() - start_time) * 1000

        avg_precision = total_relevant / total_retrieved if total_retrieved > 0 else 0
        avg_recall = total_relevant / total_relevant if total_relevant > 0 else 0

        return BenchmarkResult(
            name=self.get_name(),
            score=(avg_precision + avg_recall) / 2,
            details={"precision@k": avg_precision, "recall@k": avg_recall, "top_k": self.top_k},
            metadata={"predictions": predictions},
            execution_time_ms=execution_time,
        )


class BenchmarkRunner:
    """Orchestrates running multiple benchmarks."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.benchmarks: List[Benchmark] = []
        self._load_benchmarks()

    def _load_benchmarks(self):
        """Load configured benchmarks."""
        benchmark_configs = self.config.get("benchmarks", ["squad", "msmarco", "natural_questions"])

        if "squad" in benchmark_configs:
            self.benchmarks.append(SQuADBenchmark(self.config.get("squad_config")))

        if "msmarco" in benchmark_configs:
            self.benchmarks.append(MSMARCOBenchmark(self.config.get("msmarco_config")))

        if "natural_questions" in benchmark_configs:
            self.benchmarks.append(
                NaturalQuestionsBenchmark(self.config.get("natural_questions_config"))
            )

        if "retrieval" in benchmark_configs:
            self.benchmarks.append(RetrievalBenchmark(self.config.get("retrieval_config")))

    async def run_all(
        self, rag_pipeline, test_data: Dict[str, List[Dict]]
    ) -> List[BenchmarkResult]:
        """Run all configured benchmarks."""
        results = []

        for benchmark in self.benchmarks:
            dataset_name = benchmark.get_name().lower()
            dataset = test_data.get(dataset_name, [])

            if not dataset:
                logger.warning(f"No test data for {dataset_name}")
                continue

            logger.info(f"Running benchmark: {benchmark.get_name()}")

            try:
                result = await benchmark.run(rag_pipeline, dataset)
                results.append(result)

                logger.info(
                    f"Benchmark {result.name}: {result.score:.4f} "
                    f"(took {result.execution_time_ms:.2f}ms)"
                )
            except Exception as e:
                logger.error(f"Error running benchmark {benchmark.get_name()}: {e}")

        return results

    def get_summary(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Generate summary of benchmark results."""
        return {
            "total_benchmarks": len(results),
            "average_score": sum(r.score for r in results) / len(results) if results else 0,
            "benchmark_details": [
                {"name": r.name, "score": r.score, "details": r.details} for r in results
            ],
        }
