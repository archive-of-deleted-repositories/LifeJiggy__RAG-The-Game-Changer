"""
RAG Metrics Calculator - RAG-The-Game-Changer

Comprehensive metrics calculation for RAG evaluation.
"""

import asyncio
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import time

logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    """Result of a metric calculation."""

    name: str
    value: float
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class RAGMetrics:
    """Comprehensive RAG metrics calculator."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.rouge_available = self._check_rouge()
        self.bert_score_available = self._check_bert_score()

    def _check_rouge(self) -> bool:
        """Check if ROUGE is available."""
        try:
            from rouge_score import rouge_scorer

            return True
        except ImportError:
            return False

    def _check_bert_score(self) -> bool:
        """Check if BERTScore is available."""
        try:
            from bert_score import score as bert_score

            return True
        except ImportError:
            return False

    async def calculate_retrieval_metrics(
        self,
        retrieved_docs: List[Dict[str, Any]],
        relevant_docs: List[str],
        top_k: Optional[int] = None,
    ) -> Dict[str, MetricResult]:
        """Calculate retrieval metrics."""
        results = {}

        # Precision@K
        precision = self.calculate_precision_at_k(retrieved_docs, relevant_docs, top_k)
        results[f"precision_at_{top_k or len(retrieved_docs)}"] = MetricResult(
            name=f"Precision@{top_k or len(retrieved_docs)}",
            value=precision,
            details={"retrieved": len(retrieved_docs), "relevant": len(relevant_docs)},
        )

        # Recall@K
        recall = self.calculate_recall_at_k(retrieved_docs, relevant_docs, top_k)
        results[f"recall_at_{top_k or len(retrieved_docs)}"] = MetricResult(
            name=f"Recall@{top_k or len(retrieved_docs)}",
            value=recall,
            details={"retrieved": len(retrieved_docs), "relevant": len(relevant_docs)},
        )

        # F1@K
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        results[f"f1_at_{top_k or len(retrieved_docs)}"] = MetricResult(
            name=f"F1@{top_k or len(retrieved_docs)}",
            value=f1,
            details={"precision": precision, "recall": recall},
        )

        # NDCG@K
        ndcg = self.calculate_ndcg_at_k(retrieved_docs, relevant_docs, top_k)
        results[f"ndcg_at_{top_k or len(retrieved_docs)}"] = MetricResult(
            name=f"NDCG@{top_k or len(retrieved_docs)}",
            value=ndcg,
            details={"retrieved": len(retrieved_docs)},
        )

        return results

    def calculate_precision_at_k(
        self,
        retrieved_docs: List[Dict[str, Any]],
        relevant_docs: List[str],
        k: Optional[int] = None,
    ) -> float:
        """Calculate precision at K."""
        if not retrieved_docs or not relevant_docs:
            return 0.0

        k = k or len(retrieved_docs)
        retrieved_at_k = retrieved_docs[:k]

        retrieved_ids = [doc.get("document_id", "") for doc in retrieved_at_k]
        relevant_set = set(relevant_docs)

        relevant_retrieved = sum(1 for doc_id in retrieved_ids if doc_id in relevant_set)

        return relevant_retrieved / len(retrieved_at_k)

    def calculate_recall_at_k(
        self,
        retrieved_docs: List[Dict[str, Any]],
        relevant_docs: List[str],
        k: Optional[int] = None,
    ) -> float:
        """Calculate recall at K."""
        if not relevant_docs:
            return 0.0

        k = k or len(retrieved_docs)
        retrieved_at_k = retrieved_docs[:k]

        retrieved_ids = [doc.get("document_id", "") for doc in retrieved_at_k]
        relevant_set = set(relevant_docs)

        relevant_retrieved = sum(1 for doc_id in retrieved_ids if doc_id in relevant_set)

        return relevant_retrieved / len(relevant_set)

    def calculate_ndcg_at_k(
        self,
        retrieved_docs: List[Dict[str, Any]],
        relevant_docs: List[str],
        k: Optional[int] = None,
    ) -> float:
        """Calculate NDCG at K."""
        if not retrieved_docs:
            return 0.0

        k = k or len(retrieved_docs)
        retrieved_at_k = retrieved_docs[:k]

        # Calculate DCG
        dcg = 0.0
        for i, doc in enumerate(retrieved_at_k):
            doc_id = doc.get("document_id", "")
            relevance = 1.0 if doc_id in set(relevant_docs) else 0.0
            dcg += relevance / (i + 1)

        # Calculate IDCG (Ideal DCG)
        idcg = 0.0
        for i in range(min(k, len(relevant_docs))):
            idcg += 1.0 / (i + 1)

        return dcg / idcg if idcg > 0 else 0.0

    async def calculate_generation_metrics(
        self,
        generated_text: str,
        reference_text: str,
        sources: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, MetricResult]:
        """Calculate generation quality metrics."""
        results = {}

        # ROUGE scores
        rouge_metrics = await self.calculate_rouge_scores(generated_text, reference_text)
        results.update(rouge_metrics)

        # BERTScore
        bert_metrics = await self.calculate_bert_scores(generated_text, reference_text)
        results.update(bert_metrics)

        # Factual accuracy (if sources available)
        if sources:
            factuality = await self.calculate_factual_accuracy(
                generated_text, reference_text, sources
            )
            results["factual_accuracy"] = factuality

        # Length and complexity metrics
        length_metrics = self.calculate_text_metrics(generated_text, reference_text)
        results.update(length_metrics)

        return results

    async def calculate_rouge_scores(
        self, generated: str, reference: str
    ) -> Dict[str, MetricResult]:
        """Calculate ROUGE scores."""
        if not self.rouge_available:
            # Simple overlap fallback
            overlap = self.calculate_simple_overlap(generated, reference)
            return {
                "rouge_1": MetricResult("ROUGE-1", overlap, {"method": "simple_overlap"}),
                "rouge_2": MetricResult("ROUGE-2", overlap, {"method": "simple_overlap"}),
                "rouge_l": MetricResult("ROUGE-L", overlap, {"method": "simple_overlap"}),
            }

        try:
            from rouge_score import rouge_scorer

            scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
            scores = scorer.score(reference, generated)

            results = {}
            for metric in ["rouge1", "rouge2", "rougeL"]:
                if metric in scores:
                    results[metric] = MetricResult(
                        name=metric.upper(),
                        value=scores[metric].fmeasure,
                        details={
                            "precision": scores[metric].precision,
                            "recall": scores[metric].recall,
                            "fmeasure": scores[metric].fmeasure,
                        },
                    )

            return results

        except Exception as e:
            logger.warning(f"ROUGE calculation failed: {e}")
            overlap = self.calculate_simple_overlap(generated, reference)
            return {
                "rouge_1": MetricResult(
                    "ROUGE-1", overlap, {"method": "simple_overlap", "error": str(e)}
                ),
                "rouge_2": MetricResult(
                    "ROUGE-2", overlap, {"method": "simple_overlap", "error": str(e)}
                ),
                "rouge_l": MetricResult(
                    "ROUGE-L", overlap, {"method": "simple_overlap", "error": str(e)}
                ),
            }

    async def calculate_bert_scores(
        self, generated: str, reference: str
    ) -> Dict[str, MetricResult]:
        """Calculate BERTScore."""
        if not self.bert_score_available:
            # Simple similarity fallback
            similarity = self.calculate_simple_overlap(generated, reference)
            return {
                "bert_score_f1": MetricResult(
                    "BERTScore-F1", similarity, {"method": "simple_overlap"}
                ),
                "bert_score_precision": MetricResult(
                    "BERTScore-Precision", similarity, {"method": "simple_overlap"}
                ),
                "bert_score_recall": MetricResult(
                    "BERTScore-Recall", similarity, {"method": "simple_overlap"}
                ),
            }

        try:
            from bert_score import score as bert_score

            P, R, F1 = bert_score([generated], [reference], lang="en", rescale_with_baseline=True)

            return {
                "bert_score_f1": MetricResult("BERTScore-F1", float(F1.mean()), {"model": "bert"}),
                "bert_score_precision": MetricResult(
                    "BERTScore-Precision", float(P.mean()), {"model": "bert"}
                ),
                "bert_score_recall": MetricResult(
                    "BERTScore-Recall", float(R.mean()), {"model": "bert"}
                ),
            }

        except Exception as e:
            logger.warning(f"BERTScore calculation failed: {e}")
            similarity = self.calculate_simple_overlap(generated, reference)
            return {
                "bert_score_f1": MetricResult(
                    "BERTScore-F1", similarity, {"method": "simple_overlap", "error": str(e)}
                ),
                "bert_score_precision": MetricResult(
                    "BERTScore-Precision", similarity, {"method": "simple_overlap", "error": str(e)}
                ),
                "bert_score_recall": MetricResult(
                    "BERTScore-Recall", similarity, {"method": "simple_overlap", "error": str(e)}
                ),
            }

    async def calculate_factual_accuracy(
        self, generated: str, reference: str, sources: List[Dict[str, Any]]
    ) -> MetricResult:
        """Calculate factual accuracy based on source support."""
        try:
            # Extract claims from generated text (simplified)
            generated_claims = self._extract_claims(generated)

            # Extract facts from sources
            source_facts = []
            for source in sources[:5]:  # Top 5 sources
                content = source.get("content", "")
                facts = self._extract_facts_from_text(content)
                source_facts.extend(facts)

            # Check how many claims are supported
            supported_claims = 0
            for claim in generated_claims:
                if self._is_claim_supported(claim, source_facts):
                    supported_claims += 1

            accuracy = supported_claims / len(generated_claims) if generated_claims else 1.0

            return MetricResult(
                name="Factual Accuracy",
                value=accuracy,
                details={
                    "total_claims": len(generated_claims),
                    "supported_claims": supported_claims,
                    "source_facts": len(source_facts),
                    "sources_used": len(sources),
                },
            )

        except Exception as e:
            logger.warning(f"Factual accuracy calculation failed: {e}")
            return MetricResult("Factual Accuracy", 0.5, {"error": str(e)})

    def calculate_simple_overlap(self, text1: str, text2: str) -> float:
        """Calculate simple word overlap."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)

    def calculate_text_metrics(self, generated: str, reference: str) -> Dict[str, MetricResult]:
        """Calculate text-level metrics."""
        gen_words = generated.split()
        ref_words = reference.split()

        # Length ratio
        length_ratio = len(gen_words) / len(ref_words) if ref_words else 1.0

        # Sentence count
        gen_sentences = generated.count(".") + generated.count("!") + generated.count("?")
        ref_sentences = reference.count(".") + reference.count("!") + reference.count("?")

        # Readability (simplified)
        avg_word_length = sum(len(word) for word in gen_words) / len(gen_words) if gen_words else 0

        return {
            "length_ratio": MetricResult(
                "Length Ratio", length_ratio, {"gen_len": len(gen_words), "ref_len": len(ref_words)}
            ),
            "sentence_count": MetricResult(
                "Sentence Count",
                gen_sentences,
                {"gen_sentences": gen_sentences, "ref_sentences": ref_sentences},
            ),
            "avg_word_length": MetricResult(
                "Avg Word Length", avg_word_length, {"words": gen_words}
            ),
        }

    def _extract_claims(self, text: str) -> List[str]:
        """Extract claims from text (simplified)."""
        # Split into sentences and filter out very short ones
        sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 10]
        return sentences

    def _extract_facts_from_text(self, text: str) -> List[str]:
        """Extract facts from text (simplified)."""
        # Simple extraction - take sentences as facts
        sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 10]
        return sentences

    def _is_claim_supported(self, claim: str, facts: List[str]) -> bool:
        """Check if a claim is supported by facts."""
        # Simple keyword-based support check
        claim_words = set(claim.lower().split())

        for fact in facts:
            fact_words = set(fact.lower().split())
            # If claim shares significant words with fact, consider it supported
            overlap = len(claim_words & fact_words)
            if overlap >= 3:  # At least 3 common words
                return True

        return False

    async def calculate_latency_metrics(
        self, retrieval_times: List[float], generation_times: List[float], total_times: List[float]
    ) -> Dict[str, MetricResult]:
        """Calculate latency and performance metrics."""
        results = {}

        # Retrieval metrics
        if retrieval_times:
            results["retrieval_latency_mean"] = MetricResult(
                "Retrieval Latency Mean",
                np.mean(retrieval_times),
                {"unit": "ms", "samples": len(retrieval_times)},
            )
            results["retrieval_latency_p95"] = MetricResult(
                "Retrieval Latency P95", np.percentile(retrieval_times, 95), {"unit": "ms"}
            )
            results["retrieval_latency_p99"] = MetricResult(
                "Retrieval Latency P99", np.percentile(retrieval_times, 99), {"unit": "ms"}
            )

        # Generation metrics
        if generation_times:
            results["generation_latency_mean"] = MetricResult(
                "Generation Latency Mean",
                np.mean(generation_times),
                {"unit": "ms", "samples": len(generation_times)},
            )
            results["generation_latency_p95"] = MetricResult(
                "Generation Latency P95", np.percentile(generation_times, 95), {"unit": "ms"}
            )

        # Total metrics
        if total_times:
            results["total_latency_mean"] = MetricResult(
                "Total Latency Mean",
                np.mean(total_times),
                {"unit": "ms", "samples": len(total_times)},
            )
            results["total_latency_p95"] = MetricResult(
                "Total Latency P95", np.percentile(total_times, 95), {"unit": "ms"}
            )

            # Throughput (queries per second)
            avg_time = np.mean(total_times) / 1000  # Convert to seconds
            results["throughput"] = MetricResult(
                "Throughput", 1.0 / avg_time if avg_time > 0 else 0.0, {"unit": "queries/second"}
            )

        return results

    def calculate_confidence_metrics(
        self, confidence_scores: List[float]
    ) -> Dict[str, MetricResult]:
        """Calculate confidence-related metrics."""
        if not confidence_scores:
            return {}

        scores = np.array(confidence_scores)

        return {
            "confidence_mean": MetricResult(
                "Confidence Mean", float(np.mean(scores)), {"samples": len(scores)}
            ),
            "confidence_std": MetricResult(
                "Confidence Std Dev", float(np.std(scores)), {"samples": len(scores)}
            ),
            "confidence_min": MetricResult("Confidence Min", float(np.min(scores)), {}),
            "confidence_max": MetricResult("Confidence Max", float(np.max(scores)), {}),
        }

    def calculate_source_quality_metrics(
        self, sources: List[Dict[str, Any]]
    ) -> Dict[str, MetricResult]:
        """Calculate source quality metrics."""
        if not sources:
            return {
                "source_count": MetricResult("Source Count", 0, {}),
                "avg_source_score": MetricResult("Avg Source Score", 0.0, {}),
            }

        scores = [source.get("score", 0.0) for source in sources]
        unique_sources = set(source.get("document_id", "") for source in sources)

        return {
            "source_count": MetricResult(
                "Source Count", len(sources), {"unique_sources": len(unique_sources)}
            ),
            "avg_source_score": MetricResult(
                "Avg Source Score", np.mean(scores), {"min": min(scores), "max": max(scores)}
            ),
            "source_diversity": MetricResult(
                "Source Diversity",
                len(unique_sources) / len(sources),
                {"total_sources": len(sources), "unique_sources": len(unique_sources)},
            ),
        }


class MetricCalculator:
    """High-level interface for metrics calculation."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.metrics = RAGMetrics(config)

    async def calculate_comprehensive_metrics(
        self,
        query_results: List[Dict[str, Any]],
        ground_truths: Optional[List[str]] = None,
        relevant_docs_list: Optional[List[List[str]]] = None,
    ) -> Dict[str, Any]:
        """Calculate comprehensive metrics for multiple queries."""
        all_metrics = {}

        # Batch processing
        retrieval_metrics = []
        generation_metrics = []
        latency_metrics = []
        confidence_metrics = []
        source_quality_metrics = []

        for i, result in enumerate(query_results):
            # Retrieval metrics
            relevant_docs = relevant_docs_list[i] if relevant_docs_list else []
            retrieval_metric = await self.metrics.calculate_retrieval_metrics(
                result.get("retrieved_chunks", []), relevant_docs, result.get("top_k")
            )
            retrieval_metrics.append(retrieval_metric)

            # Generation metrics
            ground_truth = ground_truths[i] if ground_truths else None
            generation_metric = await self.metrics.calculate_generation_metrics(
                result.get("answer", ""), ground_truth or "", result.get("sources", [])
            )
            generation_metrics.append(generation_metric)

            # Latency metrics
            latencies = self.metrics.calculate_latency_metrics(
                [result.get("retrieval_time_ms", 0)],
                [result.get("generation_time_ms", 0)],
                [result.get("total_time_ms", 0)],
            )
            latency_metrics.append(latencies)

            # Confidence metrics
            confidence_scores = result.get("confidence_scores", [result.get("confidence", 0)])
            confidence_result = self.metrics.calculate_confidence_metrics(confidence_scores)
            confidence_metrics.append(confidence_result)

            # Source quality metrics
            source_quality = self.metrics.calculate_source_quality_metrics(
                result.get("sources", [])
            )
            source_quality_metrics.append(source_quality)

        # Aggregate all metrics
        all_metrics["retrieval"] = self._aggregate_metric_dicts(retrieval_metrics)
        all_metrics["generation"] = self._aggregate_metric_dicts(generation_metrics)
        all_metrics["latency"] = self._aggregate_metric_dicts(latency_metrics)
        all_metrics["confidence"] = self._aggregate_metric_dicts(confidence_metrics)
        all_metrics["source_quality"] = self._aggregate_metric_dicts(source_quality_metrics)

        return all_metrics

    def _aggregate_metric_dicts(
        self, metric_dicts: List[Dict[str, MetricResult]]
    ) -> Dict[str, Dict[str, float]]:
        """Aggregate multiple metric dictionaries."""
        aggregated = {}

        # Get all unique metric names
        all_metric_names = set()
        for metric_dict in metric_dicts:
            all_metric_names.update(metric_dict.keys())

        # Calculate statistics for each metric
        for metric_name in all_metric_names:
            values = []
            for metric_dict in metric_dicts:
                if metric_name in metric_dict:
                    values.append(metric_dict[metric_name].value)

            if values:
                aggregated[metric_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "count": len(values),
                    "median": float(np.median(values)),
                }

        return aggregated
