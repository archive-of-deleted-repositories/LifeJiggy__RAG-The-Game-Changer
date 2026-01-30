"""
Quality Assessment - RAG-The-Game-Changer

Quality scoring and assessment for RAG responses.
"""

import logging
import re
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class QualityScore:
    """Individual quality dimension score."""

    dimension: str
    score: float
    details: Dict[str, Any]
    weight: float = 1.0


@dataclass
class AssessmentConfig:
    """Configuration for quality assessment."""

    enable_relevance: bool = True
    enable_coherence: bool = True
    enable_completeness: bool = True
    enable_fluency: bool = True
    enable_correctness: bool = True
    dimensions_weights: Dict[str, float] = field(default_factory=dict)


class QualityAssessor:
    """Assess quality of RAG generated responses."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.assessment_config = AssessmentConfig(**self.config)
        self._set_default_weights()

    def _set_default_weights(self):
        """Set default dimension weights."""
        weights = {
            "relevance": 0.4,
            "coherence": 0.25,
            "completeness": 0.15,
            "fluency": 0.1,
            "correctness": 0.1,
        }
        self.assessment_config.dimensions_weights.update(weights)

    async def assess_quality(
        self,
        query: str,
        answer: str,
        retrieved_contexts: List[str],
        expected_answer: Optional[str] = None,
    ) -> List[QualityScore]:
        """Assess overall quality of RAG response."""
        scores = []

        # Assess each dimension
        if self.assessment_config.enable_relevance:
            relevance_score = await self._assess_relevance(query, answer, retrieved_contexts)
            scores.append(relevance_score)

        if self.assessment_config.enable_coherence:
            coherence_score = await self._assess_coherence(answer)
            scores.append(coherence_score)

        if self.assessment_config.enable_completeness:
            completeness_score = await self._assess_completeness(query, answer)
            scores.append(completeness_score)

        if self.assessment_config.enable_fluency:
            fluency_score = await self._assess_fluency(answer)
            scores.append(fluency_score)

        if self.assessment_config.enable_correctness and expected_answer:
            correctness_score = await self._assess_correctness(answer, expected_answer)
            scores.append(correctness_score)

        logger.info(f"Quality assessment complete. Dimensions: {len(scores)}")
        return scores

    async def _assess_relevance(self, query: str, answer: str, contexts: List[str]) -> QualityScore:
        """Assess relevance of answer to query."""
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        context_words = set(" ".join(contexts).lower().split())

        # Query coverage
        query_coverage = len(answer_words & query_words) / len(query_words) if query_words else 0

        # Context support
        context_support = (
            len(answer_words & context_words) / len(answer_words) if answer_words else 0
        )

        # Calculate score
        score = (query_coverage + context_support) / 2

        details = {
            "query_words": len(query_words),
            "answer_words": len(answer_words),
            "overlap_query": len(answer_words & query_words),
            "overlap_context": len(answer_words & context_words),
        }

        weight = self.assessment_config.dimensions_weights.get("relevance", 0.4)

        return QualityScore(
            dimension="relevance", score=score * weight, details=details, weight=weight
        )

    async def _assess_coherence(self, answer: str) -> QualityScore:
        """Assess coherence of generated answer."""
        sentences = [s.strip() for s in answer.split(".") if s.strip()]

        if len(sentences) <= 1:
            return QualityScore(
                dimension="coherence",
                score=1.0 * self.assessment_config.dimensions_weights.get("coherence", 0.25),
                details={"sentence_count": len(sentences)},
                weight=self.assessment_config.dimensions_weights.get("coherence", 0.25),
            )

        # Check for logical flow
        coherence_score = 1.0
        coherence_issues = []

        for i in range(len(sentences) - 1):
            s1 = sentences[i]
            s2 = sentences[i + 1]

            # Check pronoun reference
            if self._has_pronoun_reference(s1, s2):
                coherence_score -= 0.1
                coherence_issues.append("pronoun_mismatch")

            # Check for logical connectors
            if not self._has_logical_connector(s1, s2):
                coherence_score -= 0.1
                coherence_issues.append("poor_flow")

        details = {"sentence_count": len(sentences), "coherence_issues": coherence_issues}

        weight = self.assessment_config.dimensions_weights.get("coherence", 0.25)

        return QualityScore(
            dimension="coherence",
            score=max(0.0, coherence_score * weight),
            details=details,
            weight=weight,
        )

    def _has_pronoun_reference(self, s1: str, s2: str) -> bool:
        """Check if second sentence properly references first."""
        s1_pronouns = self._extract_pronouns(s1)
        s2_pronouns = self._extract_pronouns(s2)

        return len(set(s1_pronouns) & set(s2_pronouns)) > 0

    def _has_logical_connector(self, s1: str, s2: str) -> bool:
        """Check if sentences have logical connectors."""
        connectors = ["therefore", "however", "thus", "consequently", "moreover", "furthermore"]
        return any(connector in s1.lower() or connector in s2.lower())

    def _extract_pronouns(self, text: str) -> List[str]:
        """Extract pronouns from text."""
        pronouns = ["he", "she", "it", "they", "this", "that", "these", "those"]
        words = text.lower().split()
        return [w for w in words if w in pronouns]

    async def _assess_completeness(self, query: str, answer: str) -> QualityScore:
        """Assess completeness of answer relative to query."""
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())

        # Calculate coverage
        if len(query_words) == 0:
            return QualityScore(
                dimension="completeness",
                score=1.0 * self.assessment_config.dimensions_weights.get("completeness", 0.15),
                details={"coverage": "N/A"},
                weight=self.assessment_config.dimensions_weights.get("completeness", 0.15),
            )

        coverage = len(answer_words & query_words) / len(query_words)

        # Check for direct answers
        question_words = ["who", "what", "where", "when", "why", "how"]
        has_answer = any(word in query.lower() for word in question_words)

        # Bonus for having direct answer
        if has_answer:
            coverage += 0.1

        details = {
            "query_coverage": coverage,
            "has_answer": has_answer,
            "missing_aspects": list(query_words - answer_words),
        }

        weight = self.assessment_config.dimensions_weights.get("completeness", 0.15)

        return QualityScore(
            dimension="completeness",
            score=min(1.0, coverage * weight),
            details=details,
            weight=weight,
        )

    async def _assess_fluency(self, answer: str) -> QualityScore:
        """Assess fluency of generated answer."""
        # Readability metrics
        avg_sentence_length = len(answer.split(".")) if answer else 0
        avg_word_length = len(answer.split()) / len(answer.split(".")) if answer else 0

        # Fluency indicators
        short_sentences = sum(1 for s in answer.split(".") if len(s.split()) < 5)
        long_sentences = sum(1 for s in answer.split(".") if len(s.split()) > 20)

        # Check for awkward phrasing
        awkward_indicators = [
            r"it is (?:a | the case that?)",
            r"there (?:is | are) (?:many|several)",
            r"this (?:is | are) (?:a lot of)",
            r"very (?:much | many)",
            r"rather(?: | than)",
        ]
        awkward_count = sum(
            1 for pattern in awkward_indicators if re.search(pattern, answer, re.IGNORECASE)
        )

        # Calculate score
        ideal_sentence_length = 15
        ideal_word_length = 10

        length_score = (
            1.0 - abs(avg_sentence_length - ideal_sentence_length) / ideal_sentence_length
        )
        structure_score = (
            1.0 - (awkward_count / len(awkward_indicators)) if awkward_indicators else 0
        )

        score = (length_score + structure_score) / 2

        details = {
            "avg_sentence_length": avg_sentence_length,
            "avg_word_length": avg_word_length,
            "short_sentences": short_sentences,
            "long_sentences": long_sentences,
            "awkward_phrases": awkward_count,
        }

        weight = self.assessment_config.dimensions_weights.get("fluency", 0.1)

        return QualityScore(
            dimension="fluency", score=score * weight, details=details, weight=weight
        )

    async def _assess_correctness(self, answer: str, expected_answer: str) -> QualityScore:
        """Assess factual correctness of answer."""
        answer_lower = answer.lower().strip()
        expected_lower = expected_answer.lower().strip()

        # Exact match
        if answer_lower == expected_lower:
            return QualityScore(
                dimension="correctness",
                score=1.0 * self.assessment_config.dimensions_weights.get("correctness", 0.1),
                details={"match_type": "exact"},
                weight=self.assessment_config.dimensions_weights.get("correctness", 0.1),
            )

        # Semantic similarity (simple)
        answer_words = set(answer_lower.split())
        expected_words = set(expected_lower.split())

        overlap = len(answer_words & expected_words) / len(expected_words) if expected_words else 0

        # Check for contradictions
        has_contradiction = self._check_contradictions(answer_lower, expected_lower)

        if has_contradiction:
            score = 0.0
        else:
            score = overlap

        details = {"overlap": overlap, "contradiction": has_contradiction, "match_type": "none"}

        weight = self.assessment_config.dimensions_weights.get("correctness", 0.1)

        return QualityScore(
            dimension="correctness", score=score * weight, details=details, weight=weight
        )

    def _check_contradictions(self, answer: str, expected: str) -> bool:
        """Check for explicit contradictions."""
        negative_words = ["not", "never", "no", "nobody", "nothing", "none", "without"]

        for neg_word in negative_words:
            if neg_word in answer.lower():
                # Check if expected has positive version
                positive_words = ["yes", "always", "always", "indeed", "true"]
                for pos_word in positive_words:
                    if pos_word in expected.lower():
                        # Not a contradiction
                        return False

        return False

    def calculate_overall_score(self, quality_scores: List[QualityScore]) -> float:
        """Calculate weighted overall quality score."""
        if not quality_scores:
            return 0.0

        total_score = sum(qs.score for qs in quality_scores)

        # Normalize to 0-1 range
        max_possible = sum(qs.weight for qs in quality_scores)

        if max_possible > 0:
            normalized_score = total_score / max_possible
        else:
            normalized_score = 0.0

        return normalized_score

    def get_dimension_scores(self, quality_scores: List[QualityScore]) -> Dict[str, float]:
        """Extract individual dimension scores."""
        return {qs.dimension: qs.score for qs in quality_scores}

    def generate_report(self, quality_scores: List[QualityScore]) -> str:
        """Generate quality assessment report."""
        lines = [
            "=" * 80,
            "RAG QUALITY ASSESSMENT REPORT",
            "=" * 80,
            "",
            f"Overall Quality Score: {self.calculate_overall_score(quality_scores):.4f}",
            "",
            "-" * 80,
            "DIMENSION SCORES",
            "-" * 80,
        ]

        for qs in quality_scores:
            lines.append(f"  {qs.dimension.upper()}: {qs.score:.4f} (weight: {qs.weight:.2f})")
            if qs.details:
                for key, value in qs.details.items():
                    lines.append(f"    {key}: {value}")

        lines.extend(
            [
                "",
                "=" * 80,
                "END OF REPORT",
                "=" * 80,
            ]
        )

        return "\n".join(lines)
