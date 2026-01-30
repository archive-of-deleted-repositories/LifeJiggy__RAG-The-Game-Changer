"""
Hallucination Control - RAG-The-Game-Changer

Advanced hallucination mitigation and control strategies.
"""

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class HallucinationConfig:
    """Configuration for hallucination control."""

    enable_fact_checking: bool = True
    confidence_threshold: float = 0.7
    source_verification: bool = True
    self_consistency_check: bool = True
    max_hallucination_rate: float = 0.15
    enable_reranking: bool = True


@dataclass
class ControlledResponse:
    """Response with hallucination controls applied."""

    original_answer: str
    controlled_answer: str
    confidence_score: float
    hallucination_detected: bool
    warnings: List[str]
    sources_verified: bool


class HallucinationController:
    """Controller for mitigating and detecting hallucinations."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.control_config = HallucinationConfig(**self.config)

    async def control_hallucination(
        self,
        query: str,
        generated_answer: str,
        retrieved_contexts: List[str],
        original_confidence: float,
        original_sources: List[Dict],
    ) -> ControlledResponse:
        """Apply hallucination control to generated response."""
        warnings = []
        hallucination_detected = False
        controlled_answer = generated_answer

        # 1. Confidence threshold check
        if original_confidence < self.control_config.confidence_threshold:
            warnings.append(f"Low confidence score: {original_confidence:.2f}")
            hallucination_detected = True

        # 2. Self-consistency check
        if self.control_config.self_consistency_check:
            consistency_issue = self._check_self_consistency(generated_answer)
            if consistency_issue:
                warnings.append(consistency_issue)
                controlled_answer = self._fix_consistency_issue(generated_answer, consistency_issue)
                hallucination_detected = True

        # 3. Source verification
        sources_verified = False
        if self.control_config.source_verification:
            if self._verify_sources(generated_answer, retrieved_contexts):
                sources_verified = True
            else:
                warnings.append("Answer not supported by retrieved sources")
                hallucination_detected = True

        # 4. Fact checking
        if self.control_config.enable_fact_checking:
            fact_issues = await self._fact_check(query, generated_answer)
            if fact_issues:
                warnings.extend(fact_issues)
                hallucination_detected = True
                controlled_answer = self._add_uncertainty_to_answer(generated_answer)

        # 5. Check for hallucination patterns
        patterns = self._detect_hallucination_patterns(generated_answer)
        if patterns:
            warnings.extend(patterns)
            controlled_answer = self._add_uncertainty_to_answer(generated_answer)

        return ControlledResponse(
            original_answer=generated_answer,
            controlled_answer=controlled_answer,
            confidence_score=original_confidence,
            hallucination_detected=len(warnings) > 0,
            warnings=warnings,
            sources_verified=sources_verified,
        )

    def _check_self_consistency(self, text: str) -> Optional[str]:
        """Check answer for self-contradictions."""
        sentences = [s.strip() for s in text.split(".") if s.strip()]

        if len(sentences) < 2:
            return None

        # Check for numerical inconsistencies
        numbers = re.findall(r"\d+\.?\d*", text)
        if len(numbers) > 1:
            # Check if numbers are in similar range
            num_values = [float(n) for n in numbers]
            if max(num_values) / min(num_values) > 10:  # More than 10x difference
                return f"Numerical inconsistency detected"

        # Check for contradictory statements
        contradiction_patterns = [
            (r"(not|no)\w+(?:is|are|was|were)", r"(is|are|was|were)\w+(?:not|no)"),
            (
                r"(always|never|never)\w*(?:always|never|never)",
                r"(rarely|often|usually)\w*(?:rarely|often|usually)",
            ),
        ]

        for pattern in contradiction_patterns:
            if re.search(pattern[0], text, re.IGNORECASE) and re.search(
                pattern[1], text, re.IGNORECASE
            ):
                return "Potential contradiction detected"

        return None

    def _fix_consistency_issue(self, text: str, issue: str) -> str:
        """Attempt to fix consistency issue in answer."""
        # Add qualifiers to uncertain statements
        if "numerical" in issue.lower():
            text = re.sub(r"(\d+\.?\d*)", r"approximately \1", text)
        elif "contradiction" in issue.lower():
            text = text.replace(".", ", depending on the context.")

        return text

    def _verify_sources(self, answer: str, contexts: List[str]) -> bool:
        """Verify that answer is supported by retrieved contexts."""
        answer_words = set(answer.lower().split())
        context_words = set(" ".join(contexts).lower().split())

        # If answer contains significant terms not in context
        answer_specific = answer_words - context_words

        # Allow some novel words, but too many indicates hallucination
        if len(answer_specific) > 5:
            return False

        # Check if answer directly contradicts context
        for word in answer_words:
            if f"not {word}" in " ".join(contexts).lower():
                return False

        return True

    async def _fact_check(self, query: str, answer: str) -> List[str]:
        """Check facts in answer against known knowledge."""
        issues = []

        # Check for absolute claims without sources
        if re.search(
            r"\b(absolutely|certainly|definitely|undoubtedly|unquestionably)\b",
            answer,
            re.IGNORECASE,
        ):
            issues.append("Absolute claim detected without source")

        # Check for speculative language
        speculative_words = ["maybe", "possibly", "perhaps", "probably", "might be", "could be"]
        if any(word in answer.lower() for word in speculative_words):
            # This is good - indicates uncertainty
            pass
        else:
            # No uncertainty markers might indicate overconfidence
            if not re.search(
                r"\b(according to|based on|sources indicate)\b", answer, re.IGNORECASE
            ):
                issues.append("No source attribution for factual claims")

        # Check for dates/numbers without qualifiers
        if re.search(r"\b(in \d{4}|on \w+ \d{1,2}|at \d{1,2}:\d{2})\b", answer, re.IGNORECASE):
            if not re.search(
                r"\b(approximately|around|about|roughly|nearly)\b", answer, re.IGNORECASE
            ):
                issues.append("Specific date/time without approximate qualifier")

        return issues

    def _detect_hallucination_patterns(self, text: str) -> List[str]:
        """Detect common hallucination patterns."""
        patterns = []

        # 1. Generic filler responses
        if re.search(r"\b(i am (?:an? )?(?:ai|language model|assistant))\b", text, re.IGNORECASE):
            patterns.append("Generic AI self-identification")

        # 2. Refusal to answer with no context
        if re.search(r"\b(i (?:do not|cannot|am not sure))\b", text, re.IGNORECASE):
            patterns.append("Refusal without clear reason")

        # 3. Overly vague statements
        vague_patterns = [
            r"it depends\b",
            r"there are (?:many|several) (?:ways|factors)\b",
            r"it is (?:complex|difficult) to (?:say|determine)\b",
        ]
        for pattern in vague_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                patterns.append("Overly vague statement")

        return patterns

    def _add_uncertainty_to_answer(self, answer: str) -> str:
        """Add uncertainty qualifiers to answer."""
        # Don't modify if already has uncertainty
        uncertainty_indicators = [
            "possibly",
            "maybe",
            "probably",
            "might",
            "could",
            "approximately",
        ]
        if any(ind in answer.lower() for ind in uncertainty_indicators):
            return answer

        # Add uncertainty to first sentence
        sentences = answer.split(".")
        if sentences:
            first_sentence = sentences[0].strip()
            modified_first = f"Based on the available information, {first_sentence.lower()}"
            sentences[0] = modified_first
            return ". ".join(sentences)

        return answer

    async def apply_reranking(
        self, query: str, candidate_answers: List[str], retrieved_contexts: List[str]
    ) -> tuple[str, List[float]]:
        """Rerank answers based on reliability."""
        scores = []

        for i, answer in enumerate(candidate_answers):
            score = 0.5  # Base score

            # 1. Length penalty (too long = bad)
            if len(answer) > 500:
                score -= 0.1
            elif len(answer) < 50:
                score -= 0.1
            else:
                score += 0.1

            # 2. Source support bonus
            if self._verify_sources(answer, retrieved_contexts):
                score += 0.3

            # 3. Uncertainty bonus (good)
            if any(word in answer.lower() for word in ["possibly", "maybe", "might"]):
                score += 0.1

            # 4. Hallucination penalty
            patterns = self._detect_hallucination_patterns(answer)
            if patterns:
                score -= 0.3

            scores.append(score)

        # Select best answer
        best_idx = scores.index(max(scores))
        best_answer = candidate_answers[best_idx]

        return best_answer, scores


class RAGPipelineWrapper:
    """Wrapper for RAG pipeline with hallucination control."""

    def __init__(self, rag_pipeline, controller: HallucinationController):
        self.rag_pipeline = rag_pipeline
        self.controller = controller
        self.total_queries = 0
        self.hallucination_count = 0

    async def query_with_control(
        self, query: str, top_k: int = 5, include_sources: bool = True
    ) -> ControlledResponse:
        """Query with hallucination control applied."""
        self.total_queries += 1

        # Get original response
        result = await self.rag_pipeline.query(
            query=query, top_k=top_k, include_sources=include_sources
        )

        # Apply control
        controlled = await self.controller.control_hallucination(
            query=query,
            generated_answer=result.answer,
            retrieved_contexts=[c.get("content") for c in result.retrieved_chunks],
            original_confidence=result.confidence,
            original_sources=result.sources,
        )

        if controlled.hallucination_detected:
            self.hallucination_count += 1

        return controlled

    def get_hallucination_rate(self) -> float:
        """Get current hallucination rate."""
        if self.total_queries == 0:
            return 0.0
        return self.hallucination_count / self.total_queries

    def get_stats(self) -> Dict[str, Any]:
        """Get wrapper statistics."""
        return {
            "total_queries": self.total_queries,
            "hallucination_count": self.hallucination_count,
            "hallucination_rate": self.get_hallucination_rate(),
            "control_enabled": True,
        }
