"""
Self-Reflection RAG - Advanced RAG Pattern

RAG system with self-reflection and correction capabilities.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import time

from ..config.pipeline_configs.rag_pipeline import RAGPipeline, RAGResponse

logger = logging.getLogger(__name__)


@dataclass
class ReflectionResult:
    """Result from self-reflection process."""

    needs_correction: bool
    confidence_improvement: float
    corrected_answer: Optional[str] = None
    reasoning: Optional[str] = None
    issues_found: List[str] = field(default_factory=list)


@dataclass
class ReflectionRound:
    """Single round of reflection."""

    round_number: int
    original_query: str
    original_answer: str
    original_sources: List[Dict[str, Any]]
    reflection_result: ReflectionResult
    timestamp: float = field(default_factory=time.time)


class SelfReflectionRAG:
    """RAG system with self-reflection and correction capabilities."""

    def __init__(self, base_pipeline: RAGPipeline, config: Optional[Dict[str, Any]] = None):
        self.pipeline = base_pipeline
        self.config = config or {}

        # Reflection settings
        self.max_reflection_rounds = self.config.get("max_reflection_rounds", 2)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
        self.enable_fact_checking = self.config.get("enable_fact_checking", True)
        self.enable_coherence_checking = self.config.get("enable_coherence_checking", True)
        self.enable_source_verification = self.config.get("enable_source_verification", True)

        # LLM settings for reflection
        self.reflection_model = self.config.get("reflection_model", "gpt-4")
        self.correction_model = self.config.get("correction_model", "gpt-4")

    async def query_with_reflection(
        self, query: str, max_rounds: Optional[int] = None
    ) -> Dict[str, Any]:
        """Execute query with self-reflection and correction."""
        start_time = time.time()

        # Initial query
        reflection_rounds = []
        current_query = query
        current_answer = None
        current_sources = None
        total_confidence_improvement = 0.0

        max_rounds = min(max_rounds or self.max_reflection_rounds, self.max_reflection_rounds)

        for round_num in range(max_rounds):
            logger.info(f"Reflection round {round_num + 1}/{max_rounds}")

            # Execute query
            response = await self.pipeline.query(
                query=current_query, top_k=5, include_sources=True, include_confidence=True
            )

            current_answer = response.answer
            current_sources = response.sources
            current_confidence = response.confidence

            # Perform self-reflection
            if round_num < max_rounds - 1:  # Don't reflect on final round
                reflection_result = await self._reflect_on_answer(
                    query, current_answer, current_sources, reflection_rounds
                )

                # Decide if correction is needed
                if reflection_result.needs_correction and reflection_result.corrected_answer:
                    current_query = reflection_result.corrected_answer
                    total_confidence_improvement += reflection_result.confidence_improvement

                    # Create reflection round record
                    reflection_round = ReflectionRound(
                        round_number=round_num + 1,
                        original_query=query,
                        original_answer=current_answer,
                        original_sources=current_sources,
                        reflection_result=reflection_result,
                    )
                    reflection_rounds.append(reflection_round)
                else:
                    # No correction needed, this is our final answer
                    reflection_round = ReflectionRound(
                        round_number=round_num + 1,
                        original_query=query,
                        original_answer=current_answer,
                        original_sources=current_sources,
                        reflection_result=reflection_result,
                    )
                    reflection_rounds.append(reflection_round)
                    break
            else:
                # Final round
                reflection_round = ReflectionRound(
                    round_number=round_num + 1,
                    original_query=query,
                    original_answer=current_answer,
                    original_sources=current_sources,
                    reflection_result=ReflectionResult(needs_correction=False),
                )
                reflection_rounds.append(reflection_round)

        total_time = (time.time() - start_time) * 1000

        return {
            "original_query": query,
            "final_answer": current_answer,
            "final_sources": current_sources,
            "final_confidence": current_confidence,
            "reflection_rounds": reflection_rounds,
            "total_rounds": len(reflection_rounds),
            "total_confidence_improvement": total_confidence_improvement,
            "total_time_ms": total_time,
            "self_corrected": total_confidence_improvement > 0,
            "metadata": {
                "max_reflection_rounds": max_rounds,
                "reflection_threshold": self.confidence_threshold,
            },
        }

    async def _reflect_on_answer(
        self,
        query: str,
        answer: str,
        sources: List[Dict[str, Any]],
        previous_rounds: List[ReflectionRound],
    ) -> ReflectionResult:
        """Perform self-reflection on the answer."""
        try:
            # Analyze different aspects of the answer
            issues_found = []
            needs_correction = False
            corrected_answer = None

            # 1. Confidence analysis
            confidence_issue = await self._analyze_confidence(answer, sources)
            if confidence_issue:
                issues_found.extend(confidence_issue)

            # 2. Fact checking
            if self.enable_fact_checking:
                fact_issues = await self._check_factual_accuracy(answer, sources)
                issues_found.extend(fact_issues)

            # 3. Coherence analysis
            if self.enable_coherence_checking:
                coherence_issues = await self._check_coherence(query, answer)
                issues_found.extend(coherence_issues)

            # 4. Source verification
            if self.enable_source_verification:
                source_issues = await self._verify_sources(answer, sources)
                issues_found.extend(source_issues)

            # Determine if correction is needed
            if issues_found and self.confidence_threshold > 0.0:
                avg_confidence = await self._estimate_confidence(answer, sources)
                if avg_confidence < self.confidence_threshold:
                    needs_correction = True
                    corrected_answer = await self._generate_correction(query, answer, issues_found)

            reasoning = self._generate_reflection_reasoning(issues_found, needs_correction)

            confidence_improvement = 0.0
            if corrected_answer:
                confidence_improvement = await self._estimate_confidence_improvement(
                    answer, corrected_answer
                )

            return ReflectionResult(
                needs_correction=needs_correction,
                confidence_improvement=confidence_improvement,
                corrected_answer=corrected_answer,
                reasoning=reasoning,
                issues_found=issues_found,
            )

        except Exception as e:
            logger.error(f"Error in self-reflection: {e}")
            return ReflectionResult(
                needs_correction=False,
                confidence_improvement=0.0,
                reasoning=f"Reflection failed: {str(e)}",
            )

    async def _analyze_confidence(self, answer: str, sources: List[Dict[str, Any]]) -> List[str]:
        """Analyze confidence of the answer."""
        issues = []

        # Check for hedge words
        hedge_phrases = [
            "might be",
            "could be",
            "possibly",
            "probably",
            "seems like",
            "I think",
            "it appears",
            "roughly",
            "approximately",
        ]

        lower_answer = answer.lower()
        for phrase in hedge_phrases:
            if phrase in lower_answer:
                issues.append(f"Contains hedge phrase: '{phrase}'")

        # Check for uncertainty indicators
        uncertainty_phrases = [
            "I'm not sure",
            "I cannot confirm",
            "there is insufficient information",
            "based on limited data",
            "this is speculation",
        ]

        for phrase in uncertainty_phrases:
            if phrase in lower_answer:
                issues.append(f"Contains uncertainty: '{phrase}'")

        # Check source quality impact on confidence
        if sources:
            source_scores = [source.get("score", 0.0) for source in sources]
            avg_source_score = sum(source_scores) / len(source_scores)

            if avg_source_score < 0.6:
                issues.append(f"Low source relevance: {avg_source_score:.2f}")

        return issues

    async def _check_factual_accuracy(
        self, answer: str, sources: List[Dict[str, Any]]
    ) -> List[str]:
        """Check factual accuracy against sources."""
        issues = []

        if not sources:
            return ["No sources provided for fact-checking"]

        # Extract key claims from answer
        claims = self._extract_key_claims(answer)

        # Check each claim against sources
        for claim in claims:
            is_supported = await self._verify_claim_against_sources(claim, sources)
            if not is_supported:
                issues.append(f"Unsupported claim: {claim[:100]}...")

        return issues

    async def _check_coherence(self, query: str, answer: str) -> List[str]:
        """Check answer coherence."""
        issues = []

        # Check for contradictions within the answer
        sentences = answer.split(".")

        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue

            # Check for contradictions with previous sentences
            for j, prev_sentence in enumerate(sentences[:i]):
                prev_sentence = prev_sentence.strip()
                if len(prev_sentence) < 10:
                    continue

                contradiction = await self._detect_contradiction(prev_sentence, sentence)
                if contradiction:
                    issues.append(
                        f"Contradiction: '{prev_sentence[:50]}...' vs '{sentence[:50]}...'"
                    )

        # Check answer relevance to query
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())

        overlap = len(query_words & answer_words) / len(query_words) if query_words else 0
        if overlap < 0.3:  # Less than 30% word overlap
            issues.append(f"Low query relevance: {overlap:.1%}")

        return issues

    async def _verify_sources(self, answer: str, sources: List[Dict[str, Any]]) -> List[str]:
        """Verify source quality and relevance."""
        issues = []

        # Check source diversity
        source_ids = set(source.get("document_id", "") for source in sources)
        if len(source_ids) < 2 and len(sources) > 1:
            issues.append("Low source diversity")

        # Check source scores
        for source in sources:
            score = source.get("score", 0.0)
            if score < 0.3:
                issues.append(f"Low relevance source: {source.get('title', 'Unknown')}")

        # Check for recent sources
        # (This would require timestamp information in sources)

        return issues

    async def _generate_correction(
        self, query: str, original_answer: str, issues: List[str]
    ) -> str:
        """Generate corrected answer."""
        try:
            # Create correction prompt
            issues_text = "\n".join(f"- {issue}" for issue in issues)

            correction_prompt = f"""The following answer has identified issues:

Original Query: {query}

Original Answer: {original_answer}

Issues Found:
{issues_text}

Please provide a corrected, more accurate and confident answer that addresses these issues.
Be more specific, better supported by sources, and more confident in your response."""

            from openai import OpenAI

            client = OpenAI()

            response = client.chat.completions.create(
                model=self.correction_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at correcting and improving AI-generated answers to be more accurate and confident.",
                    },
                    {"role": "user", "content": correction_prompt},
                ],
                temperature=0.1,
                max_tokens=800,
            )

            corrected_answer = response.choices[0].message.content.strip()

            logger.info(f"Generated correction for answer")
            return corrected_answer

        except Exception as e:
            logger.error(f"Error generating correction: {e}")
            return original_answer

    def _extract_key_claims(self, text: str) -> List[str]:
        """Extract key claims from text."""
        # Simple claim extraction - split by sentences and filter
        sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 15]
        return sentences

    async def _verify_claim_against_sources(
        self, claim: str, sources: List[Dict[str, Any]]
    ) -> bool:
        """Verify if a claim is supported by sources."""
        claim_words = set(claim.lower().split())

        for source in sources:
            source_text = source.get("content", "").lower()
            source_words = set(source_text.split())

            # Check for significant overlap
            overlap = len(claim_words & source_words) / len(claim_words) if claim_words else 0
            if overlap >= 0.5:  # 50% overlap threshold
                return True

        return False

    async def _detect_contradiction(self, sentence1: str, sentence2: str) -> bool:
        """Detect contradiction between two sentences."""
        # Simple contradiction patterns
        contradictions = [
            ("not", ""),
            ("never", "always"),
            ("no", "yes"),
            ("false", "true"),
            ("incorrect", "correct"),
            ("cannot", "can"),
            ("impossible", "possible"),
        ]

        words1 = set(sentence1.lower().split())
        words2 = set(sentence2.lower().split())

        for neg, pos in contradictions:
            if (neg in words1 and pos in words2) or (pos in words1 and neg in words2):
                return True

        return False

    async def _estimate_confidence(self, answer: str, sources: List[Dict[str, Any]]) -> float:
        """Estimate confidence in the answer."""
        # Base confidence on source quality
        if sources:
            source_scores = [source.get("score", 0.0) for source in sources]
            source_confidence = sum(source_scores) / len(source_scores)
        else:
            source_confidence = 0.3  # Low confidence without sources

        # Adjust based on answer characteristics
        answer_length = len(answer.split())

        # Long answers might be more comprehensive
        length_factor = min(answer_length / 100, 1.2)

        # Hedge words reduce confidence
        hedge_words = ["might", "could", "possibly", "probably"]
        hedge_count = sum(1 for word in hedge_words if word in answer.lower())
        hedge_penalty = hedge_count * 0.1

        estimated_confidence = source_confidence * length_factor - hedge_penalty

        return max(0.0, min(1.0, estimated_confidence))

    async def _estimate_confidence_improvement(
        self, original_answer: str, corrected_answer: str
    ) -> float:
        """Estimate confidence improvement from correction."""
        # Simple heuristic based on correction characteristics
        if corrected_answer == original_answer:
            return 0.0

        # Corrections that add specificity and citations tend to improve confidence
        original_length = len(original_answer.split())
        corrected_length = len(corrected_answer.split())

        if corrected_length > original_length * 1.2:  # Significantly longer
            return 0.3
        elif corrected_length > original_length * 1.1:
            return 0.2
        elif corrected_length > original_length:
            return 0.1

        return 0.05

    def _generate_reflection_reasoning(
        self, issues_found: List[str], needs_correction: bool
    ) -> str:
        """Generate reasoning for reflection decision."""
        if not issues_found:
            return "No significant issues found in the answer."

        reasoning_parts = ["Analysis identified the following issues:"]
        reasoning_parts.extend(f"• {issue}" for issue in issues_found[:5])

        if needs_correction:
            reasoning_parts.append("Correction is recommended to improve accuracy and confidence.")
        else:
            reasoning_parts.append("No correction needed at this time.")

        return " ".join(reasoning_parts)

    async def get_reflection_stats(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics about reflection performance."""
        # This would connect to a metrics system in a full implementation
        return {
            "session_id": session_id,
            "max_reflection_rounds": self.max_reflection_rounds,
            "confidence_threshold": self.confidence_threshold,
            "features_enabled": {
                "fact_checking": self.enable_fact_checking,
                "coherence_checking": self.enable_coherence_checking,
                "source_verification": self.enable_source_verification,
            },
        }
