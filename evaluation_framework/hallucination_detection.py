"""
Hallucination Detection - RAG-The-Game-Changer

Advanced hallucination detection for RAG systems.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import re
import time

logger = logging.getLogger(__name__)


@dataclass
class HallucinationResult:
    """Result of hallucination detection."""

    is_hallucinated: bool
    confidence: float
    hallucinated_claims: List[str] = field(default_factory=list)
    supported_claims: List[str] = field(default_factory=list)
    unsupported_claims: List[str] = field(default_factory=list)
    reasoning: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClaimAnalysis:
    """Analysis of a single claim."""

    claim: str
    claim_type: str  # factual, numerical, causal, etc.
    support_level: str  # supported, partially_supported, unsupported, unknown
    supporting_sources: List[Dict[str, Any]] = field(default_factory=list)
    contradictory_sources: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0


class HallucinationDetector:
    """Advanced hallucination detection for RAG outputs."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # Detection strategies
        self.use_source_verification = self.config.get("use_source_verification", True)
        self.use_fact_checking = self.config.get("use_fact_checking", False)
        self.use_semantic_consistency = self.config.get("use_semantic_consistency", True)
        self.use_numerical_verification = self.config.get("use_numerical_verification", True)

        # Thresholds
        self.hallucination_threshold = self.config.get("hallucination_threshold", 0.5)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)

        # LLM settings for fact-checking
        self.fact_check_model = self.config.get("fact_check_model", "gpt-4")
        self.max_claims_per_analysis = self.config.get("max_claims_per_analysis", 10)

    async def detect_hallucination(
        self,
        generated_answer: str,
        sources: List[Dict[str, Any]],
        original_query: str,
        ground_truth: Optional[str] = None,
    ) -> HallucinationResult:
        """Detect hallucinations in generated answer."""
        try:
            # Extract claims from the generated answer
            claims = await self._extract_claims(generated_answer)

            if not claims:
                return HallucinationResult(
                    is_hallucinated=False, confidence=1.0, reasoning="No claims found to verify"
                )

            # Analyze each claim
            claim_analyses = []
            for claim in claims[: self.max_claims_per_analysis]:
                analysis = await self._analyze_claim(claim, sources, original_query)
                claim_analyses.append(analysis)

            # Determine overall hallucination status
            hallucination_result = await self._determine_hallucination_status(
                claim_analyses, sources, ground_truth
            )

            return hallucination_result

        except Exception as e:
            logger.error(f"Error in hallucination detection: {e}")
            return HallucinationResult(
                is_hallucinated=True, confidence=0.0, reasoning=f"Detection failed: {str(e)}"
            )

    async def _extract_claims(self, text: str) -> List[str]:
        """Extract individual claims from text."""
        # Split text into sentences and analyze each
        sentences = re.split(r"[.!?]+", text)
        claims = []

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10 and self._is_claim_sentence(sentence):
                claims.append(sentence)

        return claims

    def _is_claim_sentence(self, sentence: str) -> bool:
        """Check if sentence contains a claim."""
        # Claims typically contain:
        # - Factual statements
        # - Numerical values
        # - Causal relationships
        # - Specific information

        # Simple heuristics
        claim_indicators = [
            r"\b(is|are|was|were)\b",  # State of being
            r"\b\d+(\.\d+)?\b",  # Numbers
            r"\b(more than|less than|greater than)\b",  # Comparisons
            r"\b(because|since|due to|as a result)\b",  # Causality
            r"\b(according to|research shows|studies show)\b",  # Attribution
            r"\b(specify|exactly|precisely)\b",  # Specifics
            r"\b(increased|decreased|improved|worsened)\b",  # Changes
        ]

        return any(re.search(pattern, sentence.lower()) for pattern in claim_indicators)

    async def _analyze_claim(
        self, claim: str, sources: List[Dict[str, Any]], original_query: str
    ) -> ClaimAnalysis:
        """Analyze a single claim against sources."""
        claim_type = self._classify_claim_type(claim)

        # Source verification
        source_support = await self._verify_claim_with_sources(claim, sources)

        # Semantic consistency
        semantic_consistency = await self._check_semantic_consistency(claim, original_query)

        # Numerical verification
        if claim_type == "numerical":
            numerical_support = await self._verify_numerical_claim(claim, sources)
            source_support.support_level = numerical_support
        else:
            source_support.support_level = (
                "supported" if source_support.supporting_sources else "unsupported"
            )

        # Combine all evidence
        overall_support = self._combine_evidence(source_support, semantic_consistency)

        return ClaimAnalysis(
            claim=claim,
            claim_type=claim_type,
            support_level=overall_support,
            supporting_sources=source_support.supporting_sources,
            contradictory_sources=source_support.contradictory_sources,
            confidence=source_support.confidence,
        )

    def _classify_claim_type(self, claim: str) -> str:
        """Classify the type of claim."""
        claim_lower = claim.lower()

        # Numerical claims
        if re.search(r"\b\d+(\.\d+)?\b", claim_lower):
            return "numerical"

        # Causal claims
        if re.search(r"\b(because|since|due to|as a result|causes|leads to)\b", claim_lower):
            return "causal"

        # Comparative claims
        if re.search(r"\b(more than|less than|greater than|higher than|lower than)\b", claim_lower):
            return "comparative"

        # Attribution claims
        if re.search(r"\b(according to|research shows|studies show|experts say)\b", claim_lower):
            return "attribution"

        # Default to factual
        return "factual"

    async def _verify_claim_with_sources(
        self, claim: str, sources: List[Dict[str, Any]]
    ) -> ClaimAnalysis:
        """Verify claim against retrieved sources."""
        if not sources:
            return ClaimAnalysis(
                claim=claim, claim_type="factual", support_level="unsupported", confidence=0.0
            )

        supporting_sources = []
        contradictory_sources = []
        total_confidence = 0.0

        claim_words = set(claim.lower().split())

        for source in sources:
            source_content = source.get("content", "").lower()
            source_score = source.get("score", 0.0)

            # Simple text overlap for support detection
            content_words = set(source_content.split())
            overlap = len(claim_words & content_words) / len(claim_words) if claim_words else 0

            if overlap >= 0.5:  # 50% overlap threshold
                supporting_sources.append(
                    {
                        "source_id": source.get("document_id", ""),
                        "content": source_content[:200],  # First 200 chars
                        "score": source_score,
                        "overlap": overlap,
                    }
                )
                total_confidence += source_score
            elif self._is_contradictory(claim, source_content):
                contradictory_sources.append(
                    {
                        "source_id": source.get("document_id", ""),
                        "content": source_content[:200],
                        "score": source_score,
                    }
                )

        avg_confidence = total_confidence / len(sources) if sources else 0.0

        return ClaimAnalysis(
            claim=claim,
            claim_type="factual",
            support_level="partially_supported" if supporting_sources else "unsupported",
            supporting_sources=supporting_sources,
            contradictory_sources=contradictory_sources,
            confidence=min(avg_confidence, 1.0),
        )

    def _is_contradictory(self, claim: str, source_content: str) -> bool:
        """Simple contradiction detection."""
        # Look for negation patterns
        claim_lower = claim.lower()
        source_lower = source_content.lower()

        # Simple contradiction indicators
        contradiction_patterns = [
            (r"\bno\b", r"\bnot\b"),
            (r"\bis not\b", r"\bis never\b"),
            (r"\bfailed to\b", r"\bsucceeded in\b"),
            (r"\bincorrect\b", r"\bincorrect\b"),
            (r"\bimpossible\b", r"\bpossible\b"),
        ]

        for neg_pattern, pos_pattern in contradiction_patterns:
            if (re.search(neg_pattern, claim_lower) and re.search(pos_pattern, source_lower)) or (
                re.search(pos_pattern, claim_lower) and re.search(neg_pattern, source_lower)
            ):
                return True

        return False

    async def _check_semantic_consistency(self, claim: str, original_query: str) -> str:
        """Check semantic consistency with original query."""
        # Simple semantic check - is the claim relevant to the query?
        query_words = set(original_query.lower().split())
        claim_words = set(claim.lower().split())

        # Calculate semantic overlap
        overlap = len(query_words & claim_words) / len(query_words) if query_words else 0

        if overlap >= 0.3:  # 30% overlap threshold
            return "consistent"
        elif overlap >= 0.1:
            return "partially_consistent"
        else:
            return "inconsistent"

    async def _verify_numerical_claim(self, claim: str, sources: List[Dict[str, Any]]) -> str:
        """Verify numerical claims against sources."""
        # Extract numbers from claim
        claim_numbers = self._extract_numbers(claim)

        if not claim_numbers:
            return "unknown"

        # Extract numbers from sources
        source_numbers = []
        for source in sources:
            numbers = self._extract_numbers(source.get("content", ""))
            source_numbers.extend(numbers)

        # Check if any claim numbers appear in sources
        supported_numbers = []
        for claim_num in claim_numbers:
            for source_num in source_numbers:
                if self._numbers_similar(claim_num, source_num):
                    supported_numbers.append(claim_num)
                    break

        if len(supported_numbers) == len(claim_numbers):
            return "supported"
        elif len(supported_numbers) > 0:
            return "partially_supported"
        else:
            return "unsupported"

    def _extract_numbers(self, text: str) -> List[float]:
        """Extract numerical values from text."""
        # Find numbers with optional decimals
        number_pattern = r"\b\d+(?:\.\d+)?\b"
        matches = re.findall(number_pattern, text)
        return [float(match) for match in matches]

    def _numbers_similar(self, num1: float, num2: float, tolerance: float = 0.1) -> bool:
        """Check if two numbers are similar within tolerance."""
        if abs(num1 - num2) <= tolerance * max(abs(num1), abs(num2), 1.0):
            return True
        return False

    def _combine_evidence(self, source_analysis: ClaimAnalysis, semantic_consistency: str) -> str:
        """Combine different types of evidence."""
        if source_analysis.support_level == "supported":
            if semantic_consistency == "consistent":
                return "supported"
            elif semantic_consistency == "partially_consistent":
                return "partially_supported"
            else:
                return "questionable"

        elif source_analysis.support_level == "partially_supported":
            if semantic_consistency == "consistent":
                return "partially_supported"
            else:
                return "questionable"

        else:
            if semantic_consistency == "consistent":
                return "questionable"
            else:
                return "unsupported"

    async def _determine_hallucination_status(
        self,
        claim_analyses: List[ClaimAnalysis],
        sources: List[Dict[str, Any]],
        ground_truth: Optional[str],
    ) -> HallucinationResult:
        """Determine overall hallucination status."""
        if not claim_analyses:
            return HallucinationResult(
                is_hallucinated=False, confidence=1.0, reasoning="No claims to analyze"
            )

        # Calculate metrics
        total_claims = len(claim_analyses)
        supported_claims = sum(
            1 for analysis in claim_analyses if analysis.support_level == "supported"
        )
        partially_supported = sum(
            1 for analysis in claim_analyses if analysis.support_level == "partially_supported"
        )
        unsupported_claims = sum(
            1 for analysis in claim_analyses if analysis.support_level == "unsupported"
        )

        # Determine hallucination
        hallucination_ratio = unsupported_claims / total_claims if total_claims > 0 else 0.0
        is_hallucinated = hallucination_ratio > self.hallucination_threshold

        # Calculate confidence
        avg_confidence = sum(analysis.confidence for analysis in claim_analyses) / total_claims

        # Extract specific claims
        hallucinated_claims = [
            analysis.claim
            for analysis in claim_analyses
            if analysis.support_level in ["unsupported", "questionable"]
        ]

        supported_claims_list = [
            analysis.claim for analysis in claim_analyses if analysis.support_level == "supported"
        ]

        # Ground truth comparison if available
        ground_truth_match = 1.0
        reasoning_parts = []

        if ground_truth:
            # Simple semantic similarity with ground truth
            ground_truth_words = set(ground_truth.lower().split())
            all_claim_words = set()
            for analysis in claim_analyses:
                all_claim_words.update(analysis.claim.lower().split())

            overlap = (
                len(ground_truth_words & all_claim_words) / len(ground_truth_words)
                if ground_truth_words
                else 0
            )
            ground_truth_match = overlap

            reasoning_parts.append(f"Ground truth overlap: {overlap:.2f}")

        # Build reasoning
        reasoning_parts.extend(
            [
                f"Total claims: {total_claims}",
                f"Supported: {supported_claims}",
                f"Partially supported: {partially_supported}",
                f"Unsupported: {unsupported_claims}",
                f"Hallucination ratio: {hallucination_ratio:.2f}",
            ]
        )

        return HallucinationResult(
            is_hallucinated=is_hallucinated,
            confidence=avg_confidence,
            hallucinated_claims=hallucinated_claims,
            supported_claims=supported_claims_list,
            unsupported_claims=[
                analysis.claim
                for analysis in claim_analyses
                if analysis.support_level == "unsupported"
            ],
            reasoning=" | ".join(reasoning_parts),
            metadata={
                "total_claims": total_claims,
                "supported_claims": supported_claims,
                "partially_supported": partially_supported,
                "unsupported_claims": unsupported_claims,
                "hallucination_ratio": hallucination_ratio,
                "ground_truth_match": ground_truth_match,
                "sources_count": len(sources),
            },
        )

    async def detect_batch_hallucinations(
        self, query_responses: List[Dict[str, Any]]
    ) -> List[HallucinationResult]:
        """Detect hallucinations in multiple responses."""
        results = []

        for response in query_responses:
            result = await self.detect_hallucination(
                generated_answer=response.get("answer", ""),
                sources=response.get("sources", []),
                original_query=response.get("query", ""),
                ground_truth=response.get("ground_truth"),
            )
            results.append(result)

        return results

    def calculate_hallucination_metrics(self, results: List[HallucinationResult]) -> Dict[str, Any]:
        """Calculate hallucination-related metrics."""
        if not results:
            return {}

        total_responses = len(results)
        hallucinated_responses = sum(1 for result in results if result.is_hallucinated)
        hallucination_rate = hallucinated_responses / total_responses

        # Confidence statistics
        confidences = [result.confidence for result in results]
        avg_confidence = sum(confidences) / len(confidences)
        min_confidence = min(confidences)
        max_confidence = max(confidences)

        # Claim statistics
        total_claims = sum(
            len(result.hallucinated_claims) + len(result.supported_claims) for result in results
        )
        avg_claims_per_response = total_claims / total_responses if total_responses > 0 else 0

        return {
            "total_responses": total_responses,
            "hallucinated_responses": hallucinated_responses,
            "hallucination_rate": hallucination_rate,
            "avg_confidence": avg_confidence,
            "min_confidence": min_confidence,
            "max_confidence": max_confidence,
            "avg_claims_per_response": avg_claims_per_response,
            "total_hallucinated_claims": sum(len(result.hallucinated_claims) for result in results),
            "total_supported_claims": sum(len(result.supported_claims) for result in results),
        }
