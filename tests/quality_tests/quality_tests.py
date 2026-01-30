"""
Quality Tests - RAG-The-Game-Changer

Quality assessment and validation tests for RAG systems.
"""

import pytest
import asyncio
from evaluation_framework.quality_assessment import QualityAssessor


@pytest.fixture
def quality_assessor():
    """Quality assessor fixture."""
    return QualityAssessor()


@pytest.mark.asyncio
async def test_quality_relevance(quality_assessor):
    """Test relevance scoring."""
    query = "What is the capital of France?"
    answer = "The capital of France is Paris."
    contexts = ["Paris is a major city in France."]

    scores = await quality_assessor.assess_quality(
        query=query, answer=answer, retrieved_contexts=contexts
    )

    relevance_score = next((s for s in scores if s.dimension == "relevance"), None)

    assert relevance_score is not None
    assert 0.0 <= relevance_score.score <= 1.0
    assert relevance_score.score > 0.5  # Should be high relevance


@pytest.mark.asyncio
async def test_quality_coherence(quality_assessor):
    """Test coherence scoring."""
    answer = (
        "Paris is the capital of France. It has many famous landmarks including the Eiffel Tower."
    )

    scores = await quality_assessor.assess_quality(query="", answer=answer, retrieved_contexts=[])

    coherence_score = next((s for s in scores if s.dimension == "coherence"), None)

    assert coherence_score is not None
    assert coherence_score.score > 0.7  # Should be coherent


@pytest.mark.asyncio
async def test_quality_completeness(quality_assessor):
    """Test completeness scoring."""
    query = "What is the population of France?"
    answer = "France has a population of approximately 67 million people."

    scores = await quality_assessor.assess_quality(
        query=query, answer=answer, retrieved_contexts=[]
    )

    completeness_score = next((s for s in scores if s.dimension == "completeness"), None)

    assert completeness_score is not None
    assert completeness_score.score > 0.8  # Should be complete


@pytest.mark.asyncio
async def test_quality_fluency(quality_assessor):
    """Test fluency scoring."""
    fluent_answer = "The Eiffel Tower is located in Paris, France. It was built in 1889 and attracts millions of visitors annually."

    scores = await quality_assessor.assess_quality(
        query="", answer=fluent_answer, retrieved_contexts=[]
    )

    fluency_score = next((s for s in scores if s.dimension == "fluency"), None)

    assert fluency_score is not None
    assert fluency_score.score > 0.7  # Should be fluent


@pytest.mark.asyncio
async def test_quality_correctness(quality_assessor):
    """Test correctness scoring."""
    query = "What is 2 + 2?"
    correct_answer = "4"
    wrong_answer = "5"

    correct_scores = await quality_assessor.assess_quality(
        query=query, answer=correct_answer, expected_answer=correct_answer
    )

    wrong_scores = await quality_assessor.assess_quality(
        query=query, answer=wrong_answer, expected_answer=correct_answer
    )

    correct_score = next((s for s in correct_scores if s.dimension == "correctness"), None)
    wrong_score = next((s for s in wrong_scores if s.dimension == "correctness"), None)

    assert correct_score is not None
    assert correct_score.score > wrong_score.score  # Correct answer should have higher score


@pytest.mark.asyncio
async def test_quality_multi_dimension(quality_assessor):
    """Test multi-dimensional quality assessment."""
    query = "What are the main exports of France?"
    answer = "France exports wine, cheese, aircraft, and luxury goods."
    contexts = [
        "France is known for its wine production, particularly in regions like Bordeaux and Burgundy."
    ]

    scores = await quality_assessor.assess_quality(
        query=query, answer=answer, retrieved_contexts=contexts
    )

    assert len(scores) >= 4  # Should have multiple dimensions

    overall_score = quality_assessor.calculate_overall_score(scores)

    assert 0.0 <= overall_score <= 1.0  # Should be in valid range
    assert overall_score > 0.5  # Should be reasonably good quality


@pytest.mark.asyncio
async def test_quality_empty_query(quality_assessor):
    """Test quality assessment with empty query."""
    answer = "I don't have information about that."
    contexts = []

    scores = await quality_assessor.assess_quality(
        query="", answer=answer, retrieved_contexts=contexts
    )

    # Should handle empty query gracefully
    assert all(0.0 <= s.score <= 1.0 for s in scores)


@pytest.mark.asyncio
async def test_quality_assessment_weights(quality_assessor):
    """Test custom quality assessment weights."""
    from evaluation_framework.quality_assessment import AssessmentConfig

    config = AssessmentConfig(
        dimensions_weights={
            "relevance": 0.6,
            "coherence": 0.2,
            "completeness": 0.15,
            "fluency": 0.05,
        }
    )

    weighted_assessor = QualityAssessor(config)

    scores = await weighted_assessor.assess_quality(
        query="Test query", answer="Test answer with relevant information.", retrieved_contexts=[]
    )

    # Check that weights are applied
    total_weight = sum(s.weight for s in scores)
    assert total_weight > 0
