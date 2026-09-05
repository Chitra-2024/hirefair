"""Tests for the Matcher agent using Gemini Structured Outputs."""

from unittest.mock import patch
from pydantic import ValidationError
import pytest

from app.agents.matcher import (
    DEFAULT_MODEL,
    MATCHER_SYSTEM_PROMPT,
    Matcher,
    match_candidate,
)
from app.models.candidate import (
    CandidateProfile,
    Education,
    Project,
    WorkExperience,
)
from app.models.job_description import (
    JobRequirement,
    ParsedJobDescription,
    RequirementType,
)
from app.models.score import MatchResult, ScoreRecord


@pytest.fixture
def sample_candidate() -> CandidateProfile:
    """Fixture providing a realistic structured CandidateProfile with raw resume text."""
    raw_resume = (
        "Elena Vasquez\n"
        "Senior Data Engineer\n"
        "Crestline Data Systems (March 2022 - Present)\n"
        "- Architected and maintain 40+ Apache Airflow DAGs processing over 3B records daily into Snowflake.\n"
        "- Optimized SQL queries reducing execution time by 45%.\n"
        "Skills: Python, SQL, Apache Airflow, Snowflake, Great Expectations, Docker."
    )
    return CandidateProfile(
        candidate_id="elena_vasquez",
        candidate_name="Elena Vasquez",
        raw_text=raw_resume,
        skills=["Python", "SQL", "Apache Airflow", "Snowflake", "Great Expectations", "Docker"],
        experience=[
            WorkExperience(
                title="Senior Data Engineer",
                company="Crestline Data Systems",
                duration_months=54,
                description="Architected and maintain 40+ Apache Airflow DAGs processing over 3B records daily into Snowflake.",
            )
        ],
        education=[
            Education(
                degree="B.S. Computer Science",
                institution="University of Texas at Dallas",
                graduation_year=2017,
                field_of_study="Computer Science",
            )
        ],
        projects=[
            Project(
                name="airflow-dbt-pipeline",
                description="End-to-end data pipeline orchestrating dbt models in Airflow with automated validation.",
                technologies=["Python", "Airflow", "dbt", "Snowflake"],
            )
        ],
    )


@pytest.fixture
def sample_rubric() -> ParsedJobDescription:
    """Fixture providing a parsed job description with multiple rubric criteria."""
    return ParsedJobDescription(
        job_title="Senior Data Engineer",
        company="Crestline Data Systems",
        requirements=[
            JobRequirement(
                criterion_id="REQ-01",
                description="4+ years experience with Apache Airflow and pipeline orchestration",
                weight=5,
                type=RequirementType.MUST_HAVE,
                evidence_type="work experience",
            ),
            JobRequirement(
                criterion_id="REQ-02",
                description="Production experience with Snowflake data warehousing and SQL optimization",
                weight=4,
                type=RequirementType.MUST_HAVE,
                evidence_type="work experience",
            ),
            JobRequirement(
                criterion_id="REQ-03",
                description="Experience with Kubernetes container orchestration",
                weight=2,
                type=RequirementType.NICE_TO_HAVE,
                evidence_type="work experience or projects",
            ),
        ],
    )


@pytest.fixture
def sample_match_result(sample_candidate: CandidateProfile) -> MatchResult:
    """Fixture providing a valid MatchResult with 3 ScoreRecords."""
    return MatchResult(
        scores=[
            ScoreRecord(
                candidate_id=sample_candidate.candidate_id,
                criterion_id="REQ-01",
                score=0.95,
                evidence_quote="Architected and maintain 40+ Apache Airflow DAGs processing over 3B records daily into Snowflake.",
                confidence=0.95,
            ),
            ScoreRecord(
                candidate_id=sample_candidate.candidate_id,
                criterion_id="REQ-02",
                score=0.90,
                evidence_quote="Optimized SQL queries reducing execution time by 45%.",
                confidence=0.90,
            ),
            ScoreRecord(
                candidate_id=sample_candidate.candidate_id,
                criterion_id="REQ-03",
                score=0.15,
                evidence_quote="",  # <= 0.3 allows empty evidence
                confidence=0.80,
            ),
        ]
    )


def test_matcher_empty_or_invalid_candidate(sample_rubric: ParsedJobDescription):
    """Verify that empty or invalid candidate input raises ValueError."""
    matcher = Matcher()

    # None candidate
    with pytest.raises(ValueError, match="CandidateProfile must be provided"):
        matcher.match(None, sample_rubric)  # type: ignore

    # Non-CandidateProfile object
    with pytest.raises(ValueError, match="CandidateProfile must be provided"):
        matcher.match("not_a_candidate", sample_rubric)  # type: ignore

    # Candidate with empty raw_text
    empty_cand = CandidateProfile(candidate_id="c1", raw_text="")
    with pytest.raises(ValueError, match="raw_text cannot be empty"):
        matcher.match(empty_cand, sample_rubric)

    # Candidate with whitespace raw_text
    whitespace_cand = CandidateProfile(candidate_id="c1", raw_text="   \n\t  ")
    with pytest.raises(ValueError, match="raw_text cannot be empty"):
        matcher.match(whitespace_cand, sample_rubric)


def test_matcher_empty_or_invalid_rubric(sample_candidate: CandidateProfile):
    """Verify that empty or invalid rubric input raises ValueError."""
    matcher = Matcher()

    # None rubric
    with pytest.raises(ValueError, match="ParsedJobDescription rubric must be provided"):
        matcher.match(sample_candidate, None)  # type: ignore

    # Non-ParsedJobDescription object
    with pytest.raises(ValueError, match="ParsedJobDescription rubric must be provided"):
        matcher.match(sample_candidate, "not_a_rubric")  # type: ignore

    # Rubric with empty requirements
    empty_rubric = ParsedJobDescription(job_title="Data Engineer", requirements=[])
    with pytest.raises(ValueError, match="Rubric requirements cannot be empty"):
        matcher.match(sample_candidate, empty_rubric)


def test_matcher_passes_response_model_and_default_model(
    sample_candidate: CandidateProfile,
    sample_rubric: ParsedJobDescription,
    sample_match_result: MatchResult,
):
    """Verify that Matcher calls generate_structured with MatchResult and default model gemini-2.5-flash."""
    with patch("app.agents.matcher.generate_structured") as mock_generate:
        mock_generate.return_value = sample_match_result

        matcher = Matcher()
        result = matcher.match(sample_candidate, sample_rubric)

        assert mock_generate.called
        call_kwargs = mock_generate.call_args.kwargs

        # Verify response_model is MatchResult
        assert call_kwargs["response_model"] is MatchResult

        # Verify model is default gemini-2.5-flash
        assert call_kwargs["model"] == "gemini-2.5-flash"
        assert call_kwargs["model"] == DEFAULT_MODEL

        # Verify system prompt
        assert call_kwargs["system"] == MATCHER_SYSTEM_PROMPT

        # Verify prompt contains raw_text and rubric criteria
        prompt = call_kwargs["prompt"]
        assert sample_candidate.raw_text in prompt
        assert "REQ-01" in prompt
        assert "REQ-02" in prompt
        assert "REQ-03" in prompt

        # Result returned properly
        assert result is sample_match_result


def test_matcher_custom_model(
    sample_candidate: CandidateProfile,
    sample_rubric: ParsedJobDescription,
    sample_match_result: MatchResult,
):
    """Verify that Matcher supports overriding the model identifier."""
    with patch("app.agents.matcher.generate_structured") as mock_generate:
        mock_generate.return_value = sample_match_result

        custom_model = "gemini-1.5-pro"
        matcher = Matcher(model=custom_model)
        result = matcher.match(sample_candidate, sample_rubric)

        assert mock_generate.call_args.kwargs["model"] == custom_model
        assert result is sample_match_result


def test_matcher_structured_output_multiple_criteria(
    sample_candidate: CandidateProfile,
    sample_rubric: ParsedJobDescription,
    sample_match_result: MatchResult,
):
    """Verify structured MatchResult / ScoreRecord contract with multiple rubric criteria."""
    with patch("app.agents.matcher.generate_structured") as mock_generate:
        mock_generate.return_value = sample_match_result

        matcher = Matcher()
        result = matcher.match(sample_candidate, sample_rubric)

        # 1. Type verification
        assert isinstance(result, MatchResult)
        assert len(result.scores) == 3

        # 2. ScoreRecord structure verification
        criterion_ids = [s.criterion_id for s in result.scores]
        assert criterion_ids == ["REQ-01", "REQ-02", "REQ-03"]

        for score_rec in result.scores:
            assert isinstance(score_rec, ScoreRecord)
            assert score_rec.candidate_id == "elena_vasquez"
            assert 0.0 <= score_rec.score <= 1.0
            assert 0.0 <= score_rec.confidence <= 1.0


def test_matcher_preserves_evidence_quote(
    sample_candidate: CandidateProfile,
    sample_rubric: ParsedJobDescription,
    sample_match_result: MatchResult,
):
    """Verify evidence quotes are accurately preserved from the candidate resume."""
    with patch("app.agents.matcher.generate_structured") as mock_generate:
        mock_generate.return_value = sample_match_result

        matcher = Matcher()
        result = matcher.match(sample_candidate, sample_rubric)

        req1_score = next(s for s in result.scores if s.criterion_id == "REQ-01")
        assert (
            req1_score.evidence_quote
            == "Architected and maintain 40+ Apache Airflow DAGs processing over 3B records daily into Snowflake."
        )

        req3_score = next(s for s in result.scores if s.criterion_id == "REQ-03")
        assert req3_score.evidence_quote == ""


def test_score_record_score_bounded_0_to_1():
    """Verify ScoreRecord constrains score to [0.0, 1.0]."""
    # Valid boundaries
    s_min = ScoreRecord(candidate_id="c1", criterion_id="REQ-01", score=0.0, evidence_quote="", confidence=0.5)
    assert s_min.score == 0.0

    s_max = ScoreRecord(candidate_id="c1", criterion_id="REQ-01", score=1.0, evidence_quote="valid quote", confidence=0.5)
    assert s_max.score == 1.0

    # Below 0.0
    with pytest.raises(ValidationError):
        ScoreRecord(candidate_id="c1", criterion_id="REQ-01", score=-0.01, evidence_quote="", confidence=0.5)

    # Above 1.0
    with pytest.raises(ValidationError):
        ScoreRecord(candidate_id="c1", criterion_id="REQ-01", score=1.01, evidence_quote="quote", confidence=0.5)


def test_score_record_confidence_bounded_0_to_1():
    """Verify ScoreRecord constrains confidence to [0.0, 1.0]."""
    # Valid boundaries
    c_min = ScoreRecord(candidate_id="c1", criterion_id="REQ-01", score=0.2, evidence_quote="", confidence=0.0)
    assert c_min.confidence == 0.0

    c_max = ScoreRecord(candidate_id="c1", criterion_id="REQ-01", score=0.2, evidence_quote="", confidence=1.0)
    assert c_max.confidence == 1.0

    # Below 0.0
    with pytest.raises(ValidationError):
        ScoreRecord(candidate_id="c1", criterion_id="REQ-01", score=0.2, evidence_quote="", confidence=-0.1)

    # Above 1.0
    with pytest.raises(ValidationError):
        ScoreRecord(candidate_id="c1", criterion_id="REQ-01", score=0.2, evidence_quote="", confidence=1.1)


def test_score_record_rejects_score_above_threshold_without_evidence():
    """Verify ScoreRecord rejects score > 0.3 when evidence_quote is empty or whitespace."""
    # Empty string
    with pytest.raises(ValidationError, match="requires a non-empty direct evidence_quote"):
        ScoreRecord(
            candidate_id="c1",
            criterion_id="REQ-01",
            score=0.31,
            evidence_quote="",
            confidence=0.8,
        )

    # Whitespace only
    with pytest.raises(ValidationError, match="requires a non-empty direct evidence_quote"):
        ScoreRecord(
            candidate_id="c1",
            criterion_id="REQ-01",
            score=0.85,
            evidence_quote="   \n\t  ",
            confidence=0.9,
        )

    # High score (1.0) with empty evidence
    with pytest.raises(ValidationError, match="requires a non-empty direct evidence_quote"):
        ScoreRecord(
            candidate_id="c1",
            criterion_id="REQ-01",
            score=1.0,
            evidence_quote="",
            confidence=0.95,
        )


def test_score_record_accepts_score_above_threshold_with_direct_evidence():
    """Verify ScoreRecord accepts score > 0.3 when evidence_quote contains non-whitespace text."""
    rec1 = ScoreRecord(
        candidate_id="c1",
        criterion_id="REQ-01",
        score=0.31,
        evidence_quote="Built automated ETL pipelines",
        confidence=0.7,
    )
    assert rec1.score == 0.31
    assert rec1.evidence_quote == "Built automated ETL pipelines"

    rec2 = ScoreRecord(
        candidate_id="c1",
        criterion_id="REQ-02",
        score=0.95,
        evidence_quote="Architected 40+ Apache Airflow DAGs",
        confidence=0.95,
    )
    assert rec2.score == 0.95
    assert rec2.evidence_quote == "Architected 40+ Apache Airflow DAGs"


def test_score_record_allows_score_at_or_below_threshold_without_evidence():
    """Verify ScoreRecord allows score <= 0.3 with empty evidence_quote."""
    rec_exact_threshold = ScoreRecord(
        candidate_id="c1",
        criterion_id="REQ-01",
        score=0.3,
        evidence_quote="",
        confidence=0.5,
    )
    assert rec_exact_threshold.score == 0.3
    assert rec_exact_threshold.evidence_quote == ""

    rec_low = ScoreRecord(
        candidate_id="c1",
        criterion_id="REQ-01",
        score=0.1,
        evidence_quote="",
        confidence=0.8,
    )
    assert rec_low.score == 0.1

    rec_zero = ScoreRecord(
        candidate_id="c1",
        criterion_id="REQ-01",
        score=0.0,
        evidence_quote="",
        confidence=0.9,
    )
    assert rec_zero.score == 0.0


def test_matcher_surfaces_validation_failure_when_llm_violates_evidence_rule(
    sample_candidate: CandidateProfile,
    sample_rubric: ParsedJobDescription,
):
    """Verify that Matcher surfaces validation failure when output violates >0.3 evidence rule.

    Does NOT clamp the score, fabricate quotes, or attempt silent self-repair.
    """
    # Create an invalid record and MatchResult using model_construct to simulate invalid LLM output
    invalid_record = ScoreRecord.model_construct(
        candidate_id="elena_vasquez",
        criterion_id="REQ-01",
        score=0.85,
        evidence_quote="",  # Violates rule!
        confidence=0.9,
    )
    invalid_match_result = MatchResult.model_construct(scores=[invalid_record])

    with patch("app.agents.matcher.generate_structured") as mock_generate:
        mock_generate.return_value = invalid_match_result

        matcher = Matcher()
        # Must surface validation failure as ValueError / ValidationError
        with pytest.raises(ValueError, match="without a non-whitespace evidence quote"):
            matcher.match(sample_candidate, sample_rubric)


def test_no_high_score_can_be_returned_as_valid_score_record_without_evidence():
    """Verify that no high score (>0.3) can be validated as a valid ScoreRecord without evidence."""
    invalid_json = """
    {
        "scores": [
            {
                "candidate_id": "cand_01",
                "criterion_id": "REQ-01",
                "score": 0.85,
                "evidence_quote": "",
                "confidence": 0.9
            }
        ]
    }
    """
    with pytest.raises(ValidationError, match="requires a non-empty direct evidence_quote"):
        MatchResult.model_validate_json(invalid_json)


def test_match_candidate_convenience_function(
    sample_candidate: CandidateProfile,
    sample_rubric: ParsedJobDescription,
    sample_match_result: MatchResult,
):
    """Verify that match_candidate convenience wrapper functions correctly."""
    with patch("app.agents.matcher.generate_structured") as mock_generate:
        mock_generate.return_value = sample_match_result

        result = match_candidate(sample_candidate, sample_rubric)

        assert isinstance(result, MatchResult)
        assert len(result.scores) == 3
        assert mock_generate.called
        assert mock_generate.call_args.kwargs["model"] == "gemini-2.5-flash"
