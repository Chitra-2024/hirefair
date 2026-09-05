"""Comprehensive offline tests for the Fairness Auditor agent.

All tests are fully offline — generate_structured() and/or Matcher.match() are mocked.
No GEMINI_API_KEY is required.
"""

from unittest.mock import MagicMock, call, patch
import pytest

from app.agents.citation_checker import CitationCheckBatch, CitationCheckResult
from app.agents.fairness_auditor import (
    COUNTERFACTUAL_DELTA_THRESHOLD,
    NON_TRADITIONAL_SCORE_THRESHOLD,
    FairnessAuditor,
    _weighted_aggregate,
    audit_candidate,
)
from app.models.audit import AuditRecord
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


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


RAW_RESUME = (
    "Elena Vasquez\n"
    "Senior Data Engineer at Crestline Data Systems (2022 – Present)\n"
    "Architected 40+ Apache Airflow DAGs processing over 3B records daily into Snowflake.\n"
    "Optimized SQL queries reducing execution time by 45%.\n"
    "Education: B.S. Computer Science, University of Texas at Dallas, 2017\n"
    "Skills: Python, SQL, Apache Airflow, Snowflake, Docker"
)


@pytest.fixture
def candidate_with_raw_text() -> CandidateProfile:
    """A candidate with a realistic raw_text containing name, institution, and year."""
    return CandidateProfile(
        candidate_id="elena_vasquez",
        candidate_name="Elena Vasquez",
        raw_text=RAW_RESUME,
        skills=["Python", "SQL", "Apache Airflow", "Snowflake", "Docker"],
        experience=[
            WorkExperience(
                title="Senior Data Engineer",
                company="Crestline Data Systems",
                duration_months=36,
                description="Architected 40+ Apache Airflow DAGs processing over 3B records daily into Snowflake.",
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
                name="airflow-pipeline",
                description="End-to-end pipeline orchestrating dbt models with automated validation.",
                technologies=["Python", "Airflow", "dbt"],
            )
        ],
    )


@pytest.fixture
def candidate_with_non_traditional() -> CandidateProfile:
    """A candidate whose experience/projects contain non-traditional evidence keywords."""
    return CandidateProfile(
        candidate_id="alex_freelance",
        candidate_name="Alex Freelance",
        raw_text=(
            "Alex Freelance\n"
            "Self-taught data engineer with freelance and open-source contributions.\n"
            "Contributed to open-source Airflow plugins on GitHub. "
            "Built personal projects using Python and Snowflake.\n"
            "Skills: Python, SQL, Apache Airflow, Snowflake"
        ),
        skills=["Python", "SQL", "Apache Airflow", "Snowflake"],
        experience=[
            WorkExperience(
                title="Freelance Data Engineer",
                company="Self-employed",
                duration_months=24,
                description=(
                    "Freelance and open-source data pipeline work using Python and Airflow. "
                    "Self-taught engineer with GitHub contributions."
                ),
            )
        ],
        education=[
            Education(
                degree="B.S. Computer Science",
                institution="State University",
                graduation_year=2018,
                field_of_study="Computer Science",
            )
        ],
        projects=[
            Project(
                name="airflow-contrib",
                description="Open-source contribution to Apache Airflow operators on GitHub.",
                technologies=["Python", "Airflow"],
            )
        ],
    )


@pytest.fixture
def simple_rubric() -> ParsedJobDescription:
    """Two-criterion rubric with known weights."""
    return ParsedJobDescription(
        job_title="Data Engineer",
        company="Acme Corp",
        requirements=[
            JobRequirement(
                criterion_id="REQ-01",
                description="Experience with Apache Airflow pipeline orchestration",
                weight=4,
                type=RequirementType.MUST_HAVE,
                evidence_type="work experience",
            ),
            JobRequirement(
                criterion_id="REQ-02",
                description="Production experience with Snowflake and SQL optimization",
                weight=2,
                type=RequirementType.MUST_HAVE,
                evidence_type="work experience",
            ),
        ],
    )


@pytest.fixture
def clean_original_result(candidate_with_raw_text: CandidateProfile) -> MatchResult:
    """A Matcher result where both scores are high with valid evidence."""
    return MatchResult(
        scores=[
            ScoreRecord(
                candidate_id=candidate_with_raw_text.candidate_id,
                criterion_id="REQ-01",
                score=0.90,
                evidence_quote="Architected 40+ Apache Airflow DAGs processing over 3B records daily into Snowflake.",
                confidence=0.95,
            ),
            ScoreRecord(
                candidate_id=candidate_with_raw_text.candidate_id,
                criterion_id="REQ-02",
                score=0.80,
                evidence_quote="Optimized SQL queries reducing execution time by 45%.",
                confidence=0.85,
            ),
        ]
    )


@pytest.fixture
def clean_anonymized_result() -> MatchResult:
    """A similar Matcher result for the anonymized profile (small delta)."""
    return MatchResult(
        scores=[
            ScoreRecord(
                candidate_id="elena_vasquez",
                criterion_id="REQ-01",
                score=0.88,
                evidence_quote="Architected 40+ Apache Airflow DAGs processing over 3B records daily into Snowflake.",
                confidence=0.90,
            ),
            ScoreRecord(
                candidate_id="elena_vasquez",
                criterion_id="REQ-02",
                score=0.78,
                evidence_quote="Optimized SQL queries reducing execution time by 45%.",
                confidence=0.85,
            ),
        ]
    )


@pytest.fixture
def all_citations_valid() -> CitationCheckBatch:
    return CitationCheckBatch(
        results=[
            CitationCheckResult(criterion_id="REQ-01", is_valid=True, reason="Quote directly demonstrates Airflow expertise."),
            CitationCheckResult(criterion_id="REQ-02", is_valid=True, reason="Quote demonstrates SQL optimization skills."),
        ]
    )


@pytest.fixture
def one_citation_invalid() -> CitationCheckBatch:
    return CitationCheckBatch(
        results=[
            CitationCheckResult(criterion_id="REQ-01", is_valid=True, reason="Valid Airflow quote."),
            CitationCheckResult(criterion_id="REQ-02", is_valid=False, reason="Quote mentions SQL but does not demonstrate optimization competency."),
        ]
    )


# ---------------------------------------------------------------------------
# Tests: CandidateProfile.anonymize() — raw_text scrubbing
# ---------------------------------------------------------------------------


def test_anonymize_removes_candidate_name_from_raw_text(
    candidate_with_raw_text: CandidateProfile,
):
    """Anonymization replaces candidate name in raw_text with [CANDIDATE]."""
    anon = candidate_with_raw_text.anonymize()
    assert "Elena Vasquez" not in anon.raw_text
    assert "[CANDIDATE]" in anon.raw_text


def test_anonymize_removes_institution_from_raw_text(
    candidate_with_raw_text: CandidateProfile,
):
    """Anonymization replaces institution name in raw_text with [INSTITUTION]."""
    anon = candidate_with_raw_text.anonymize()
    assert "University of Texas at Dallas" not in anon.raw_text
    assert "[INSTITUTION]" in anon.raw_text


def test_anonymize_removes_graduation_year_from_raw_text(
    candidate_with_raw_text: CandidateProfile,
):
    """Anonymization replaces graduation year in raw_text with [YEAR]."""
    anon = candidate_with_raw_text.anonymize()
    assert "2017" not in anon.raw_text
    assert "[YEAR]" in anon.raw_text


def test_anonymize_preserves_competency_evidence_in_raw_text(
    candidate_with_raw_text: CandidateProfile,
):
    """Anonymization preserves competency-relevant content in raw_text."""
    anon = candidate_with_raw_text.anonymize()
    # Skills and experience details remain
    assert "Apache Airflow" in anon.raw_text
    assert "Snowflake" in anon.raw_text
    assert "SQL" in anon.raw_text
    # Company name remains (companies are not identity context)
    assert "Crestline Data Systems" in anon.raw_text
    # Work evidence remains
    assert "Architected 40+ Apache Airflow DAGs" in anon.raw_text


def test_anonymize_preserves_structured_competency_fields(
    candidate_with_raw_text: CandidateProfile,
):
    """Anonymized structured fields preserve all competency data."""
    anon = candidate_with_raw_text.anonymize()
    assert anon.candidate_name is None
    assert anon.education[0].institution is None
    assert anon.education[0].graduation_year is None

    # Competency fields intact
    assert anon.skills == candidate_with_raw_text.skills
    assert anon.experience[0].title == "Senior Data Engineer"
    assert anon.experience[0].company == "Crestline Data Systems"
    assert anon.education[0].degree == "B.S. Computer Science"
    assert anon.education[0].field_of_study == "Computer Science"
    assert anon.projects[0].name == "airflow-pipeline"
    assert "Python" in anon.projects[0].technologies


def test_anonymize_does_not_modify_original(
    candidate_with_raw_text: CandidateProfile,
):
    """Anonymization returns a copy; original is unchanged."""
    original_raw = candidate_with_raw_text.raw_text
    _ = candidate_with_raw_text.anonymize()
    assert candidate_with_raw_text.raw_text == original_raw
    assert candidate_with_raw_text.candidate_name == "Elena Vasquez"


# ---------------------------------------------------------------------------
# Tests: Weighted aggregate
# ---------------------------------------------------------------------------


def test_weighted_aggregate_correct_calculation():
    """Weighted aggregate is sum(score*weight)/sum(weight)."""
    scores = [
        ScoreRecord(candidate_id="c1", criterion_id="REQ-01", score=0.9, evidence_quote="quote", confidence=0.8),
        ScoreRecord(candidate_id="c1", criterion_id="REQ-02", score=0.2, evidence_quote="", confidence=0.7),
    ]
    weights = {"REQ-01": 4, "REQ-02": 2}
    # (0.9*4 + 0.2*2) / (4+2) = (3.6 + 0.4) / 6 = 4.0/6 ≈ 0.6667
    result = _weighted_aggregate(scores, weights)
    assert result is not None
    assert abs(result - (0.9 * 4 + 0.2 * 2) / 6) < 1e-9


def test_weighted_aggregate_zero_weight_returns_none():
    """Zero total weight returns None, not a fabricated score."""
    scores = [
        ScoreRecord(candidate_id="c1", criterion_id="REQ-01", score=0.9, evidence_quote="quote", confidence=0.8),
    ]
    weights = {"REQ-01": 0}
    assert _weighted_aggregate(scores, weights) is None


def test_weighted_aggregate_missing_criterion_treated_as_zero_weight():
    """Criteria not in weights dict are treated as weight=0."""
    scores = [
        ScoreRecord(candidate_id="c1", criterion_id="UNKNOWN", score=0.9, evidence_quote="quote", confidence=0.8),
    ]
    weights = {}
    assert _weighted_aggregate(scores, weights) is None


# ---------------------------------------------------------------------------
# Tests: Full audit — clean candidate (no flags)
# ---------------------------------------------------------------------------


def test_clean_candidate_not_flagged(
    candidate_with_raw_text: CandidateProfile,
    simple_rubric: ParsedJobDescription,
    clean_original_result: MatchResult,
    clean_anonymized_result: MatchResult,
    all_citations_valid: CitationCheckBatch,
):
    """A candidate with small delta and valid citations is not flagged."""
    auditor = FairnessAuditor()

    with patch.object(auditor._matcher, "match", return_value=clean_anonymized_result), \
         patch("app.agents.fairness_auditor.check_citations", return_value=all_citations_valid):

        record = auditor.audit(candidate_with_raw_text, simple_rubric, clean_original_result)

    assert isinstance(record, AuditRecord)
    assert record.candidate_id == "elena_vasquez"
    assert record.flagged is False
    assert record.flagged_reason == ""
    assert record.citation_valid is True
    assert record.repair_applied is False


# ---------------------------------------------------------------------------
# Tests: Check A — Counterfactual re-scoring
# ---------------------------------------------------------------------------


def test_counterfactual_scoring_uses_anonymized_profile(
    candidate_with_raw_text: CandidateProfile,
    simple_rubric: ParsedJobDescription,
    clean_original_result: MatchResult,
    clean_anonymized_result: MatchResult,
    all_citations_valid: CitationCheckBatch,
):
    """Matcher.match is called with the anonymized profile, not the original."""
    auditor = FairnessAuditor()

    with patch.object(auditor._matcher, "match", return_value=clean_anonymized_result) as mock_match, \
         patch("app.agents.fairness_auditor.check_citations", return_value=all_citations_valid):

        auditor.audit(candidate_with_raw_text, simple_rubric, clean_original_result)

    mock_match.assert_called_once()
    passed_candidate = mock_match.call_args[1]["candidate"]
    # anonymized profile should NOT contain the candidate's name
    assert passed_candidate.candidate_name is None
    assert "Elena Vasquez" not in passed_candidate.raw_text


def test_aggregate_scores_computed_correctly(
    candidate_with_raw_text: CandidateProfile,
    simple_rubric: ParsedJobDescription,
    clean_original_result: MatchResult,
    clean_anonymized_result: MatchResult,
    all_citations_valid: CitationCheckBatch,
):
    """original_score and anonymized_score are weighted aggregates."""
    auditor = FairnessAuditor()

    with patch.object(auditor._matcher, "match", return_value=clean_anonymized_result), \
         patch("app.agents.fairness_auditor.check_citations", return_value=all_citations_valid):

        record = auditor.audit(candidate_with_raw_text, simple_rubric, clean_original_result)

    # REQ-01 weight=4, REQ-02 weight=2
    expected_original = (0.90 * 4 + 0.80 * 2) / 6
    expected_anonymized = (0.88 * 4 + 0.78 * 2) / 6
    assert abs(record.original_score - expected_original) < 1e-9
    assert abs(record.anonymized_score - expected_anonymized) < 1e-9
    assert record.delta is not None
    assert abs(record.delta - abs(expected_original - expected_anonymized)) < 1e-9


def test_zero_total_weight_produces_none_scores(
    candidate_with_raw_text: CandidateProfile,
    all_citations_valid: CitationCheckBatch,
):
    """When all rubric criteria have weight=0, scores are None — not fabricated."""
    zero_weight_rubric = ParsedJobDescription(
        job_title="Test",
        requirements=[
            JobRequirement(
                criterion_id="REQ-01",
                description="Some criterion",
                weight=1,  # Pydantic ge=1 enforced, so use weight=1 but override via dict
                type=RequirementType.MUST_HAVE,
                evidence_type="work experience",
            )
        ],
    )
    # Manually create a rubric-like object that would yield zero weight
    # We test _weighted_aggregate directly (already covered) and the None propagation
    original_result = MatchResult(
        scores=[
            ScoreRecord(
                candidate_id="elena_vasquez",
                criterion_id="REQ-01",
                score=0.9,
                evidence_quote="Architected 40+ Apache Airflow DAGs",
                confidence=0.9,
            )
        ]
    )
    anon_result = MatchResult(
        scores=[
            ScoreRecord(
                candidate_id="elena_vasquez",
                criterion_id="REQ-01",
                score=0.88,
                evidence_quote="Architected 40+ Apache Airflow DAGs",
                confidence=0.88,
            )
        ]
    )
    auditor = FairnessAuditor()

    with patch.object(auditor._matcher, "match", return_value=anon_result), \
         patch("app.agents.fairness_auditor.check_citations", return_value=all_citations_valid):
        # Override weights internally by patching _weighted_aggregate to return None
        with patch("app.agents.fairness_auditor._weighted_aggregate", return_value=None):
            record = auditor.audit(candidate_with_raw_text, zero_weight_rubric, original_result)

    assert record.original_score is None
    assert record.anonymized_score is None
    assert record.delta is None


def test_large_delta_is_detected(
    candidate_with_raw_text: CandidateProfile,
    simple_rubric: ParsedJobDescription,
    clean_original_result: MatchResult,
    all_citations_valid: CitationCheckBatch,
):
    """A delta >= COUNTERFACTUAL_DELTA_THRESHOLD is detected and causes flagging."""
    # Make anonymized scores substantially lower → large delta
    large_delta_result = MatchResult(
        scores=[
            ScoreRecord(
                candidate_id="elena_vasquez",
                criterion_id="REQ-01",
                score=0.55,  # original is 0.90; delta per criterion
                evidence_quote="Architected 40+ Apache Airflow DAGs processing over 3B records daily into Snowflake.",
                confidence=0.8,
            ),
            ScoreRecord(
                candidate_id="elena_vasquez",
                criterion_id="REQ-02",
                score=0.40,  # original is 0.80
                evidence_quote="Optimized SQL queries reducing execution time by 45%.",
                confidence=0.7,
            ),
        ]
    )

    auditor = FairnessAuditor()
    with patch.object(auditor._matcher, "match", return_value=large_delta_result), \
         patch("app.agents.fairness_auditor.check_citations", return_value=all_citations_valid):

        record = auditor.audit(candidate_with_raw_text, simple_rubric, clean_original_result)

    # original: (0.90*4 + 0.80*2)/6 = 0.8667
    # anonymized: (0.55*4 + 0.40*2)/6 = 0.5000
    # delta = 0.3667 >> threshold of 0.15
    assert record.delta is not None
    assert record.delta >= COUNTERFACTUAL_DELTA_THRESHOLD
    assert record.flagged is True
    assert "delta" in record.flagged_reason.lower() or "counterfactual" in record.flagged_reason.lower()


def test_small_delta_is_not_flagged_as_large(
    candidate_with_raw_text: CandidateProfile,
    simple_rubric: ParsedJobDescription,
    clean_original_result: MatchResult,
    clean_anonymized_result: MatchResult,
    all_citations_valid: CitationCheckBatch,
):
    """A delta below COUNTERFACTUAL_DELTA_THRESHOLD does not trigger a large-delta flag."""
    auditor = FairnessAuditor()
    with patch.object(auditor._matcher, "match", return_value=clean_anonymized_result), \
         patch("app.agents.fairness_auditor.check_citations", return_value=all_citations_valid):

        record = auditor.audit(candidate_with_raw_text, simple_rubric, clean_original_result)

    # original ≈ 0.8667, anonymized ≈ 0.8467, delta ≈ 0.02 < 0.15
    assert record.delta is not None
    assert record.delta < COUNTERFACTUAL_DELTA_THRESHOLD
    # Should not be flagged for delta alone
    assert "Large counterfactual" not in record.flagged_reason


# ---------------------------------------------------------------------------
# Tests: Check B — Citation validity
# ---------------------------------------------------------------------------


def test_citation_validity_passes_when_all_valid(
    candidate_with_raw_text: CandidateProfile,
    simple_rubric: ParsedJobDescription,
    clean_original_result: MatchResult,
    clean_anonymized_result: MatchResult,
    all_citations_valid: CitationCheckBatch,
):
    """citation_valid=True when all applicable citations pass."""
    auditor = FairnessAuditor()
    with patch.object(auditor._matcher, "match", return_value=clean_anonymized_result), \
         patch("app.agents.fairness_auditor.check_citations", return_value=all_citations_valid):

        record = auditor.audit(candidate_with_raw_text, simple_rubric, clean_original_result)

    assert record.citation_valid is True


def test_citation_validity_fails_when_any_citation_invalid(
    candidate_with_raw_text: CandidateProfile,
    simple_rubric: ParsedJobDescription,
    clean_original_result: MatchResult,
    clean_anonymized_result: MatchResult,
    one_citation_invalid: CitationCheckBatch,
):
    """citation_valid=False when at least one citation fails."""
    auditor = FairnessAuditor()
    with patch.object(auditor._matcher, "match", return_value=clean_anonymized_result), \
         patch("app.agents.fairness_auditor.check_citations", return_value=one_citation_invalid):

        record = auditor.audit(candidate_with_raw_text, simple_rubric, clean_original_result)

    assert record.citation_valid is False
    assert record.flagged is True


def test_failed_citation_reason_in_flagged_reason(
    candidate_with_raw_text: CandidateProfile,
    simple_rubric: ParsedJobDescription,
    clean_original_result: MatchResult,
    clean_anonymized_result: MatchResult,
    one_citation_invalid: CitationCheckBatch,
):
    """flagged_reason identifies the failed criterion ID and reason."""
    auditor = FairnessAuditor()
    with patch.object(auditor._matcher, "match", return_value=clean_anonymized_result), \
         patch("app.agents.fairness_auditor.check_citations", return_value=one_citation_invalid):

        record = auditor.audit(candidate_with_raw_text, simple_rubric, clean_original_result)

    assert "REQ-02" in record.flagged_reason
    assert "SQL" in record.flagged_reason or "optimization" in record.flagged_reason.lower() or "citation" in record.flagged_reason.lower()


def test_low_score_without_evidence_does_not_cause_citation_failure(
    candidate_with_raw_text: CandidateProfile,
    simple_rubric: ParsedJobDescription,
    all_citations_valid: CitationCheckBatch,
    clean_anonymized_result: MatchResult,
):
    """A score <= 0.3 with no evidence quote must NOT cause citation_valid=False."""
    # Result where REQ-01 is high (valid evidence) and REQ-02 is low (no evidence required)
    low_score_result = MatchResult(
        scores=[
            ScoreRecord(
                candidate_id="elena_vasquez",
                criterion_id="REQ-01",
                score=0.85,
                evidence_quote="Architected 40+ Apache Airflow DAGs processing over 3B records daily into Snowflake.",
                confidence=0.9,
            ),
            ScoreRecord(
                candidate_id="elena_vasquez",
                criterion_id="REQ-02",
                score=0.2,  # <= 0.3, empty evidence is valid
                evidence_quote="",
                confidence=0.5,
            ),
        ]
    )

    auditor = FairnessAuditor()
    with patch.object(auditor._matcher, "match", return_value=clean_anonymized_result), \
         patch("app.agents.fairness_auditor.check_citations", return_value=all_citations_valid):

        record = auditor.audit(candidate_with_raw_text, simple_rubric, low_score_result)

    # citation_valid should be True — the low-score record is not evaluated for citations
    assert record.citation_valid is True


# ---------------------------------------------------------------------------
# Tests: Check C — Non-traditional evidence
# ---------------------------------------------------------------------------


def test_non_traditional_evidence_detected_for_under_scored_criterion(
    candidate_with_non_traditional: CandidateProfile,
    simple_rubric: ParsedJobDescription,
    all_citations_valid: CitationCheckBatch,
):
    """Non-traditional evidence is flagged when relevant and original score is low."""
    # Low original scores for both criteria
    low_score_result = MatchResult(
        scores=[
            ScoreRecord(
                candidate_id="alex_freelance",
                criterion_id="REQ-01",
                score=0.3,  # <= NON_TRADITIONAL_SCORE_THRESHOLD (0.5)
                evidence_quote="",
                confidence=0.5,
            ),
            ScoreRecord(
                candidate_id="alex_freelance",
                criterion_id="REQ-02",
                score=0.2,
                evidence_quote="",
                confidence=0.4,
            ),
        ]
    )
    anon_result = MatchResult(
        scores=[
            ScoreRecord(
                candidate_id="alex_freelance",
                criterion_id="REQ-01",
                score=0.28,
                evidence_quote="",
                confidence=0.5,
            ),
            ScoreRecord(
                candidate_id="alex_freelance",
                criterion_id="REQ-02",
                score=0.18,
                evidence_quote="",
                confidence=0.4,
            ),
        ]
    )

    auditor = FairnessAuditor()
    with patch.object(auditor._matcher, "match", return_value=anon_result), \
         patch("app.agents.fairness_auditor.check_citations", return_value=all_citations_valid):

        record = auditor.audit(candidate_with_non_traditional, simple_rubric, low_score_result)

    assert record.non_traditional_evidence_found is True
    assert record.flagged is True
    assert "non-traditional" in record.flagged_reason.lower()


def test_non_traditional_evidence_does_not_flag_high_scoring_criterion(
    candidate_with_non_traditional: CandidateProfile,
    simple_rubric: ParsedJobDescription,
    all_citations_valid: CitationCheckBatch,
):
    """Non-traditional evidence does NOT flag criteria where original score is already high."""
    # High original scores → non-traditional evidence should not cause a flag
    high_score_result = MatchResult(
        scores=[
            ScoreRecord(
                candidate_id="alex_freelance",
                criterion_id="REQ-01",
                score=0.85,  # > NON_TRADITIONAL_SCORE_THRESHOLD (0.5)
                evidence_quote="Contributed to open-source Airflow plugins on GitHub.",
                confidence=0.85,
            ),
            ScoreRecord(
                candidate_id="alex_freelance",
                criterion_id="REQ-02",
                score=0.75,  # > NON_TRADITIONAL_SCORE_THRESHOLD (0.5)
                evidence_quote="Freelance and open-source data pipeline work using Python and Airflow.",
                confidence=0.75,
            ),
        ]
    )
    anon_result = MatchResult(
        scores=[
            ScoreRecord(
                candidate_id="alex_freelance",
                criterion_id="REQ-01",
                score=0.83,
                evidence_quote="Contributed to open-source Airflow plugins on GitHub.",
                confidence=0.82,
            ),
            ScoreRecord(
                candidate_id="alex_freelance",
                criterion_id="REQ-02",
                score=0.73,
                evidence_quote="Freelance and open-source data pipeline work using Python and Airflow.",
                confidence=0.73,
            ),
        ]
    )

    auditor = FairnessAuditor()
    with patch.object(auditor._matcher, "match", return_value=anon_result), \
         patch("app.agents.fairness_auditor.check_citations", return_value=all_citations_valid):

        record = auditor.audit(candidate_with_non_traditional, simple_rubric, high_score_result)

    # Non-traditional evidence exists but scores are above threshold → no flag from Check C
    assert record.non_traditional_evidence_found is False


def test_non_traditional_evidence_does_not_automatically_flag_candidate(
    candidate_with_non_traditional: CandidateProfile,
    simple_rubric: ParsedJobDescription,
    all_citations_valid: CitationCheckBatch,
):
    """Mere presence of non-traditional evidence must NOT cause a flag when scores are high."""
    high_score_result = MatchResult(
        scores=[
            ScoreRecord(
                candidate_id="alex_freelance",
                criterion_id="REQ-01",
                score=0.90,
                evidence_quote="Contributed to open-source Airflow plugins on GitHub.",
                confidence=0.9,
            ),
            ScoreRecord(
                candidate_id="alex_freelance",
                criterion_id="REQ-02",
                score=0.80,
                evidence_quote="Freelance and open-source data pipeline work using Python and Airflow.",
                confidence=0.8,
            ),
        ]
    )
    anon_result = MatchResult(
        scores=[
            ScoreRecord(
                candidate_id="alex_freelance",
                criterion_id="REQ-01",
                score=0.88,
                evidence_quote="Contributed to open-source Airflow plugins on GitHub.",
                confidence=0.87,
            ),
            ScoreRecord(
                candidate_id="alex_freelance",
                criterion_id="REQ-02",
                score=0.78,
                evidence_quote="Freelance and open-source data pipeline work using Python and Airflow.",
                confidence=0.77,
            ),
        ]
    )

    auditor = FairnessAuditor()
    with patch.object(auditor._matcher, "match", return_value=anon_result), \
         patch("app.agents.fairness_auditor.check_citations", return_value=all_citations_valid):

        record = auditor.audit(candidate_with_non_traditional, simple_rubric, high_score_result)

    assert record.flagged is False
    assert record.non_traditional_evidence_found is False


# ---------------------------------------------------------------------------
# Tests: Flagging — large delta
# ---------------------------------------------------------------------------


def test_candidate_with_large_delta_is_flagged(
    candidate_with_raw_text: CandidateProfile,
    simple_rubric: ParsedJobDescription,
    clean_original_result: MatchResult,
    all_citations_valid: CitationCheckBatch,
):
    """A large counterfactual delta flags the candidate."""
    large_drop_result = MatchResult(
        scores=[
            ScoreRecord(
                candidate_id="elena_vasquez",
                criterion_id="REQ-01",
                score=0.45,
                evidence_quote="Architected 40+ Apache Airflow DAGs processing over 3B records daily into Snowflake.",
                confidence=0.7,
            ),
            ScoreRecord(
                candidate_id="elena_vasquez",
                criterion_id="REQ-02",
                score=0.35,
                evidence_quote="Optimized SQL queries reducing execution time by 45%.",
                confidence=0.65,
            ),
        ]
    )

    auditor = FairnessAuditor()
    with patch.object(auditor._matcher, "match", return_value=large_drop_result), \
         patch("app.agents.fairness_auditor.check_citations", return_value=all_citations_valid):

        record = auditor.audit(candidate_with_raw_text, simple_rubric, clean_original_result)

    assert record.flagged is True
    assert record.delta >= COUNTERFACTUAL_DELTA_THRESHOLD


# ---------------------------------------------------------------------------
# Tests: Flagging — citation failure
# ---------------------------------------------------------------------------


def test_candidate_with_failed_citation_is_flagged(
    candidate_with_raw_text: CandidateProfile,
    simple_rubric: ParsedJobDescription,
    clean_original_result: MatchResult,
    clean_anonymized_result: MatchResult,
    one_citation_invalid: CitationCheckBatch,
):
    """A citation failure flags the candidate."""
    auditor = FairnessAuditor()
    with patch.object(auditor._matcher, "match", return_value=clean_anonymized_result), \
         patch("app.agents.fairness_auditor.check_citations", return_value=one_citation_invalid):

        record = auditor.audit(candidate_with_raw_text, simple_rubric, clean_original_result)

    assert record.flagged is True
    assert record.citation_valid is False


# ---------------------------------------------------------------------------
# Tests: Option A fallback
# ---------------------------------------------------------------------------


def _make_large_delta_anon_result() -> MatchResult:
    """Helper: anonymized result producing a large delta."""
    return MatchResult(
        scores=[
            ScoreRecord(
                candidate_id="elena_vasquez",
                criterion_id="REQ-01",
                score=0.45,
                evidence_quote="Architected 40+ Apache Airflow DAGs processing over 3B records daily into Snowflake.",
                confidence=0.6,
            ),
            ScoreRecord(
                candidate_id="elena_vasquez",
                criterion_id="REQ-02",
                score=0.30,
                evidence_quote="",
                confidence=0.6,
            ),
        ]
    )


def test_option_a_fallback_applied_when_large_delta_and_citation_failure(
    candidate_with_raw_text: CandidateProfile,
    simple_rubric: ParsedJobDescription,
    clean_original_result: MatchResult,
    one_citation_invalid: CitationCheckBatch,
):
    """Option A fallback is triggered when BOTH large delta AND citation failure occur."""
    large_drop = _make_large_delta_anon_result()

    auditor = FairnessAuditor()
    with patch.object(auditor._matcher, "match", return_value=large_drop), \
         patch("app.agents.fairness_auditor.check_citations", return_value=one_citation_invalid):

        record = auditor.audit(candidate_with_raw_text, simple_rubric, clean_original_result)

    assert record.repair_applied is True
    assert "fallback" in record.flagged_reason.lower() or "option a" in record.flagged_reason.lower()


def test_option_a_fallback_adopts_anonymized_score(
    candidate_with_raw_text: CandidateProfile,
    simple_rubric: ParsedJobDescription,
    clean_original_result: MatchResult,
    one_citation_invalid: CitationCheckBatch,
):
    """The fallback adopts the already-computed anonymized aggregate score."""
    large_drop = _make_large_delta_anon_result()

    auditor = FairnessAuditor()
    with patch.object(auditor._matcher, "match", return_value=large_drop), \
         patch("app.agents.fairness_auditor.check_citations", return_value=one_citation_invalid):

        record = auditor.audit(candidate_with_raw_text, simple_rubric, clean_original_result)

    # anonymized_score should equal the weighted aggregate of large_drop scores
    # REQ-01: 0.45*4=1.8, REQ-02: 0.30*2=0.60, total weight=6 → 2.4/6=0.40
    expected_anon = (0.45 * 4 + 0.30 * 2) / 6
    assert record.anonymized_score is not None
    assert abs(record.anonymized_score - expected_anon) < 1e-9


def test_option_a_fallback_does_not_make_additional_matcher_call(
    candidate_with_raw_text: CandidateProfile,
    simple_rubric: ParsedJobDescription,
    clean_original_result: MatchResult,
    one_citation_invalid: CitationCheckBatch,
):
    """Option A fallback must NOT make a second Matcher/Gemini call."""
    large_drop = _make_large_delta_anon_result()

    auditor = FairnessAuditor()
    with patch.object(auditor._matcher, "match", return_value=large_drop) as mock_match, \
         patch("app.agents.fairness_auditor.check_citations", return_value=one_citation_invalid):

        record = auditor.audit(candidate_with_raw_text, simple_rubric, clean_original_result)

    # Matcher.match should only have been called ONCE (Check A counterfactual re-score)
    assert mock_match.call_count == 1
    assert record.repair_applied is True


def test_option_a_fallback_preserves_citation_valid_and_non_traditional(
    candidate_with_raw_text: CandidateProfile,
    simple_rubric: ParsedJobDescription,
    clean_original_result: MatchResult,
    one_citation_invalid: CitationCheckBatch,
):
    """After Option A fallback, citation_valid and non_traditional_evidence_found are unchanged."""
    large_drop = _make_large_delta_anon_result()

    auditor = FairnessAuditor()
    with patch.object(auditor._matcher, "match", return_value=large_drop), \
         patch("app.agents.fairness_auditor.check_citations", return_value=one_citation_invalid):

        record = auditor.audit(candidate_with_raw_text, simple_rubric, clean_original_result)

    # citation_valid remains False (from Check B), not silently repaired
    assert record.citation_valid is False
    # non_traditional_evidence_found was determined before fallback and should be unchanged
    # (for this candidate/rubric pair, may be True or False, just must exist)
    assert isinstance(record.non_traditional_evidence_found, bool)


def test_option_a_fallback_not_applied_with_large_delta_only(
    candidate_with_raw_text: CandidateProfile,
    simple_rubric: ParsedJobDescription,
    clean_original_result: MatchResult,
    all_citations_valid: CitationCheckBatch,
):
    """Option A fallback is NOT triggered when there is large delta but all citations pass."""
    large_drop = _make_large_delta_anon_result()

    auditor = FairnessAuditor()
    with patch.object(auditor._matcher, "match", return_value=large_drop), \
         patch("app.agents.fairness_auditor.check_citations", return_value=all_citations_valid):

        record = auditor.audit(candidate_with_raw_text, simple_rubric, clean_original_result)

    assert record.repair_applied is False


def test_option_a_fallback_not_applied_with_citation_failure_only(
    candidate_with_raw_text: CandidateProfile,
    simple_rubric: ParsedJobDescription,
    clean_original_result: MatchResult,
    clean_anonymized_result: MatchResult,
    one_citation_invalid: CitationCheckBatch,
):
    """Option A fallback is NOT triggered when citation fails but delta is small."""
    auditor = FairnessAuditor()
    with patch.object(auditor._matcher, "match", return_value=clean_anonymized_result), \
         patch("app.agents.fairness_auditor.check_citations", return_value=one_citation_invalid):

        record = auditor.audit(candidate_with_raw_text, simple_rubric, clean_original_result)

    assert record.repair_applied is False


def test_no_retry_loop_occurs(
    candidate_with_raw_text: CandidateProfile,
    simple_rubric: ParsedJobDescription,
    clean_original_result: MatchResult,
    one_citation_invalid: CitationCheckBatch,
):
    """Regardless of conditions, Matcher.match is called at most once."""
    large_drop = _make_large_delta_anon_result()

    auditor = FairnessAuditor()
    with patch.object(auditor._matcher, "match", return_value=large_drop) as mock_match, \
         patch("app.agents.fairness_auditor.check_citations", return_value=one_citation_invalid):

        auditor.audit(candidate_with_raw_text, simple_rubric, clean_original_result)

    # Exactly 1 call regardless of fallback
    assert mock_match.call_count == 1


# ---------------------------------------------------------------------------
# Tests: audit_candidate convenience wrapper
# ---------------------------------------------------------------------------


def test_audit_candidate_convenience_wrapper(
    candidate_with_raw_text: CandidateProfile,
    simple_rubric: ParsedJobDescription,
    clean_original_result: MatchResult,
    clean_anonymized_result: MatchResult,
    all_citations_valid: CitationCheckBatch,
):
    """audit_candidate() convenience function works correctly."""
    with patch("app.agents.fairness_auditor.Matcher") as MockMatcher, \
         patch("app.agents.fairness_auditor.check_citations", return_value=all_citations_valid):

        mock_instance = MockMatcher.return_value
        mock_instance.match.return_value = clean_anonymized_result

        record = audit_candidate(
            candidate=candidate_with_raw_text,
            rubric=simple_rubric,
            original_result=clean_original_result,
        )

    assert isinstance(record, AuditRecord)
    assert record.candidate_id == "elena_vasquez"
