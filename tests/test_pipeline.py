"""Comprehensive offline tests for the LangGraph orchestration pipeline.

All tests run fully offline without live Gemini calls.
"""

from unittest.mock import MagicMock, call, patch
import pytest

from app.models.audit import AuditRecord
from app.models.candidate import CandidateProfile, WorkExperience
from app.models.job_description import (
    JobRequirement,
    ParsedJobDescription,
    RequirementType,
)
from app.models.result import (
    CandidateInput,
    CandidateResult,
    FailedCandidate,
    PipelineResult,
)
from app.models.score import MatchResult, ScoreRecord
from app.pipeline.graph import (
    candidate_graph,
    create_candidate_graph,
    fairness_auditor_node,
    matcher_node,
    resume_parser_node,
    run_full_pipeline,
    run_pipeline,
)
from app.pipeline.state import CandidateGraphState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_rubric() -> ParsedJobDescription:
    """Sample parsed job rubric."""
    return ParsedJobDescription(
        job_title="Backend Engineer",
        company="HireFair Labs",
        requirements=[
            JobRequirement(
                criterion_id="REQ-01",
                description="Experience building REST APIs with Python",
                weight=5,
                type=RequirementType.MUST_HAVE,
                evidence_type="work experience",
            ),
            JobRequirement(
                criterion_id="REQ-02",
                description="Experience with SQL databases",
                weight=3,
                type=RequirementType.NICE_TO_HAVE,
                evidence_type="work experience",
            ),
        ],
    )


def make_profile(candidate_id: str = "cand_1") -> CandidateProfile:
    """Factory helper to create a mock CandidateProfile."""
    return CandidateProfile(
        candidate_id=candidate_id,
        candidate_name="Jane Doe",
        raw_text=f"Resume of {candidate_id}. Built APIs in Python and SQL databases.",
        skills=["Python", "SQL"],
        experience=[
            WorkExperience(
                title="Backend Developer",
                company="Acme Corp",
                duration_months=36,
                description="Built high-throughput APIs using Python.",
            )
        ],
    )


def make_match_result(candidate_id: str = "cand_1") -> MatchResult:
    """Factory helper to create a mock MatchResult."""
    return MatchResult(
        scores=[
            ScoreRecord(
                candidate_id=candidate_id,
                criterion_id="REQ-01",
                score=0.9,
                evidence_quote="Built high-throughput APIs using Python.",
                confidence=0.95,
            ),
            ScoreRecord(
                candidate_id=candidate_id,
                criterion_id="REQ-02",
                score=0.7,
                evidence_quote="Built APIs in Python and SQL databases.",
                confidence=0.85,
            ),
        ]
    )


def make_audit_record(candidate_id: str = "cand_1") -> AuditRecord:
    """Factory helper to create a mock AuditRecord."""
    return AuditRecord(
        candidate_id=candidate_id,
        original_score=0.825,
        anonymized_score=0.825,
        delta=0.0,
        citation_valid=True,
        flagged=False,
        flagged_reason="",
        repair_applied=False,
    )


# ---------------------------------------------------------------------------
# Test 1: Single-candidate node order
# ResumeParser -> Matcher -> FairnessAuditor
# ---------------------------------------------------------------------------


def test_single_candidate_node_order(sample_rubric: ParsedJobDescription):
    """Verify execution order is strictly ResumeParser -> Matcher -> FairnessAuditor."""
    call_order = []

    profile = make_profile("cand_order")
    match_result = make_match_result("cand_order")
    audit_rec = make_audit_record("cand_order")

    def mock_parse(*args, **kwargs):
        call_order.append("ResumeParser")
        return profile

    def mock_match(*args, **kwargs):
        call_order.append("Matcher")
        return match_result

    def mock_audit(*args, **kwargs):
        call_order.append("FairnessAuditor")
        return audit_rec

    with patch("app.pipeline.graph.ResumeParser") as mock_rp_cls, \
         patch("app.pipeline.graph.Matcher") as mock_m_cls, \
         patch("app.pipeline.graph.FairnessAuditor") as mock_fa_cls:

        mock_rp_cls.return_value.parse.side_effect = mock_parse
        mock_m_cls.return_value.match.side_effect = mock_match
        mock_fa_cls.return_value.audit.side_effect = mock_audit

        initial_state: CandidateGraphState = {
            "candidate_id": "cand_order",
            "resume_text": "Sample resume text",
            "rubric": sample_rubric,
        }

        graph = create_candidate_graph()
        final_state = graph.invoke(initial_state)

        assert call_order == ["ResumeParser", "Matcher", "FairnessAuditor"]
        assert final_state["candidate_profile"] == profile
        assert final_state["match_result"] == match_result
        assert final_state["audit_record"] == audit_rec


# ---------------------------------------------------------------------------
# Test 2: State accumulation across nodes
# ---------------------------------------------------------------------------


def test_state_accumulation(sample_rubric: ParsedJobDescription):
    """Verify state accumulates correctly and reaches final CandidateResult."""
    profile = make_profile("cand_accum")
    match_res = make_match_result("cand_accum")
    audit_rec = make_audit_record("cand_accum")

    matcher_received = {}
    auditor_received = {}

    def mock_match(candidate, rubric):
        matcher_received["candidate"] = candidate
        matcher_received["rubric"] = rubric
        return match_res

    def mock_audit(candidate, rubric, original_result):
        auditor_received["candidate"] = candidate
        auditor_received["rubric"] = rubric
        auditor_received["original_result"] = original_result
        return audit_rec

    with patch("app.pipeline.graph.ResumeParser") as mock_rp_cls, \
         patch("app.pipeline.graph.Matcher") as mock_m_cls, \
         patch("app.pipeline.graph.FairnessAuditor") as mock_fa_cls:

        mock_rp_cls.return_value.parse.return_value = profile
        mock_m_cls.return_value.match.side_effect = mock_match
        mock_fa_cls.return_value.audit.side_effect = mock_audit

        candidates = [
            {"candidate_id": "cand_accum", "resume_text": "Accumulation resume text"}
        ]
        result = run_pipeline(rubric=sample_rubric, candidates=candidates)

        # Verify rubric remained intact through Matcher and Auditor
        assert matcher_received["rubric"] == sample_rubric
        assert auditor_received["rubric"] == sample_rubric

        # Verify CandidateProfile reached Matcher and Auditor
        assert matcher_received["candidate"] == profile
        assert auditor_received["candidate"] == profile

        # Verify MatchResult reached Auditor
        assert auditor_received["original_result"] == match_res

        # Verify final result contains the unwrapped scores and audit_record
        assert len(result.results) == 1
        cand_result = result.results[0]
        assert cand_result.candidate_id == "cand_accum"
        assert cand_result.scores == match_res.scores
        assert cand_result.audit_record == audit_rec


# ---------------------------------------------------------------------------
# Test 3: JD Parser called exactly once per batch
# ---------------------------------------------------------------------------


def test_jd_parser_called_exactly_once(sample_rubric: ParsedJobDescription):
    """Verify run_full_pipeline calls JDParser once, and run_pipeline does not call it."""
    with patch("app.pipeline.graph.JDParser") as mock_jd_cls, \
         patch("app.pipeline.graph.ResumeParser") as mock_rp_cls, \
         patch("app.pipeline.graph.Matcher") as mock_m_cls, \
         patch("app.pipeline.graph.FairnessAuditor") as mock_fa_cls:

        mock_jd_parser = mock_jd_cls.return_value
        mock_jd_parser.parse.return_value = sample_rubric

        mock_rp_cls.return_value.parse.side_effect = lambda resume_text, candidate_id=None: make_profile(candidate_id or "c")
        mock_m_cls.return_value.match.side_effect = lambda candidate, rubric: make_match_result(candidate.candidate_id)
        mock_fa_cls.return_value.audit.side_effect = lambda candidate, rubric, original_result: make_audit_record(candidate.candidate_id)

        candidates = [
            {"candidate_id": "c1", "resume_text": "Resume 1"},
            {"candidate_id": "c2", "resume_text": "Resume 2"},
            {"candidate_id": "c3", "resume_text": "Resume 3"},
        ]

        full_result = run_full_pipeline(
            jd_text="Looking for Backend Engineer with Python",
            candidates=candidates,
        )

        # JDParser was called exactly once for all 3 candidates
        assert mock_jd_parser.parse.call_count == 1
        assert len(full_result.results) == 3

        # Direct run_pipeline call must NOT call JDParser
        mock_jd_parser.reset_mock()
        sub_result = run_pipeline(rubric=sample_rubric, candidates=candidates)
        assert mock_jd_parser.parse.call_count == 0
        assert len(sub_result.results) == 3


# ---------------------------------------------------------------------------
# Test 4: Strictly sequential candidate processing
# ---------------------------------------------------------------------------


def test_sequential_candidate_processing(sample_rubric: ParsedJobDescription):
    """Verify candidates are processed strictly in input order sequentially."""
    processing_events = []

    def mock_parse(resume_text, candidate_id=None):
        processing_events.append((candidate_id, "start"))
        return make_profile(candidate_id)

    def mock_audit(candidate, rubric, original_result):
        processing_events.append((candidate.candidate_id, "end"))
        return make_audit_record(candidate.candidate_id)

    with patch("app.pipeline.graph.ResumeParser") as mock_rp_cls, \
         patch("app.pipeline.graph.Matcher") as mock_m_cls, \
         patch("app.pipeline.graph.FairnessAuditor") as mock_fa_cls:

        mock_rp_cls.return_value.parse.side_effect = mock_parse
        mock_m_cls.return_value.match.side_effect = lambda candidate, rubric: make_match_result(candidate.candidate_id)
        mock_fa_cls.return_value.audit.side_effect = mock_audit

        candidate_ids = ["cand_A", "cand_B", "cand_C", "cand_D"]
        candidates = [
            {"candidate_id": cid, "resume_text": f"Resume of {cid}"}
            for cid in candidate_ids
        ]

        result = run_pipeline(rubric=sample_rubric, candidates=candidates)

        assert [r.candidate_id for r in result.results] == candidate_ids

        # Verify sequential non-overlapping execution
        expected_events = [
            ("cand_A", "start"), ("cand_A", "end"),
            ("cand_B", "start"), ("cand_B", "end"),
            ("cand_C", "start"), ("cand_C", "end"),
            ("cand_D", "start"), ("cand_D", "end"),
        ]
        assert processing_events == expected_events


# ---------------------------------------------------------------------------
# Test 5: Failure isolation across candidates
# ---------------------------------------------------------------------------


def test_failure_isolation(sample_rubric: ParsedJobDescription):
    """Verify that a failure in one candidate does not terminate the batch and records reason."""
    def mock_parse(resume_text, candidate_id=None):
        if candidate_id == "cand_fail":
            raise ValueError("Corrupted PDF text: unable to parse resume")
        return make_profile(candidate_id)

    with patch("app.pipeline.graph.ResumeParser") as mock_rp_cls, \
         patch("app.pipeline.graph.Matcher") as mock_m_cls, \
         patch("app.pipeline.graph.FairnessAuditor") as mock_fa_cls:

        mock_rp_cls.return_value.parse.side_effect = mock_parse
        mock_m_cls.return_value.match.side_effect = lambda candidate, rubric: make_match_result(candidate.candidate_id)
        mock_fa_cls.return_value.audit.side_effect = lambda candidate, rubric, original_result: make_audit_record(candidate.candidate_id)

        candidates = [
            {"candidate_id": "cand_1", "resume_text": "Good resume 1"},
            {"candidate_id": "cand_fail", "resume_text": "Bad resume"},
            {"candidate_id": "cand_3", "resume_text": "Good resume 3"},
        ]

        result = run_pipeline(rubric=sample_rubric, candidates=candidates)

        # Successful candidates
        assert len(result.results) == 2
        assert [r.candidate_id for r in result.results] == ["cand_1", "cand_3"]

        # Failed candidate recorded
        assert len(result.failed_candidates) == 1
        failed = result.failed_candidates[0]
        assert failed.candidate_id == "cand_fail"
        assert "Corrupted PDF text" in failed.reason

        # No fabricated CandidateResult for the failed candidate
        result_ids = {r.candidate_id for r in result.results}
        assert "cand_fail" not in result_ids


def test_failure_isolation_in_matcher_or_auditor(sample_rubric: ParsedJobDescription):
    """Verify failure isolation when Matcher or Auditor raises an exception."""
    def mock_match(candidate, rubric):
        if candidate.candidate_id == "cand_matcher_fail":
            raise RuntimeError("Matcher network timeout during criteria evaluation")
        return make_match_result(candidate.candidate_id)

    with patch("app.pipeline.graph.ResumeParser") as mock_rp_cls, \
         patch("app.pipeline.graph.Matcher") as mock_m_cls, \
         patch("app.pipeline.graph.FairnessAuditor") as mock_fa_cls:

        mock_rp_cls.return_value.parse.side_effect = lambda resume_text, candidate_id=None: make_profile(candidate_id)
        mock_m_cls.return_value.match.side_effect = mock_match
        mock_fa_cls.return_value.audit.side_effect = lambda candidate, rubric, original_result: make_audit_record(candidate.candidate_id)

        candidates = [
            {"candidate_id": "cand_ok", "resume_text": "Good resume"},
            {"candidate_id": "cand_matcher_fail", "resume_text": "Fails at matcher"},
        ]

        result = run_pipeline(rubric=sample_rubric, candidates=candidates)

        assert len(result.results) == 1
        assert result.results[0].candidate_id == "cand_ok"
        assert len(result.failed_candidates) == 1
        assert result.failed_candidates[0].candidate_id == "cand_matcher_fail"
        assert "Matcher network timeout" in result.failed_candidates[0].reason


# ---------------------------------------------------------------------------
# Test 6: CandidateResult contract
# ---------------------------------------------------------------------------


def test_candidate_result_contract(sample_rubric: ParsedJobDescription):
    """Verify CandidateResult contains exact fields: candidate_id, scores, audit_record."""
    with patch("app.pipeline.graph.ResumeParser") as mock_rp_cls, \
         patch("app.pipeline.graph.Matcher") as mock_m_cls, \
         patch("app.pipeline.graph.FairnessAuditor") as mock_fa_cls:

        mock_rp_cls.return_value.parse.return_value = make_profile("cand_contract")
        mock_m_cls.return_value.match.return_value = make_match_result("cand_contract")
        mock_fa_cls.return_value.audit.return_value = make_audit_record("cand_contract")

        candidates = [CandidateInput(candidate_id="cand_contract", resume_text="Resume text")]
        result = run_pipeline(rubric=sample_rubric, candidates=candidates)

        cand_result = result.results[0]
        assert isinstance(cand_result, CandidateResult)
        assert cand_result.candidate_id == "cand_contract"
        assert isinstance(cand_result.scores, list)
        assert all(isinstance(s, ScoreRecord) for s in cand_result.scores)
        assert len(cand_result.scores) == 2
        assert isinstance(cand_result.audit_record, AuditRecord)
        assert cand_result.audit_record.candidate_id == "cand_contract"


# ---------------------------------------------------------------------------
# Test 7: PipelineResult contract
# ---------------------------------------------------------------------------


def test_pipeline_result_contract(sample_rubric: ParsedJobDescription):
    """Verify PipelineResult contract, input ordering preservation, and failure tracking."""
    def mock_parse(resume_text, candidate_id=None):
        if candidate_id == "c2_bad":
            raise ValueError("Invalid candidate")
        return make_profile(candidate_id)

    with patch("app.pipeline.graph.ResumeParser") as mock_rp_cls, \
         patch("app.pipeline.graph.Matcher") as mock_m_cls, \
         patch("app.pipeline.graph.FairnessAuditor") as mock_fa_cls:

        mock_rp_cls.return_value.parse.side_effect = mock_parse
        mock_m_cls.return_value.match.side_effect = lambda candidate, rubric: make_match_result(candidate.candidate_id)
        mock_fa_cls.return_value.audit.side_effect = lambda candidate, rubric, original_result: make_audit_record(candidate.candidate_id)

        candidates = [
            {"candidate_id": "c1_good", "resume_text": "text1"},
            {"candidate_id": "c2_bad", "resume_text": "text2"},
            {"candidate_id": "c3_good", "resume_text": "text3"},
            {"candidate_id": "c4_good", "resume_text": "text4"},
        ]

        pipeline_result = run_pipeline(rubric=sample_rubric, candidates=candidates)

        assert isinstance(pipeline_result, PipelineResult)
        assert [r.candidate_id for r in pipeline_result.results] == ["c1_good", "c3_good", "c4_good"]
        assert len(pipeline_result.failed_candidates) == 1
        assert pipeline_result.failed_candidates[0].candidate_id == "c2_bad"
        assert isinstance(pipeline_result.failed_candidates[0], FailedCandidate)


# ---------------------------------------------------------------------------
# Test 8: Empty candidate behavior
# ---------------------------------------------------------------------------


def test_empty_candidate_behavior_run_pipeline(sample_rubric: ParsedJobDescription):
    """run_pipeline with candidates=[] returns empty PipelineResult without invoking graph."""
    mock_graph = MagicMock()
    result = run_pipeline(rubric=sample_rubric, candidates=[], graph=mock_graph)

    assert isinstance(result, PipelineResult)
    assert result.results == []
    assert result.failed_candidates == []
    mock_graph.invoke.assert_not_called()


def test_empty_candidate_behavior_run_full_pipeline(sample_rubric: ParsedJobDescription):
    """run_full_pipeline with candidates=[] calls JDParser once and returns empty PipelineResult."""
    with patch("app.pipeline.graph.JDParser") as mock_jd_cls:
        mock_jd = mock_jd_cls.return_value
        mock_jd.parse.return_value = sample_rubric

        result = run_full_pipeline(
            jd_text="Job Description for Empty Batch",
            candidates=[],
        )

        assert mock_jd.parse.call_count == 1
        assert isinstance(result, PipelineResult)
        assert result.results == []
        assert result.failed_candidates == []


# ---------------------------------------------------------------------------
# Test 9: Node functions thin wrapper behavior
# ---------------------------------------------------------------------------


def test_node_functions_are_thin_wrappers(sample_rubric: ParsedJobDescription):
    """Verify each node function delegates directly to the underlying agent."""
    profile = make_profile("test_cand")
    match_result = make_match_result("test_cand")
    audit_rec = make_audit_record("test_cand")

    # resume_parser_node
    with patch("app.pipeline.graph.ResumeParser") as mock_rp_cls:
        mock_rp_cls.return_value.parse.return_value = profile
        state: CandidateGraphState = {
            "candidate_id": "test_cand",
            "resume_text": "Sample text",
            "rubric": sample_rubric,
        }
        res = resume_parser_node(state)
        mock_rp_cls.return_value.parse.assert_called_once_with(
            resume_text="Sample text", candidate_id="test_cand"
        )
        assert res == {"candidate_profile": profile}

    # matcher_node
    with patch("app.pipeline.graph.Matcher") as mock_m_cls:
        mock_m_cls.return_value.match.return_value = match_result
        state["candidate_profile"] = profile
        res = matcher_node(state)
        mock_m_cls.return_value.match.assert_called_once_with(
            candidate=profile, rubric=sample_rubric
        )
        assert res == {"match_result": match_result}

    # fairness_auditor_node
    with patch("app.pipeline.graph.FairnessAuditor") as mock_fa_cls:
        mock_fa_cls.return_value.audit.return_value = audit_rec
        state["match_result"] = match_result
        res = fairness_auditor_node(state)
        mock_fa_cls.return_value.audit.assert_called_once_with(
            candidate=profile, rubric=sample_rubric, original_result=match_result
        )
        assert res == {"audit_record": audit_rec}
