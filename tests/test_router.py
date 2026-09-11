"""Comprehensive offline tests for the Router agent."""

from copy import deepcopy
import pytest

from app.agents.router import (
    Router,
    check_qualification,
    detect_duplicates,
    get_final_score,
    is_incomplete_profile,
    route_candidate,
    route_pipeline_result,
)
from app.agents.router_config import (
    DUPLICATE_NAME_MATCH_SIMILARITY_THRESHOLD,
    DUPLICATE_TEXT_SIMILARITY_THRESHOLD,
    MIN_MUST_HAVE_SCORE,
    QUALIFICATION_THRESHOLD,
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
from app.models.result import CandidateResult, FailedCandidate, PipelineResult
from app.models.routing import RouteDecision, RoutingDecision, RoutingResult
from app.models.score import ScoreRecord


# ---------------------------------------------------------------------------
# Test Fixtures & Factory Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def standard_rubric() -> ParsedJobDescription:
    """Standard rubric with two must-haves and one nice-to-have."""
    return ParsedJobDescription(
        job_title="Senior Python Engineer",
        company="HireFair Inc",
        requirements=[
            JobRequirement(
                criterion_id="REQ-PYTHON",
                description="Advanced Python backend development",
                weight=4,
                type=RequirementType.MUST_HAVE,
                evidence_type="work experience",
            ),
            JobRequirement(
                criterion_id="REQ-SQL",
                description="PostgreSQL database query optimization",
                weight=3,
                type=RequirementType.MUST_HAVE,
                evidence_type="work experience",
            ),
            JobRequirement(
                criterion_id="REQ-K8S",
                description="Kubernetes container orchestration",
                weight=1,
                type=RequirementType.NICE_TO_HAVE,
                evidence_type="projects or work experience",
            ),
        ],
    )


def make_profile(
    candidate_id: str = "cand_1",
    name: str = "Jordan Smith",
    raw_text: str = "Experienced Python engineer with SQL and Docker.",
    skills: list[str] = None,
    experience: list[WorkExperience] = None,
    projects: list[Project] = None,
    education: list[Education] = None,
) -> CandidateProfile:
    """Helper to build CandidateProfile with sensible complete defaults."""
    if skills is None:
        skills = ["Python", "PostgreSQL", "Docker"]
    if experience is None:
        experience = [
            WorkExperience(
                title="Backend Developer",
                company="Acme Software",
                duration_months=36,
                description="Built high-scale APIs with Python and PostgreSQL.",
            )
        ]
    if projects is None:
        projects = []
    if education is None:
        education = [
            Education(
                degree="B.S. Computer Science",
                institution="State University",
                graduation_year=2020,
            )
        ]

    return CandidateProfile(
        candidate_id=candidate_id,
        candidate_name=name,
        raw_text=raw_text,
        skills=skills,
        experience=experience,
        projects=projects,
        education=education,
    )


def make_candidate_result(
    candidate_id: str = "cand_1",
    profile: CandidateProfile = None,
    scores: list[ScoreRecord] = None,
    original_score: float = 0.8,
    anonymized_score: float = 0.8,
    flagged: bool = False,
    flagged_reason: str = "",
    repair_applied: bool = False,
) -> CandidateResult:
    """Helper to construct CandidateResult with populated fields."""
    if profile is None:
        profile = make_profile(candidate_id=candidate_id)

    if scores is None:
        scores = [
            ScoreRecord(
                candidate_id=candidate_id,
                criterion_id="REQ-PYTHON",
                score=0.85,
                evidence_quote="Built high-scale APIs with Python and PostgreSQL.",
                confidence=0.9,
            ),
            ScoreRecord(
                candidate_id=candidate_id,
                criterion_id="REQ-SQL",
                score=0.75,
                evidence_quote="Built high-scale APIs with Python and PostgreSQL.",
                confidence=0.85,
            ),
            ScoreRecord(
                candidate_id=candidate_id,
                criterion_id="REQ-K8S",
                score=0.60,
                evidence_quote="Docker containerization",
                confidence=0.75,
            ),
        ]

    delta = (
        abs(original_score - anonymized_score)
        if (original_score is not None and anonymized_score is not None)
        else None
    )

    audit_rec = AuditRecord(
        candidate_id=candidate_id,
        original_score=original_score,
        anonymized_score=anonymized_score,
        delta=delta,
        citation_valid=not flagged,
        flagged=flagged,
        flagged_reason=flagged_reason,
        repair_applied=repair_applied,
    )

    return CandidateResult(
        candidate_id=candidate_id,
        candidate_profile=profile,
        scores=scores,
        audit_record=audit_rec,
    )


# ---------------------------------------------------------------------------
# Test A: CandidateResult contains CandidateProfile
# ---------------------------------------------------------------------------


def test_candidate_result_contains_candidate_profile():
    """Verify CandidateResult contract includes candidate_profile without modification."""
    prof = make_profile(candidate_id="c_contract", name="Alex Rivera")
    cand_res = make_candidate_result(candidate_id="c_contract", profile=prof)

    assert hasattr(cand_res, "candidate_profile")
    assert isinstance(cand_res.candidate_profile, CandidateProfile)
    assert cand_res.candidate_profile.candidate_name == "Alex Rivera"
    assert cand_res.candidate_profile.skills == ["Python", "PostgreSQL", "Docker"]


# ---------------------------------------------------------------------------
# Test B: Final Score Helper
# ---------------------------------------------------------------------------


def test_final_score_helper_repair_false():
    """repair_applied=False returns original_score."""
    audit = AuditRecord(
        candidate_id="c1",
        original_score=0.78,
        anonymized_score=0.62,
        repair_applied=False,
    )
    assert get_final_score(audit) == 0.78


def test_final_score_helper_repair_true():
    """repair_applied=True returns anonymized_score."""
    audit = AuditRecord(
        candidate_id="c1",
        original_score=0.85,
        anonymized_score=0.65,
        repair_applied=True,
    )
    assert get_final_score(audit) == 0.65


def test_final_score_helper_none_handling():
    """Returns None safely if audit_record is None or scores are None."""
    assert get_final_score(None) is None

    audit_none = AuditRecord(candidate_id="c1", original_score=None, anonymized_score=None)
    assert get_final_score(audit_none) is None


# ---------------------------------------------------------------------------
# Test C: Qualification Boundaries
# ---------------------------------------------------------------------------


def test_qualification_boundary_exact_thresholds(standard_rubric: ParsedJobDescription):
    """Candidate with overall score exactly 0.5 and all must-haves exactly 0.3 qualifies."""
    # Rubric: REQ-PYTHON (wt 4), REQ-SQL (wt 3), REQ-K8S (wt 1) -> total weight 8
    # Scores: must-haves tested at exactly 0.30, and overall final_score tested at boundary 0.50
    scores = [
        ScoreRecord(
            candidate_id="c_boundary",
            criterion_id="REQ-PYTHON",
            score=0.30,
            evidence_quote="Python experience",
            confidence=0.8,
        ),
        ScoreRecord(
            candidate_id="c_boundary",
            criterion_id="REQ-SQL",
            score=0.30,
            evidence_quote="SQL experience",
            confidence=0.8,
        ),
        ScoreRecord(
            candidate_id="c_boundary",
            criterion_id="REQ-K8S",
            score=0.30,
            evidence_quote="K8s experience",
            confidence=0.8,
        ),
    ]
    # Set final_score on audit_record to exactly 0.5
    cand_res = make_candidate_result(
        candidate_id="c_boundary",
        scores=scores,
        original_score=0.50,
        anonymized_score=0.50,
    )

    is_qual, reason = check_qualification(cand_res, standard_rubric)
    assert is_qual is True
    assert "meets all qualification requirements" in reason

    # Router decision should be CLEARED
    decision = route_candidate(cand_res, standard_rubric)
    assert decision.decision == RoutingDecision.CLEARED
    assert decision.final_score == 0.50


def test_qualification_overall_below_threshold(standard_rubric: ParsedJobDescription):
    """Overall score 0.499 fails qualification even if must-haves are high."""
    scores = [
        ScoreRecord(
            candidate_id="c_low_overall",
            criterion_id="REQ-PYTHON",
            score=0.49,
            evidence_quote="Python exp",
            confidence=0.8,
        ),
        ScoreRecord(
            candidate_id="c_low_overall",
            criterion_id="REQ-SQL",
            score=0.49,
            evidence_quote="SQL exp",
            confidence=0.8,
        ),
        ScoreRecord(
            candidate_id="c_low_overall",
            criterion_id="REQ-K8S",
            score=0.49,
            evidence_quote="K8s exp",
            confidence=0.8,
        ),
    ]
    cand_res = make_candidate_result(
        candidate_id="c_low_overall",
        scores=scores,
        original_score=0.499,
    )

    is_qual, reason = check_qualification(cand_res, standard_rubric)
    assert is_qual is False
    assert "below qualification threshold" in reason

    decision = route_candidate(cand_res, standard_rubric)
    assert decision.decision == RoutingDecision.NOT_QUALIFIED


def test_qualification_must_have_below_threshold(standard_rubric: ParsedJobDescription):
    """Overall score 0.8 fails if a must-have score is 0.299 (< 0.3)."""
    scores = [
        ScoreRecord(
            candidate_id="c_low_must_have",
            criterion_id="REQ-PYTHON",
            score=0.90,
            evidence_quote="Python exp",
            confidence=0.9,
        ),
        ScoreRecord(
            candidate_id="c_low_must_have",
            criterion_id="REQ-SQL",
            score=0.299,
            evidence_quote="",  # <= 0.3 allows empty evidence quote
            confidence=0.5,
        ),
        ScoreRecord(
            candidate_id="c_low_must_have",
            criterion_id="REQ-K8S",
            score=0.80,
            evidence_quote="K8s exp",
            confidence=0.8,
        ),
    ]
    cand_res = make_candidate_result(
        candidate_id="c_low_must_have",
        scores=scores,
        original_score=0.80,
    )

    is_qual, reason = check_qualification(cand_res, standard_rubric)
    assert is_qual is False
    assert "Must-have criterion 'REQ-SQL'" in reason
    assert "below minimum 0.3" in reason

    decision = route_candidate(cand_res, standard_rubric)
    assert decision.decision == RoutingDecision.NOT_QUALIFIED


def test_qualification_cleared_when_both_conditions_met(standard_rubric: ParsedJobDescription):
    """Overall >= 0.5 and all must-haves >= 0.3 results in CLEARED."""
    cand_res = make_candidate_result(
        candidate_id="c_cleared",
        original_score=0.82,
    )
    decision = route_candidate(cand_res, standard_rubric)
    assert decision.decision == RoutingDecision.CLEARED
    assert decision.final_score == 0.82


# ---------------------------------------------------------------------------
# Test D: Must-Have vs Nice-To-Have Compensation
# ---------------------------------------------------------------------------


def test_nice_to_have_cannot_compensate_failed_must_have(standard_rubric: ParsedJobDescription):
    """High score in nice-to-have criterion cannot compensate for failed must-have."""
    scores = [
        ScoreRecord(
            candidate_id="c_nice_comp",
            criterion_id="REQ-PYTHON",
            score=0.25,  # FAILS MUST-HAVE (< 0.3)
            evidence_quote="",
            confidence=0.5,
        ),
        ScoreRecord(
            candidate_id="c_nice_comp",
            criterion_id="REQ-SQL",
            score=0.90,  # PASSES MUST-HAVE
            evidence_quote="SQL expertise",
            confidence=0.9,
        ),
        ScoreRecord(
            candidate_id="c_nice_comp",
            criterion_id="REQ-K8S",
            score=1.00,  # PERFECT NICE-TO-HAVE
            evidence_quote="Kubernetes maintainer",
            confidence=1.0,
        ),
    ]
    cand_res = make_candidate_result(
        candidate_id="c_nice_comp",
        scores=scores,
        original_score=0.70,  # Overall score is very high (0.70 >= 0.50)
    )

    decision = route_candidate(cand_res, standard_rubric)
    assert decision.decision == RoutingDecision.NOT_QUALIFIED
    assert "REQ-PYTHON" in decision.reason


def test_missing_must_have_score_fails_qualification(standard_rubric: ParsedJobDescription):
    """Missing score for a must-have criterion fails qualification."""
    # Only REQ-SQL and REQ-K8S scored, REQ-PYTHON is omitted
    scores = [
        ScoreRecord(
            candidate_id="c_missing_must",
            criterion_id="REQ-SQL",
            score=0.90,
            evidence_quote="SQL expertise",
            confidence=0.9,
        ),
        ScoreRecord(
            candidate_id="c_missing_must",
            criterion_id="REQ-K8S",
            score=0.90,
            evidence_quote="K8s expertise",
            confidence=0.9,
        ),
    ]
    cand_res = make_candidate_result(
        candidate_id="c_missing_must",
        scores=scores,
        original_score=0.90,
    )

    is_qual, reason = check_qualification(cand_res, standard_rubric)
    assert is_qual is False
    assert "Missing score record for must-have criterion 'REQ-PYTHON'" in reason


# ---------------------------------------------------------------------------
# Test E: Fairness Precedence Over Qualification
# ---------------------------------------------------------------------------


def test_fairness_precedence_over_qualification(standard_rubric: ParsedJobDescription):
    """Fairness flagged candidate does not become CLEARED even if fully qualified."""
    cand_res = make_candidate_result(
        candidate_id="c_flagged",
        original_score=0.92,
        flagged=True,
        flagged_reason="Citation invalid: quoted text does not support score.",
    )

    decision = route_candidate(cand_res, standard_rubric)
    assert decision.decision == RoutingDecision.FLAGGED_FOR_REVIEW
    assert "Citation invalid" in decision.reason
    assert decision.decision != RoutingDecision.CLEARED


# ---------------------------------------------------------------------------
# Test F: Duplicate Precedence Over Cleared
# ---------------------------------------------------------------------------


def test_duplicate_precedence_over_cleared(standard_rubric: ParsedJobDescription):
    """Duplicate candidate that is otherwise CLEARED must be routed to DUPLICATE with duplicate_of."""
    prof1 = make_profile(candidate_id="c1", name="Taylor Reed", raw_text="Exact same resume text for Taylor.")
    prof2 = make_profile(candidate_id="c2", name="Taylor Reed", raw_text="Exact same resume text for Taylor.")

    cand1 = make_candidate_result(candidate_id="c1", profile=prof1, original_score=0.85)
    cand2 = make_candidate_result(candidate_id="c2", profile=prof2, original_score=0.85)

    pipeline_res = PipelineResult(results=[cand1, cand2], failed_candidates=[])
    routing_res = route_pipeline_result(pipeline_res, standard_rubric)

    dec1 = routing_res.routed_candidates[0]
    dec2 = routing_res.routed_candidates[1]

    assert dec1.decision == RoutingDecision.DUPLICATE
    assert dec1.duplicate_of == "c2"
    assert dec2.decision == RoutingDecision.DUPLICATE
    assert dec2.duplicate_of == "c1"


# ---------------------------------------------------------------------------
# Test G: Incomplete-Data Precedence Over Qualification
# ---------------------------------------------------------------------------


def test_incomplete_data_precedence_over_qualification(standard_rubric: ParsedJobDescription):
    """Incomplete candidate that is otherwise qualified routes to INCOMPLETE_DATA."""
    # Experience has duration_months=None (missing duration)
    prof_incomplete = make_profile(
        candidate_id="c_inc",
        experience=[
            WorkExperience(
                title="Software Engineer",
                company="Tech Co",
                duration_months=None,  # MISSING DURATION
                description="Engineered backend systems with Python.",
            )
        ],
    )
    cand_res = make_candidate_result(
        candidate_id="c_inc",
        profile=prof_incomplete,
        original_score=0.90,  # High score
    )

    decision = route_candidate(cand_res, standard_rubric)
    assert decision.decision == RoutingDecision.INCOMPLETE_DATA
    assert "missing duration_months" in decision.reason


# ---------------------------------------------------------------------------
# Test H: Full Precedence Ordering Hierarchy
# 1. DUPLICATE > 2. INCOMPLETE > 3. FAIRNESS > 4. NOT_QUALIFIED > 5. CLEARED
# ---------------------------------------------------------------------------


def test_duplicate_beats_incomplete(standard_rubric: ParsedJobDescription):
    """Duplicate (1) beats Incomplete Data (2)."""
    prof1 = make_profile(
        candidate_id="c_dup_inc1",
        name="Morgan Lee",
        raw_text="Identical Morgan Lee resume text.",
        experience=[WorkExperience(title="Dev", company="Co", duration_months=None, description="work")],
    )
    prof2 = make_profile(
        candidate_id="c_dup_inc2",
        name="Morgan Lee",
        raw_text="Identical Morgan Lee resume text.",
        experience=[WorkExperience(title="Dev", company="Co", duration_months=None, description="work")],
    )
    cand1 = make_candidate_result(candidate_id="c_dup_inc1", profile=prof1)
    cand2 = make_candidate_result(candidate_id="c_dup_inc2", profile=prof2)

    pipeline_res = PipelineResult(results=[cand1, cand2], failed_candidates=[])
    routing_res = route_pipeline_result(pipeline_res, standard_rubric)

    # Must be DUPLICATE, not INCOMPLETE_DATA
    assert routing_res.routed_candidates[0].decision == RoutingDecision.DUPLICATE
    assert routing_res.routed_candidates[1].decision == RoutingDecision.DUPLICATE


def test_incomplete_beats_fairness(standard_rubric: ParsedJobDescription):
    """Incomplete Data (2) beats Fairness Flagged (3)."""
    prof = make_profile(
        candidate_id="c_inc_fair",
        experience=[WorkExperience(title="Dev", company="Co", duration_months=None, description="work")],
    )
    cand_res = make_candidate_result(
        candidate_id="c_inc_fair",
        profile=prof,
        flagged=True,
        flagged_reason="Auditor flagged large counterfactual delta",
    )

    decision = route_candidate(cand_res, standard_rubric)
    assert decision.decision == RoutingDecision.INCOMPLETE_DATA


def test_fairness_beats_not_qualified(standard_rubric: ParsedJobDescription):
    """Fairness Flagged (3) beats Not Qualified (4)."""
    # Candidate score is 0.2 (would fail qualification), but fairness flagged
    cand_res = make_candidate_result(
        candidate_id="c_fair_unqual",
        original_score=0.20,
        flagged=True,
        flagged_reason="Potential demographic bias detected",
    )

    decision = route_candidate(cand_res, standard_rubric)
    assert decision.decision == RoutingDecision.FLAGGED_FOR_REVIEW
    assert "demographic bias" in decision.reason


def test_not_qualified_beats_cleared(standard_rubric: ParsedJobDescription):
    """Not Qualified (4) beats Cleared (5)."""
    cand_res = make_candidate_result(
        candidate_id="c_unqual",
        original_score=0.35,  # below 0.50
    )
    decision = route_candidate(cand_res, standard_rubric)
    assert decision.decision == RoutingDecision.NOT_QUALIFIED


# ---------------------------------------------------------------------------
# Test I: Duplicate Detection Rules & Scope
# ---------------------------------------------------------------------------


def test_duplicate_detection_excludes_failed_candidates(standard_rubric: ParsedJobDescription):
    """Failed candidates are not compared for duplicates."""
    prof = make_profile(candidate_id="c_valid", name="Same Person", raw_text="Identical text string.")
    cand_res = make_candidate_result(candidate_id="c_valid", profile=prof, original_score=0.85)

    failed_cand = FailedCandidate(candidate_id="c_failed", reason="Parsing exception")

    pipeline_res = PipelineResult(results=[cand_res], failed_candidates=[failed_cand])
    routing_res = route_pipeline_result(pipeline_res, standard_rubric)

    # c_valid must NOT be marked duplicate of c_failed
    assert routing_res.routed_candidates[0].decision == RoutingDecision.CLEARED
    assert routing_res.routed_candidates[0].duplicate_of is None


def test_duplicate_both_routed_no_merge(standard_rubric: ParsedJobDescription):
    """Duplicate pairs are both routed for human review without merging or winner selection."""
    prof1 = make_profile(candidate_id="c1", name="Pat Doe", raw_text="Common developer resume text.")
    prof2 = make_profile(candidate_id="c2", name="Pat Doe", raw_text="Common developer resume text.")

    cand1 = make_candidate_result(candidate_id="c1", profile=prof1, original_score=0.80)
    cand2 = make_candidate_result(candidate_id="c2", profile=prof2, original_score=0.90)

    pipeline_res = PipelineResult(results=[cand1, cand2], failed_candidates=[])
    routing_res = route_pipeline_result(pipeline_res, standard_rubric)

    assert len(routing_res.routed_candidates) == 2
    c1_dec, c2_dec = routing_res.routed_candidates

    assert c1_dec.candidate_id == "c1"
    assert c1_dec.decision == RoutingDecision.DUPLICATE
    assert c1_dec.duplicate_of == "c2"
    assert c1_dec.candidate_result == cand1  # Original data intact

    assert c2_dec.candidate_id == "c2"
    assert c2_dec.decision == RoutingDecision.DUPLICATE
    assert c2_dec.duplicate_of == "c1"
    assert c2_dec.candidate_result == cand2  # Original data intact


def test_duplicate_name_match_and_text_match():
    """Verify duplicate detection helper directly with exact thresholds."""
    # Pair 1: Same name, text similarity >= 0.70
    p1 = make_profile(candidate_id="a1", name="Chris Paul", raw_text="Chris Paul Software Engineer Python SQL Docker AWS")
    p2 = make_profile(candidate_id="a2", name="Chris Paul", raw_text="Chris Paul Software Engineer Python SQL Docker GCP")
    cand1 = make_candidate_result("a1", profile=p1)
    cand2 = make_candidate_result("a2", profile=p2)

    dup_map = detect_duplicates([cand1, cand2])
    assert "a1" in dup_map and dup_map["a1"] == "a2"
    assert "a2" in dup_map and dup_map["a2"] == "a1"

    # Pair 2: Different names, completely different text -> NOT duplicate
    p3 = make_profile(candidate_id="b1", name="Alice", raw_text="Alice specialized in frontend React CSS Webpack")
    p4 = make_profile(candidate_id="b2", name="Bob", raw_text="Bob specialized in embedded systems C firmware RTOS")
    cand3 = make_candidate_result("b1", profile=p3)
    cand4 = make_candidate_result("b2", profile=p4)

    dup_map2 = detect_duplicates([cand3, cand4])
    assert dup_map2 == {}


# ---------------------------------------------------------------------------
# Test J: Incomplete-Data Rule
# ---------------------------------------------------------------------------


def test_incomplete_profile_representative_cases():
    """Verify is_incomplete_profile against representative cases."""
    # 1. Complete profile -> passes
    complete = make_profile()
    is_inc, reason = is_incomplete_profile(complete)
    assert is_inc is False
    assert reason is None

    # 2. Missing name -> incomplete
    missing_name = make_profile(name="")
    is_inc, reason = is_incomplete_profile(missing_name)
    assert is_inc is True
    assert "missing candidate name" in reason

    # 3. Missing experience duration -> incomplete
    missing_duration = make_profile(
        experience=[WorkExperience(title="Eng", company="Corp", duration_months=None, description="desc")]
    )
    is_inc, reason = is_incomplete_profile(missing_duration)
    assert is_inc is True
    assert "missing duration_months" in reason

    # 4. Zero skills -> incomplete
    zero_skills = make_profile(skills=[])
    is_inc, reason = is_incomplete_profile(zero_skills)
    assert is_inc is True
    assert "minimum 1 required" in reason

    # 5. Neither experience nor projects -> incomplete
    no_exp_no_proj = make_profile(experience=[], projects=[])
    is_inc, reason = is_incomplete_profile(no_exp_no_proj)
    assert is_inc is True
    assert "neither work experience nor projects" in reason

    # 6. No experience, but has personal/open-source projects -> passes!
    has_projects_only = make_profile(
        experience=[],
        projects=[Project(name="oss-tool", description="Python open-source tool", technologies=["Python"])],
    )
    is_inc, reason = is_incomplete_profile(has_projects_only)
    assert is_inc is False
    assert reason is None


# ---------------------------------------------------------------------------
# Test K: Failed Candidates Passthrough
# ---------------------------------------------------------------------------


def test_failed_candidates_carried_forward(standard_rubric: ParsedJobDescription):
    """PipelineResult.failed_candidates pass through untouched without fabricating results."""
    valid_cand = make_candidate_result("c_ok", original_score=0.85)
    failed1 = FailedCandidate(candidate_id="c_err1", reason="Unreadable PDF stream")
    failed2 = FailedCandidate(candidate_id="c_err2", reason="Rate limit exhausted")

    pipeline_res = PipelineResult(
        results=[valid_cand],
        failed_candidates=[failed1, failed2],
    )

    routing_res = route_pipeline_result(pipeline_res, standard_rubric)

    assert len(routing_res.routed_candidates) == 1
    assert routing_res.routed_candidates[0].candidate_id == "c_ok"

    assert len(routing_res.failed_candidates) == 2
    assert routing_res.failed_candidates[0].candidate_id == "c_err1"
    assert routing_res.failed_candidates[0].reason == "Unreadable PDF stream"
    assert routing_res.failed_candidates[1].candidate_id == "c_err2"
    assert routing_res.failed_candidates[1].reason == "Rate limit exhausted"


# ---------------------------------------------------------------------------
# Test L: Input Ordering Preserved
# ---------------------------------------------------------------------------


def test_input_ordering_preserved(standard_rubric: ParsedJobDescription):
    """Routed candidates preserve the exact input order from PipelineResult.results."""
    p1 = make_profile("id_gamma", name="Gamma Smith", raw_text="Gamma resume text for python.")
    p2 = make_profile("id_alpha", name="Alpha Jones", raw_text="Alpha resume text for database.")
    p3 = make_profile("id_beta", name="Beta Brown", raw_text="Beta resume text for infrastructure.")

    c1 = make_candidate_result("id_gamma", profile=p1, original_score=0.90)  # CLEARED
    c2 = make_candidate_result("id_alpha", profile=p2, original_score=0.20)  # NOT_QUALIFIED
    c3 = make_candidate_result("id_beta", profile=p3, original_score=0.85, flagged=True, flagged_reason="Audit flag")

    pipeline_res = PipelineResult(results=[c1, c2, c3], failed_candidates=[])
    routing_res = route_pipeline_result(pipeline_res, standard_rubric)

    ordered_ids = [d.candidate_id for d in routing_res.routed_candidates]
    assert ordered_ids == ["id_gamma", "id_alpha", "id_beta"]
    assert routing_res.routed_candidates[0].decision == RoutingDecision.CLEARED
    assert routing_res.routed_candidates[1].decision == RoutingDecision.NOT_QUALIFIED
    assert routing_res.routed_candidates[2].decision == RoutingDecision.FLAGGED_FOR_REVIEW


# ---------------------------------------------------------------------------
# Test M: No Mutation of Input Objects
# ---------------------------------------------------------------------------


def test_no_mutation_of_input_objects(standard_rubric: ParsedJobDescription):
    """Router does not mutate the incoming PipelineResult or CandidateResult objects."""
    c1 = make_candidate_result("c_immut", original_score=0.88)
    pipeline_res = PipelineResult(results=[c1], failed_candidates=[])

    snapshot_before = deepcopy(pipeline_res)
    _ = route_pipeline_result(pipeline_res, standard_rubric)

    assert pipeline_res.model_dump() == snapshot_before.model_dump()


# ---------------------------------------------------------------------------
# Test N: Pure Offline Execution (No Live Gemini Calls)
# ---------------------------------------------------------------------------


def test_no_live_gemini_calls(standard_rubric: ParsedJobDescription):
    """Router execution is purely deterministic and calls no LLM or external APIs."""
    c1 = make_candidate_result("c_offline", original_score=0.85)
    pipeline_res = PipelineResult(results=[c1], failed_candidates=[])

    # No patches needed because Router contains 0 LLM dependencies
    routing_res = route_pipeline_result(pipeline_res, standard_rubric)
    assert routing_res.routed_candidates[0].decision == RoutingDecision.CLEARED


# ---------------------------------------------------------------------------
# Test O: 3-Way Duplicate Handling & Single ID Contract
# ---------------------------------------------------------------------------


def test_duplicate_three_way_group_single_id(standard_rubric: ParsedJobDescription):
    """Verify that 3 duplicate candidates each retain only a single candidate ID in duplicate_of."""
    p1 = make_profile("c_dup1", name="Sam Jones", raw_text="Identical full resume text for Sam Jones.")
    p2 = make_profile("c_dup2", name="Sam Jones", raw_text="Identical full resume text for Sam Jones.")
    p3 = make_profile("c_dup3", name="Sam Jones", raw_text="Identical full resume text for Sam Jones.")

    c1 = make_candidate_result("c_dup1", profile=p1, original_score=0.85)
    c2 = make_candidate_result("c_dup2", profile=p2, original_score=0.85)
    c3 = make_candidate_result("c_dup3", profile=p3, original_score=0.85)

    pipeline_res = PipelineResult(results=[c1, c2, c3], failed_candidates=[])
    routing_res = route_pipeline_result(pipeline_res, standard_rubric)

    assert len(routing_res.routed_candidates) == 3
    for dec in routing_res.routed_candidates:
        assert dec.decision == RoutingDecision.DUPLICATE
        assert dec.duplicate_of is not None
        # Must be a single candidate ID, NOT a comma-separated list
        assert "," not in dec.duplicate_of
        assert dec.duplicate_of in {"c_dup1", "c_dup2", "c_dup3"}
        assert dec.duplicate_of != dec.candidate_id


# ---------------------------------------------------------------------------
# Test P: Fallback Score Calculated Against Full Rubric Weight
# ---------------------------------------------------------------------------


def test_fallback_score_calculated_against_full_rubric_weight():
    """When audit_record has no score, weighted score must divide by total rubric weight."""
    rubric = ParsedJobDescription(
        job_title="Data Scientist",
        requirements=[
            JobRequirement(
                criterion_id="REQ-01",
                description="Machine Learning",
                weight=5,
                type=RequirementType.MUST_HAVE,
                evidence_type="work experience",
            ),
            JobRequirement(
                criterion_id="REQ-02",
                description="Data Modeling",
                weight=3,
                type=RequirementType.MUST_HAVE,
                evidence_type="work experience",
            ),
            JobRequirement(
                criterion_id="REQ-03",
                description="Cloud Platforms",
                weight=2,
                type=RequirementType.NICE_TO_HAVE,
                evidence_type="work experience",
            ),
        ],
    )
    # Total rubric weight = 5 + 3 + 2 = 10.
    # Candidate only scored on REQ-01 with score 0.80 (weighted sum = 0.80 * 5 = 4.0).
    # AuditRecord has None for original_score and anonymized_score.
    scores = [
        ScoreRecord(
            candidate_id="c_fallback",
            criterion_id="REQ-01",
            score=0.80,
            evidence_quote="ML experience",
            confidence=0.9,
        )
    ]
    cand_res = make_candidate_result(
        candidate_id="c_fallback",
        scores=scores,
        original_score=None,
        anonymized_score=None,
    )

    is_qual, reason = check_qualification(cand_res, rubric)

    # Against full rubric weight (10): 4.0 / 10 = 0.40 (< 0.50 qualification threshold).
    # If it had erroneously divided only by scored weight (5), it would be 4.0 / 5 = 0.80 (>= 0.50).
    assert is_qual is False
    assert "Final score 0.400 is below qualification threshold 0.5" in reason

