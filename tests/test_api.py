"""Focused offline tests for the HireFair FastAPI boundary."""

from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.models.audit import AuditRecord
from app.models.candidate import CandidateProfile, WorkExperience
from app.models.result import CandidateResult, FailedCandidate
from app.models.routing import RouteDecision, RoutingDecision, RoutingResult
from app.models.score import ScoreRecord

client = TestClient(app)


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------


def make_fake_route_decision(
    candidate_id: str,
    decision: RoutingDecision,
    reason: str,
    score: float = 0.85,
    duplicate_of: str = None,
    flagged: bool = False,
    flagged_reason: str = "",
) -> RouteDecision:
    """Construct a fake RouteDecision for testing serialization."""
    profile = CandidateProfile(
        candidate_id=candidate_id,
        candidate_name=f"Candidate {candidate_id}",
        raw_text=f"Resume text for {candidate_id}",
        skills=["Python", "SQL"],
        experience=[
            WorkExperience(
                title="Software Engineer",
                company="Tech Corp",
                duration_months=36,
                description="Built high-performance APIs",
            )
        ],
    )
    scores = [
        ScoreRecord(
            candidate_id=candidate_id,
            criterion_id="REQ-01",
            score=score,
            evidence_quote="Built high-performance APIs",
            confidence=0.95,
        )
    ]
    audit_record = AuditRecord(
        candidate_id=candidate_id,
        original_score=score,
        anonymized_score=score,
        delta=0.0,
        citation_valid=not flagged,
        flagged=flagged,
        flagged_reason=flagged_reason,
        repair_applied=False,
    )
    cand_res = CandidateResult(
        candidate_id=candidate_id,
        candidate_profile=profile,
        scores=scores,
        audit_record=audit_record,
    )
    return RouteDecision(
        candidate_id=candidate_id,
        decision=decision,
        reason=reason,
        final_score=score,
        duplicate_of=duplicate_of,
        candidate_result=cand_res,
    )


# ---------------------------------------------------------------------------
# Test 1: Health Endpoint
# ---------------------------------------------------------------------------


def test_health_endpoint():
    """Verify minimal server health check returns HTTP 200 and ok status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# Test 2: Successful /screen Request with Job Description and Resumes
# ---------------------------------------------------------------------------


def test_successful_screen_request():
    """Verify successful multipart upload and execution flow."""
    fake_decisions = [
        make_fake_route_decision("alice", RoutingDecision.CLEARED, "Fully qualified"),
        make_fake_route_decision("bob", RoutingDecision.NOT_QUALIFIED, "Below threshold", score=0.40),
    ]
    fake_routing_result = RoutingResult(
        routed_candidates=fake_decisions,
        failed_candidates=[],
    )

    with patch("app.api.main.screen_documents", return_value=fake_routing_result) as mock_screen:
        files = [
            ("job_description", ("job.txt", b"Senior Python Developer with 3+ years experience.", "text/plain")),
            ("resumes", ("alice.txt", b"Alice's resume text with Python and SQL.", "text/plain")),
            ("resumes", ("bob.txt", b"Bob's resume text with Junior experience.", "text/plain")),
        ]
        response = client.post("/screen", files=files)

        assert response.status_code == 200
        data = response.json()

        assert len(data["results"]) == 2
        assert data["results"][0]["candidate_id"] == "alice"
        assert data["results"][0]["decision"] == "cleared"
        assert data["results"][1]["candidate_id"] == "bob"
        assert data["results"][1]["decision"] == "not_qualified"

        # Verify mock received in-memory text and candidates
        mock_screen.assert_called_once()
        call_kwargs = mock_screen.call_args[1]
        assert "Senior Python Developer" in call_kwargs["jd_text"]
        assert len(call_kwargs["candidates"]) == 2
        assert call_kwargs["candidates"][0].candidate_id == "alice"
        assert call_kwargs["candidates"][1].candidate_id == "bob"


# ---------------------------------------------------------------------------
# Test 3: Rejection of Unsupported File Formats (.pdf)
# ---------------------------------------------------------------------------


def test_screen_rejects_non_txt_job_description():
    """Rejects job description file with non-.txt extension (.pdf)."""
    files = [
        ("job_description", ("job.pdf", b"%PDF-1.4 binary content", "application/pdf")),
        ("resumes", ("alice.txt", b"Alice resume", "text/plain")),
    ]
    response = client.post("/screen", files=files)
    assert response.status_code == 400
    assert "Only .txt files are accepted" in response.json()["detail"]


def test_screen_rejects_non_txt_resume():
    """Rejects candidate resume file with non-.txt extension (.pdf)."""
    files = [
        ("job_description", ("job.txt", b"Job text", "text/plain")),
        ("resumes", ("resume.pdf", b"%PDF-1.4 binary content", "application/pdf")),
    ]
    response = client.post("/screen", files=files)
    assert response.status_code == 400
    assert "Only .txt files are accepted" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Test 4: Missing and Empty File Inputs
# ---------------------------------------------------------------------------


def test_screen_rejects_empty_job_description_text():
    """Rejects job description with empty whitespace content."""
    files = [
        ("job_description", ("job.txt", b"   \n  \t  ", "text/plain")),
        ("resumes", ("alice.txt", b"Alice resume", "text/plain")),
    ]
    response = client.post("/screen", files=files)
    assert response.status_code == 400
    assert "cannot be empty" in response.json()["detail"]


def test_screen_missing_job_description():
    """FastAPI validation rejects request missing job_description."""
    files = [
        ("resumes", ("alice.txt", b"Alice resume", "text/plain")),
    ]
    response = client.post("/screen", files=files)
    assert response.status_code in (400, 422)


def test_screen_missing_resumes():
    """FastAPI validation rejects request missing resumes."""
    files = [
        ("job_description", ("job.txt", b"Job text", "text/plain")),
    ]
    response = client.post("/screen", files=files)
    assert response.status_code in (400, 422)


# ---------------------------------------------------------------------------
# Test 5: Serialization of All Decision Types
# ---------------------------------------------------------------------------


def test_screen_serializes_all_decision_categories():
    """Verify clean serialization of CLEARED, FLAGGED, NOT_QUALIFIED, INCOMPLETE, DUPLICATE."""
    decisions = [
        make_fake_route_decision("c1", RoutingDecision.CLEARED, "Meets all criteria", score=0.9),
        make_fake_route_decision(
            "c2",
            RoutingDecision.FLAGGED_FOR_REVIEW,
            "Citation invalid",
            score=0.85,
            flagged=True,
            flagged_reason="Citation quote does not support criterion",
        ),
        make_fake_route_decision("c3", RoutingDecision.NOT_QUALIFIED, "Must-have unmet", score=0.25),
        make_fake_route_decision("c4", RoutingDecision.INCOMPLETE_DATA, "Missing duration_months", score=0.7),
        make_fake_route_decision("c5", RoutingDecision.DUPLICATE, "Matches c1", score=0.9, duplicate_of="c1"),
    ]
    fake_result = RoutingResult(routed_candidates=decisions, failed_candidates=[])

    with patch("app.api.main.screen_documents", return_value=fake_result):
        files = [
            ("job_description", ("jd.txt", b"Job description text", "text/plain")),
            ("resumes", ("c1.txt", b"text 1", "text/plain")),
            ("resumes", ("c2.txt", b"text 2", "text/plain")),
            ("resumes", ("c3.txt", b"text 3", "text/plain")),
            ("resumes", ("c4.txt", b"text 4", "text/plain")),
            ("resumes", ("c5.txt", b"text 5", "text/plain")),
        ]
        response = client.post("/screen", files=files)

        assert response.status_code == 200
        results = response.json()["results"]
        assert len(results) == 5

        dec_types = [r["decision"] for r in results]
        assert dec_types == [
            "cleared",
            "flagged_for_review",
            "not_qualified",
            "incomplete_data",
            "duplicate",
        ]
        assert results[4]["duplicate_of"] == "c1"


# ---------------------------------------------------------------------------
# Test 6: Separate Preservation of Failed Candidates
# ---------------------------------------------------------------------------


def test_screen_preserves_failed_candidates_separately():
    """Verify failed candidates are returned in failed_candidates without fabricating results."""
    routed = [make_fake_route_decision("good_cand", RoutingDecision.CLEARED, "All good")]
    failed = [FailedCandidate(candidate_id="bad_cand", reason="ResumeParser parsing error")]

    fake_result = RoutingResult(routed_candidates=routed, failed_candidates=failed)

    with patch("app.api.main.screen_documents", return_value=fake_result):
        files = [
            ("job_description", ("jd.txt", b"Job text", "text/plain")),
            ("resumes", ("good_cand.txt", b"Good resume", "text/plain")),
            ("resumes", ("bad_cand.txt", b"Bad resume", "text/plain")),
        ]
        response = client.post("/screen", files=files)

        assert response.status_code == 200
        data = response.json()

        assert len(data["results"]) == 1
        assert data["results"][0]["candidate_id"] == "good_cand"

        assert len(data["failed_candidates"]) == 1
        assert data["failed_candidates"][0]["candidate_id"] == "bad_cand"
        assert data["failed_candidates"][0]["reason"] == "ResumeParser parsing error"


# ---------------------------------------------------------------------------
# Test 7: Preservation of Candidate Profile, Scores, and Audit Information
# ---------------------------------------------------------------------------


def test_screen_preserves_audit_and_profile_context():
    """Verify CandidateProfile, ScoreRecord quotes, and AuditRecord are preserved for future dashboard."""
    decision = make_fake_route_decision(
        "elena",
        RoutingDecision.FLAGGED_FOR_REVIEW,
        "Counterfactual delta exceeded threshold",
        score=0.88,
        flagged=True,
        flagged_reason="Counterfactual delta 0.22 >= 0.15",
    )
    fake_result = RoutingResult(routed_candidates=[decision], failed_candidates=[])

    with patch("app.api.main.screen_documents", return_value=fake_result):
        files = [
            ("job_description", ("jd.txt", b"Job text", "text/plain")),
            ("resumes", ("elena.txt", b"Elena resume text", "text/plain")),
        ]
        response = client.post("/screen", files=files)

        assert response.status_code == 200
        cand_resp = response.json()["results"][0]

        # Profile is preserved
        assert cand_resp["candidate_profile"]["candidate_name"] == "Candidate elena"
        assert cand_resp["candidate_profile"]["skills"] == ["Python", "SQL"]

        # Scores and evidence quotes are preserved
        assert len(cand_resp["scores"]) == 1
        assert cand_resp["scores"][0]["evidence_quote"] == "Built high-performance APIs"

        # AuditRecord is preserved
        assert cand_resp["audit_record"]["flagged"] is True
        assert cand_resp["audit_record"]["flagged_reason"] == "Counterfactual delta 0.22 >= 0.15"


# ---------------------------------------------------------------------------
# Test 8: Internal Pipeline Error Handling (No Stack Trace Leak)
# ---------------------------------------------------------------------------


def test_screen_internal_error_hides_stack_trace():
    """Verify unexpected internal exception returns HTTP 500 without leaking stack traces."""
    with patch("app.api.main.screen_documents", side_effect=RuntimeError("Secret internal database failed at /var/app/db.py:102")):
        files = [
            ("job_description", ("jd.txt", b"Job text", "text/plain")),
            ("resumes", ("candidate.txt", b"Resume text", "text/plain")),
        ]
        response = client.post("/screen", files=files)

        assert response.status_code == 500
        detail = response.json()["detail"]
        assert "An error occurred during candidate screening" in detail
        # Ensure private implementation details and stack traces are NOT exposed
        assert "/var/app/db.py" not in detail
        assert "RuntimeError" not in detail
