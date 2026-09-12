from datetime import datetime, timezone
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

from app.api.main import app, calendar
from app.models.audit import AuditRecord
from app.models.candidate import CandidateProfile, WorkExperience
from app.models.result import CandidateResult, FailedCandidate
from app.models.routing import RouteDecision, RoutingDecision, RoutingResult
from app.models.scheduling import SchedulingStatus
from app.models.score import ScoreRecord

client = TestClient(app)

FIXED_REF_TIME = datetime(2025, 1, 6, 9, 0, 0, tzinfo=timezone.utc)  # Monday 09:00 UTC


@pytest.fixture(autouse=True)
def reset_calendar():
    """Isolate and reset shared app-level calendar before and after each test."""
    calendar.clear()
    yield
    calendar.clear()


@pytest.fixture(autouse=True)
def default_fixed_reference_time():
    """Ensure tests use a deterministic fixed timezone-aware reference time rather than system clock."""
    with patch("app.api.main.get_current_reference_time", return_value=FIXED_REF_TIME):
        yield



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
        assert data["results"][0]["scheduling_result"] is not None
        assert data["results"][0]["scheduling_result"]["status"] == "scheduled"
        assert data["results"][0]["scheduling_result"]["scheduled_at"] == "2025-01-06T09:00:00Z"

        assert data["results"][1]["candidate_id"] == "bob"
        assert data["results"][1]["decision"] == "not_qualified"
        assert data["results"][1]["scheduling_result"] is not None
        assert data["results"][1]["scheduling_result"]["status"] == "ineligible"

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


# ---------------------------------------------------------------------------
# Test 9: CLEARED Candidate Receives SCHEDULED
# ---------------------------------------------------------------------------


def test_cleared_candidate_receives_scheduled():
    """Verify a CLEARED candidate receives SCHEDULED with the earliest available working slot."""
    decisions = [
        make_fake_route_decision("cand_cleared", RoutingDecision.CLEARED, "Exemplary candidate", score=0.92)
    ]
    fake_result = RoutingResult(routed_candidates=decisions, failed_candidates=[])

    with patch("app.api.main.screen_documents", return_value=fake_result):
        files = [
            ("job_description", ("jd.txt", b"Job text", "text/plain")),
            ("resumes", ("cand_cleared.txt", b"Resume text", "text/plain")),
        ]
        response = client.post("/screen", files=files)

        assert response.status_code == 200
        cand = response.json()["results"][0]
        sched = cand["scheduling_result"]

        assert sched is not None
        assert sched["status"] == "scheduled"
        assert sched["scheduled_at"] == "2025-01-06T09:00:00Z"
        assert sched["duration_minutes"] == 30
        assert "cand_cleared" in sched["reason"]


# ---------------------------------------------------------------------------
# Test 10: Multiple CLEARED Candidates in One Request Receive Sequential Slots
# ---------------------------------------------------------------------------


def test_multiple_cleared_candidates_receive_sequential_slots():
    """Verify multiple CLEARED candidates in the same request compete and book sequential slots."""
    decisions = [
        make_fake_route_decision("c1", RoutingDecision.CLEARED, "Qualified 1", score=0.90),
        make_fake_route_decision("c2", RoutingDecision.CLEARED, "Qualified 2", score=0.88),
        make_fake_route_decision("c3", RoutingDecision.CLEARED, "Qualified 3", score=0.85),
    ]
    fake_result = RoutingResult(routed_candidates=decisions, failed_candidates=[])

    with patch("app.api.main.screen_documents", return_value=fake_result):
        files = [
            ("job_description", ("jd.txt", b"Job text", "text/plain")),
            ("resumes", ("c1.txt", b"Resume 1", "text/plain")),
            ("resumes", ("c2.txt", b"Resume 2", "text/plain")),
            ("resumes", ("c3.txt", b"Resume 3", "text/plain")),
        ]
        response = client.post("/screen", files=files)

        assert response.status_code == 200
        results = response.json()["results"]

        sched_slots = [r["scheduling_result"]["scheduled_at"] for r in results]
        assert sched_slots == [
            "2025-01-06T09:00:00Z",
            "2025-01-06T09:30:00Z",
            "2025-01-06T10:00:00Z",
        ]
        assert all(r["scheduling_result"]["status"] == "scheduled" for r in results)


# ---------------------------------------------------------------------------
# Test 11: No-Availability Produces NOT_SCHEDULED_NO_AVAILABILITY
# ---------------------------------------------------------------------------


def test_no_availability_produces_not_scheduled_no_availability():
    """Verify that when no working slots exist in 48h (e.g. Friday 17:00), status is NOT_SCHEDULED_NO_AVAILABILITY."""
    decisions = [
        make_fake_route_decision("c_friday", RoutingDecision.CLEARED, "Qualified candidate", score=0.90)
    ]
    fake_result = RoutingResult(routed_candidates=decisions, failed_candidates=[])
    friday_evening = datetime(2025, 1, 10, 17, 0, 0, tzinfo=timezone.utc)

    with patch("app.api.main.screen_documents", return_value=fake_result):
        with patch("app.api.main.get_current_reference_time", return_value=friday_evening):
            files = [
                ("job_description", ("jd.txt", b"Job text", "text/plain")),
                ("resumes", ("c_friday.txt", b"Resume", "text/plain")),
            ]
            response = client.post("/screen", files=files)

            assert response.status_code == 200
            cand = response.json()["results"][0]
            sched = cand["scheduling_result"]

            assert sched["status"] == "not_scheduled_no_availability"
            assert sched["scheduled_at"] is None
            assert "No available interview slot found" in sched["reason"]


# ---------------------------------------------------------------------------
# Test 12: FLAGGED, NOT_QUALIFIED, INCOMPLETE_DATA, DUPLICATE Receive INELIGIBLE
# ---------------------------------------------------------------------------


def test_non_cleared_decisions_receive_ineligible():
    """Verify all non-CLEARED routed candidates receive INELIGIBLE without booking calendar slots."""
    decisions = [
        make_fake_route_decision("flagged_c", RoutingDecision.FLAGGED_FOR_REVIEW, "Audit flag", score=0.85, flagged=True),
        make_fake_route_decision("unqual_c", RoutingDecision.NOT_QUALIFIED, "Score low", score=0.35),
        make_fake_route_decision("inc_c", RoutingDecision.INCOMPLETE_DATA, "Missing fields", score=0.70),
        make_fake_route_decision("dup_c", RoutingDecision.DUPLICATE, "Duplicate resume", score=0.85, duplicate_of="c1"),
    ]
    fake_result = RoutingResult(routed_candidates=decisions, failed_candidates=[])

    with patch("app.api.main.screen_documents", return_value=fake_result):
        files = [
            ("job_description", ("jd.txt", b"Job text", "text/plain")),
            ("resumes", ("f.txt", b"f", "text/plain")),
            ("resumes", ("u.txt", b"u", "text/plain")),
            ("resumes", ("i.txt", b"i", "text/plain")),
            ("resumes", ("d.txt", b"d", "text/plain")),
        ]
        response = client.post("/screen", files=files)

        assert response.status_code == 200
        results = response.json()["results"]

        for r in results:
            sched = r["scheduling_result"]
            assert sched is not None
            assert sched["status"] == "ineligible"
            assert sched["scheduled_at"] is None
            assert "not CLEARED" in sched["reason"]

        # Calendar must have had zero bookings
        assert len(calendar.booked_slots) == 0


# ---------------------------------------------------------------------------
# Test 13: Failed Candidates Have No scheduling_result
# ---------------------------------------------------------------------------


def test_failed_candidates_have_no_scheduling_result():
    """Verify failed candidates never receive a scheduling_result and retain pipeline failure reason."""
    routed = [make_fake_route_decision("c_ok", RoutingDecision.CLEARED, "OK", score=0.8)]
    failed = [FailedCandidate(candidate_id="c_fail", reason="Corrupted resume text")]
    fake_result = RoutingResult(routed_candidates=routed, failed_candidates=failed)

    with patch("app.api.main.screen_documents", return_value=fake_result):
        files = [
            ("job_description", ("jd.txt", b"Job text", "text/plain")),
            ("resumes", ("c_ok.txt", b"ok", "text/plain")),
            ("resumes", ("c_fail.txt", b"fail", "text/plain")),
        ]
        response = client.post("/screen", files=files)

        assert response.status_code == 200
        data = response.json()

        # Routed candidate has scheduling_result
        assert "scheduling_result" in data["results"][0]

        # Failed candidate does NOT have scheduling_result
        failed_entry = data["failed_candidates"][0]
        assert "scheduling_result" not in failed_entry
        assert failed_entry["candidate_id"] == "c_fail"
        assert failed_entry["reason"] == "Corrupted resume text"


# ---------------------------------------------------------------------------
# Test 14: Calendar Bookings Persist Across Sequential Requests
# ---------------------------------------------------------------------------


def test_calendar_bookings_persist_across_sequential_requests():
    """Verify bookings in the app-level calendar persist across multiple independent HTTP requests."""
    res1 = RoutingResult(
        routed_candidates=[make_fake_route_decision("alice", RoutingDecision.CLEARED, "Good", score=0.9)],
        failed_candidates=[],
    )
    res2 = RoutingResult(
        routed_candidates=[make_fake_route_decision("bob", RoutingDecision.CLEARED, "Good", score=0.9)],
        failed_candidates=[],
    )

    # First request: Alice is screened and booked for 09:00
    with patch("app.api.main.screen_documents", return_value=res1):
        files1 = [
            ("job_description", ("jd.txt", b"Job text", "text/plain")),
            ("resumes", ("alice.txt", b"Alice", "text/plain")),
        ]
        resp1 = client.post("/screen", files=files1)
        assert resp1.status_code == 200
        assert resp1.json()["results"][0]["scheduling_result"]["scheduled_at"] == "2025-01-06T09:00:00Z"

    # Second request: Bob is screened with same reference time, receives 09:30 because 09:00 is taken
    with patch("app.api.main.screen_documents", return_value=res2):
        files2 = [
            ("job_description", ("jd.txt", b"Job text", "text/plain")),
            ("resumes", ("bob.txt", b"Bob", "text/plain")),
        ]
        resp2 = client.post("/screen", files=files2)
        assert resp2.status_code == 200
        assert resp2.json()["results"][0]["scheduling_result"]["scheduled_at"] == "2025-01-06T09:30:00Z"


# ---------------------------------------------------------------------------
# Test 15: API Response Correctly Serializes All scheduling_result Fields
# ---------------------------------------------------------------------------


def test_api_response_correctly_serializes_scheduling_result():
    """Verify all fields of SchedulingResult serialize accurately into JSON."""
    decisions = [
        make_fake_route_decision("dana", RoutingDecision.CLEARED, "Cleared candidate", score=0.89)
    ]
    fake_result = RoutingResult(routed_candidates=decisions, failed_candidates=[])

    with patch("app.api.main.screen_documents", return_value=fake_result):
        files = [
            ("job_description", ("jd.txt", b"Job text", "text/plain")),
            ("resumes", ("dana.txt", b"Dana", "text/plain")),
        ]
        response = client.post("/screen", files=files)

        assert response.status_code == 200
        sched = response.json()["results"][0]["scheduling_result"]

        assert set(sched.keys()) == {
            "candidate_id",
            "status",
            "scheduled_at",
            "duration_minutes",
            "reason",
        }
        assert sched["candidate_id"] == "dana"
        assert sched["status"] == "scheduled"
        assert sched["duration_minutes"] == 30
        assert sched["scheduled_at"] == "2025-01-06T09:00:00Z"
        assert isinstance(sched["reason"], str)


# ---------------------------------------------------------------------------
# Test 16: Query Parameter reference_time Override
# ---------------------------------------------------------------------------


def test_query_parameter_reference_time_override():
    """Verify passing reference_time as a query parameter overrides default reference time."""
    decisions = [
        make_fake_route_decision("c_tuesday", RoutingDecision.CLEARED, "Cleared candidate", score=0.90)
    ]
    fake_result = RoutingResult(routed_candidates=decisions, failed_candidates=[])

    with patch("app.api.main.screen_documents", return_value=fake_result):
        files = [
            ("job_description", ("jd.txt", b"Job text", "text/plain")),
            ("resumes", ("c_tuesday.txt", b"Tuesday", "text/plain")),
        ]
        # Override reference_time to Tuesday 11:00 UTC
        response = client.post("/screen?reference_time=2025-01-07T11:00:00Z", files=files)

        assert response.status_code == 200
        sched = response.json()["results"][0]["scheduling_result"]

        assert sched["status"] == "scheduled"
        assert sched["scheduled_at"] == "2025-01-07T11:00:00Z"

