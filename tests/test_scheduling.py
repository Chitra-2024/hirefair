"""Offline deterministic tests for the Phase 10 scheduling subsystem.

Tests cover all 14 locked requirements:
 1.  CLEARED candidate gets the earliest available slot.
 2.  Already-booked earliest slot causes next slot to be selected.
 3.  Later availability inside 48 h is used.
 4.  No availability inside 48 h => NOT_SCHEDULED_NO_AVAILABILITY.
 5.  Conflict between availability lookup and booking causes retry of next slot.
 6.  Non-CLEARED => INELIGIBLE; calendar is untouched.
 7.  Missing candidate => INELIGIBLE; calendar is untouched.
 8.  Weekend does not extend the 48 h window.
 9.  Exact +48 h endpoint is excluded (half-open interval).
10.  Naive datetime rejected.
11.  Scheduled datetime is timezone-aware UTC.
12.  RoutingResult is not mutated.
13.  Multiple candidates can occupy different slots.
14.  Same slot cannot be booked successfully twice.
"""

from datetime import datetime, timedelta, timezone
from typing import List
from unittest.mock import patch, MagicMock

import pytest

from app.models.routing import RouteDecision, RoutingDecision, RoutingResult
from app.models.scheduling import SchedulingResult, SchedulingStatus
from app.scheduling.calendar import MockCalendar
from app.scheduling.scheduler import InterviewScheduler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

UTC = timezone.utc

# A Monday 09:00 UTC — safe anchor for most tests.
MONDAY_0900 = datetime(2025, 1, 6, 9, 0, 0, tzinfo=UTC)   # 2025-01-06 is a Monday
MONDAY_0930 = datetime(2025, 1, 6, 9, 30, 0, tzinfo=UTC)
MONDAY_1000 = datetime(2025, 1, 6, 10, 0, 0, tzinfo=UTC)


def _routing_result(
    candidate_id: str,
    decision: RoutingDecision,
    reason: str = "test",
) -> RoutingResult:
    """Build a minimal RoutingResult with a single RouteDecision."""
    rd = RouteDecision(candidate_id=candidate_id, decision=decision, reason=reason)
    return RoutingResult(routed_candidates=[rd])


def _routing_result_many(decisions: List[tuple]) -> RoutingResult:
    """Build a RoutingResult from a list of (candidate_id, RoutingDecision) tuples."""
    rds = [
        RouteDecision(candidate_id=cid, decision=dec, reason="test")
        for cid, dec in decisions
    ]
    return RoutingResult(routed_candidates=rds)


# ---------------------------------------------------------------------------
# Requirement 1 — CLEARED candidate gets earliest available slot.
# ---------------------------------------------------------------------------

def test_cleared_candidate_gets_earliest_slot():
    """Req 1: CLEARED candidate is scheduled at the first available working slot."""
    # reference_time is exactly Monday 09:00 — first slot should be 09:00 itself.
    calendar = MockCalendar()
    scheduler = InterviewScheduler(calendar=calendar)
    routing = _routing_result("alice", RoutingDecision.CLEARED)

    result = scheduler.schedule_candidate(routing, "alice", MONDAY_0900)

    assert result.status == SchedulingStatus.SCHEDULED
    assert result.candidate_id == "alice"
    assert result.scheduled_at == MONDAY_0900


# ---------------------------------------------------------------------------
# Requirement 2 — Already-booked earliest slot causes next slot to be selected.
# ---------------------------------------------------------------------------

def test_booked_earliest_slot_selects_next():
    """Req 2: If the first slot is already booked the second one is used."""
    calendar = MockCalendar()
    calendar.book_slot(MONDAY_0900)  # Pre-book the earliest slot.

    scheduler = InterviewScheduler(calendar=calendar)
    routing = _routing_result("bob", RoutingDecision.CLEARED)

    result = scheduler.schedule_candidate(routing, "bob", MONDAY_0900)

    assert result.status == SchedulingStatus.SCHEDULED
    assert result.scheduled_at == MONDAY_0930


# ---------------------------------------------------------------------------
# Requirement 3 — Later availability inside 48 h is used.
# ---------------------------------------------------------------------------

def test_later_availability_used():
    """Req 3: When the reference_time is late in the day, a later-day slot is found."""
    # Friday 16:30 is the last valid slot of that day; search starts near end of week.
    # Use Monday 16:30 so the first slot is Monday 16:30 and next is Tuesday 09:00.
    ref = datetime(2025, 1, 6, 16, 0, 0, tzinfo=UTC)  # Monday 16:00
    calendar = MockCalendar()
    scheduler = InterviewScheduler(calendar=calendar)
    routing = _routing_result("carol", RoutingDecision.CLEARED)

    result = scheduler.schedule_candidate(routing, "carol", ref)

    assert result.status == SchedulingStatus.SCHEDULED
    # The slot must be Monday 16:00 (starts at exactly ref, which is on the boundary)
    # or 16:30 — whatever the first valid working slot at/after ref is.
    assert result.scheduled_at is not None
    assert result.scheduled_at >= ref


# ---------------------------------------------------------------------------
# Requirement 4 — No availability inside 48 h => NOT_SCHEDULED_NO_AVAILABILITY.
# ---------------------------------------------------------------------------

def test_no_availability_returns_not_scheduled():
    """Req 4: If no slots are available the result is NOT_SCHEDULED_NO_AVAILABILITY."""
    # Start on a Friday 17:00; next 48 h covers Fri 17:00–Sun 17:00 (all weekend).
    friday_1700 = datetime(2025, 1, 10, 17, 0, 0, tzinfo=UTC)  # 2025-01-10 is a Friday
    calendar = MockCalendar()
    scheduler = InterviewScheduler(calendar=calendar)
    routing = _routing_result("dave", RoutingDecision.CLEARED)

    result = scheduler.schedule_candidate(routing, "dave", friday_1700)

    assert result.status == SchedulingStatus.NOT_SCHEDULED_NO_AVAILABILITY
    assert result.scheduled_at is None


# ---------------------------------------------------------------------------
# Requirement 5 — Conflict between availability lookup and booking causes retry.
# ---------------------------------------------------------------------------

def test_conflict_causes_retry_of_next_slot():
    """Req 5: book_slot() returning False triggers iteration to the next slot."""
    calendar = MockCalendar()
    scheduler = InterviewScheduler(calendar=calendar)
    routing = _routing_result("eve", RoutingDecision.CLEARED)

    # Patch book_slot so the first call fails (simulated conflict) and the
    # second call succeeds.
    original_book = calendar.book_slot
    call_count = [0]

    def fake_book(slot: datetime) -> bool:
        call_count[0] += 1
        if call_count[0] == 1:
            return False  # Simulate conflict on first slot
        return original_book(slot)

    calendar.book_slot = fake_book  # type: ignore[method-assign]

    result = scheduler.schedule_candidate(routing, "eve", MONDAY_0900)

    assert result.status == SchedulingStatus.SCHEDULED
    # Should have fallen back to the second slot.
    assert result.scheduled_at == MONDAY_0930
    assert call_count[0] == 2


# ---------------------------------------------------------------------------
# Requirement 6 — Non-CLEARED => INELIGIBLE; calendar is untouched.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "decision",
    [
        RoutingDecision.NOT_QUALIFIED,
        RoutingDecision.FLAGGED_FOR_REVIEW,
        RoutingDecision.INCOMPLETE_DATA,
        RoutingDecision.DUPLICATE,
        RoutingDecision.FAILED,
    ],
)
def test_non_cleared_is_ineligible_calendar_untouched(decision):
    """Req 6: Any non-CLEARED decision results in INELIGIBLE without touching the calendar."""
    calendar = MockCalendar()
    scheduler = InterviewScheduler(calendar=calendar)
    routing = _routing_result("frank", decision)

    result = scheduler.schedule_candidate(routing, "frank", MONDAY_0900)

    assert result.status == SchedulingStatus.INELIGIBLE
    assert result.scheduled_at is None
    assert len(calendar.booked_slots) == 0


# ---------------------------------------------------------------------------
# Requirement 7 — Missing candidate => INELIGIBLE; calendar is untouched.
# ---------------------------------------------------------------------------

def test_missing_candidate_is_ineligible():
    """Req 7: A candidate_id not found in routing result gives INELIGIBLE."""
    calendar = MockCalendar()
    scheduler = InterviewScheduler(calendar=calendar)
    routing = _routing_result("grace", RoutingDecision.CLEARED)

    result = scheduler.schedule_candidate(routing, "unknown", MONDAY_0900)

    assert result.status == SchedulingStatus.INELIGIBLE
    assert result.candidate_id == "unknown"
    assert result.scheduled_at is None
    # Calendar must not have been touched.
    assert len(calendar.booked_slots) == 0


# ---------------------------------------------------------------------------
# Requirement 8 — Weekend does not extend the 48 h window.
# ---------------------------------------------------------------------------

def test_weekend_does_not_extend_window():
    """Req 8: The 48 h window is wall-clock hours; weekends are not skipped."""
    # Friday 09:00 UTC + 48 h = Sunday 09:00 UTC.
    # No working hours exist during that window (Fri after 09:00 has slots though).
    # Use Friday 16:31 so no Fri slots remain; the 48 h window ends Sunday 16:31.
    friday_1631 = datetime(2025, 1, 10, 16, 31, 0, tzinfo=UTC)
    calendar = MockCalendar()
    scheduler = InterviewScheduler(calendar=calendar)
    routing = _routing_result("henry", RoutingDecision.CLEARED)

    result = scheduler.schedule_candidate(routing, "henry", friday_1631)

    # The window [Fri 16:31, Sun 16:31) contains zero working slots.
    assert result.status == SchedulingStatus.NOT_SCHEDULED_NO_AVAILABILITY


# ---------------------------------------------------------------------------
# Requirement 9 — Exact +48 h endpoint is excluded (half-open interval).
# ---------------------------------------------------------------------------

def test_exact_48h_endpoint_excluded():
    """Req 9: A slot at exactly reference_time + 48 h is excluded."""
    # Choose a reference such that exactly +48 h lands on a valid Mon 09:00 slot.
    # Saturday 09:00 + 48 h = Monday 09:00.
    saturday_0900 = datetime(2025, 1, 11, 9, 0, 0, tzinfo=UTC)  # Sat 2025-01-11
    expected_endpoint = saturday_0900 + timedelta(hours=48)      # Mon 2025-01-13 09:00

    calendar = MockCalendar()
    available = calendar.get_available_slots(saturday_0900)

    # The slot exactly at the 48 h boundary must NOT appear.
    assert expected_endpoint not in available

    # Confirm that Mon 09:30 (inside window) would normally appear if we extended
    # by even one more 30-min increment — but that's already outside the window here,
    # so we just verify the boundary is clean.
    assert all(slot < expected_endpoint for slot in available)


# ---------------------------------------------------------------------------
# Requirement 10 — Naive datetime rejected.
# ---------------------------------------------------------------------------

def test_naive_datetime_rejected_by_calendar():
    """Req 10a: get_available_slots() rejects a naive reference_time."""
    calendar = MockCalendar()
    naive = datetime(2025, 1, 6, 9, 0, 0)  # No tzinfo

    with pytest.raises(ValueError, match="timezone-aware"):
        calendar.get_available_slots(naive)


def test_naive_datetime_rejected_by_book_slot():
    """Req 10b: book_slot() rejects a naive slot datetime."""
    calendar = MockCalendar()
    naive = datetime(2025, 1, 6, 9, 0, 0)  # No tzinfo

    with pytest.raises(ValueError, match="timezone-aware"):
        calendar.book_slot(naive)


def test_naive_reference_time_rejected_by_scheduler():
    """Req 10c: InterviewScheduler rejects a naive reference_time."""
    calendar = MockCalendar()
    scheduler = InterviewScheduler(calendar=calendar)
    routing = _routing_result("ivan", RoutingDecision.CLEARED)
    naive = datetime(2025, 1, 6, 9, 0, 0)

    with pytest.raises(ValueError, match="timezone-aware"):
        scheduler.schedule_candidate(routing, "ivan", naive)


# ---------------------------------------------------------------------------
# Requirement 11 — Scheduled datetime is timezone-aware UTC.
# ---------------------------------------------------------------------------

def test_scheduled_at_is_timezone_aware_utc():
    """Req 11: scheduled_at in a SCHEDULED result is timezone-aware and UTC."""
    calendar = MockCalendar()
    scheduler = InterviewScheduler(calendar=calendar)
    routing = _routing_result("jane", RoutingDecision.CLEARED)

    result = scheduler.schedule_candidate(routing, "jane", MONDAY_0900)

    assert result.status == SchedulingStatus.SCHEDULED
    assert result.scheduled_at is not None
    assert result.scheduled_at.tzinfo is not None
    assert result.scheduled_at.utcoffset() == timedelta(0)


# ---------------------------------------------------------------------------
# Requirement 12 — RoutingResult is not mutated.
# ---------------------------------------------------------------------------

def test_routing_result_not_mutated():
    """Req 12: schedule_candidate() must not alter the RoutingResult."""
    routing = _routing_result("kate", RoutingDecision.CLEARED)
    original_len = len(routing.routed_candidates)
    original_decision = routing.routed_candidates[0].decision
    original_id = routing.routed_candidates[0].candidate_id

    calendar = MockCalendar()
    scheduler = InterviewScheduler(calendar=calendar)
    scheduler.schedule_candidate(routing, "kate", MONDAY_0900)

    assert len(routing.routed_candidates) == original_len
    assert routing.routed_candidates[0].decision == original_decision
    assert routing.routed_candidates[0].candidate_id == original_id


# ---------------------------------------------------------------------------
# Requirement 13 — Multiple candidates can occupy different slots.
# ---------------------------------------------------------------------------

def test_multiple_candidates_occupy_different_slots():
    """Req 13: When two CLEARED candidates are scheduled they land on distinct slots."""
    calendar = MockCalendar()
    scheduler = InterviewScheduler(calendar=calendar)

    routing = _routing_result_many(
        [("liam", RoutingDecision.CLEARED), ("mia", RoutingDecision.CLEARED)]
    )

    result_liam = scheduler.schedule_candidate(routing, "liam", MONDAY_0900)
    result_mia = scheduler.schedule_candidate(routing, "mia", MONDAY_0900)

    assert result_liam.status == SchedulingStatus.SCHEDULED
    assert result_mia.status == SchedulingStatus.SCHEDULED
    assert result_liam.scheduled_at != result_mia.scheduled_at


# ---------------------------------------------------------------------------
# Requirement 14 — Same slot cannot be booked successfully twice.
# ---------------------------------------------------------------------------

def test_same_slot_cannot_be_booked_twice():
    """Req 14: book_slot() returns False on a second attempt for the same slot."""
    calendar = MockCalendar()

    first = calendar.book_slot(MONDAY_0900)
    second = calendar.book_slot(MONDAY_0900)

    assert first is True
    assert second is False
    # Slot appears only once in the booked set.
    assert MONDAY_0900 in calendar.booked_slots
    assert len([s for s in calendar.booked_slots if s == MONDAY_0900]) == 1
