"""Interview scheduling policy layer.

Locked Phase 10 implementation decisions:
- InterviewScheduler independently verifies CLEARED eligibility from RoutingResult.
- calendar.get_available_slots() is called once; the returned list is iterated in
  chronological order and each slot attempted via book_slot() which is authoritative.
- book_slot() failure on a candidate slot (race/conflict) causes iteration to continue
  to the next slot rather than giving up immediately.
- RoutingResult and individual RouteDecision objects are never mutated.
- No datetime.now() / utcnow() inside this module; reference_time is always supplied
  by the caller.
- 48-hour half-open window [reference_time, reference_time + 48h) is enforced by
  MockCalendar; the scheduler simply passes reference_time through.
- Naive datetimes are rejected by MockCalendar._validate_timezone_aware(); the
  scheduler surfaces that ValueError unchanged.
"""

from datetime import datetime

from app.models.routing import RoutingDecision, RoutingResult
from app.models.scheduling import SchedulingResult, SchedulingStatus
from app.scheduling.calendar import MockCalendar


class InterviewScheduler:
    """Policy layer that schedules CLEARED candidates using a MockCalendar.

    Each InterviewScheduler instance owns one MockCalendar.  Booking state is
    kept in memory and does not survive process restart (v1 limitation).
    """

    def __init__(self, calendar: MockCalendar | None = None) -> None:
        """Initialise with an optional pre-configured calendar.

        If *calendar* is not supplied a fresh MockCalendar is created.  Passing
        a shared calendar instance allows multiple scheduler calls to share
        booking state (e.g. for multi-candidate batch scheduling in tests).
        """
        self._calendar: MockCalendar = calendar if calendar is not None else MockCalendar()

    @property
    def calendar(self) -> MockCalendar:
        """Expose the underlying calendar for inspection in tests."""
        return self._calendar

    def schedule_candidate(
        self,
        routing_result: RoutingResult,
        candidate_id: str,
        reference_time: datetime,
    ) -> SchedulingResult:
        """Attempt to schedule a single candidate for an interview.

        Args:
            routing_result:  The full RoutingResult from the Router phase.
                             This object (and its nested decisions) is never mutated.
            candidate_id:    The candidate to schedule.  Must match a RouteDecision
                             in *routing_result* with decision == CLEARED.
            reference_time:  Timezone-aware UTC datetime representing "now".
                             Naive datetimes cause ValueError (propagated from
                             MockCalendar._validate_timezone_aware).

        Returns:
            SchedulingResult with status SCHEDULED, NOT_SCHEDULED_NO_AVAILABILITY,
            or INELIGIBLE.
        """
        # --- Validate reference_time is timezone-aware (surfaced from calendar) ---
        # We do a lightweight check here so the error message is consistent even
        # before the calendar is touched for INELIGIBLE paths.
        if reference_time.tzinfo is None or reference_time.tzinfo.utcoffset(reference_time) is None:
            raise ValueError("reference_time must be a timezone-aware datetime")

        # --- Eligibility check (independent; never touches the calendar) ----------
        route_decision = None
        for rd in routing_result.routed_candidates:
            if rd.candidate_id == candidate_id:
                route_decision = rd
                break

        if route_decision is None:
            return SchedulingResult(
                candidate_id=candidate_id,
                status=SchedulingStatus.INELIGIBLE,
                reason=f"Candidate '{candidate_id}' not found in routing result.",
            )

        if route_decision.decision != RoutingDecision.CLEARED:
            return SchedulingResult(
                candidate_id=candidate_id,
                status=SchedulingStatus.INELIGIBLE,
                reason=(
                    f"Candidate '{candidate_id}' is not CLEARED "
                    f"(decision: {route_decision.decision.value}); scheduling skipped."
                ),
            )

        # --- Calendar interaction (CLEARED path only) ----------------------------
        available_slots = self._calendar.get_available_slots(reference_time)

        for slot in available_slots:
            booked = self._calendar.book_slot(slot)
            if booked:
                return SchedulingResult(
                    candidate_id=candidate_id,
                    status=SchedulingStatus.SCHEDULED,
                    scheduled_at=slot,
                    reason=(
                        f"Interview scheduled at {slot.isoformat()} UTC "
                        f"for candidate '{candidate_id}'."
                    ),
                )
            # book_slot() returned False: slot was taken between availability lookup
            # and booking attempt (conflict); continue to the next slot.

        # No slot could be successfully booked within the 48-hour window.
        return SchedulingResult(
            candidate_id=candidate_id,
            status=SchedulingStatus.NOT_SCHEDULED_NO_AVAILABILITY,
            reason=(
                f"No available interview slot found within 48 hours of "
                f"{reference_time.isoformat()} for candidate '{candidate_id}'."
            ),
        )
