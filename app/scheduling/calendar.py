"""Mock in-memory calendar connector for interview scheduling.

Locked Phase 10 implementation decisions:
- In-memory state: Stores booked slots in memory without database or external API dependencies.
- Strict working hours: 09:00–17:00 Monday–Friday UTC.
- 30-minute fixed slot duration.
- Strict 48-hour wall-clock search window [reference_time, reference_time + 48h).
- Timezone-aware UTC enforcement: Naive datetimes are strictly rejected.
- Authoritative booking: book_slot() verifies slot availability at booking time
  and serves as the single source of truth against booking conflicts.
"""

from datetime import datetime, time, timedelta, timezone
from typing import List, Set

# Timezone constant
SCHEDULING_TIMEZONE = timezone.utc

# Working hours constants
WORKING_HOURS_START = time(9, 0)
WORKING_HOURS_END = time(17, 0)  # 17:00 UTC
LATEST_SLOT_START = time(16, 30)  # 30-min slot ends at 17:00

# Scheduling constraints
INTERVIEW_DURATION_MINUTES = 30
AVAILABILITY_WINDOW_HOURS = 48


def _validate_timezone_aware(dt: datetime, param_name: str = "datetime") -> datetime:
    """Validate that a datetime is timezone-aware and convert to UTC.

    Raises ValueError if dt is naive.
    """
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        raise ValueError(f"{param_name} must be a timezone-aware datetime")
    return dt.astimezone(SCHEDULING_TIMEZONE)


def _is_valid_working_slot(dt: datetime) -> bool:
    """Check if a datetime corresponds to a valid 30-minute weekday working slot."""
    # Weekday check: Monday (0) to Friday (4)
    if dt.weekday() >= 5:
        return False

    # 30-minute alignment check
    if dt.second != 0 or dt.microsecond != 0 or dt.minute not in (0, 30):
        return False

    # Working hours check: 09:00 to 16:30 start
    slot_time = dt.time()
    return WORKING_HOURS_START <= slot_time <= LATEST_SLOT_START


class MockCalendar:
    """Deterministic in-memory calendar connector."""

    def __init__(self) -> None:
        self._booked_slots: Set[datetime] = set()

    @property
    def booked_slots(self) -> Set[datetime]:
        """Return a copy of currently booked slots."""
        return set(self._booked_slots)

    def is_booked(self, slot: datetime) -> bool:
        """Check whether a given slot is already booked."""
        slot_utc = _validate_timezone_aware(slot, "slot")
        return slot_utc in self._booked_slots

    def get_available_slots(self, reference_time: datetime) -> List[datetime]:
        """Generate all unbooked 30-minute interview slots in [reference_time, reference_time + 48h).

        Rules:
        - Search window is half-open: [reference_time, reference_time + 48 wall-clock hours).
        - No weekend extension: strictly 48 wall-clock hours.
        - Working hours only: Mon–Fri 09:00–17:00 UTC (latest slot starts at 16:30).
        - Excludes already-booked slots.
        - Returns slots in chronological order.
        """
        ref_utc = _validate_timezone_aware(reference_time, "reference_time")
        window_end = ref_utc + timedelta(hours=AVAILABILITY_WINDOW_HOURS)

        # Align reference_time up to the next 30-minute boundary
        candidate = ref_utc.replace(second=0, microsecond=0)
        if candidate < ref_utc:
            candidate += timedelta(minutes=1)

        rem = candidate.minute % INTERVIEW_DURATION_MINUTES
        if rem != 0:
            candidate += timedelta(minutes=(INTERVIEW_DURATION_MINUTES - rem))

        available: List[datetime] = []

        # Iterate in 30-minute increments across the half-open window
        while candidate < window_end:
            if _is_valid_working_slot(candidate):
                if candidate not in self._booked_slots:
                    available.append(candidate)
            candidate += timedelta(minutes=INTERVIEW_DURATION_MINUTES)

        return available

    def book_slot(self, slot: datetime) -> bool:
        """Authoritative booking operation.

        Independently validates slot format, working hours, and current booking state.
        Returns True if the slot was free and is now booked; False otherwise.
        """
        slot_utc = _validate_timezone_aware(slot, "slot")

        if not _is_valid_working_slot(slot_utc):
            return False

        if slot_utc in self._booked_slots:
            return False

        self._booked_slots.add(slot_utc)
        return True

    def clear(self) -> None:
        """Clear all bookings in memory (useful for testing)."""
        self._booked_slots.clear()
