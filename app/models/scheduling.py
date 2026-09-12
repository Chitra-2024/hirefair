"""Pydantic models for candidate interview scheduling."""

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class SchedulingStatus(str, Enum):
    """Categorical outcome of an interview scheduling attempt."""

    SCHEDULED = "scheduled"
    NOT_SCHEDULED_NO_AVAILABILITY = "not_scheduled_no_availability"
    INELIGIBLE = "ineligible"


class SchedulingResult(BaseModel):
    """Structured result of an interview scheduling attempt for a candidate."""

    candidate_id: str = Field(..., description="ID of the candidate being scheduled")
    status: SchedulingStatus = Field(..., description="Outcome status of the scheduling attempt")
    scheduled_at: Optional[datetime] = Field(
        default=None,
        description="Timezone-aware UTC datetime of the booked 30-minute slot, if scheduled",
    )
    duration_minutes: int = Field(
        default=30,
        description="Duration of the interview in minutes (fixed to 30 for v1)",
    )
    reason: str = Field(..., description="Human-readable explanation of the scheduling outcome")

    @field_validator("scheduled_at")
    @classmethod
    def validate_timezone_aware(cls, v: Optional[datetime]) -> Optional[datetime]:
        """Ensure scheduled_at datetime is timezone-aware."""
        if v is not None and (v.tzinfo is None or v.tzinfo.utcoffset(v) is None):
            raise ValueError("scheduled_at must be a timezone-aware datetime")
        return v
