"""Data models for Router screening decisions and batch routing output."""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

from app.models.result import CandidateResult, FailedCandidate


class RoutingDecision(str, Enum):
    """Categorical outcome of the Router screening evaluation."""

    CLEARED = "cleared"
    FLAGGED_FOR_REVIEW = "flagged_for_review"
    NOT_QUALIFIED = "not_qualified"
    INCOMPLETE_DATA = "incomplete_data"
    DUPLICATE = "duplicate"
    FAILED = "failed"

    # Convenient alias for fairness-flagged outcomes
    FAIRNESS_FLAGGED = "flagged_for_review"


class RouteDecision(BaseModel):
    """Routing evaluation decision for a single candidate.

    Composes CandidateResult to preserve CandidateProfile, scores, and AuditRecord
    for downstream auto-scheduling and human reviewer workflows without duplicate fields.
    """

    candidate_id: str
    decision: RoutingDecision
    reason: str
    final_score: Optional[float] = None
    duplicate_of: Optional[str] = None
    candidate_result: Optional[CandidateResult] = None


class RoutingResult(BaseModel):
    """Batch screening summary produced by the Router."""

    routed_candidates: List[RouteDecision] = Field(default_factory=list)
    failed_candidates: List[FailedCandidate] = Field(default_factory=list)
