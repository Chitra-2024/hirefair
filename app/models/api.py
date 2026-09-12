"""Pydantic request and response models for the HireFair FastAPI boundary."""

from typing import List, Optional
from pydantic import BaseModel, Field

from app.models.audit import AuditRecord
from app.models.candidate import CandidateProfile
from app.models.result import FailedCandidate
from app.models.routing import RouteDecision, RoutingDecision
from app.models.score import ScoreRecord


from app.models.scheduling import SchedulingResult


class CandidateDecisionResponse(BaseModel):
    """Screening outcome for a single processed candidate.

    Surfaces the decision, reason, score, original profile/audit context,
    and interview scheduling outcome.
    """

    candidate_id: str
    decision: RoutingDecision
    reason: str
    final_score: Optional[float] = None
    duplicate_of: Optional[str] = None
    candidate_profile: Optional[CandidateProfile] = None
    scores: List[ScoreRecord] = Field(default_factory=list)
    audit_record: Optional[AuditRecord] = None
    scheduling_result: Optional[SchedulingResult] = None

    @classmethod
    def from_route_decision(
        cls,
        decision: RouteDecision,
        scheduling_result: Optional[SchedulingResult] = None,
    ) -> "CandidateDecisionResponse":
        """Construct response from an internal RouteDecision object and optional scheduling result."""
        cand_res = decision.candidate_result
        return cls(
            candidate_id=decision.candidate_id,
            decision=decision.decision,
            reason=decision.reason,
            final_score=decision.final_score,
            duplicate_of=decision.duplicate_of,
            candidate_profile=cand_res.candidate_profile if cand_res else None,
            scores=cand_res.scores if cand_res else [],
            audit_record=cand_res.audit_record if cand_res else None,
            scheduling_result=scheduling_result,
        )


class ScreenResponse(BaseModel):
    """Flat batch response representing screening results for all submitted candidates."""

    results: List[CandidateDecisionResponse] = Field(
        default_factory=list,
        description="Flat list of candidate routing decisions in input order",
    )
    failed_candidates: List[FailedCandidate] = Field(
        default_factory=list,
        description="Candidates whose processing failed during pipeline execution",
    )

    @property
    def routed_candidates(self) -> List[CandidateDecisionResponse]:
        """Convenience alias for compatibility with RoutingResult attribute naming."""
        return self.results
