"""Data models for candidate orchestration inputs and pipeline outputs."""

from typing import List
from pydantic import BaseModel, Field

from app.models.audit import AuditRecord
from app.models.candidate import CandidateProfile
from app.models.score import ScoreRecord


class CandidateInput(BaseModel):
    """Input payload representing an unscored candidate resume."""

    candidate_id: str
    resume_text: str


class FailedCandidate(BaseModel):
    """Record of a candidate that failed processing during pipeline execution."""

    candidate_id: str
    reason: str


class CandidateResult(BaseModel):
    """Evaluation result for a single candidate that completed the pipeline.

    Downstream consumers (such as the Router) inspect the unwrapped ScoreRecord
    list, candidate_profile, and audit_record directly without relying on internal
    agent batching wrappers.
    """

    candidate_id: str
    candidate_profile: CandidateProfile
    scores: List[ScoreRecord]
    audit_record: AuditRecord


class PipelineResult(BaseModel):
    """Batch execution summary returned by the HireFair pipeline."""

    results: List[CandidateResult] = Field(default_factory=list)
    failed_candidates: List[FailedCandidate] = Field(default_factory=list)
