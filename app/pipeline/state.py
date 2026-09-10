"""LangGraph state schema for per-candidate processing."""

from typing import Optional, TypedDict

from app.models.audit import AuditRecord
from app.models.candidate import CandidateProfile
from app.models.job_description import ParsedJobDescription
from app.models.score import MatchResult


class CandidateGraphState(TypedDict, total=False):
    """Explicit state representation accumulated during per-candidate LangGraph execution.

    Contains initial inputs and intermediate outputs produced by agent nodes.
    Errors are deliberately excluded from this state: exceptions propagate to the
    outer batch driver, which isolates failures at the candidate boundary.
    """

    candidate_id: str
    resume_text: str
    rubric: ParsedJobDescription
    candidate_profile: Optional[CandidateProfile]
    match_result: Optional[MatchResult]
    audit_record: Optional[AuditRecord]
