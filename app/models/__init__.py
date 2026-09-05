# Pydantic schemas (shared data contracts)

from app.models.candidate import (
    CandidateProfile,
    Education,
    Project,
    WorkExperience,
)
from app.models.job_description import (
    JobRequirement,
    JobRubric,
    ParsedJobDescription,
    RequirementType,
)
from app.models.score import (
    MatchResult,
    ScoreRecord,
)

__all__ = [
    "CandidateProfile",
    "Education",
    "JobRequirement",
    "JobRubric",
    "MatchResult",
    "ParsedJobDescription",
    "Project",
    "RequirementType",
    "ScoreRecord",
    "WorkExperience",
]

