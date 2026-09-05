"""Pydantic models for job description parsing and evaluation rubric."""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class RequirementType(str, Enum):
    """Classification of requirement importance."""

    MUST_HAVE = "must_have"
    NICE_TO_HAVE = "nice_to_have"


class JobRequirement(BaseModel):
    """A single criterion/requirement in the job description rubric."""

    criterion_id: str = Field(
        ...,
        description="Unique identifier for the requirement (e.g., REQ-01, python_exp)",
    )
    description: str = Field(
        ...,
        description="Clear textual description of the criterion",
    )
    weight: int = Field(
        ...,
        ge=1,
        le=5,
        description="Importance weight from 1 (lowest) to 5 (highest)",
    )
    type: RequirementType = Field(
        ...,
        description="Requirement category: must_have or nice_to_have",
    )
    evidence_type: str = Field(
        ...,
        description="Expected form of competency evidence (e.g., work experience, projects, education, certifications)",
    )


class ParsedJobDescription(BaseModel):
    """Top-level model representing a complete parsed job description and its rubric."""

    job_title: Optional[str] = Field(
        default=None,
        description="Title of the role (e.g., Senior Data Engineer)",
    )
    company: Optional[str] = Field(
        default=None,
        description="Hiring company name",
    )
    requirements: List[JobRequirement] = Field(
        default_factory=list,
        description="List of rubric requirements extracted from the job description",
    )

    @property
    def must_haves(self) -> List[JobRequirement]:
        """Return list of must-have requirements."""
        return [r for r in self.requirements if r.type == RequirementType.MUST_HAVE]

    @property
    def nice_to_haves(self) -> List[JobRequirement]:
        """Return list of nice-to-have requirements."""
        return [r for r in self.requirements if r.type == RequirementType.NICE_TO_HAVE]


# Convenience alias
JobRubric = ParsedJobDescription
