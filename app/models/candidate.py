"""Pydantic models for structured candidate profiles extracted from resumes."""

from typing import List, Optional
from pydantic import BaseModel, Field


class WorkExperience(BaseModel):
    """Structured work experience record."""

    title: str = Field(..., description="Job title / role name")
    company: str = Field(..., description="Company or organization name")
    duration_months: Optional[int] = Field(
        default=None,
        description="Total duration in months. None if dates or duration are missing or unclear — do not guess.",
    )
    description: str = Field(
        ...,
        description="Detailed description of responsibilities, achievements, and technologies used (preserving quotes/metrics)",
    )


class Education(BaseModel):
    """Structured education record."""

    degree: str = Field(..., description="Degree or certificate name (e.g., B.S. Computer Science)")
    institution: Optional[str] = Field(
        default=None,
        description="University, college, or educational institution name",
    )
    graduation_year: Optional[int] = Field(
        default=None,
        description="Graduation year (4 digits), or None if missing or unclear",
    )
    field_of_study: Optional[str] = Field(
        default=None,
        description="Major or field of study if distinct from degree name",
    )


class Project(BaseModel):
    """Structured personal or open-source project record."""

    name: str = Field(..., description="Project name or repository name")
    description: str = Field(
        ...,
        description="Detailed description of technical scope, architecture, and impact",
    )
    url: Optional[str] = Field(
        default=None,
        description="Repository URL or project link if available",
    )
    technologies: List[str] = Field(
        default_factory=list,
        description="Technologies, frameworks, and tools used",
    )


class CandidateProfile(BaseModel):
    """Complete structured candidate profile extracted from a resume."""

    candidate_id: str = Field(
        default="",
        description="Unique identifier for the candidate (e.g. slug or candidate ID)",
    )
    candidate_name: Optional[str] = Field(
        default=None,
        description="Candidate's full name if present",
    )
    raw_text: str = Field(
        default="",
        description="Original unparsed resume text",
    )
    skills: List[str] = Field(
        default_factory=list,
        description="Extracted technical and domain skills",
    )
    experience: List[WorkExperience] = Field(
        default_factory=list,
        description="List of employment history records",
    )
    education: List[Education] = Field(
        default_factory=list,
        description="List of educational qualifications",
    )
    projects: List[Project] = Field(
        default_factory=list,
        description="List of personal projects, open-source work, and code contributions",
    )

    def anonymize(self) -> "CandidateProfile":
        """Produce an anonymized copy for counterfactual fairness auditing.

        Strips identity and demographic context fields:
        - candidate_name: set to None
        - education institution: set to None
        - education graduation_year: set to None

        Preserves all competency-relevant evidence:
        - skills
        - work experience (titles, companies, durations, descriptions)
        - education degrees and fields of study
        - projects, URLs, and technologies
        """
        anonymized = self.model_copy(deep=True)
        anonymized.candidate_name = None
        for edu in anonymized.education:
            edu.institution = None
            edu.graduation_year = None
        return anonymized
