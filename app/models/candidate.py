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

        Strips identity and demographic context from both structured fields and raw_text:
        - candidate_name: set to None; occurrences in raw_text replaced with [CANDIDATE]
        - education institution: set to None; occurrences in raw_text replaced with [INSTITUTION]
        - education graduation_year: set to None; occurrences in raw_text replaced with [YEAR]

        Preserves all competency-relevant evidence:
        - skills
        - work experience (titles, companies, durations, descriptions)
        - education degrees and fields of study
        - projects, URLs, and technologies
        - company names (not anonymized — companies are not identity context)

        Note: raw_text scrubbing uses exact-string replacement of known values.
        It does not detect indirect references or every possible re-identification path.
        """
        anonymized = self.model_copy(deep=True)

        # Scrub raw_text: replace known identity tokens with placeholders
        scrubbed = anonymized.raw_text

        # Replace candidate name
        if anonymized.candidate_name:
            scrubbed = scrubbed.replace(anonymized.candidate_name, "[CANDIDATE]")

        # Replace institution names and graduation years before zeroing out structured fields
        for edu in anonymized.education:
            if edu.institution:
                scrubbed = scrubbed.replace(edu.institution, "[INSTITUTION]")
            if edu.graduation_year is not None:
                scrubbed = scrubbed.replace(str(edu.graduation_year), "[YEAR]")

        anonymized.raw_text = scrubbed

        # Zero out structured identity fields
        anonymized.candidate_name = None
        for edu in anonymized.education:
            edu.institution = None
            edu.graduation_year = None

        return anonymized
