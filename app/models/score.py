"""Pydantic models for scoring candidate profiles against job rubric criteria."""

from typing import List
from pydantic import BaseModel, Field, model_validator


class ScoreRecord(BaseModel):
    """Evaluation score record for a single rubric criterion against a candidate."""

    candidate_id: str = Field(
        ...,
        description="Unique identifier for the candidate being evaluated",
    )
    criterion_id: str = Field(
        ...,
        description="Unique identifier of the rubric criterion evaluated",
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Competency score from 0.0 to 1.0",
    )
    evidence_quote: str = Field(
        default="",
        description="Verbatim direct quote from the resume supporting the score. Required if score > 0.3.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence in the evaluation from 0.0 to 1.0",
    )

    @model_validator(mode="after")
    def validate_evidence_quote(self) -> "ScoreRecord":
        """Enforce the project invariant: score > 0.3 requires non-empty evidence quote."""
        if self.score > 0.3:
            if not self.evidence_quote or not self.evidence_quote.strip():
                raise ValueError(
                    f"Criterion '{self.criterion_id}': score {self.score} > 0.3 "
                    "requires a non-empty direct evidence_quote from the resume."
                )
        return self


class MatchResult(BaseModel):
    """Wrapper containing all criterion scores produced for a candidate."""

    scores: List[ScoreRecord] = Field(
        default_factory=list,
        description="List of criterion evaluation scores for the candidate",
    )
