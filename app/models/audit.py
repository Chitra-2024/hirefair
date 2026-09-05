"""Pydantic models for Fairness Auditor results."""

from typing import Optional
from pydantic import BaseModel, Field


class AuditRecord(BaseModel):
    """Complete result of a fairness audit performed on a Matcher result.

    Design-document fields (original specification):
    - candidate_id
    - original_score
    - anonymized_score
    - delta
    - citation_valid
    - non_traditional_evidence_found
    - flagged_reason

    Implementation additions (not in original design doc):
    - flagged (bool): Explicit machine-readable downstream signal for the Router.
      Avoids requiring downstream components to infer flagged status from
      whether flagged_reason is non-empty.
    - repair_applied (bool): Records whether Option A fallback was applied
      (anonymized_score adopted in place of a second Matcher call).
    """

    candidate_id: str = Field(
        ...,
        description="Unique identifier for the candidate being audited",
    )

    # --- Counterfactual scoring ---
    original_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Weighted aggregate score from the original (non-anonymized) Matcher result. "
            "None if total rubric weight is zero."
        ),
    )
    anonymized_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Weighted aggregate score from the anonymized counterfactual Matcher result. "
            "None if total rubric weight is zero."
        ),
    )
    delta: Optional[float] = Field(
        default=None,
        description=(
            "Absolute difference between original_score and anonymized_score. "
            "None if either score is unavailable."
        ),
    )

    # --- Citation validity ---
    citation_valid: bool = Field(
        default=True,
        description=(
            "True only if all applicable criterion citations are verified as valid. "
            "False if any applicable citation is judged not to actually support its criterion."
        ),
    )

    # --- Non-traditional evidence ---
    non_traditional_evidence_found: bool = Field(
        default=False,
        description=(
            "True if relevant non-traditional competency evidence exists for any rubric "
            "criterion where the original score is at or below NON_TRADITIONAL_SCORE_THRESHOLD."
        ),
    )

    # --- Flags ---
    flagged: bool = Field(
        default=False,
        description=(
            "Machine-readable downstream signal. True if any fairness concern was detected. "
            "Implementation addition: prevents downstream Router from inferring flag status "
            "from whether flagged_reason is non-empty."
        ),
    )
    flagged_reason: str = Field(
        default="",
        description=(
            "Human-readable explanation of why the candidate was flagged. "
            "Empty string when flagged=False."
        ),
    )

    # --- Self-repair ---
    repair_applied: bool = Field(
        default=False,
        description=(
            "True when Option A fallback was triggered: both large counterfactual delta "
            "AND citation failure were detected, and anonymized_score was adopted as the "
            "authoritative final score without making an additional Matcher/Gemini call. "
            "Implementation addition."
        ),
    )
