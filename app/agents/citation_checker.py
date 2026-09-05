"""LLM-based citation validity checker for the Fairness Auditor (Check B)."""

from typing import List
from pydantic import BaseModel, Field

from app.models.score import ScoreRecord
from app.utils.llm_client import generate_structured


CITATION_CHECKER_SYSTEM_PROMPT = """You are an independent citation validity auditor for HireFair, a fairness-aware resume screening system.

Your task is to independently verify whether a provided evidence quote actually supports a specific rubric criterion.

Rules:
1. Evaluate only whether the given evidence quote demonstrates the criterion — not whether the candidate is good or bad.
2. A citation is VALID only if the quote provides clear, direct evidence that the candidate meets the criterion.
3. A citation is INVALID if:
   - The quote is a keyword coincidence (contains a relevant word but provides no actual competency demonstration).
   - The quote is from a different context than the criterion requires (e.g., mentions a tool passively, not as active competency).
   - The quote does not logically support the criterion score.
4. Do NOT invent evidence. Do NOT fabricate citations.
5. Provide a brief, clear reason for your decision.
"""


class CitationCheckResult(BaseModel):
    """Result of checking one criterion's citation validity."""

    criterion_id: str = Field(..., description="The rubric criterion being evaluated")
    is_valid: bool = Field(
        ...,
        description="True if the evidence quote directly and meaningfully supports the criterion",
    )
    reason: str = Field(
        ...,
        description="Brief explanation of why the citation is valid or invalid",
    )


class CitationCheckBatch(BaseModel):
    """Batch result for all citation checks in one audit pass."""

    results: List[CitationCheckResult] = Field(
        default_factory=list,
        description="One CitationCheckResult per evaluated criterion",
    )


def check_citations(
    score_records: List[ScoreRecord],
    rubric_descriptions: dict,
    model: str = "gemini-2.5-flash",
) -> CitationCheckBatch:
    """Run LLM-based citation validity checks for all applicable score records.

    Only evaluates records where score > 0.3 (i.e., where evidence is required and expected).
    Records with score <= 0.3 are skipped — a missing evidence quote at low scores is not a failure.

    Args:
        score_records: All ScoreRecord instances from the Matcher result.
        rubric_descriptions: dict mapping criterion_id -> criterion description string.
        model: Gemini model identifier.

    Returns:
        CitationCheckBatch containing one result per applicable (score > 0.3) criterion.
    """
    # Only check citations where a quote is required (score > 0.3)
    applicable = [r for r in score_records if r.score > 0.3]

    if not applicable:
        return CitationCheckBatch(results=[])

    lines = []
    for rec in applicable:
        desc = rubric_descriptions.get(rec.criterion_id, "(no description)")
        lines.append(
            f"Criterion ID: {rec.criterion_id}\n"
            f"Criterion Description: {desc}\n"
            f"Score Assigned: {rec.score:.2f}\n"
            f"Evidence Quote Provided:\n\"\"\"{rec.evidence_quote}\"\"\""
        )

    checks_text = "\n\n---\n\n".join(lines)
    prompt = (
        f"For each of the following criterion/evidence pairs, determine whether the evidence quote "
        f"actually and directly supports the criterion.\n\n"
        f"{checks_text}\n\n"
        f"Return a CitationCheckBatch with one CitationCheckResult per criterion listed above."
    )

    return generate_structured(
        prompt=prompt,
        system=CITATION_CHECKER_SYSTEM_PROMPT,
        response_model=CitationCheckBatch,
        model=model,
    )
