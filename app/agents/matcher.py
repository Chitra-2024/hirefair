"""Matcher agent module using Gemini Structured Outputs to evaluate candidate profiles against job rubrics."""

from typing import Optional
from app.models.candidate import CandidateProfile
from app.models.job_description import ParsedJobDescription
from app.models.score import MatchResult, ScoreRecord
from app.utils.llm_client import generate_structured

DEFAULT_MODEL = "gemini-3.6-flash"

MATCHER_SYSTEM_PROMPT = """You are an expert technical evaluator and Matcher agent for HireFair, a fairness-aware resume screening system.

Your task is to independently evaluate a candidate's qualifications against every criterion in a job description rubric.

Follow these strict scoring and evidence rules:

1. One ScoreRecord Per Rubric Criterion:
   - You must produce exactly one ScoreRecord for every criterion specified in the rubric.
   - Use the exact criterion_id defined in the rubric.
   - Set candidate_id to the candidate's ID.

2. Scoring Scale (0.0 to 1.0):
   - Score each criterion independently on a continuous scale from 0.0 (no evidence / completely unqualified) to 1.0 (exceptional evidence fully satisfying or exceeding requirements).
   - Consider both MUST_HAVE and NICE_TO_HAVE criteria fairly according to their descriptions, weights, and expected evidence types.

3. Evidence Requirement (CRITICAL CODE-LEVEL INVARIANT):
   - If score > 0.3, evidence_quote MUST contain an exact, verbatim direct quote from the candidate's original resume text.
   - If score <= 0.3 (e.g. criterion unmet, no evidence, or negligible match), set evidence_quote to an empty string "".
   - NEVER fabricate, hallucinate, invent, or extrapolate quotes.
   - NEVER paraphrase text and present it as a quote.
   - Quotes must be drawn directly from the candidate's original resume text provided.

4. Legitimate Evidence Sources:
   - Valid evidence includes work experience (responsibilities, achievements), personal projects, open-source contributions, technical skills, and academic coursework/degrees as documented in the resume.
   - Evaluate technical competencies objectively without penalizing career changes, employment gaps, non-traditional education, or lack of traditional corporate job titles if legitimate evidence of the required skill is present.

5. Confidence (0.0 to 1.0):
   - Assign a confidence score between 0.0 and 1.0 reflecting your certainty in the score and available evidence.
"""


def _format_matcher_prompt(
    candidate: CandidateProfile, rubric: ParsedJobDescription
) -> str:
    """Format the evaluation prompt for the Matcher containing candidate details and rubric criteria."""
    rubric_lines = []
    for req in rubric.requirements:
        rubric_lines.append(
            f"- Criterion ID: {req.criterion_id}\n"
            f"  Type: {req.type.value}\n"
            f"  Weight: {req.weight}/5\n"
            f"  Expected Evidence: {req.evidence_type}\n"
            f"  Description: {req.description}"
        )
    rubric_text = "\n".join(rubric_lines)

    prompt = (
        f"JOB EVALUATION RUBRIC:\n"
        f"Role: {rubric.job_title or 'Not specified'}\n"
        f"Company: {rubric.company or 'Not specified'}\n"
        f"Criteria ({len(rubric.requirements)} total):\n"
        f"{rubric_text}\n\n"
        f"==================================================\n"
        f"CANDIDATE INFORMATION:\n"
        f"Candidate ID: {candidate.candidate_id}\n"
        f"Candidate Name: {candidate.candidate_name or 'N/A'}\n\n"
        f"CANDIDATE ORIGINAL RESUME TEXT (Source of truth for direct quotes):\n"
        f"\"\"\"\n{candidate.raw_text}\n\"\"\"\n\n"
        f"EVALUATION INSTRUCTIONS:\n"
        f"Evaluate candidate '{candidate.candidate_id}' against all {len(rubric.requirements)} rubric criteria listed above.\n"
        f"Return a MatchResult containing a ScoreRecord for each criterion in the rubric.\n"
        f"Remember: If score > 0.3, evidence_quote must contain a verbatim quote directly from the resume text above."
    )
    return prompt


class Matcher:
    """Agent responsible for scoring candidate profiles against job rubric criteria."""

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model

    def match(
        self, candidate: CandidateProfile, rubric: ParsedJobDescription
    ) -> MatchResult:
        """Score a candidate profile against all criteria in the job rubric.

        Args:
            candidate: Structured candidate profile including raw_text.
            rubric: Parsed job description containing criteria requirements.

        Returns:
            MatchResult containing one ScoreRecord per rubric criterion.

        Raises:
            ValueError: If candidate or rubric is invalid/empty, or if score records violate evidence rules.
        """
        if candidate is None or not isinstance(candidate, CandidateProfile):
            raise ValueError("CandidateProfile must be provided.")
        if not candidate.raw_text or not candidate.raw_text.strip():
            raise ValueError("Candidate raw_text cannot be empty.")
        if rubric is None or not isinstance(rubric, ParsedJobDescription):
            raise ValueError("ParsedJobDescription rubric must be provided.")
        if not rubric.requirements:
            raise ValueError("Rubric requirements cannot be empty.")

        prompt = _format_matcher_prompt(candidate, rubric)

        result: MatchResult = generate_structured(
            prompt=prompt,
            system=MATCHER_SYSTEM_PROMPT,
            response_model=MatchResult,
            model=self.model,
        )

        # Enforce candidate_id and invariant check on all records
        for record in result.scores:
            if not record.candidate_id:
                record.candidate_id = candidate.candidate_id
            if record.score > 0.3 and (
                not record.evidence_quote or not record.evidence_quote.strip()
            ):
                raise ValueError(
                    f"Matcher validation error: criterion '{record.criterion_id}' has score "
                    f"{record.score} > 0.3 without a non-whitespace evidence quote."
                )

        return result

    # Method aliases for flexibility
    score = match
    match_candidate = match


def match_candidate(
    candidate: CandidateProfile,
    rubric: ParsedJobDescription,
    model: str = DEFAULT_MODEL,
) -> MatchResult:
    """Convenience functional wrapper around Matcher."""
    matcher = Matcher(model=model)
    return matcher.match(candidate=candidate, rubric=rubric)
