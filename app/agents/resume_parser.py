"""Resume Parser agent module using Gemini Structured Outputs."""

from typing import Optional
from app.models.candidate import CandidateProfile
from app.utils.llm_client import generate_structured

DEFAULT_MODEL = "gemini-2.5-flash"

RESUME_PARSER_SYSTEM_PROMPT = """You are an expert HR and technical resume parser for HireFair, a fairness-aware resume screening system.

Your task is to parse raw resume text into a structured, faithful CandidateProfile.

Follow these strict extraction rules:
1. Fidelity and No Fabrication:
   - Extract ONLY information explicitly stated in the resume text.
   - Do NOT invent, assume, or extrapolate any skills, experiences, degrees, or metrics.
   - Preserve rich descriptive detail, achievements, and technical metrics in the experience and project descriptions so the downstream Matcher can cite them as direct evidence.

2. Incomplete Data Handling (CRITICAL):
   - For experience duration_months: Calculate the duration in months ONLY if clear start and end dates (or an unambiguous duration) are explicitly provided in the resume text.
   - If dates or durations are missing, ambiguous, or unclear, set duration_months to null/None.
   - NEVER guess, assume, or fabricate a default duration.
   - If graduation year is missing from education, set graduation_year to null/None.

3. Experience:
   - Extract every employment position with job title, company name, duration in months (or null if missing/unclear), and full description of responsibilities and accomplishments.

4. Education:
   - Extract all academic degrees, institutions, graduation years (or null if absent), and fields of study.
   - Do not invent institutions or graduation years if not mentioned.

5. Projects and Open-Source Work:
   - Extract personal projects, open-source maintainership or contributions, and technical portfolios.
   - Capture technologies used and scope of work in the description.

6. Skills:
   - Extract all programming languages, technologies, frameworks, tools, and domain competencies mentioned.
"""


class ResumeParser:
    """Agent responsible for parsing resumes into structured CandidateProfiles."""

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model

    def parse(
        self, resume_text: str, candidate_id: Optional[str] = None
    ) -> CandidateProfile:
        """Parse raw resume text into a structured CandidateProfile."""
        if not resume_text or not resume_text.strip():
            raise ValueError("Resume text cannot be empty.")

        prompt = (
            f"Please parse the following resume into a structured candidate profile:\n\n"
            f"{resume_text}"
        )

        profile = generate_structured(
            prompt=prompt,
            system=RESUME_PARSER_SYSTEM_PROMPT,
            response_model=CandidateProfile,
            model=self.model,
        )

        # Ensure raw_text is faithfully populated with the original unadulterated resume text
        profile.raw_text = resume_text

        # Set candidate_id if provided, otherwise derive fallback if empty
        if candidate_id:
            profile.candidate_id = candidate_id
        elif not profile.candidate_id:
            fallback = profile.candidate_name or "candidate"
            profile.candidate_id = fallback.lower().replace(" ", "_")

        return profile


def parse_resume(
    resume_text: str,
    candidate_id: Optional[str] = None,
    model: str = DEFAULT_MODEL,
) -> CandidateProfile:
    """Convenience functional wrapper around ResumeParser."""
    parser = ResumeParser(model=model)
    return parser.parse(resume_text, candidate_id=candidate_id)
