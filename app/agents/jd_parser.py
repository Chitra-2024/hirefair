"""JD Parser agent module using Gemini Structured Outputs."""

from app.models.job_description import ParsedJobDescription
from app.utils.llm_client import generate_structured

DEFAULT_MODEL = "gemini-2.5-flash"

JD_PARSER_SYSTEM_PROMPT = """You are an expert HR and technical recruiting rubric extractor for HireFair, a fairness-aware resume screening system.

Your task is to analyze a Job Description (JD) and extract a structured, weighted evaluation rubric.

Follow these strict guidelines:
1. Criteria Extraction:
   - Identify every distinct requirement from the job description.
   - Capture technical proficiencies, workflow/tooling experience, data modeling capabilities, data quality practices, and collaboration/communication skills.
   - Assign each requirement a unique criterion_id (e.g., "REQ-01", "REQ-02", etc.).

2. Requirement Type Classification:
   - Classify each criterion as either "must_have" or "nice_to_have".
   - Carefully respect the explicit structure of the job description (e.g., "Must-Have Requirements" vs. "Nice-to-Have Requirements") and semantic signals.

3. Criterion Weighting (1–5 scale):
   - Assign an integer weight from 1 to 5 to each requirement based on emphasis, repetition, and criticality to the role:
     * 5: Core indispensable qualification; emphasized throughout and fundamental to the position (e.g., primary language/tooling, core required years of experience).
     * 4: Major must-have requirement essential for performing standard responsibilities.
     * 3: Standard requirement; important must-have or high-value nice-to-have.
     * 2: Beneficial nice-to-have qualification.
     * 1: Minor bonus or peripheral preference.

4. Evidence Types:
   - Specify the expected competency evidence type (e.g., "work_experience", "projects", "education", "certifications", "open_source", "leadership").
   - Remember that in HireFair, evidence can be demonstrated through traditional employment, personal projects, open-source contributions, or self-directed learning.

5. Metadata:
   - Extract the job_title and company name if present in the text.
"""


class JDParser:
    """Agent responsible for parsing job descriptions into structured rubrics."""

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model

    def parse(self, jd_text: str) -> ParsedJobDescription:
        """Parse raw job description text into a structured rubric."""
        if not jd_text or not jd_text.strip():
            raise ValueError("Job description text cannot be empty.")

        prompt = (
            f"Please extract the structured evaluation rubric from the following "
            f"job description:\n\n{jd_text}"
        )

        return generate_structured(
            prompt=prompt,
            system=JD_PARSER_SYSTEM_PROMPT,
            response_model=ParsedJobDescription,
            model=self.model,
        )


def parse_job_description(
    jd_text: str,
    model: str = DEFAULT_MODEL,
) -> ParsedJobDescription:
    """Convenience functional wrapper around JDParser."""
    parser = JDParser(model=model)
    return parser.parse(jd_text)
