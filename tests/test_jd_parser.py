"""Tests for the JD Parser agent using Gemini Structured Outputs."""

from pathlib import Path
from unittest.mock import patch
import pytest

from app.agents.jd_parser import (
    DEFAULT_MODEL,
    JD_PARSER_SYSTEM_PROMPT,
    JDParser,
    parse_job_description,
)
from app.models.job_description import (
    JobRequirement,
    ParsedJobDescription,
    RequirementType,
)


@pytest.fixture
def sample_jd_text() -> str:
    """Load the sample job description from data/job_description.txt."""
    jd_path = Path(__file__).resolve().parent.parent / "data" / "job_description.txt"
    with open(jd_path, "r", encoding="utf-8") as f:
        return f.read()


@pytest.fixture
def mock_parsed_jd() -> ParsedJobDescription:
    """Return a realistic ParsedJobDescription object for mocked responses."""
    return ParsedJobDescription(
        job_title="Senior Data Engineer",
        company="Meridian Analytics",
        requirements=[
            JobRequirement(
                criterion_id="REQ-01",
                description="4+ years of professional experience building production data pipelines",
                weight=5,
                type=RequirementType.MUST_HAVE,
                evidence_type="work_experience",
            ),
            JobRequirement(
                criterion_id="REQ-02",
                description="Strong proficiency in Python for data engineering tasks",
                weight=5,
                type=RequirementType.MUST_HAVE,
                evidence_type="work_experience_or_projects",
            ),
            JobRequirement(
                criterion_id="REQ-03",
                description="Strong proficiency in SQL including window functions, CTEs, and query optimization",
                weight=5,
                type=RequirementType.MUST_HAVE,
                evidence_type="work_experience_or_projects",
            ),
            JobRequirement(
                criterion_id="REQ-04",
                description="Hands-on experience with Apache Airflow for workflow orchestration",
                weight=4,
                type=RequirementType.MUST_HAVE,
                evidence_type="work_experience_or_open_source",
            ),
            JobRequirement(
                criterion_id="REQ-05",
                description="Experience with cloud data warehouse, preferably Snowflake",
                weight=4,
                type=RequirementType.MUST_HAVE,
                evidence_type="work_experience",
            ),
            JobRequirement(
                criterion_id="REQ-06",
                description="Experience implementing data quality frameworks or validation checks",
                weight=4,
                type=RequirementType.MUST_HAVE,
                evidence_type="work_experience_or_projects",
            ),
            JobRequirement(
                criterion_id="REQ-07",
                description="Experience with streaming data technologies (Kafka, Spark Streaming, Flink)",
                weight=3,
                type=RequirementType.NICE_TO_HAVE,
                evidence_type="projects_or_experience",
            ),
            JobRequirement(
                criterion_id="REQ-08",
                description="Familiarity with containerization (Docker) and IaC (Terraform)",
                weight=2,
                type=RequirementType.NICE_TO_HAVE,
                evidence_type="projects_or_experience",
            ),
            JobRequirement(
                criterion_id="REQ-09",
                description="Experience with dbt for data transformation",
                weight=3,
                type=RequirementType.NICE_TO_HAVE,
                evidence_type="work_experience_or_open_source",
            ),
            JobRequirement(
                criterion_id="REQ-10",
                description="Contributions to open-source data tooling or mentoring",
                weight=2,
                type=RequirementType.NICE_TO_HAVE,
                evidence_type="open_source_or_leadership",
            ),
        ],
    )


def test_jd_parser_empty_input():
    """Verify that empty or whitespace JD text raises a ValueError without API call."""
    parser = JDParser()
    with pytest.raises(ValueError, match="cannot be empty"):
        parser.parse("")

    with pytest.raises(ValueError, match="cannot be empty"):
        parser.parse("   \n\t  ")

    with pytest.raises(ValueError, match="cannot be empty"):
        parse_job_description("")


def test_jd_parser_passes_response_model_and_default_model(
    sample_jd_text: str, mock_parsed_jd: ParsedJobDescription
):
    """Verify that JDParser calls generate_structured with ParsedJobDescription and Gemini default model."""
    with patch("app.agents.jd_parser.generate_structured") as mock_generate:
        mock_generate.return_value = mock_parsed_jd

        parser = JDParser()
        result = parser.parse(sample_jd_text)

        assert mock_generate.called
        call_kwargs = mock_generate.call_args.kwargs

        # Verify model selection: default Gemini model
        assert call_kwargs["model"] == "gemini-2.5-flash"
        assert call_kwargs["model"] == DEFAULT_MODEL

        # Verify response_model: ParsedJobDescription
        assert call_kwargs["response_model"] is ParsedJobDescription

        # Verify prompt and system instruction
        assert sample_jd_text in call_kwargs["prompt"]
        assert call_kwargs["system"] == JD_PARSER_SYSTEM_PROMPT

        # Verify result is the returned parsed object
        assert result is mock_parsed_jd


def test_jd_parser_structured_output_contract(
    sample_jd_text: str, mock_parsed_jd: ParsedJobDescription
):
    """Verify the full parsed structured output contract (must-haves, nice-to-haves, weights, evidence)."""
    with patch("app.agents.jd_parser.generate_structured") as mock_generate:
        mock_generate.return_value = mock_parsed_jd

        parser = JDParser()
        result = parser.parse(sample_jd_text)

        # 1. Type and metadata verification
        assert isinstance(result, ParsedJobDescription)
        assert result.job_title == "Senior Data Engineer"
        assert result.company == "Meridian Analytics"

        # 2. Requirements separation
        must_haves = result.must_haves
        nice_to_haves = result.nice_to_haves

        assert len(must_haves) == 6
        assert len(nice_to_haves) == 4
        assert len(must_haves) + len(nice_to_haves) == len(result.requirements)

        # 3. Criterion-level schema verification
        for req in result.requirements:
            assert isinstance(req.criterion_id, str) and len(req.criterion_id) > 0
            assert isinstance(req.description, str) and len(req.description) > 0
            assert isinstance(req.weight, int)
            assert 1 <= req.weight <= 5
            assert req.type in (RequirementType.MUST_HAVE, RequirementType.NICE_TO_HAVE)
            assert isinstance(req.evidence_type, str) and len(req.evidence_type) > 0


def test_parse_job_description_convenience_function(
    sample_jd_text: str, mock_parsed_jd: ParsedJobDescription
):
    """Verify that the convenience functional interface works as expected."""
    with patch("app.agents.jd_parser.generate_structured") as mock_generate:
        mock_generate.return_value = mock_parsed_jd

        result = parse_job_description(sample_jd_text)

        assert isinstance(result, ParsedJobDescription)
        assert len(result.requirements) == 10
        assert mock_generate.called
        assert mock_generate.call_args.kwargs["model"] == "gemini-2.5-flash"


def test_jd_parser_custom_model(
    sample_jd_text: str, mock_parsed_jd: ParsedJobDescription
):
    """Verify that JDParser allows overriding the model name."""
    with patch("app.agents.jd_parser.generate_structured") as mock_generate:
        mock_generate.return_value = mock_parsed_jd

        custom_model = "gemini-1.5-pro"
        parser = JDParser(model=custom_model)
        result = parser.parse(sample_jd_text)

        assert mock_generate.call_args.kwargs["model"] == custom_model
        assert result is mock_parsed_jd
