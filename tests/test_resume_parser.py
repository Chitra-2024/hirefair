"""Tests for the Resume Parser agent using Gemini Structured Outputs."""

from pathlib import Path
from unittest.mock import patch
import pytest

from app.agents.resume_parser import (
    DEFAULT_MODEL,
    RESUME_PARSER_SYSTEM_PROMPT,
    ResumeParser,
    parse_resume,
)
from app.models.candidate import (
    CandidateProfile,
    Education,
    Project,
    WorkExperience,
)


@pytest.fixture
def sample_resume_text() -> str:
    """Load a sample resume text from data/resumes/elena_vasquez.txt."""
    resume_path = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "resumes"
        / "elena_vasquez.txt"
    )
    with open(resume_path, "r", encoding="utf-8") as f:
        return f.read()


@pytest.fixture
def sample_incomplete_resume_text() -> str:
    """Load the incomplete resume text from data/resumes/nathaniel_brooks.txt."""
    resume_path = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "resumes"
        / "nathaniel_brooks.txt"
    )
    with open(resume_path, "r", encoding="utf-8") as f:
        return f.read()


@pytest.fixture
def mock_complete_profile() -> CandidateProfile:
    """Return a realistic CandidateProfile for a complete candidate."""
    return CandidateProfile(
        candidate_id="elena_vasquez",
        candidate_name="Elena Vasquez",
        raw_text="",
        skills=[
            "Python",
            "SQL",
            "Apache Airflow",
            "Snowflake",
            "Great Expectations",
            "GitHub Actions",
            "Docker",
            "Terraform",
            "dbt",
        ],
        experience=[
            WorkExperience(
                title="Senior Data Engineer",
                company="Crestline Data Systems",
                duration_months=54,  # March 2022 – Present (~4.5 yrs)
                description="Architected and maintain 40+ Apache Airflow DAGs processing over 3B records daily into Snowflake.",
            ),
            WorkExperience(
                title="Data Engineer",
                company="Broadleaf Technologies",
                duration_months=32,  # June 2019 – February 2022
                description="Developed Python-based ETL pipelines into Redshift, wrote complex SQL transformations, Great Expectations.",
            ),
        ],
        education=[
            Education(
                degree="B.S. Computer Science",
                institution="University of Texas at Dallas",
                graduation_year=2017,
                field_of_study="Computer Science",
            )
        ],
        projects=[
            Project(
                name="airflow-dbt-pipeline",
                description="End-to-end data pipeline orchestrating dbt models in Airflow with automated Great Expectations validation.",
                technologies=["Python", "Airflow", "dbt", "Snowflake"],
            )
        ],
    )


@pytest.fixture
def mock_incomplete_profile() -> CandidateProfile:
    """Return a realistic CandidateProfile for an incomplete resume (e.g. Nathaniel Brooks)."""
    return CandidateProfile(
        candidate_id="nathaniel_brooks",
        candidate_name="Nathaniel Brooks",
        raw_text="",
        skills=["Python", "SQL", "Apache Airflow", "Snowflake", "Great Expectations"],
        experience=[
            WorkExperience(
                title="Data Engineer",
                company="Vertex Analytics",
                duration_months=None,  # Missing dates/duration
                description="Designed and maintained data pipelines using Python and Apache Airflow.",
            ),
            WorkExperience(
                title="Data Analyst",
                company="Pinnacle Systems",
                duration_months=None,  # Missing dates/duration
                description="Built reporting dashboards and wrote SQL queries for business analytics.",
            ),
        ],
        education=[
            Education(
                degree="B.S. Computer Science",
                institution="Westfield State University",
                graduation_year=None,  # Missing graduation year
                field_of_study="Computer Science",
            )
        ],
        projects=[],
    )


def test_resume_parser_empty_input():
    """Verify that empty or whitespace resume text raises a ValueError without API call."""
    parser = ResumeParser()
    with pytest.raises(ValueError, match="cannot be empty"):
        parser.parse("")

    with pytest.raises(ValueError, match="cannot be empty"):
        parser.parse("   \n\t  ")

    with pytest.raises(ValueError, match="cannot be empty"):
        parse_resume("")


def test_resume_parser_passes_response_model_and_default_model(
    sample_resume_text: str, mock_complete_profile: CandidateProfile
):
    """Verify that ResumeParser calls generate_structured with CandidateProfile and Gemini default model."""
    with patch("app.agents.resume_parser.generate_structured") as mock_generate:
        mock_generate.return_value = mock_complete_profile

        parser = ResumeParser()
        result = parser.parse(sample_resume_text, candidate_id="elena_vasquez")

        assert mock_generate.called
        call_kwargs = mock_generate.call_args.kwargs

        # Verify model selection: default Gemini model
        assert call_kwargs["model"] == "gemini-2.5-flash"
        assert call_kwargs["model"] == DEFAULT_MODEL

        # Verify response_model is CandidateProfile
        assert call_kwargs["response_model"] is CandidateProfile

        # Verify prompt and system instruction
        assert sample_resume_text in call_kwargs["prompt"]
        assert call_kwargs["system"] == RESUME_PARSER_SYSTEM_PROMPT

        # Verify raw_text and candidate_id are properly preserved
        assert result.candidate_id == "elena_vasquez"
        assert result.raw_text == sample_resume_text


def test_resume_parser_structured_contract(
    sample_resume_text: str, mock_complete_profile: CandidateProfile
):
    """Verify extraction contract: skills, experience, education, projects."""
    with patch("app.agents.resume_parser.generate_structured") as mock_generate:
        mock_generate.return_value = mock_complete_profile

        parser = ResumeParser()
        result = parser.parse(sample_resume_text)

        # 1. Type verification
        assert isinstance(result, CandidateProfile)

        # 2. Skills verification
        assert len(result.skills) == 9
        assert "Python" in result.skills
        assert "Snowflake" in result.skills

        # 3. Experience verification
        assert len(result.experience) == 2
        for exp in result.experience:
            assert isinstance(exp.title, str) and len(exp.title) > 0
            assert isinstance(exp.company, str) and len(exp.company) > 0
            assert isinstance(exp.duration_months, int) and exp.duration_months > 0
            assert isinstance(exp.description, str) and len(exp.description) > 0

        # 4. Education verification
        assert len(result.education) == 1
        edu = result.education[0]
        assert edu.degree == "B.S. Computer Science"
        assert edu.institution == "University of Texas at Dallas"
        assert edu.graduation_year == 2017

        # 5. Projects verification
        assert len(result.projects) == 1
        proj = result.projects[0]
        assert proj.name == "airflow-dbt-pipeline"
        assert len(proj.technologies) > 0


def test_resume_parser_missing_duration_incomplete_data(
    sample_incomplete_resume_text: str, mock_incomplete_profile: CandidateProfile
):
    """Verify that incomplete resume data leaves duration_months and dates as None without guessing."""
    with patch("app.agents.resume_parser.generate_structured") as mock_generate:
        mock_generate.return_value = mock_incomplete_profile

        parser = ResumeParser()
        result = parser.parse(
            sample_incomplete_resume_text, candidate_id="nathaniel_brooks"
        )

        assert isinstance(result, CandidateProfile)
        assert result.candidate_id == "nathaniel_brooks"
        assert result.raw_text == sample_incomplete_resume_text

        # Both experiences must have duration_months as None (not 0, not guessed)
        assert len(result.experience) == 2
        for exp in result.experience:
            assert exp.duration_months is None

        # Verify education graduation year is None
        assert len(result.education) == 1
        assert result.education[0].graduation_year is None


def test_parse_resume_convenience_function(
    sample_resume_text: str, mock_complete_profile: CandidateProfile
):
    """Verify that the convenience functional wrapper parse_resume works properly."""
    with patch("app.agents.resume_parser.generate_structured") as mock_generate:
        mock_generate.return_value = mock_complete_profile

        result = parse_resume(sample_resume_text, candidate_id="elena_vasquez")

        assert isinstance(result, CandidateProfile)
        assert result.candidate_id == "elena_vasquez"
        assert result.raw_text == sample_resume_text
        assert mock_generate.called
        assert mock_generate.call_args.kwargs["model"] == "gemini-2.5-flash"


def test_resume_parser_custom_model(
    sample_resume_text: str, mock_complete_profile: CandidateProfile
):
    """Verify that ResumeParser allows overriding the model identifier."""
    with patch("app.agents.resume_parser.generate_structured") as mock_generate:
        mock_generate.return_value = mock_complete_profile

        custom_model = "gemini-1.5-pro"
        parser = ResumeParser(model=custom_model)
        result = parser.parse(sample_resume_text)

        assert mock_generate.call_args.kwargs["model"] == custom_model
        assert result is mock_complete_profile


def test_candidate_profile_anonymize(mock_complete_profile: CandidateProfile):
    """Verify anonymization for counterfactual fairness auditing.

    Strips:
    - candidate_name
    - education institution
    - education graduation_year

    Preserves competency-relevant information:
    - skills
    - work experience (job titles, companies, durations, descriptions)
    - education degree and field of study
    - projects (names, descriptions, technologies)
    """
    anonymized = mock_complete_profile.anonymize()

    # Original remains unmodified
    assert mock_complete_profile.candidate_name == "Elena Vasquez"
    assert (
        mock_complete_profile.education[0].institution
        == "University of Texas at Dallas"
    )
    assert mock_complete_profile.education[0].graduation_year == 2017

    # Anonymized profile strips ONLY the identity/demographic context fields
    assert anonymized.candidate_name is None
    assert anonymized.education[0].institution is None
    assert anonymized.education[0].graduation_year is None

    # Preserves competency-relevant information:
    # 1. Skills
    assert anonymized.skills == mock_complete_profile.skills

    # 2. Work experience (job titles, companies, durations, descriptions)
    assert len(anonymized.experience) == len(mock_complete_profile.experience)
    for anon_exp, orig_exp in zip(anonymized.experience, mock_complete_profile.experience):
        assert anon_exp.title == orig_exp.title
        assert anon_exp.company == orig_exp.company
        assert anon_exp.duration_months == orig_exp.duration_months
        assert anon_exp.description == orig_exp.description

    # 3. Education degree and field of study
    assert len(anonymized.education) == len(mock_complete_profile.education)
    assert (
        anonymized.education[0].degree == mock_complete_profile.education[0].degree
    )
    assert (
        anonymized.education[0].field_of_study
        == mock_complete_profile.education[0].field_of_study
    )

    # 4. Projects and technical descriptions
    assert len(anonymized.projects) == len(mock_complete_profile.projects)
    for anon_proj, orig_proj in zip(anonymized.projects, mock_complete_profile.projects):
        assert anon_proj.name == orig_proj.name
        assert anon_proj.description == orig_proj.description
        assert anon_proj.technologies == orig_proj.technologies
