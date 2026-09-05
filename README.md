# HireFair

A fairness-aware resume screening and interview scheduling system.

## Purpose

HireFair evaluates candidate resumes against a job description, scores them on a structured rubric, and independently audits those scores for bias before routing candidates to auto-scheduling or human review.

## High-Level Architecture

```
Job Description ──► JD Parser ──► Weighted Rubric ─┐
                                                    │
                                                    ▼
Resume Text ──► Resume Parser ──► Candidate Profile ──► Matcher ──► Scored Evaluation
                                                                         │
                                                                         ▼
                                                                  Fairness Auditor
                                                                         │
                                                          ┌──────────────┴──────────────┐
                                                          ▼                             ▼
                                                   Clean → Router              Flagged → Router
                                                          │                             │
                                                          ▼                             ▼
                                                   Auto-Schedule              Human Review Queue
```

### Agents

| Agent             | Responsibility                                                  |
| ----------------- | --------------------------------------------------------------- |
| JD Parser         | Extracts a weighted rubric from the job description             |
| Resume Parser     | Converts resume text into a structured candidate profile        |
| Matcher           | Scores the candidate per criterion with quoted evidence         |
| Fairness Auditor  | Independently audits scores via counterfactual, citation, and evidence checks |
| Router            | Sends clean candidates to scheduling, flagged to human review   |

### Orchestration

A LangGraph pipeline connects the agents with a capped self-repair loop — if the Auditor detects both a score discrepancy and a citation failure, the candidate is re-scored once on an anonymized profile.

## Technology Stack

- **Python** — core language
- **LangGraph** — agent orchestration
- **Google Gemini API** — LLM with structured outputs (`gemini-2.5-flash` via `google-genai` SDK)
- **FastAPI** — backend API
- **SQLite** — persistence
- **Streamlit** — reviewer interface

### LLM Structured Outputs & Resilience

All agent interactions with Google Gemini are routed through a shared utility (`app/utils/llm_client.py`):
- **Structured Outputs**: Uses `google-genai` with `response_schema` mapped to Pydantic models, validated on parse.
- **Rate-Limit Resilience**: Incorporates `tenacity` retries specifically for HTTP 429 (`ClientError` with code 429) using exponential backoff up to 5 attempts. Non-retryable 4xx errors fail immediately.
- **Provider Choice**: Google Gemini via Google AI Studio's free tier was adopted as a development constraint due to exhausted OpenAI API credits, preserving identical agent behavior and pipeline specifications.

## Environment Setup

1. Configure environment variables in `.env`:
   ```bash
   GEMINI_API_KEY=your_google_ai_studio_api_key_here
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
HireFair/
├── app/                  # Application source code
│   ├── agents/           # Individual agent implementations
│   ├── api/              # FastAPI routes and server
│   ├── db/               # SQLite models and queries
│   ├── models/           # Pydantic schemas (shared data contracts)
│   ├── pipeline/         # LangGraph orchestration
│   └── utils/            # Shared helpers (llm_client, etc.)
├── frontend/             # Streamlit reviewer interface
├── data/                 # Synthetic test data (resumes, job descriptions)
├── tests/                # Automated tests
├── PROJECT_SPEC.md       # Full project specification
├── README.md
├── requirements.txt
├── .env.example
└── .gitignore
```

## Development Phases

1. **Phase 1 — Foundation** *(current)*
   Project structure, specification, and tooling setup.

2. **Phase 2 — Data Models**
   Pydantic schemas for rubrics, candidate profiles, scores, and audit results.

3. **Phase 3 — JD Parser Agent**
   Extract a weighted rubric from a job description using Gemini structured outputs.

4. **Phase 4 — Resume Parser Agent**
   Convert resume text into a structured candidate profile.

5. **Phase 5 — Matcher Agent**
   Score candidates against the rubric with quoted evidence.

6. **Phase 6 — Fairness Auditor Agent**
   Counterfactual re-scoring, citation validation, and non-traditional evidence scan.

7. **Phase 7 — Router**
   Routing logic and decision output.

8. **Phase 8 — LangGraph Pipeline**
   Wire agents together with the capped self-repair loop.

9. **Phase 9 — SQLite Persistence**
   Store rubrics, profiles, scores, audits, and routing decisions.

10. **Phase 10 — FastAPI Backend**
    API endpoints for submitting jobs, uploading resumes, and retrieving results.

11. **Phase 11 — Streamlit Frontend**
    Reviewer dashboard for human review of flagged candidates.

12. **Phase 12 — Synthetic Test Data and Tests**
    Generate test resumes/JDs and write automated tests.

13. **Phase 13 — Extended Features**
    Mock scheduling, 48-hour fallback, duplicate detection, demo scenarios.

## License

TBD
