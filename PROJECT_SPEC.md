# HireFair — Project Specification

## Purpose

HireFair is a fairness-aware resume screening and interview scheduling system. It takes a job description and a pool of candidate resumes and produces scored, audited evaluations with automatic routing — clean candidates go to auto-scheduling, flagged candidates go to human review.

The key differentiator is an **independent Fairness Auditor Critic** that checks whether the Matcher may have produced an unreliable or biased score.

---

## System Inputs and Outputs

### Inputs

1. One job description (text).
2. A pool of candidate resumes (text).

### Outputs

1. A structured, weighted job rubric.
2. Structured candidate profiles.
3. Criterion-level candidate scores supported by direct resume evidence.
4. An independent fairness audit of those scores.
5. A routing decision:
   - **Clean candidate** → auto-scheduling.
   - **Flagged candidate** → human review with the Auditor's specific rationale and evidence.

---

## Core Agents

### 1. JD Parser

- Extracts a structured weighted rubric from the job description.
- Separates **must-have** and **nice-to-have** criteria.
- Assigns criterion weights from **1–5** based on emphasis and repetition in the job description.

### 2. Resume Parser

- Converts raw resume text into a structured candidate profile.
- Uses schema-constrained structured extraction.
- **Must not invent information** that is absent from the resume.

### 3. Matcher

- Scores the candidate against every rubric criterion.
- Produces a score from **0–1** per criterion.
- Every score above **0.3** must have a **direct quote** from the resume as evidence.
- No fabricated evidence.

### 4. Fairness Auditor Critic

Independently checks the Matcher's output using **three mechanisms**:

#### a. Counterfactual Re-Scoring

- Remove candidate name, institution names, and graduation year.
- Re-score the anonymized profile.
- Compare the original and anonymized scores.

#### b. Citation Validity

- Independently check whether each cited quote actually supports the criterion and score.
- Keyword coincidence alone is **not sufficient**.

#### c. Non-Traditional Evidence Scan

- Look for competency evidence that may have been underweighted.
- This includes personal projects, open-source work, self-taught experience, and career-change experience.

### 5. Router

- Routes clean candidates toward auto-scheduling.
- Routes flagged candidates to human review.
- Flagged candidates must retain the Auditor's specific rationale and evidence.

---

## Fairness Requirements

The system must **NOT** automatically treat these as competency deficits:

- Employment gaps.
- Lesser-known institutions.
- Career changes.
- Self-taught backgrounds.
- Lack of a traditional job title.

Personal projects and open-source work **can** be valid evidence of competency when they actually demonstrate the required skill.

---

## Incomplete Data Handling

If important resume information is missing or unclear (e.g., dates, experience duration):

- **Do not guess.**
- **Do not assign a fabricated default.**
- Route the candidate to **manual review for incomplete data**.

---

## Self-Repair Loop

The LangGraph pipeline contains a **capped repair loop**.

If the fairness audit finds **both**:

1. A significant counterfactual score discrepancy, **and**
2. A failed citation check,

the candidate may be **re-scored once** using the anonymized profile.

- The repair loop is capped at exactly **one retry per candidate**.
- It must **never loop indefinitely**.

---

## Technology Stack

| Layer            | Technology                                    |
| ---------------- | --------------------------------------------- |
| Language         | Python                                        |
| Orchestration    | LangGraph                                     |
| LLM Integration  | Google Gemini API (`gemini-2.5-flash` via `google-genai` SDK) |
| Backend API      | FastAPI                                       |
| Database         | SQLite                                        |
| Frontend         | Streamlit                                     |

> [!NOTE]
> **Implementation Note (LLM Provider)**: The system utilizes Google Gemini (`gemini-2.5-flash`) via the `google-genai` SDK and Google AI Studio's free tier. This choice was adopted as a development constraint due to exhausted OpenAI API credits. All LLM calls are routed through a centralized utility (`app/utils/llm_client.py`) using Gemini's `response_schema` alongside Pydantic models for structured outputs, protected by Tenacity retry logic targeting HTTP 429 rate limits (up to 5 attempts with exponential backoff). The core product design, agents, scoring rules, and fairness criteria remain unchanged.

No additional infrastructure (PostgreSQL, Redis, Docker, Kubernetes, cloud services) unless explicitly requested.

---

## Core Scope

- JD Parser
- Resume Parser
- Matcher
- Fairness Auditor
- Router
- LangGraph orchestration
- SQLite persistence
- FastAPI backend
- Streamlit reviewer interface
- Synthetic test data
- Automated tests

---

## Extended Scope (Future)

- Mock calendar scheduling connector.
- 48-hour fallback when no immediate scheduling slot is available.
- Duplicate application detection.
- Polished demo scenarios.
