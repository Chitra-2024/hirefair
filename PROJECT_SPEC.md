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

the pipeline applies **Option A**: it adopts the already-computed anonymized aggregate score as the final score (`repair_applied=True`). **No second Gemini/Matcher call is made.** The anonymized score has already been computed during Check A (counterfactual re-scoring), so a second identical call would consume quota without providing a meaningfully different signal.

- The repair is capped at exactly **one application per candidate**.
- It must **never loop indefinitely**.

---

## Technology Stack

| Layer            | Technology                                                          |
| ---------------- | ------------------------------------------------------------------- |
| Language         | Python                                                              |
| Orchestration    | LangGraph                                                           |
| LLM Integration  | Google Gemini API (`gemini-3.5-flash-lite` via `google-genai` SDK)  |
| Backend API      | FastAPI                                                             |
| Database         | SQLite *(deferred — v1 uses in-memory state only)*                  |
| Frontend         | Streamlit                                                           |

> [!NOTE]
> **LLM Provider**: The system uses `gemini-3.5-flash-lite` via the `google-genai` SDK and Google AI Studio's free tier. All LLM calls are routed through `app/utils/llm_client.py` using Gemini's `response_schema` alongside Pydantic models for structured outputs, protected by Tenacity retry logic targeting HTTP 429 rate limits (up to 5 attempts with exponential backoff). The core product design, agents, scoring rules, and fairness criteria remain unchanged.

> [!NOTE]
> **Persistence**: SQLite persistence is deferred. All state in v1 — routing decisions, audit records, and calendar bookings — is held in memory for the lifetime of the FastAPI process.

No additional infrastructure (PostgreSQL, Redis, Docker, Kubernetes, cloud services) unless explicitly requested.

---

## Core Scope

- JD Parser
- Resume Parser
- Matcher
- Fairness Auditor
- Router
- LangGraph orchestration
- SQLite persistence *(deferred — not implemented in v1)*
- FastAPI backend
- Streamlit reviewer interface
- Synthetic test data
- Automated tests

---

## Extended Scope

The following were originally listed as future features and are **implemented in v1**:

- **Mock calendar scheduling connector** — `MockCalendar` + `InterviewScheduler` in `app/scheduling/`.
- **48-hour window scheduling** — strict half-open window `[reference_time, reference_time + 48h)` enforced by `MockCalendar`.
- **Duplicate application detection** — text-similarity and name-match heuristics in the Router.
- **Demo scenarios** — synthetic resume dataset covers CLEARED, FLAGGED, INCOMPLETE_DATA, DUPLICATE, and no-availability cases.

---

## v1 Implementation Thresholds

These values are **v1 implementation choices** — named constants in the source code, not specified by this design document. They can be tuned.

| Threshold                                           | Value              | Constant / Location                                    |
| --------------------------------------------------- | ------------------ | ------------------------------------------------------ |
| Counterfactual delta (triggers repair consideration) | `0.15`            | `COUNTERFACTUAL_DELTA_THRESHOLD` in `fairness_auditor.py` |
| Qualification floor (overall weighted score)        | `>= 0.50`         | `QUALIFICATION_THRESHOLD` in `router_config.py`        |
| Must-have criterion floor                           | `>= 0.30`         | `MIN_MUST_HAVE_SCORE` in `router_config.py`            |
| Non-traditional evidence surfacing threshold        | `<= 0.50` score   | `NON_TRADITIONAL_SCORE_THRESHOLD` in `fairness_auditor.py` |
