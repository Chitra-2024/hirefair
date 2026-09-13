# HireFair

A fairness-aware resume screening and interview scheduling system.

## Purpose

HireFair evaluates candidate resumes against a job description, scores them on a structured rubric, and independently audits those scores for bias before routing candidates to auto-scheduling or a human review queue.

## High-Level Architecture

```
Job Description ──► JD Parser ──► Weighted Rubric ─┐
                                                    │
                                                    ▼
Resume Text ──► LangGraph (Resume Parser ──► Matcher ──► Fairness Auditor)
                                                    │
                                                    ▼
                                                 Router
                                           ┌────────────────┐
                                           ▼                ▼
                                        CLEARED          FLAGGED /
                                           │            INCOMPLETE /
                                           ▼           NOT_QUALIFIED /
                                      Scheduler         DUPLICATE
                                           │                │
                                           ▼                ▼
                                    Mock Calendar      Reviewer Queue
```

The full end-to-end flow for one `/screen` request:

**JD Parser → LangGraph (Resume Parser → Matcher → Fairness Auditor) → Router → InterviewScheduler → MockCalendar**

### Agents

| Agent            | Responsibility                                                                           |
| ---------------- | ---------------------------------------------------------------------------------------- |
| JD Parser        | Extracts a weighted rubric from the job description (Gemini structured output)           |
| Resume Parser    | Converts resume text into a structured candidate profile (Gemini structured output)      |
| Matcher          | Scores the candidate per criterion with quoted evidence (Gemini structured output)       |
| Fairness Auditor | Independently audits scores via counterfactual re-scoring, citation validity, and non-traditional evidence detection |
| Router           | Deterministic routing based on audit results and qualification rules — no LLM calls      |

### LangGraph Orchestration

A LangGraph pipeline connects Resume Parser → Matcher → Fairness Auditor per candidate. The Fairness Auditor uses a **capped self-repair loop (Option A)**: if the Auditor detects both a significant counterfactual score discrepancy **and** a citation check failure, it adopts the already-computed anonymized score as the final score (`repair_applied=True`). **No second Gemini call is made.** The repair is capped at one application per candidate and never loops indefinitely.

### Router Routing Decisions (Strict Precedence)

1. **DUPLICATE** — resume text similarity above threshold
2. **INCOMPLETE_DATA** — missing name, skills, experience duration, or work/project entries
3. **FLAGGED_FOR_REVIEW** — Fairness Auditor raised a fairness concern
4. **NOT_QUALIFIED** — score below qualification thresholds
5. **CLEARED** — meets all requirements; forwarded to `InterviewScheduler`

### Scheduling

CLEARED candidates are passed to an `InterviewScheduler` backed by a `MockCalendar`. The calendar generates 30-minute slots at 09:00 and 14:00 UTC within a strict 48-hour half-open window `[reference_time, reference_time + 48h)`. Bookings persist in memory across requests for the lifetime of the FastAPI process.

### Reviewer Dashboard

A Streamlit frontend (`frontend/app.py`) connects to the FastAPI backend over HTTP. It displays each candidate's routing decision with rationale, final score, and — for CLEARED candidates — the scheduled interview time or a "no availability" message.

## Technology Stack

| Layer           | Technology                                                                   |
| --------------- | ---------------------------------------------------------------------------- |
| Language        | Python                                                                       |
| Orchestration   | LangGraph                                                                    |
| LLM Integration | Google Gemini (`gemini-3.5-flash-lite` via `google-genai` SDK)              |
| Backend API     | FastAPI                                                                      |
| Frontend        | Streamlit                                                                    |

> [!NOTE]
> **LLM Provider**: All Gemini calls are routed through `app/utils/llm_client.py` using `response_schema` with Pydantic models for structured outputs, protected by Tenacity retry logic targeting HTTP 429 rate limits (up to 5 attempts with exponential backoff). The active model is `gemini-3.5-flash-lite`.

> [!NOTE]
> **No database**: SQLite persistence is deferred. All state — scoring results, routing decisions, and calendar bookings — is held in memory for the lifetime of the FastAPI process.

## Environment Setup

1. Configure environment variables in `.env`:
   ```bash
   GEMINI_API_KEY=your_google_ai_studio_api_key_here
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

### 1. Start the FastAPI Backend
```bash
uvicorn app.api.main:app --reload
```
The API serves interactive documentation at `http://127.0.0.1:8000/docs` and exposes:
- `GET /health`: Minimal server liveness check.
- `POST /screen`: Synchronous candidate screening from multipart `.txt` file uploads.

### 2. Start the Streamlit Frontend
```bash
streamlit run frontend/app.py
```
The Streamlit app connects to the FastAPI backend over HTTP using `requests`. It displays the full routing result including scheduling outcomes and reviewer queue entries.

## v1 Implementation Thresholds

These values are **v1 implementation choices** — named constants in the source code, not specified by the original design document. They can be tuned.

| Threshold                                          | Value    | Location                          |
| -------------------------------------------------- | -------- | --------------------------------- |
| Counterfactual delta (triggers repair consideration) | `0.15` | `app/agents/fairness_auditor.py`  |
| Qualification floor (overall weighted score)       | `>= 0.50` | `app/agents/router_config.py`    |
| Must-have criterion floor                          | `>= 0.30` | `app/agents/router_config.py`    |
| Non-traditional evidence surfacing threshold       | `<= 0.50` criterion score | `app/agents/fairness_auditor.py` |

## Project Structure

```
HireFair/
├── app/                  # Application source code
│   ├── agents/           # Individual agent implementations
│   ├── api/              # FastAPI routes and server
│   ├── db/               # Deferred (empty placeholder — SQLite not implemented)
│   ├── models/           # Pydantic schemas (shared data contracts)
│   ├── pipeline/         # LangGraph orchestration
│   ├── scheduling/       # MockCalendar and InterviewScheduler
│   └── utils/            # Shared helpers (llm_client, etc.)
├── frontend/             # Streamlit reviewer interface
├── data/                 # Synthetic test data (resumes, job descriptions)
├── tests/                # Automated tests (139 offline, zero Gemini calls)
├── PROJECT_SPEC.md       # Full project specification
├── README.md
├── requirements.txt
├── .env.example
└── .gitignore
```

## Synthetic Test Dataset

The `data/resumes/` directory contains candidates designed to exercise every routing path:

| File                         | Intended routing                              |
| ---------------------------- | --------------------------------------------- |
| `alex_rivera.txt`            | CLEARED (clean, qualified)                    |
| `daniel_yeboah.txt`          | FLAGGED — non-traditional background          |
| `hannah_eriksen.txt`         | FLAGGED — fairness flag                       |
| `nina_kozlova.txt`           | Scheduling / no-availability demo             |
| `laura_petrov.txt`           | INCOMPLETE_DATA                               |
| `elena_vasquez.txt`          | CLEARED or qualified baseline                 |
| `elena_vasquez_copy.txt`     | DUPLICATE (exact copy of elena_vasquez.txt)   |

## Known v1 Limitations

- **Filename-derived candidate IDs**: The candidate ID is derived from the uploaded filename stem (`alex_rivera.txt` → `alex_rivera`). No stable UUID or database-assigned ID is used.
- **In-memory state**: Routing results, audit records, and calendar bookings exist only for the lifetime of the FastAPI process. Restarting the server clears all state.
- **UTC-only scheduling**: `MockCalendar` generates and validates slots in UTC only. Candidate or interviewer timezone preferences are not supported.
- **Strict 48-hour scheduling window**: Slots are only offered within a `[reference_time, reference_time + 48h)` half-open window. No fallback beyond 48 hours.
- **Synchronous `/screen` endpoint**: The entire pipeline runs synchronously per HTTP request. Large resume batches block the request for the full Gemini call duration.
- **`.txt`-only input**: Plain-text files are accepted exclusively. PDF, DOCX, and other formats are rejected with HTTP 400.
- **Read-only reviewer UI**: The Streamlit dashboard displays decisions but has no controls to override them, escalate, or mark a candidate as reviewed.
- **Heuristic non-traditional evidence detection**: Non-traditional evidence (open-source work, side projects, self-taught experience) is detected by keyword matching, not semantic analysis.
- **Exact-string anonymization**: The Fairness Auditor anonymizes candidate profiles by exact-string removal of names, institutions, and graduation years. Paraphrased or abbreviated references may survive anonymization.

## License

TBD
