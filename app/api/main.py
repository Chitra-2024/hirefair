"""FastAPI application for HireFair resume screening.

Locked Phase 9 implementation decisions:
- Synchronous/blocking /screen execution: Processes in-memory without task queues or background workers.
- .txt uploads only: Rejects PDF or other non-.txt files with HTTP 400 Bad Request.
- In-memory processing: Uploaded files are decoded in-memory and never written to disk.
- Pure delegation: Connects directly to existing LangGraph pipeline and Router flow.
- Flat response structure: Surfaces per-candidate routing decisions without nested decision buckets.
- Secure error handling: Does not expose internal stack traces to the API client.
"""

import logging
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile, status

from app.api.service import screen_documents
from app.models.api import CandidateDecisionResponse, ScreenResponse
from app.models.result import CandidateInput

logger = logging.getLogger("hirefair.api")

app = FastAPI(
    title="HireFair API",
    description="Fairness-Aware Resume Screening & Interview Scheduling Agent",
    version="0.1.0",
)


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check() -> dict:
    """Minimal server health check endpoint."""
    return {"status": "ok"}


@app.post(
    "/screen",
    response_model=ScreenResponse,
    status_code=status.HTTP_200_OK,
    summary="Screen a batch of candidate resumes against a job description",
)
async def screen_candidates(
    job_description: UploadFile = File(..., description="Job description text file (.txt only)"),
    resumes: List[UploadFile] = File(..., description="Candidate resume text files (.txt only)"),
) -> ScreenResponse:
    """Screen candidate resumes synchronously through the HireFair pipeline and Router.

    Accepts:
    - job_description: 1 .txt file
    - resumes: 1 or more .txt files

    Returns a flat list of candidate routing decisions and any pipeline failures.
    """
    # 1. Validate job description file extension
    if not job_description.filename or not job_description.filename.lower().endswith(".txt"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type for job description '{job_description.filename}'. Only .txt files are accepted.",
        )

    # 2. Validate resumes presence and file extensions
    if not resumes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one resume file must be provided.",
        )

    for r_file in resumes:
        if not r_file.filename or not r_file.filename.lower().endswith(".txt"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file type for resume '{r_file.filename}'. Only .txt files are accepted.",
            )

    # 3. Read and decode job description in-memory
    try:
        jd_bytes = await job_description.read()
        jd_text = jd_bytes.decode("utf-8", errors="replace")
    except Exception as err:
        logger.warning("Failed to decode job description file: %s", err)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to read job description as UTF-8 text.",
        )

    if not jd_text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Job description file content cannot be empty.",
        )

    # 4. Read and decode candidate resumes in-memory
    candidates: List[CandidateInput] = []
    for idx, r_file in enumerate(resumes):
        try:
            r_bytes = await r_file.read()
            r_text = r_bytes.decode("utf-8", errors="replace")
        except Exception as err:
            logger.warning("Failed to decode resume file '%s': %s", r_file.filename, err)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to read resume file '{r_file.filename}' as UTF-8 text.",
            )

        cid = Path(r_file.filename).stem if r_file.filename else f"candidate_{idx + 1}"
        if not cid.strip():
            cid = f"candidate_{idx + 1}"

        candidates.append(CandidateInput(candidate_id=cid, resume_text=r_text))

    # 5. Delegate to existing screening pipeline + Router
    try:
        routing_result = screen_documents(jd_text=jd_text, candidates=candidates)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Screening execution failed: %s", exc, exc_info=True)
        # Never expose internal stack traces or exception details to client
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during candidate screening.",
        )

    # 6. Format flat API response preserving all decisions, scores, and audit records
    decision_responses = [
        CandidateDecisionResponse.from_route_decision(d)
        for d in routing_result.routed_candidates
    ]

    return ScreenResponse(
        results=decision_responses,
        failed_candidates=routing_result.failed_candidates,
    )
