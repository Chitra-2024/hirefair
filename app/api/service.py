"""Service layer connecting FastAPI endpoints to the LangGraph pipeline and Router.

Locked Phase 9 architectural choices:
- Synchronous/blocking execution: Requests are processed synchronously in memory
  without external task queues or background polling workers.
- In-memory processing: Uploaded resume and JD files are read into memory and
  passed directly to the pipeline. Nothing is written to disk.
- Pure orchestration delegation: Does not duplicate agent parsing, matching,
  auditing, or routing logic.
"""

from typing import Sequence
from app.agents.jd_parser import JDParser
from app.agents.router import route_pipeline_result
from app.models.result import CandidateInput
from app.models.routing import RoutingResult
from app.pipeline.graph import run_pipeline


def screen_documents(
    jd_text: str,
    candidates: Sequence[CandidateInput],
) -> RoutingResult:
    """Execute end-to-end candidate screening: JD Parser -> LangGraph Pipeline -> Router.

    1. Parses job description into a structured rubric exactly once.
    2. Runs candidates sequentially through the per-candidate LangGraph.
    3. Applies deterministic Router precedence to produce the final RoutingResult.

    Args:
        jd_text: Raw job description text.
        candidates: Sequence of CandidateInput instances with candidate_id and resume_text.

    Returns:
        RoutingResult with routed candidate decisions and failed candidates.
    """
    parser = JDParser()
    rubric = parser.parse(jd_text)
    pipeline_result = run_pipeline(rubric=rubric, candidates=candidates)
    routing_result = route_pipeline_result(pipeline_result=pipeline_result, rubric=rubric)
    return routing_result
