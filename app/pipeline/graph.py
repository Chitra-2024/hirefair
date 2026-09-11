"""LangGraph orchestration pipeline for HireFair.

Provides per-candidate graph execution (Resume Parser -> Matcher -> Fairness Auditor)
and batch pipeline drivers with failure isolation.
"""

from typing import Any, Optional, Sequence
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.fairness_auditor import FairnessAuditor
from app.agents.jd_parser import JDParser
from app.agents.matcher import Matcher
from app.agents.resume_parser import ResumeParser
from app.models.job_description import ParsedJobDescription
from app.models.result import CandidateResult, FailedCandidate, PipelineResult
from app.pipeline.state import CandidateGraphState


# ---------------------------------------------------------------------------
# Per-candidate Graph Nodes
# Thin wrappers around existing agents; no agent logic moved here.
# ---------------------------------------------------------------------------


def resume_parser_node(state: CandidateGraphState) -> dict:
    """Parse candidate resume into a structured CandidateProfile."""
    parser = ResumeParser()
    profile = parser.parse(
        resume_text=state["resume_text"],
        candidate_id=state.get("candidate_id"),
    )
    return {"candidate_profile": profile}


def matcher_node(state: CandidateGraphState) -> dict:
    """Evaluate candidate profile against rubric criteria."""
    candidate_profile = state["candidate_profile"]
    rubric = state["rubric"]
    matcher = Matcher()
    match_result = matcher.match(
        candidate=candidate_profile,
        rubric=rubric,
    )
    return {"match_result": match_result}


def fairness_auditor_node(state: CandidateGraphState) -> dict:
    """Audit candidate match result for bias, citation fidelity, and non-traditional evidence."""
    candidate_profile = state["candidate_profile"]
    rubric = state["rubric"]
    match_result = state["match_result"]
    auditor = FairnessAuditor()
    audit_record = auditor.audit(
        candidate=candidate_profile,
        rubric=rubric,
        original_result=match_result,
    )
    return {"audit_record": audit_record}


# ---------------------------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------------------------


def create_candidate_graph() -> CompiledStateGraph:
    """Build and compile the linear per-candidate processing graph.

    Graph topology:
    START -> resume_parser -> matcher -> fairness_auditor -> END
    """
    workflow = StateGraph(CandidateGraphState)

    workflow.add_node("resume_parser", resume_parser_node)
    workflow.add_node("matcher", matcher_node)
    workflow.add_node("fairness_auditor", fairness_auditor_node)

    workflow.add_edge(START, "resume_parser")
    workflow.add_edge("resume_parser", "matcher")
    workflow.add_edge("matcher", "fairness_auditor")
    workflow.add_edge("fairness_auditor", END)

    return workflow.compile()


# Shared compiled graph instance for reuse across runs
candidate_graph = create_candidate_graph()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_candidate_info(candidate_raw: Any, index: int) -> tuple[str, str]:
    """Extract candidate_id and resume_text from input format (model, dict, tuple)."""
    if isinstance(candidate_raw, dict):
        cid = candidate_raw.get("candidate_id") or f"candidate_{index}"
        text = candidate_raw.get("resume_text", "")
        return str(cid), str(text)
    elif hasattr(candidate_raw, "candidate_id") and hasattr(candidate_raw, "resume_text"):
        return str(candidate_raw.candidate_id), str(candidate_raw.resume_text)
    elif isinstance(candidate_raw, (tuple, list)) and len(candidate_raw) >= 2:
        return str(candidate_raw[0]), str(candidate_raw[1])
    else:
        return f"candidate_{index}", str(candidate_raw)


# ---------------------------------------------------------------------------
# Batch Entry Points
# ---------------------------------------------------------------------------


def run_pipeline(
    rubric: ParsedJobDescription,
    candidates: Sequence[Any],
    graph: Optional[CompiledStateGraph] = None,
) -> PipelineResult:
    """Execute candidate screening sequentially through the per-candidate LangGraph.

    - Uses an already-parsed rubric (does NOT call JDParser).
    - Processes candidates strictly sequentially in input order.
    - Isolates candidate failures: exceptions for one candidate are captured in
      failed_candidates and do not abort the rest of the batch.
    - For empty candidate input, returns an empty PipelineResult without invoking the graph.

    Args:
        rubric: The pre-parsed job description rubric.
        candidates: Sequence of candidates (CandidateInput, dict, or tuple).
        graph: Optional compiled LangGraph (defaults to shared candidate_graph).

    Returns:
        PipelineResult with successful CandidateResults and FailedCandidate records.
    """
    if not candidates:
        return PipelineResult(results=[], failed_candidates=[])

    active_graph = graph or candidate_graph
    results: list[CandidateResult] = []
    failed_candidates: list[FailedCandidate] = []

    for idx, candidate_raw in enumerate(candidates):
        cid = f"candidate_{idx}"
        try:
            cid, resume_text = _extract_candidate_info(candidate_raw, idx)
            initial_state: CandidateGraphState = {
                "candidate_id": cid,
                "resume_text": resume_text,
                "rubric": rubric,
            }
            final_state = active_graph.invoke(initial_state)

            scores = (
                final_state["match_result"].scores
                if final_state.get("match_result") is not None
                else []
            )

            results.append(
                CandidateResult(
                    candidate_id=final_state.get("candidate_id", cid),
                    candidate_profile=final_state["candidate_profile"],
                    scores=scores,
                    audit_record=final_state["audit_record"],
                )
            )
        except Exception as exc:
            failed_candidates.append(
                FailedCandidate(
                    candidate_id=cid,
                    reason=str(exc) or exc.__class__.__name__,
                )
            )

    return PipelineResult(results=results, failed_candidates=failed_candidates)


def run_full_pipeline(
    jd_text: str,
    candidates: Sequence[Any],
    graph: Optional[CompiledStateGraph] = None,
) -> PipelineResult:
    """Execute complete screening pipeline starting from raw job description text.

    Calls JDParser exactly once on jd_text, then delegates candidate batch processing
    to run_pipeline.

    Args:
        jd_text: Raw job description text.
        candidates: Sequence of candidates (CandidateInput, dict, or tuple).
        graph: Optional compiled LangGraph (defaults to shared candidate_graph).

    Returns:
        PipelineResult.
    """
    parser = JDParser()
    rubric = parser.parse(jd_text)
    return run_pipeline(rubric=rubric, candidates=candidates, graph=graph)
