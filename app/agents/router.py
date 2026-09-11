"""Router agent for HireFair.

Evaluates pipeline results against calibration rules and applies strict precedence:
1. DUPLICATE
2. INCOMPLETE_DATA
3. FLAGGED_FOR_REVIEW (fairness flagged)
4. NOT_QUALIFIED
5. CLEARED

Failed candidates pass through untouched from PipelineResult.failed_candidates.
All logic is purely deterministic and runs completely offline with no LLM calls.
"""

from difflib import SequenceMatcher
from typing import Dict, List, Optional, Sequence, Tuple

from app.agents.router_config import (
    DUPLICATE_NAME_MATCH_SIMILARITY_THRESHOLD,
    DUPLICATE_TEXT_SIMILARITY_THRESHOLD,
    MIN_MUST_HAVE_SCORE,
    MIN_SKILLS_COUNT,
    QUALIFICATION_THRESHOLD,
    REQUIRE_EXPERIENCE_DURATION,
    REQUIRE_NAME,
    REQUIRE_WORK_OR_PROJECTS,
)
from app.models.audit import AuditRecord
from app.models.candidate import CandidateProfile
from app.models.job_description import ParsedJobDescription, RequirementType
from app.models.result import CandidateResult, FailedCandidate, PipelineResult
from app.models.routing import RouteDecision, RoutingDecision, RoutingResult


# ---------------------------------------------------------------------------
# Pure Helper: Final Score Derivation
# ---------------------------------------------------------------------------


def get_final_score(audit_record: Optional[AuditRecord]) -> Optional[float]:
    """Derive the authoritative final score from an AuditRecord.

    If repair_applied is True, adopts the anonymized_score from counterfactual
    auditing. Otherwise, uses original_score. Returns None if audit_record is None
    or the selected score is None.
    """
    if audit_record is None:
        return None
    if audit_record.repair_applied:
        return audit_record.anonymized_score
    return audit_record.original_score


# ---------------------------------------------------------------------------
# Pure Helper: Incomplete Profile Detection
# ---------------------------------------------------------------------------


def is_incomplete_profile(candidate: CandidateProfile) -> Tuple[bool, Optional[str]]:
    """Determine whether a candidate profile lacks critical information to trust screening.

    Rules:
    - Must have a non-empty candidate name (if REQUIRE_NAME).
    - Must have at least MIN_SKILLS_COUNT extracted skills.
    - If work experience is present and REQUIRE_EXPERIENCE_DURATION is True,
      every position must have a non-null duration_months (never guessed).
    - Must have at least one work experience or project entry (if REQUIRE_WORK_OR_PROJECTS).

    Returns:
        (True, reason_string) if incomplete; (False, None) if complete.
    """
    if REQUIRE_NAME and (not candidate.candidate_name or not candidate.candidate_name.strip()):
        return True, "Profile is missing candidate name"

    if len(candidate.skills) < MIN_SKILLS_COUNT:
        return (
            True,
            f"Profile contains {len(candidate.skills)} skills (minimum {MIN_SKILLS_COUNT} required)",
        )

    if REQUIRE_EXPERIENCE_DURATION:
        for exp in candidate.experience:
            if exp.duration_months is None:
                return (
                    True,
                    f"Work experience '{exp.title} at {exp.company}' is missing duration_months",
                )

    if REQUIRE_WORK_OR_PROJECTS:
        if len(candidate.experience) == 0 and len(candidate.projects) == 0:
            return True, "Profile contains neither work experience nor projects"

    return False, None


# ---------------------------------------------------------------------------
# Pure Helper: Duplicate Detection
# ---------------------------------------------------------------------------


def detect_duplicates(
    candidates: Sequence[CandidateResult],
    text_threshold: float = DUPLICATE_TEXT_SIMILARITY_THRESHOLD,
    name_threshold: float = DUPLICATE_NAME_MATCH_SIMILARITY_THRESHOLD,
) -> Dict[str, str]:
    """Detect duplicate candidate pairs among successfully processed profiles.

    Operates strictly on successfully parsed candidates in the batch.
    Failed candidates are excluded from duplicate checking.

    Two candidates are considered duplicates if:
    1. Both have matching normalized names and text similarity >= name_threshold, OR
    2. Normalized resume text similarity >= text_threshold (regardless of name).

    Returns:
        Mapping of candidate_id -> duplicate_of candidate_id.
    """
    duplicate_map: Dict[str, str] = {}
    n = len(candidates)

    for i in range(n):
        for j in range(i + 1, n):
            cand_a = candidates[i]
            cand_b = candidates[j]

            prof_a = cand_a.candidate_profile
            prof_b = cand_b.candidate_profile

            # Normalize raw texts
            text_a = " ".join(prof_a.raw_text.lower().split())
            text_b = " ".join(prof_b.raw_text.lower().split())

            ratio = SequenceMatcher(None, text_a, text_b).ratio()

            name_a = prof_a.candidate_name.strip().lower() if prof_a.candidate_name else ""
            name_b = prof_b.candidate_name.strip().lower() if prof_b.candidate_name else ""
            names_match = bool(name_a) and bool(name_b) and (name_a == name_b)

            is_dup = (names_match and ratio >= name_threshold) or (ratio >= text_threshold)

            if is_dup:
                # Retain only the first matching duplicate ID for each candidate
                if cand_a.candidate_id not in duplicate_map:
                    duplicate_map[cand_a.candidate_id] = cand_b.candidate_id

                if cand_b.candidate_id not in duplicate_map:
                    duplicate_map[cand_b.candidate_id] = cand_a.candidate_id

    return duplicate_map


# ---------------------------------------------------------------------------
# Pure Helper: Qualification Evaluation
# ---------------------------------------------------------------------------


def check_qualification(
    candidate_result: CandidateResult,
    rubric: ParsedJobDescription,
    qualification_threshold: float = QUALIFICATION_THRESHOLD,
    min_must_have_score: float = MIN_MUST_HAVE_SCORE,
) -> Tuple[bool, str]:
    """Check if candidate meets overall score and must-have requirements.

    Conditions for passing:
    A. Final weighted score >= qualification_threshold (0.5)
    B. Every must-have criterion score >= min_must_have_score (0.3)

    Returns:
        (True, reason) if qualified; (False, reason) if not qualified.
    """
    final_score = get_final_score(candidate_result.audit_record)

    # If final score is not present in audit_record, compute from rubric weights & scores against total rubric weight
    if final_score is None:
        total_rubric_weight = sum(req.weight for req in rubric.requirements)
        if total_rubric_weight > 0:
            weights = {req.criterion_id: req.weight for req in rubric.requirements}
            final_score = (
                sum(s.score * weights.get(s.criterion_id, 0) for s in candidate_result.scores)
                / total_rubric_weight
            )
        else:
            final_score = 0.0

    # Condition A: Overall score check
    if final_score < qualification_threshold:
        return (
            False,
            f"Final score {final_score:.3f} is below qualification threshold {qualification_threshold}",
        )

    # Condition B: Must-have criteria check
    score_by_criterion = {s.criterion_id: s.score for s in candidate_result.scores}
    for req in rubric.must_haves:
        if req.criterion_id not in score_by_criterion:
            return (
                False,
                f"Missing score record for must-have criterion '{req.criterion_id}' ({req.description})",
            )
        score = score_by_criterion[req.criterion_id]
        if score < min_must_have_score:
            return (
                False,
                f"Must-have criterion '{req.criterion_id}' score {score:.3f} is below minimum {min_must_have_score}",
            )

    return True, "Candidate meets all qualification requirements"


# ---------------------------------------------------------------------------
# Router Agent
# ---------------------------------------------------------------------------


class Router:
    """Deterministic routing agent for candidate screening decisions."""

    def __init__(
        self,
        qualification_threshold: float = QUALIFICATION_THRESHOLD,
        min_must_have_score: float = MIN_MUST_HAVE_SCORE,
        duplicate_text_threshold: float = DUPLICATE_TEXT_SIMILARITY_THRESHOLD,
        duplicate_name_threshold: float = DUPLICATE_NAME_MATCH_SIMILARITY_THRESHOLD,
    ):
        self.qualification_threshold = qualification_threshold
        self.min_must_have_score = min_must_have_score
        self.duplicate_text_threshold = duplicate_text_threshold
        self.duplicate_name_threshold = duplicate_name_threshold

    def route(
        self,
        pipeline_result: PipelineResult,
        rubric: ParsedJobDescription,
    ) -> RoutingResult:
        """Route all candidates in a completed PipelineResult.

        Applies precedence to successful candidates:
        1. DUPLICATE
        2. INCOMPLETE_DATA
        3. FLAGGED_FOR_REVIEW (fairness flagged)
        4. NOT_QUALIFIED
        5. CLEARED

        Failed candidates pass through directly without modification.

        Args:
            pipeline_result: The batch output from LangGraph orchestration.
            rubric: The parsed job description rubric.

        Returns:
            RoutingResult containing routed_candidates and failed_candidates.
        """
        # Step 1: Detect duplicates among successfully parsed candidates
        duplicate_map = detect_duplicates(
            candidates=pipeline_result.results,
            text_threshold=self.duplicate_text_threshold,
            name_threshold=self.duplicate_name_threshold,
        )

        routed_candidates: List[RouteDecision] = []

        # Step 2: Evaluate candidates in exact input order
        for cand_result in pipeline_result.results:
            decision, reason, duplicate_of = self._evaluate_candidate(
                cand_result=cand_result,
                rubric=rubric,
                duplicate_map=duplicate_map,
            )

            final_score = get_final_score(cand_result.audit_record)

            routed_candidates.append(
                RouteDecision(
                    candidate_id=cand_result.candidate_id,
                    decision=decision,
                    reason=reason,
                    final_score=final_score,
                    duplicate_of=duplicate_of,
                    candidate_result=cand_result,
                )
            )

        # Step 3: Pass through failed candidates without modification
        return RoutingResult(
            routed_candidates=routed_candidates,
            failed_candidates=list(pipeline_result.failed_candidates),
        )

    def _evaluate_candidate(
        self,
        cand_result: CandidateResult,
        rubric: ParsedJobDescription,
        duplicate_map: Dict[str, str],
    ) -> Tuple[RoutingDecision, str, Optional[str]]:
        """Apply strict routing precedence to a single candidate."""
        # Precedence 1: DUPLICATE
        if cand_result.candidate_id in duplicate_map:
            dup_id = duplicate_map[cand_result.candidate_id]
            return (
                RoutingDecision.DUPLICATE,
                f"Duplicate candidate detected: matches '{dup_id}'",
                dup_id,
            )

        # Precedence 2: INCOMPLETE_DATA
        is_incomplete, inc_reason = is_incomplete_profile(cand_result.candidate_profile)
        if is_incomplete:
            return (
                RoutingDecision.INCOMPLETE_DATA,
                inc_reason or "Profile incomplete",
                None,
            )

        # Precedence 3: FAIRNESS_FLAGGED
        if cand_result.audit_record.flagged:
            return (
                RoutingDecision.FLAGGED_FOR_REVIEW,
                cand_result.audit_record.flagged_reason or "Flagged by Fairness Auditor",
                None,
            )

        # Precedence 4: NOT_QUALIFIED
        is_qualified, qual_reason = check_qualification(
            candidate_result=cand_result,
            rubric=rubric,
            qualification_threshold=self.qualification_threshold,
            min_must_have_score=self.min_must_have_score,
        )
        if not is_qualified:
            return (
                RoutingDecision.NOT_QUALIFIED,
                qual_reason,
                None,
            )

        # Precedence 5: CLEARED
        return (
            RoutingDecision.CLEARED,
            "Candidate meets all qualification requirements and is cleared for auto-scheduling.",
            None,
        )


# ---------------------------------------------------------------------------
# Convenience Functional Wrappers
# ---------------------------------------------------------------------------


def route_pipeline_result(
    pipeline_result: PipelineResult,
    rubric: ParsedJobDescription,
    qualification_threshold: float = QUALIFICATION_THRESHOLD,
    min_must_have_score: float = MIN_MUST_HAVE_SCORE,
    duplicate_text_threshold: float = DUPLICATE_TEXT_SIMILARITY_THRESHOLD,
    duplicate_name_threshold: float = DUPLICATE_NAME_MATCH_SIMILARITY_THRESHOLD,
) -> RoutingResult:
    """Convenience function to route a PipelineResult using default or custom thresholds."""
    router = Router(
        qualification_threshold=qualification_threshold,
        min_must_have_score=min_must_have_score,
        duplicate_text_threshold=duplicate_text_threshold,
        duplicate_name_threshold=duplicate_name_threshold,
    )
    return router.route(pipeline_result=pipeline_result, rubric=rubric)


def route_candidate(
    candidate_result: CandidateResult,
    rubric: ParsedJobDescription,
    duplicate_of: Optional[str] = None,
    qualification_threshold: float = QUALIFICATION_THRESHOLD,
    min_must_have_score: float = MIN_MUST_HAVE_SCORE,
) -> RouteDecision:
    """Convenience function to route a single CandidateResult."""
    router = Router(
        qualification_threshold=qualification_threshold,
        min_must_have_score=min_must_have_score,
    )
    duplicate_map = {candidate_result.candidate_id: duplicate_of} if duplicate_of else {}
    decision, reason, dup_id = router._evaluate_candidate(
        cand_result=candidate_result,
        rubric=rubric,
        duplicate_map=duplicate_map,
    )
    final_score = get_final_score(candidate_result.audit_record)
    return RouteDecision(
        candidate_id=candidate_result.candidate_id,
        decision=decision,
        reason=reason,
        final_score=final_score,
        duplicate_of=dup_id,
        candidate_result=candidate_result,
    )
