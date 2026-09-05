"""Fairness Auditor agent for HireFair.

The Fairness Auditor independently reviews Matcher output through three checks:
  A. Counterfactual re-scoring using an anonymized candidate profile.
  B. LLM-based citation validity verification.
  C. Non-traditional evidence detection for under-scored criteria.

Implementation decisions (not explicitly specified in PROJECT_SPEC.md):

1. COUNTERFACTUAL_DELTA_THRESHOLD = 0.15
   A weighted-aggregate score difference of 0.15 or more is treated as a "large delta"
   indicating a potential fairness concern. The design document says "large delta" but does
   not specify a numeric value. 0.15 (15 percentage points) was chosen as a reasonable
   initial value; it is defined here as a named constant and can be tuned.

2. NON_TRADITIONAL_SCORE_THRESHOLD = 0.5
   Non-traditional evidence is only flagged when the candidate's ORIGINAL criterion score
   is at or below 0.5. This prevents flagging candidates who already received a strong
   score from a criterion that happens to also have non-traditional evidence. 0.5 was chosen
   as a mid-range threshold; it is a named constant and can be tuned.

3. Weighted aggregate formula:
   aggregate = sum(score * weight) / sum(weight)
   Confidence is NOT used in the aggregate. Confidence remains a ScoreRecord field but
   does not influence the fairness delta calculation. The design document does not specify
   how aggregate scores should be calculated; weighting by rubric importance is the most
   defensible choice.

4. Option A fallback (deviation from literal design document wording):
   The design document describes a "re-score" when both large delta AND citation failure
   occur. This implementation does NOT make a second Matcher/Gemini call. Instead, it
   adopts the already-computed anonymized aggregate score as the fallback final score.
   Reason: The anonymized score has already been computed in Check A. Making another
   identical deterministic Matcher call would consume Gemini quota without providing a
   meaningfully different signal. repair_applied=True records that the fallback was used.

5. Non-traditional evidence detection (Check C):
   Detection is heuristic-based using keyword matching on candidate projects and experience.
   This is intentionally kept lightweight for v1. It checks for evidence not its quality.

6. flagged (bool) and repair_applied (bool) were added to AuditRecord.
   The design document lists flagged_reason but does not include an explicit boolean flag.
   flagged is added so the Router can read a clear boolean instead of checking
   whether flagged_reason is non-empty.
"""

from typing import Dict, List, Optional

from app.agents.citation_checker import check_citations
from app.agents.matcher import Matcher
from app.models.audit import AuditRecord
from app.models.candidate import CandidateProfile
from app.models.job_description import ParsedJobDescription
from app.models.score import MatchResult, ScoreRecord

# ---------------------------------------------------------------------------
# Named thresholds — not specified by PROJECT_SPEC.md; see module docstring.
# ---------------------------------------------------------------------------

# Weighted aggregate score difference that constitutes a "large delta".
COUNTERFACTUAL_DELTA_THRESHOLD: float = 0.15

# Original criterion score at or below which non-traditional evidence is flagged.
NON_TRADITIONAL_SCORE_THRESHOLD: float = 0.5

# Keywords and patterns used for non-traditional evidence heuristic detection.
_NON_TRADITIONAL_KEYWORDS = [
    "open-source",
    "open source",
    "opensource",
    "personal project",
    "side project",
    "self-taught",
    "self taught",
    "freelance",
    "freelancer",
    "independent",
    "career change",
    "bootcamp",
    "boot camp",
    "community",
    "volunteer",
    "hobby",
    "github",
    "gitlab",
    "contribution",
    "contributor",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _weighted_aggregate(
    scores: List[ScoreRecord],
    weights: Dict[str, int],
) -> Optional[float]:
    """Compute a rubric-weight-adjusted aggregate score.

    aggregate = sum(score * weight) / sum(weight)

    Returns None if total weight is zero to avoid ZeroDivisionError.
    Confidence is intentionally excluded from this calculation.
    """
    total_weight = sum(weights.get(r.criterion_id, 0) for r in scores)
    if total_weight == 0:
        return None
    return sum(r.score * weights.get(r.criterion_id, 0) for r in scores) / total_weight


def _has_non_traditional_evidence(candidate: CandidateProfile) -> bool:
    """Heuristic check: does the candidate profile contain non-traditional evidence?

    Searches project descriptions and work experience descriptions for keywords
    that indicate non-traditional competency pathways.
    """
    text_corpus = " ".join(
        [
            " ".join(p.description for p in candidate.projects),
            " ".join(e.description for e in candidate.experience),
            candidate.raw_text,
        ]
    ).lower()

    return any(kw in text_corpus for kw in _NON_TRADITIONAL_KEYWORDS)


def _non_traditional_relevant_for_criterion(
    candidate: CandidateProfile,
    criterion_description: str,
) -> bool:
    """Check whether any non-traditional evidence is plausibly relevant to a criterion.

    For v1, this checks whether any project or non-traditional-keyword-containing
    experience description contains words also present in the criterion description.
    This is a lightweight relevance heuristic, not semantic search.
    """
    criterion_words = set(criterion_description.lower().split())
    # Stop words to ignore in relevance matching
    stop_words = {
        "a", "an", "the", "and", "or", "of", "to", "in", "for", "with",
        "on", "at", "by", "from", "as", "is", "are", "was", "were", "be",
        "been", "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "shall", "can",
    }
    meaningful_words = criterion_words - stop_words

    if not meaningful_words:
        return False

    # Check projects for non-traditional evidence relevant to criterion
    for proj in candidate.projects:
        proj_text = (proj.description + " " + " ".join(proj.technologies)).lower()
        if any(word in proj_text for word in meaningful_words):
            return True

    # Check experience descriptions containing non-traditional keywords
    for exp in candidate.experience:
        desc_lower = exp.description.lower()
        has_non_trad = any(kw in desc_lower for kw in _NON_TRADITIONAL_KEYWORDS)
        if has_non_trad and any(word in desc_lower for word in meaningful_words):
            return True

    return False


# ---------------------------------------------------------------------------
# FairnessAuditor
# ---------------------------------------------------------------------------


class FairnessAuditor:
    """Independent critic that audits Matcher output for fairness concerns.

    Performs:
      Check A — Counterfactual re-scoring using anonymized profile.
      Check B — LLM-based citation validity verification.
      Check C — Non-traditional evidence detection for under-scored criteria.
    """

    def __init__(self, model: str = "gemini-2.5-flash"):
        self.model = model
        self._matcher = Matcher(model=model)

    def audit(
        self,
        candidate: CandidateProfile,
        rubric: ParsedJobDescription,
        original_result: MatchResult,
    ) -> AuditRecord:
        """Run a full fairness audit on a completed Matcher result.

        Args:
            candidate: The original (non-anonymized) candidate profile.
            rubric: The parsed job description used for matching.
            original_result: The Matcher's output for the original profile.

        Returns:
            AuditRecord summarising all checks and any fairness flags.
        """
        if candidate is None or not isinstance(candidate, CandidateProfile):
            raise ValueError("CandidateProfile must be provided.")
        if rubric is None or not isinstance(rubric, ParsedJobDescription):
            raise ValueError("ParsedJobDescription rubric must be provided.")
        if original_result is None or not isinstance(original_result, MatchResult):
            raise ValueError("MatchResult must be provided.")

        # Build rubric lookup maps
        weights: Dict[str, int] = {r.criterion_id: r.weight for r in rubric.requirements}
        descriptions: Dict[str, str] = {r.criterion_id: r.description for r in rubric.requirements}

        # ----------------------------------------------------------------
        # CHECK A — Counterfactual re-scoring
        # ----------------------------------------------------------------
        original_aggregate = _weighted_aggregate(original_result.scores, weights)

        anonymized_profile = candidate.anonymize()
        anonymized_result: MatchResult = self._matcher.match(
            candidate=anonymized_profile,
            rubric=rubric,
        )
        anonymized_aggregate = _weighted_aggregate(anonymized_result.scores, weights)

        if original_aggregate is not None and anonymized_aggregate is not None:
            delta = abs(original_aggregate - anonymized_aggregate)
        else:
            delta = None

        large_delta = (delta is not None) and (delta >= COUNTERFACTUAL_DELTA_THRESHOLD)

        # ----------------------------------------------------------------
        # CHECK B — Citation validity
        # ----------------------------------------------------------------
        citation_batch = check_citations(
            score_records=original_result.scores,
            rubric_descriptions=descriptions,
            model=self.model,
        )

        failed_citations: List[str] = []
        for check in citation_batch.results:
            if not check.is_valid:
                failed_citations.append(
                    f"criterion '{check.criterion_id}': {check.reason}"
                )

        citation_valid = len(failed_citations) == 0

        # ----------------------------------------------------------------
        # CHECK C — Non-traditional evidence scan
        # ----------------------------------------------------------------
        # Map original scores for threshold comparison
        original_scores_by_id: Dict[str, float] = {
            r.criterion_id: r.score for r in original_result.scores
        }

        non_traditional_found = False
        non_traditional_reasons: List[str] = []

        for req in rubric.requirements:
            orig_score = original_scores_by_id.get(req.criterion_id, 0.0)
            if orig_score <= NON_TRADITIONAL_SCORE_THRESHOLD:
                if _non_traditional_relevant_for_criterion(candidate, req.description):
                    non_traditional_found = True
                    non_traditional_reasons.append(
                        f"criterion '{req.criterion_id}' (original score {orig_score:.2f}): "
                        f"relevant non-traditional evidence exists but may be under-reflected"
                    )

        # ----------------------------------------------------------------
        # Determine flagged status and build flagged_reason
        # ----------------------------------------------------------------
        flag_parts: List[str] = []

        if large_delta:
            flag_parts.append(
                f"Large counterfactual score delta detected: "
                f"original={original_aggregate:.3f}, "
                f"anonymized={anonymized_aggregate:.3f}, "
                f"delta={delta:.3f} (threshold={COUNTERFACTUAL_DELTA_THRESHOLD})"
            )

        if not citation_valid:
            flag_parts.append(
                "Citation validity failure — the following citations were judged invalid: "
                + "; ".join(failed_citations)
            )

        if non_traditional_found:
            flag_parts.append(
                "Non-traditional competency evidence may be under-reflected — "
                + "; ".join(non_traditional_reasons)
            )

        flagged = len(flag_parts) > 0
        flagged_reason = " | ".join(flag_parts) if flag_parts else ""

        # ----------------------------------------------------------------
        # OPTION A FALLBACK: large delta AND citation failure
        # ----------------------------------------------------------------
        repair_applied = False
        if large_delta and not citation_valid:
            # Do NOT call Matcher again. Adopt the already-computed anonymized aggregate.
            # anonymized_aggregate is already set above; no additional LLM call is made.
            repair_applied = True
            fallback_note = (
                f" | Option A fallback applied: anonymized score "
                f"{anonymized_aggregate:.3f} adopted as authoritative fallback "
                f"(no additional Matcher call was made)."
            )
            flagged_reason = flagged_reason + fallback_note

        return AuditRecord(
            candidate_id=candidate.candidate_id,
            original_score=original_aggregate,
            anonymized_score=anonymized_aggregate,
            delta=delta,
            citation_valid=citation_valid,
            non_traditional_evidence_found=non_traditional_found,
            flagged=flagged,
            flagged_reason=flagged_reason,
            repair_applied=repair_applied,
        )


def audit_candidate(
    candidate: CandidateProfile,
    rubric: ParsedJobDescription,
    original_result: MatchResult,
    model: str = "gemini-2.5-flash",
) -> AuditRecord:
    """Convenience functional wrapper around FairnessAuditor."""
    auditor = FairnessAuditor(model=model)
    return auditor.audit(
        candidate=candidate,
        rubric=rubric,
        original_result=original_result,
    )
