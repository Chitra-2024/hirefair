"""Calibration constants and configuration for the HireFair Router agent.

All numerical thresholds and evaluation rules for routing live here together.
These are implementation choices and calibration parameters.
"""

# ---------------------------------------------------------------------------
# Qualification Thresholds
# ---------------------------------------------------------------------------
# Overall final weighted score required for CLEARED status
QUALIFICATION_THRESHOLD: float = 0.5

# Minimum score required for each must-have criterion (ScoreRecord.score >= 0.3)
MIN_MUST_HAVE_SCORE: float = 0.3

# ---------------------------------------------------------------------------
# Duplicate Detection Thresholds
# ---------------------------------------------------------------------------
# Similarity ratio (via SequenceMatcher on normalized resume text) to classify
# candidates as duplicate submissions even with different/missing candidate names
DUPLICATE_TEXT_SIMILARITY_THRESHOLD: float = 0.85

# Similarity ratio required when candidate names match (case-insensitive)
DUPLICATE_NAME_MATCH_SIMILARITY_THRESHOLD: float = 0.70

# ---------------------------------------------------------------------------
# Incomplete Profile Calibration Rules
# ---------------------------------------------------------------------------
# Minimum number of extracted skills required
MIN_SKILLS_COUNT: int = 1

# Whether any work experience entry with missing duration_months triggers incomplete status
REQUIRE_EXPERIENCE_DURATION: bool = True

# Whether a non-empty candidate name is required
REQUIRE_NAME: bool = True

# Whether at least one work experience or project entry is required
REQUIRE_WORK_OR_PROJECTS: bool = True
