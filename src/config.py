"""Central configuration: paths, class definitions, constants.

Everything in the project should import paths from here so that we have a
single source of truth.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"
GESTURES_DIR = DATASET_DIR / "gestures"
FATIGUE_DIR = DATASET_DIR / "fatigue"

EXTERNAL_DATA_DIR = PROJECT_ROOT / "external_datasets"
PROCESSED_DIR = PROJECT_ROOT / "data_processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Specific output subfolders (created lazily)
FRAMES_DIR = PROCESSED_DIR / "frames"
SPLITS_DIR = PROCESSED_DIR / "splits"
MODELS_DIR = OUTPUTS_DIR / "models"
REPORTS_DIR = OUTPUTS_DIR / "reports"
FIGURES_DIR = OUTPUTS_DIR / "figures"

# ---------------------------------------------------------------------------
# Class definitions
# ---------------------------------------------------------------------------
# Folder names use spaces; we keep mapping from folder → canonical class id.

GESTURE_FOLDERS = {
    "open palm":         "open_palm",
    "thumbs up":         "thumbs_up",
    "correct sequence":  "correct_sequence",
    "wrong sequence":    "wrong_sequence",
    "too slow":          "too_slow",
    "incomplete":        "incomplete",
    "random hands":      "random_hands",
    "no hands":          "no_hands",
}

# For training the per-frame gesture classifier we collapse to:
#   open_palm | thumbs_up | negative
GESTURE_CLASSIFIER_LABELS = {
    "open palm":     "open_palm",
    "thumbs up":     "thumbs_up",
    "random hands":  "negative",
    "no hands":      "negative",
}

# Sequence-level test categories (for evaluating the state machine).
GESTURE_SEQUENCE_LABELS = {
    "correct sequence": "valid",
    "wrong sequence":   "invalid",
    "too slow":         "invalid",
    "incomplete":       "invalid",
}

FATIGUE_FOLDERS = {
    "calibration":   "calibration",
    "alert":         "alert",
    "alert looking": "alert_looking",
    "talking":       "talking",
    "phone":         "phone",
    "slow blinks":   "slow_blinks",
    "eyes closed":   "eyes_closed",
    "yawning":       "yawning",
    "head drooping": "head_drooping",
    "head tilting":  "head_tilting",
    "microsleep":    "microsleep",
    "fatigue face":  "fatigue_face",
    "transition":    "transition",
}

# Coarse 3-class label used for the final classifier.
# alert vs drowsy vs yawning is a common formulation in the literature.
FATIGUE_COARSE_LABEL = {
    "alert":         "alert",
    "alert_looking": "alert",
    "talking":       "alert",
    "phone":         "alert",          # distractor still "not drowsy"
    "calibration":   "alert",
    "slow_blinks":   "drowsy",
    "eyes_closed":   "drowsy",
    "head_drooping": "drowsy",
    "head_tilting":  "drowsy",
    "microsleep":    "drowsy",
    "fatigue_face":  "drowsy",
    "yawning":       "yawning",
    "transition":    "transition",     # held out for temporal eval
}

# ---------------------------------------------------------------------------
# Frame extraction defaults
# ---------------------------------------------------------------------------
DEFAULT_FRAME_STRIDE = 5     # extract every 5th frame
DEFAULT_TARGET_FPS = 6       # alternative: resample to N fps
DEFAULT_RESIZE = (640, 360)  # downscale for processing (None = keep original)

# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
# We have 2 subjects: person1 and person2.
# For our small dataset we use leave-one-person-out CV across these 2 splits.
PERSONS = ["person1", "person2"]

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".m4v", ".mkv"}


def ensure_dirs() -> None:
    """Create derivable output directories if missing."""
    for d in (PROCESSED_DIR, FRAMES_DIR, SPLITS_DIR,
              OUTPUTS_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
