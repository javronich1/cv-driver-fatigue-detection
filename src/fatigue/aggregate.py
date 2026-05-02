"""Stage 5E: clip-level aggregate features for fatigue classification.

Per-frame features (Stage 4B) capture an instantaneous facial state but
miss the *temporal* signature of fatigue: long eye-closures, head nods,
jaw bursts, blink rate. The heuristic baseline (Stage 5D) compresses a
buffer with two summary statistics (`blink_mean`, `jaw_p90`) and beats
both classical and modern trained models on LOSO. The natural next step
is to give a small classifier the same level of abstraction with a
richer set of summary statistics — so it can learn to e.g. distinguish
"high mean blink + low full-closure ratio" (looking at phone) from
"high mean blink + high full-closure ratio" (drowsy).

This module turns one ``(T, FEATURE_DIM)`` per-frame buffer into a fixed
``AGG_FEATURE_DIM``-d vector the trained model consumes. The same code
path is used by the offline trainer and by the realtime predictor.

Aggregate features (15 total)::

    blink_mean, blink_p75, blink_p90,
    look_down_mean, look_down_p75,
    ecdr_05, ecdr_07,                    # eye-closure-duration ratios
    longest_closure_ratio,               # longest run of blink>=0.5
                                         # divided by buffer length
    blink_rate_per_s,                    # number of distinct blink events / s
    jaw_p75, jaw_p90,
    longest_jaw_burst_ratio,             # longest run of jawOpen>=0.3 / T
    pitch_std,                           # head-pitch std (nodding)
    pitch_excursion,                     # max - min pitch
    mar_p90,                             # mouth aspect ratio peak

The realtime system always passes ``fps=target_fps`` (default 30) so the
``blink_rate_per_s`` feature stays consistent between training and
deployment.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from .features import FEATURE_NAMES


# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------
AGG_FEATURE_NAMES: Tuple[str, ...] = (
    "blink_mean", "blink_p75", "blink_p90",
    "look_down_mean", "look_down_p75",
    "ecdr_05", "ecdr_07",
    "longest_closure_ratio",
    "blink_rate_per_s",
    "jaw_p75", "jaw_p90",
    "longest_jaw_burst_ratio",
    "pitch_std", "pitch_excursion",
    "mar_p90",
)
AGG_FEATURE_DIM = len(AGG_FEATURE_NAMES)


# Per-frame feature indices (resolved once, cached at import).
_IDX_BLINK_L = FEATURE_NAMES.index("eyeBlinkLeft")
_IDX_BLINK_R = FEATURE_NAMES.index("eyeBlinkRight")
_IDX_LOOK_DL = FEATURE_NAMES.index("eyeLookDownLeft")
_IDX_LOOK_DR = FEATURE_NAMES.index("eyeLookDownRight")
_IDX_JAW = FEATURE_NAMES.index("jawOpen")
_IDX_MAR = FEATURE_NAMES.index("mar")
_IDX_PITCH = FEATURE_NAMES.index("head_pitch")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _longest_run_above(x: np.ndarray, thresh: float) -> int:
    """Length of the longest contiguous run with ``x >= thresh``."""
    if x.size == 0:
        return 0
    above = (x >= thresh).astype(np.int8)
    if not above.any():
        return 0
    # Run-length encode.
    diffs = np.diff(np.concatenate(([0], above, [0])))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    return int((ends - starts).max())


def _blink_count(blink: np.ndarray, *, low: float = 0.3, high: float = 0.5) -> int:
    """Hysteresis-style count of distinct blink events in the buffer.

    A blink is counted each time the signal crosses from below ``low`` to
    above ``high``. Hysteresis keeps near-threshold jitter from inflating
    the count.
    """
    if blink.size < 2:
        return 0
    state = "open"  # open / closing
    count = 0
    for v in blink:
        if state == "open" and v >= high:
            count += 1
            state = "closing"
        elif state == "closing" and v < low:
            state = "open"
    return count


def _safe_quantile(x: np.ndarray, q: float) -> float:
    return float(np.quantile(x, q)) if x.size else 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def aggregate_features(buf: np.ndarray, *, fps: float = 30.0) -> np.ndarray:
    """Turn a per-frame feature buffer into a fixed-length aggregate vector.

    Args:
        buf: ``(T, FEATURE_DIM)`` float array. ``T`` may be any length.
        fps: frames per second for the buffer — used to convert frame
             counts into seconds for ``blink_rate_per_s``. Default 30.

    Returns:
        ``(AGG_FEATURE_DIM,)`` float32 array in the order of
        ``AGG_FEATURE_NAMES``.
    """
    out = np.zeros(AGG_FEATURE_DIM, dtype=np.float32)
    T = buf.shape[0]
    if T == 0:
        return out
    blink = (buf[:, _IDX_BLINK_L] + buf[:, _IDX_BLINK_R]) / 2.0
    look_down = (buf[:, _IDX_LOOK_DL] + buf[:, _IDX_LOOK_DR]) / 2.0
    jaw = buf[:, _IDX_JAW]
    mar = buf[:, _IDX_MAR]
    pitch = buf[:, _IDX_PITCH]

    seconds = max(T / max(fps, 1e-6), 1e-6)
    longest_closure = _longest_run_above(blink, 0.5) / max(T, 1)
    longest_jaw = _longest_run_above(jaw, 0.3) / max(T, 1)
    n_blinks = _blink_count(blink)

    out[0] = float(blink.mean())
    out[1] = _safe_quantile(blink, 0.75)
    out[2] = _safe_quantile(blink, 0.90)
    out[3] = float(look_down.mean())
    out[4] = _safe_quantile(look_down, 0.75)
    out[5] = float((blink > 0.5).mean())
    out[6] = float((blink > 0.7).mean())
    out[7] = float(longest_closure)
    out[8] = float(n_blinks / seconds)
    out[9] = _safe_quantile(jaw, 0.75)
    out[10] = _safe_quantile(jaw, 0.90)
    out[11] = float(longest_jaw)
    out[12] = float(pitch.std())
    out[13] = float(pitch.max() - pitch.min())
    out[14] = _safe_quantile(mar, 0.90)
    return out


def aggregate_features_from_dataframe(
    df,
    *,
    fps: float = 30.0,
):
    """Compute one aggregate vector per clip from a per-frame ``DataFrame``.

    Convenience wrapper used by the offline trainer. Returns a numpy
    array ``(n_clips, AGG_FEATURE_DIM)`` and a meta DataFrame indexed
    by ``video`` carrying ``person`` and ``coarse_label``.
    """
    import pandas as pd  # local: keep this module free of pandas at import

    rows = []
    metas = []
    feature_cols = list(FEATURE_NAMES)
    for video, sub in df.groupby("video", sort=False):
        buf = sub[feature_cols].to_numpy(dtype=np.float32)
        rows.append(aggregate_features(buf, fps=fps))
        metas.append({
            "video": video,
            "person": str(sub["person"].iloc[0]),
            "folder_label": str(sub["folder_label"].iloc[0]),
            "fine_label": str(sub["fine_label"].iloc[0]),
            "coarse_label": str(sub["coarse_label"].iloc[0]),
            "n_frames": int(len(sub)),
        })
    X = np.stack(rows, axis=0).astype(np.float32) if rows \
        else np.zeros((0, AGG_FEATURE_DIM), dtype=np.float32)
    meta = pd.DataFrame(metas)
    return X, meta
