"""End-to-end realtime fatigue-detection orchestrator (Stage 6).

The state of the system at any moment is one of three things:

    GESTURE_WAITING    waiting for the user's activation gesture sequence.
    FATIGUE_MONITOR    activated, running the fatigue classifier on a
                       rolling window of face frames.
    ALERT              fatigue persisted long enough to raise an alarm.

Frame contract: callers pass an ``np.ndarray`` of shape (H, W, 3) in RGB
uint8, plus a wall-clock timestamp in seconds. Everything else is owned
by this orchestrator.

Pipelines are pluggable via constructor:
    * ``gesture_predictor``  any ``FramePredictor`` (classical / CNN)
    * ``fatigue_predictor``  callable: deque[features] -> (label, conf)
                             (we ship two factories below for the
                             classical-aggregator and the temporal-CNN).

The orchestrator does not own the webcam or the display — those live in
``scripts/run_realtime.py``. This keeps the core testable and lets the
same class power both webcam and offline-video demos.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch

from ..gestures.landmarks import HandLandmarks, HandLandmarkExtractor
from ..gestures.state_machine import (
    FrameEvent, GestureSequenceStateMachine, State, StateMachineConfig,
)
from ..fatigue.face_landmarks import FaceLandmarks, FaceLandmarkExtractor
from ..fatigue.features import FEATURE_DIM, landmarks_to_features
from ..fatigue.temporal_cnn import (
    CLASSES as FATIGUE_CLASSES, IDX_TO_LABEL, SEQ_LEN, best_device,
)


# ---------------------------------------------------------------------------
# Pipeline state
# ---------------------------------------------------------------------------
class SystemState(Enum):
    GESTURE_WAITING = "gesture_waiting"
    FATIGUE_MONITOR = "fatigue_monitor"
    ALERT = "alert"


@dataclass
class GestureFrameInfo:
    hand: Optional[HandLandmarks]
    label: Optional[str]
    confidence: Optional[float]
    probs: Optional[Dict[str, float]] = None


@dataclass
class FatigueFrameInfo:
    face_present: bool
    fatigue_label: Optional[str]
    fatigue_confidence: Optional[float]
    fatigue_probs: Optional[Dict[str, float]]


@dataclass
class FrameOutcome:
    """Everything the renderer needs to draw the HUD for one frame."""
    timestamp_s: float
    system_state: SystemState
    sm_state: State
    gesture: GestureFrameInfo
    fatigue: FatigueFrameInfo
    alert_on: bool
    alert_age_s: float


# ---------------------------------------------------------------------------
# Fatigue predictors (over a rolling buffer of per-frame features)
# ---------------------------------------------------------------------------
FatigueBufferPredictor = Callable[
    [np.ndarray],   # (T, FEATURE_DIM) float32
    Tuple[str, float, Dict[str, float]],
]


def make_temporal_cnn_predictor(
    model_path,
    *,
    seq_len: int = SEQ_LEN,
    device: Optional[torch.device] = None,
) -> FatigueBufferPredictor:
    """Wrap the saved temporal-CNN checkpoint for online inference."""
    from ..fatigue.temporal_cnn import load_model
    device = device or best_device()
    model, mean, std = load_model(model_path, device=device)
    mean = mean.astype(np.float32)
    std = np.maximum(std.astype(np.float32), 1e-6)

    def predict(buf: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        # buf: (T, F). Center-pad to seq_len for stable behaviour.
        T = buf.shape[0]
        if T == 0:
            return FATIGUE_CLASSES[0], 0.0, {c: 0.0 for c in FATIGUE_CLASSES}
        if T >= seq_len:
            x = buf[-seq_len:]
            real_len = seq_len
        else:
            pad = np.zeros((seq_len - T, FEATURE_DIM), dtype=np.float32)
            x = np.concatenate([buf, pad], axis=0)
            real_len = T
        x = (x - mean) / std
        x_t = torch.from_numpy(x.T.astype(np.float32))[None, ...].to(device)
        mask = np.zeros(seq_len, dtype=np.float32)
        mask[:real_len] = 1.0
        m_t = torch.from_numpy(mask)[None, ...].to(device)
        with torch.no_grad():
            logits = model(x_t, m_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        idx = int(np.argmax(probs))
        return (
            IDX_TO_LABEL[idx],
            float(probs[idx]),
            {c: float(p) for c, p in zip(FATIGUE_CLASSES, probs)},
        )

    return predict


def make_heuristic_predictor(
    *,
    drowsy_blink_threshold: float = 0.45,
    yawn_jaw_threshold: float = 0.25,
    jaw_quantile: float = 0.90,
) -> FatigueBufferPredictor:
    """Person-agnostic fatigue scorer based on MediaPipe blendshapes.

    The temporal CNN over-fits to the 2 training subjects' specific face
    appearance and fails on unseen people. This heuristic side-steps that:
    blendshapes are normalised per-face by MediaPipe, so the scores below
    have the same meaning for everyone.

    Two-stage decision (matches Stage 5D ``eval_fatigue_heuristic.py``):
      1. Yawn — peak (``jaw_quantile``) ``jawOpen`` over the buffer
         crosses ``yawn_jaw_threshold``. A yawn is episodic (a 1-3 s
         burst inside an otherwise closed-mouth window), so the high
         quantile catches it where the mean dilutes it.
      2. Drowsy vs alert — soft competition between
         ``drowsy_sig = blink_mean / drowsy_blink_threshold`` and
         ``alert_sig = 1 - drowsy_sig``. The effective drowsy-vs-alert
         boundary is ``drowsy_blink_threshold / 2``, which catches
         moderately-elevated blink (e.g. our person2 drowsy clips have
         blink≈0.40 — well above alert≈0.18 but below 0.45).

    Probabilities are normalised continuous scores so the HUD bars
    still update smoothly.
    """
    from ..fatigue.features import FEATURE_NAMES
    idx_blink_l = FEATURE_NAMES.index("eyeBlinkLeft")
    idx_blink_r = FEATURE_NAMES.index("eyeBlinkRight")
    idx_jaw = FEATURE_NAMES.index("jawOpen")

    def predict(buf: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        if buf.shape[0] == 0:
            return FATIGUE_CLASSES[0], 0.0, {c: 0.0 for c in FATIGUE_CLASSES}
        blink_mean = float((buf[:, idx_blink_l] + buf[:, idx_blink_r]).mean() / 2)
        jaw_peak = float(np.quantile(buf[:, idx_jaw], jaw_quantile))
        # Continuous, bounded "evidence" for each class.
        drowsy_sig = min(1.0, max(0.0, blink_mean / max(drowsy_blink_threshold, 1e-3)))
        yawn_sig = min(1.0, max(0.0, jaw_peak / max(yawn_jaw_threshold, 1e-3)))
        alert_sig = max(0.0, 1.0 - drowsy_sig)
        # Two-stage decision: yawn first (peak signal), then drowsy/alert
        # by argmax of the soft competition.
        if yawn_sig >= 1.0:
            label = "yawning"
        else:
            label = "drowsy" if drowsy_sig >= alert_sig else "alert"
        # HUD probabilities — keep all three so the bars are informative.
        # When yawning fires, give it dominant mass; otherwise share
        # mass between drowsy and alert.
        if label == "yawning":
            scores = {"alert": 0.0, "drowsy": drowsy_sig, "yawning": yawn_sig}
        else:
            scores = {"alert": alert_sig, "drowsy": drowsy_sig, "yawning": yawn_sig}
        total = sum(scores.values()) + 1e-6
        probs = {c: v / total for c, v in scores.items()}
        return label, probs[label], probs

    return predict


def make_classical_aggregator_predictor(
    sklearn_model,
) -> FatigueBufferPredictor:
    """Per-frame SVM/RF + mean-probability aggregation over the buffer.
    Mirrors Stage 4D's ``aggregate_mean_prob`` for the deployment model."""
    classes = list(sklearn_model.classes_)

    def predict(buf: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        if buf.shape[0] == 0:
            return classes[0], 0.0, {c: 0.0 for c in classes}
        probs = sklearn_model.predict_proba(buf)
        avg = probs.mean(axis=0)
        idx = int(np.argmax(avg))
        return (
            classes[idx],
            float(avg[idx]),
            {c: float(p) for c, p in zip(classes, avg)},
        )

    return predict


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
@dataclass
class RealtimeConfig:
    fatigue_buffer_seconds: float = 2.5      # how much history to keep
    target_fps: float = 30.0                 # used to size the buffer
    sm_config: StateMachineConfig = field(default_factory=StateMachineConfig)
    # Fatigue alert: persistence + dwell.
    alert_classes: Tuple[str, ...] = ("drowsy", "yawning")
    alert_min_persist_s: float = 1.5         # consecutive seconds in alert_classes
    alert_min_confidence: float = 0.55
    # Cool-down after activation before the fatigue model is queried.
    fatigue_warmup_s: float = 0.5


class RealtimeFatigueSystem:
    """End-to-end realtime orchestrator. Frame-rate agnostic — pass real
    timestamps and the rolling buffer adapts to whatever FPS you feed."""

    def __init__(
        self,
        *,
        gesture_predictor: Callable[[np.ndarray, HandLandmarks], Tuple[str, float]],
        fatigue_predictor: FatigueBufferPredictor,
        config: Optional[RealtimeConfig] = None,
        hand_extractor: Optional[HandLandmarkExtractor] = None,
        face_extractor: Optional[FaceLandmarkExtractor] = None,
        gesture_probs_fn: Optional[Callable[
            [np.ndarray, HandLandmarks], Dict[str, float]
        ]] = None,
    ) -> None:
        self.cfg = config or RealtimeConfig()
        self.gesture_predictor = gesture_predictor
        self.gesture_probs_fn = gesture_probs_fn
        self.fatigue_predictor = fatigue_predictor
        self._hand = hand_extractor or HandLandmarkExtractor(
            static_image_mode=True, max_num_hands=1,
        )
        self._face = face_extractor or FaceLandmarkExtractor()
        self._sm = GestureSequenceStateMachine(self.cfg.sm_config)
        # Buffer is sized so we cover ``fatigue_buffer_seconds`` at target_fps.
        n = max(int(self.cfg.fatigue_buffer_seconds * self.cfg.target_fps), 16)
        self._fatigue_buf: Deque[np.ndarray] = deque(maxlen=n)
        self._activated_at: Optional[float] = None
        # Alert hysteresis.
        self._alert_streak_started_at: Optional[float] = None
        self._alert_on: bool = False
        self._alert_started_at: Optional[float] = None

    # -- explicit cleanup so cv2 can reuse the camera --
    def close(self) -> None:
        self._hand.close()
        self._face.close()

    def __enter__(self) -> "RealtimeFatigueSystem":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------ core
    def _system_state(self) -> SystemState:
        if self._alert_on:
            return SystemState.ALERT
        if self._sm.state == State.ACTIVATED:
            return SystemState.FATIGUE_MONITOR
        return SystemState.GESTURE_WAITING

    def reset(self) -> None:
        self._sm.reset()
        self._fatigue_buf.clear()
        self._activated_at = None
        self._alert_streak_started_at = None
        self._alert_on = False
        self._alert_started_at = None

    def step(self, rgb: np.ndarray, timestamp_s: float) -> FrameOutcome:
        """Process one frame. Returns a :class:`FrameOutcome` for rendering."""
        gesture_info = GestureFrameInfo(hand=None, label=None, confidence=None)
        fatigue_info = FatigueFrameInfo(
            face_present=False, fatigue_label=None,
            fatigue_confidence=None, fatigue_probs=None,
        )

        # --- 1. Gesture path: only run before activation ---
        if self._sm.state != State.ACTIVATED:
            hand = self._hand.detect(rgb)
            gesture_info.hand = hand
            if hand is None:
                ev = FrameEvent(
                    timestamp_s=timestamp_s, label="negative", confidence=1.0,
                )
            else:
                label, conf = self.gesture_predictor(rgb, hand)
                gesture_info.label = label
                gesture_info.confidence = conf
                if self.gesture_probs_fn is not None:
                    try:
                        gesture_info.probs = self.gesture_probs_fn(rgb, hand)
                    except Exception:
                        gesture_info.probs = None
                ev = FrameEvent(
                    timestamp_s=timestamp_s, label=label, confidence=conf,
                )
            decision = self._sm.update(ev)
            if decision.activated and self._activated_at is None:
                self._activated_at = timestamp_s

        # --- 2. Fatigue path: only after activation + warmup ---
        if (
            self._sm.state == State.ACTIVATED
            and self._activated_at is not None
            and (timestamp_s - self._activated_at) >= self.cfg.fatigue_warmup_s
        ):
            face = self._face.detect(rgb)
            if face is not None:
                fatigue_info.face_present = True
                feats = landmarks_to_features(face).astype(np.float32)
                self._fatigue_buf.append(feats)
                if len(self._fatigue_buf) >= 8:
                    buf = np.stack(self._fatigue_buf, axis=0)
                    label, conf, probs = self.fatigue_predictor(buf)
                    fatigue_info.fatigue_label = label
                    fatigue_info.fatigue_confidence = conf
                    fatigue_info.fatigue_probs = probs
                    self._update_alert(label, conf, timestamp_s)

        alert_age = (
            (timestamp_s - self._alert_started_at)
            if (self._alert_on and self._alert_started_at is not None)
            else 0.0
        )
        return FrameOutcome(
            timestamp_s=timestamp_s,
            system_state=self._system_state(),
            sm_state=self._sm.state,
            gesture=gesture_info,
            fatigue=fatigue_info,
            alert_on=self._alert_on,
            alert_age_s=alert_age,
        )

    # --------------------------------------------------------- alert dwell
    def _update_alert(
        self, label: str, conf: float, timestamp_s: float,
    ) -> None:
        in_alert_class = (
            label in self.cfg.alert_classes
            and conf >= self.cfg.alert_min_confidence
        )
        if in_alert_class:
            if self._alert_streak_started_at is None:
                self._alert_streak_started_at = timestamp_s
            elif (timestamp_s - self._alert_streak_started_at
                  >= self.cfg.alert_min_persist_s):
                if not self._alert_on:
                    self._alert_on = True
                    self._alert_started_at = timestamp_s
        else:
            # Recover only if we drop the alert class for at least 0.5 s.
            self._alert_streak_started_at = None
            if self._alert_on:
                # Hysteresis: stay on for the duration of this frame's call;
                # caller resets when label leaves alert_classes consistently.
                self._alert_on = False
                self._alert_started_at = None
