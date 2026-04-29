"""Gesture-sequence state machine for the activation step.

The fatigue detector is OFF by default. To activate it, the user must perform
the configured gesture sequence (default: ``open_palm`` then ``thumbs_up``)
within a limited time window. This module implements that logic on top of
per-frame classifier predictions.

Design
------
We don't trust any single classifier prediction. Instead, we keep a small
sliding window and require ``min_consecutive`` frames of the same gesture
above ``min_confidence`` before we accept it.

States:

    IDLE           - waiting for the first gesture
    GOT_GESTURE_1  - first gesture detected, waiting for the second
    ACTIVATED      - sequence completed successfully

Transitions:

    IDLE
      └─ on stable detection of gesture[0] → GOT_GESTURE_1 (start timer)

    GOT_GESTURE_1
      ├─ on stable detection of gesture[1] within `window_s` → ACTIVATED
      ├─ on stable detection of gesture[0] again            → stay
      ├─ on stable detection of an unexpected gesture       → IDLE  (reject)
      └─ on timer expiry  (`window_s`)                      → IDLE

    ACTIVATED
      └─ stays activated; the host system reads ``is_activated`` and starts
         the fatigue detector. Reset is manual (``reset()``).

The state machine is *deterministic* and operates on time-stamped frame
predictions. It does not care whether the predictions came from the
classical pipeline or the modern one — both are interchangeable.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, List, Optional, Tuple


class State(Enum):
    IDLE = "idle"
    GOT_GESTURE_1 = "got_gesture_1"
    ACTIVATED = "activated"


@dataclass
class FrameEvent:
    timestamp_s: float
    label: str            # predicted class id, e.g. "open_palm" / "thumbs_up" / "negative"
    confidence: float     # in [0, 1]


@dataclass
class StateMachineConfig:
    # Sequence to recognise (in order).
    sequence: Tuple[str, ...] = ("open_palm", "thumbs_up")
    # Maximum seconds allowed between gesture[0] and gesture[1].
    window_s: float = 5.0
    # Minimum classifier confidence per frame to count it.
    min_confidence: float = 0.6
    # How many consecutive in-window frames of a gesture are required.
    min_consecutive: int = 4
    # Sliding window length (frames) used for stability checks.
    history_size: int = 16
    # Labels that should *not* abort GOT_GESTURE_1 if seen briefly.
    # 'negative' includes "no hand" and random hand poses; we don't want a
    # blink of negative to reset the state.
    transient_labels: Tuple[str, ...] = ("negative",)


@dataclass
class StateMachineDecision:
    state: State
    activated: bool
    rejected: bool          # True for the single tick when a sequence was rejected
    accepted_gesture: Optional[str] = None   # gesture confirmed this tick (if any)


class GestureSequenceStateMachine:
    """Convert a stream of per-frame predictions into an activation decision.

    Usage::

        sm = GestureSequenceStateMachine()
        for ev in events:
            decision = sm.update(ev)
            if decision.activated:
                start_fatigue_detector()

    For evaluation on a clip, see :func:`run_on_events`.
    """

    def __init__(self, config: Optional[StateMachineConfig] = None) -> None:
        self.cfg = config or StateMachineConfig()
        self.state: State = State.IDLE
        self._history: Deque[FrameEvent] = deque(maxlen=self.cfg.history_size)
        self._first_gesture_time: Optional[float] = None

    # ------------------------------------------------------------------ utils

    def _stable_label(self) -> Optional[str]:
        """Return the label confirmed by the last ``min_consecutive`` frames.

        Confirmation requires *all* of the last K frames to share the same
        non-transient label and to exceed ``min_confidence``.
        """
        K = self.cfg.min_consecutive
        if len(self._history) < K:
            return None
        last = list(self._history)[-K:]
        if any(ev.confidence < self.cfg.min_confidence for ev in last):
            return None
        labels = {ev.label for ev in last}
        if len(labels) != 1:
            return None
        label = labels.pop()
        return label

    # --------------------------------------------------------------- public

    def reset(self) -> None:
        self.state = State.IDLE
        self._history.clear()
        self._first_gesture_time = None

    def update(self, ev: FrameEvent) -> StateMachineDecision:
        """Feed one frame's prediction; advance the state machine; return decision."""
        self._history.append(ev)
        rejected = False
        accepted: Optional[str] = None

        # In ACTIVATED we don't change anything until reset() is called.
        if self.state == State.ACTIVATED:
            return StateMachineDecision(self.state, activated=True, rejected=False)

        stable = self._stable_label()

        if self.state == State.IDLE:
            if stable == self.cfg.sequence[0]:
                self.state = State.GOT_GESTURE_1
                self._first_gesture_time = ev.timestamp_s
                accepted = stable

        elif self.state == State.GOT_GESTURE_1:
            assert self._first_gesture_time is not None
            elapsed = ev.timestamp_s - self._first_gesture_time
            if elapsed > self.cfg.window_s:
                # Timed out -> reject and reset.
                self.state = State.IDLE
                self._first_gesture_time = None
                rejected = True
            elif stable == self.cfg.sequence[1]:
                self.state = State.ACTIVATED
                accepted = stable
            elif stable == self.cfg.sequence[0]:
                # User holds gesture 1 — keep waiting (refresh nothing).
                pass
            elif stable is not None and stable not in self.cfg.transient_labels:
                # Some other confirmed gesture — wrong sequence.
                self.state = State.IDLE
                self._first_gesture_time = None
                rejected = True

        return StateMachineDecision(
            self.state,
            activated=(self.state == State.ACTIVATED),
            rejected=rejected,
            accepted_gesture=accepted,
        )

    # Convenience: run on a list of events and return final decision.
    def run(self, events: List[FrameEvent]) -> StateMachineDecision:
        last: Optional[StateMachineDecision] = None
        for ev in events:
            last = self.update(ev)
            if last.activated:
                break
        if last is None:
            last = StateMachineDecision(self.state, activated=False, rejected=False)
        return last
