"""Unit tests for the gesture-sequence state machine."""
from src.gestures.state_machine import (
    FrameEvent, GestureSequenceStateMachine, State, StateMachineConfig,
)


def _evs(triplets):
    """Helper: build events from list of (timestamp_s, label, conf)."""
    return [FrameEvent(t, l, c) for (t, l, c) in triplets]


def test_correct_sequence_activates():
    cfg = StateMachineConfig(min_consecutive=3, window_s=5.0)
    sm = GestureSequenceStateMachine(cfg)
    events = _evs([
        (0.0, "open_palm", 0.9),
        (0.1, "open_palm", 0.9),
        (0.2, "open_palm", 0.9),     # confirms gesture 1
        (1.0, "negative", 0.9),      # transient, ignored
        (1.5, "thumbs_up", 0.9),
        (1.6, "thumbs_up", 0.9),
        (1.7, "thumbs_up", 0.9),     # confirms gesture 2 -> ACTIVATED
    ])
    decision = sm.run(events)
    assert decision.activated
    assert decision.state == State.ACTIVATED


def test_wrong_sequence_rejected():
    """thumbs_up → open_palm should NOT activate.

    The state machine sees thumbs_up first (ignored in IDLE because it's not
    the expected first gesture) and then open_palm (advances to
    GOT_GESTURE_1). The clip ends without thumbs_up after — so the system
    never activates.
    """
    cfg = StateMachineConfig(min_consecutive=3, window_s=5.0)
    sm = GestureSequenceStateMachine(cfg)
    events = _evs([
        (0.0, "thumbs_up", 0.9),
        (0.1, "thumbs_up", 0.9),
        (0.2, "thumbs_up", 0.9),
        (1.0, "open_palm", 0.9),
        (1.1, "open_palm", 0.9),
        (1.2, "open_palm", 0.9),
    ])
    decision = sm.run(events)
    assert not decision.activated


def test_timeout_rejects():
    cfg = StateMachineConfig(min_consecutive=2, window_s=2.0)
    sm = GestureSequenceStateMachine(cfg)
    events = _evs([
        (0.0, "open_palm", 0.9),
        (0.1, "open_palm", 0.9),     # gesture 1 confirmed @ ~0.1
        (3.0, "thumbs_up", 0.9),     # too late
        (3.1, "thumbs_up", 0.9),
    ])
    decision = sm.run(events)
    assert not decision.activated


def test_incomplete_does_not_activate():
    cfg = StateMachineConfig(min_consecutive=2, window_s=5.0)
    sm = GestureSequenceStateMachine(cfg)
    events = _evs([
        (0.0, "open_palm", 0.9),
        (0.1, "open_palm", 0.9),
        (0.5, "negative", 0.9),
        (0.6, "negative", 0.9),
        (1.0, "negative", 0.9),
    ])
    decision = sm.run(events)
    assert not decision.activated


def test_low_confidence_does_not_confirm():
    cfg = StateMachineConfig(min_consecutive=2, min_confidence=0.7)
    sm = GestureSequenceStateMachine(cfg)
    events = _evs([
        (0.0, "open_palm", 0.5),
        (0.1, "open_palm", 0.5),
    ])
    decision = sm.run(events)
    assert decision.state == State.IDLE
