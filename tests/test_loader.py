"""Smoke tests that don't require the full dataset to be present.

We only check the helpers in isolation. Heavier integration is exercised by
running ``python scripts/inventory.py``.
"""
from pathlib import Path

from src.data.loader import _infer_person


def test_infer_person_lowercases():
    assert _infer_person("Person1_yawning_03.mp4") == "person1"
    assert _infer_person("person2_openpalm_5.mp4") == "person2"
    assert _infer_person("nogroup.mp4") is None


def test_video_extensions_match_known_set():
    from src import config
    assert ".mp4" in config.VIDEO_EXTENSIONS
    assert ".mov" in config.VIDEO_EXTENSIONS
