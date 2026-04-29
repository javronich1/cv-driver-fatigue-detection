"""Discover and load videos from the local dataset folder.

The dataset is laid out as::

    dataset/
        gestures/<class folder with spaces>/personN_<...>.mp4
        fatigue/<class folder with spaces>/personN_<...>.mp4

This module walks the tree and yields :class:`VideoRecord` objects with the
relevant metadata pulled out of the path (class folder, person id).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .. import config


PERSON_RE = re.compile(r"(person\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class VideoRecord:
    """One video on disk plus the labels we infer from its path / filename."""
    path: Path
    dataset: str          # "gestures" | "fatigue"
    folder_label: str     # original folder name (with spaces)
    canonical_label: str  # snake_case label
    person: Optional[str] # "person1" | "person2" | None if not parseable
    filename: str

    def to_dict(self) -> Dict[str, str]:
        d = asdict(self)
        d["path"] = str(self.path)
        return d


def _infer_person(filename: str) -> Optional[str]:
    m = PERSON_RE.search(filename)
    return m.group(1).lower() if m else None


def _list_videos_in(folder: Path) -> List[Path]:
    if not folder.is_dir():
        return []
    files: List[Path] = []
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in config.VIDEO_EXTENSIONS:
            files.append(p)
    return sorted(files)


def discover_dataset_split(
    root: Path,
    label_map: Dict[str, str],
    dataset_name: str,
) -> List[VideoRecord]:
    """Walk ``root`` and return :class:`VideoRecord` objects for each video.

    Folders that aren't in ``label_map`` are skipped (e.g. dotfiles).
    """
    records: List[VideoRecord] = []
    if not root.is_dir():
        return records
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        folder_name = folder.name
        canonical = label_map.get(folder_name)
        if canonical is None:
            # Unknown folder — keep folder name as label, normalised.
            canonical = folder_name.replace(" ", "_").lower()
        for video in _list_videos_in(folder):
            records.append(
                VideoRecord(
                    path=video,
                    dataset=dataset_name,
                    folder_label=folder_name,
                    canonical_label=canonical,
                    person=_infer_person(video.stem),
                    filename=video.name,
                )
            )
    return records


def discover_gestures() -> List[VideoRecord]:
    return discover_dataset_split(
        config.GESTURES_DIR, config.GESTURE_FOLDERS, "gestures"
    )


def discover_fatigue() -> List[VideoRecord]:
    return discover_dataset_split(
        config.FATIGUE_DIR, config.FATIGUE_FOLDERS, "fatigue"
    )


def discover_all() -> List[VideoRecord]:
    return discover_gestures() + discover_fatigue()


def group_by(records: Iterable[VideoRecord], key: str) -> Dict[str, List[VideoRecord]]:
    out: Dict[str, List[VideoRecord]] = {}
    for r in records:
        k = getattr(r, key)
        out.setdefault(str(k), []).append(r)
    return out
