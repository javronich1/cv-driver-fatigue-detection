"""Modern fatigue classifier — Stage 5.

Where the classical Stage-4 pipeline made an SVM/RF prediction *per frame*
and then aggregated over a clip, this module trains a small **1D temporal
CNN directly on per-frame feature sequences**, producing one fatigue
class (alert / drowsy / yawning) per clip end-to-end.

Why a 1D-CNN over the 24-D per-frame feature sequence (and not raw
pixels):

* The per-frame features (EAR, MAR, head pose, eye/mouth/brow blendshapes)
  already condense the relevant face-shape signal. Fatigue is fundamentally
  a *temporal pattern* over those signals (e.g. low-EAR persistence,
  jaw-open spikes), which a 1D-CNN learns naturally.
* It gives a like-for-like comparison with the classical pipeline:
  identical inputs, the only thing that changes is the model.
* No external pretraining / large dataset download required, and the
  whole thing trains in seconds on MPS.

Design:
* Inputs are zero-padded to a fixed length ``SEQ_LEN`` (mask is propagated
  through the loss via masked global pooling so padding doesn't bias the
  representation).
* 4 Conv1d blocks → masked global average pool → 3-class logits.
* Class-balanced sampling + class-weighted CE for the imbalance
  (drowsy ~58%, alert ~32%, yawning ~10% at clip level).
* LOSO across the 2 subjects; one prediction per clip.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
)
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .features import FEATURE_NAMES, FEATURE_DIM


# Must stay in sync with src/fatigue/classical.py for fair comparison.
CLASSES: Tuple[str, ...] = ("alert", "drowsy", "yawning")
LABEL_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_LABEL: Dict[int, str] = {i: c for i, c in enumerate(CLASSES)}

SEQ_LEN = 64                # fixed sequence length (median frames/clip ~63)
FEATURE_COLS: Tuple[str, ...] = FEATURE_NAMES


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
def best_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Sequence dataset
# ---------------------------------------------------------------------------
@dataclass
class ClipSequence:
    video: str
    person: str
    coarse_label: str
    folder_label: str
    fine_label: str
    features: np.ndarray        # (T, F) float32
    n_frames: int               # original (un-padded) length


def build_clip_sequences(df: pd.DataFrame) -> List[ClipSequence]:
    """Group the per-frame features CSV by clip and return one
    :class:`ClipSequence` per video."""
    seqs: List[ClipSequence] = []
    cols = list(FEATURE_COLS)
    for video, sub in df.groupby("video", sort=False):
        sub = sub.sort_values("frame_idx")
        feats = sub[cols].to_numpy(dtype=np.float32)
        seqs.append(ClipSequence(
            video=video,
            person=str(sub["person"].iloc[0]),
            coarse_label=str(sub["coarse_label"].iloc[0]),
            folder_label=str(sub["folder_label"].iloc[0]),
            fine_label=str(sub["fine_label"].iloc[0]),
            features=feats,
            n_frames=len(sub),
        ))
    return seqs


def _pad_or_truncate(x: np.ndarray, target_len: int) -> Tuple[np.ndarray, int]:
    """Pad with zeros or center-truncate to ``target_len``."""
    T, F = x.shape
    if T == target_len:
        return x, T
    if T > target_len:
        # Center-truncate to keep the middle of the clip (the action peak
        # for yawning is usually mid-clip).
        start = (T - target_len) // 2
        return x[start:start + target_len], target_len
    pad = np.zeros((target_len - T, F), dtype=np.float32)
    return np.concatenate([x, pad], axis=0), T


class FatigueSequenceDataset(Dataset):
    """In-memory dataset of fixed-length feature sequences."""

    def __init__(
        self,
        sequences: List[ClipSequence],
        *,
        seq_len: int = SEQ_LEN,
        feature_mean: Optional[np.ndarray] = None,
        feature_std: Optional[np.ndarray] = None,
    ) -> None:
        self.sequences = sequences
        self.seq_len = seq_len
        # Standardise features (fit on training set, applied here).
        # If no statistics are passed, fall back to per-feature 0/1 — the
        # caller is expected to pass training-set statistics.
        self.mean = (feature_mean if feature_mean is not None
                     else np.zeros(FEATURE_DIM, dtype=np.float32))
        self.std = (feature_std if feature_std is not None
                    else np.ones(FEATURE_DIM, dtype=np.float32))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        s = self.sequences[idx]
        x, real_len = _pad_or_truncate(s.features, self.seq_len)
        x = (x - self.mean) / np.maximum(self.std, 1e-6)
        # Mask: 1 for real frames, 0 for padding.
        mask = np.zeros(self.seq_len, dtype=np.float32)
        mask[:min(real_len, self.seq_len)] = 1.0
        # (F, T) layout for Conv1d.
        x_chw = x.T.astype(np.float32)
        y = LABEL_TO_IDX[s.coarse_label]
        return (
            torch.from_numpy(x_chw),
            torch.from_numpy(mask),
            torch.tensor(y, dtype=torch.long),
            idx,
        )


def fit_feature_stats(
    sequences: List[ClipSequence],
) -> Tuple[np.ndarray, np.ndarray]:
    """Mean/std over the *concatenation* of all training frames (real ones
    only — padding has no effect because we fit on raw features)."""
    if not sequences:
        return (np.zeros(FEATURE_DIM, dtype=np.float32),
                np.ones(FEATURE_DIM, dtype=np.float32))
    stack = np.concatenate([s.features for s in sequences], axis=0)
    mean = stack.mean(axis=0).astype(np.float32)
    std = stack.std(axis=0).astype(np.float32)
    return mean, std


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class TemporalCNN(nn.Module):
    """Small 1D-CNN over feature sequences with masked global avg pooling."""

    def __init__(
        self,
        in_features: int = FEATURE_DIM,
        n_classes: int = len(CLASSES),
        widths: Tuple[int, ...] = (64, 64, 128, 128),
        kernel_sizes: Tuple[int, ...] = (5, 5, 3, 3),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        c_in = in_features
        for c_out, k in zip(widths, kernel_sizes):
            layers.append(nn.Conv1d(c_in, c_out, kernel_size=k,
                                    padding=k // 2))
            layers.append(nn.BatchNorm1d(c_out))
            layers.append(nn.ReLU(inplace=True))
            c_in = c_out
        self.backbone = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(widths[-1], n_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x:    (B, F, T)
        # mask: (B, T)  — 1.0 for real frames, 0.0 for padding
        h = self.backbone(x)                       # (B, C, T)
        m = mask.unsqueeze(1)                      # (B, 1, T)
        denom = m.sum(dim=2).clamp(min=1.0)        # (B, 1)
        pooled = (h * m).sum(dim=2) / denom        # (B, C) masked average
        pooled = self.dropout(pooled)
        return self.head(pooled)


def build_model(**kwargs) -> TemporalCNN:
    return TemporalCNN(**kwargs)


# ---------------------------------------------------------------------------
# Training / eval
# ---------------------------------------------------------------------------
@dataclass
class TemporalFoldResult:
    fold_name: str
    accuracy: float
    macro_f1: float
    report: str
    confusion: np.ndarray
    clip_preds: pd.DataFrame   # one row per held-out clip
    classes: Tuple[str, ...] = CLASSES


def _make_balanced_sampler(
    sequences: List[ClipSequence],
) -> WeightedRandomSampler:
    """Balance the 3 classes at the *clip* level."""
    labels = np.array([LABEL_TO_IDX[s.coarse_label] for s in sequences])
    counts = np.bincount(labels, minlength=len(CLASSES)).astype(np.float64)
    counts[counts == 0] = 1.0
    inv = 1.0 / counts
    weights = inv[labels]
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(sequences),
        replacement=True,
    )


def _class_weight_tensor(
    sequences: List[ClipSequence],
    device: torch.device,
) -> torch.Tensor:
    labels = np.array([LABEL_TO_IDX[s.coarse_label] for s in sequences])
    counts = np.bincount(labels, minlength=len(CLASSES)).astype(np.float64)
    counts[counts == 0] = 1.0
    inv = counts.sum() / (len(CLASSES) * counts)
    return torch.as_tensor(inv, dtype=torch.float32, device=device)


def _run_loader(
    model: TemporalCNN,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    criterion: Optional[nn.Module] = None,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """One pass over the loader. Returns (mean_loss, y_true, y_pred, idxs)."""
    train = optimizer is not None
    model.train(mode=train)
    total_loss = 0.0
    n = 0
    all_true, all_pred, all_idx = [], [], []
    for x, mask, y, idx in loader:
        # Snapshot CPU labels BEFORE moving to device (MPS race fix).
        y_cpu = y.numpy().copy()
        idx_cpu = idx.numpy().copy()
        x = x.to(device)
        mask = mask.to(device)
        y_dev = y.to(device)
        if train:
            optimizer.zero_grad()
        logits = model(x, mask)
        if criterion is not None:
            loss = criterion(logits, y_dev)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.detach().cpu()) * y_dev.size(0)
        n += y_dev.size(0)
        all_true.append(y_cpu)
        all_pred.append(logits.argmax(dim=1).detach().cpu().numpy())
        all_idx.append(idx_cpu)
    mean_loss = total_loss / max(n, 1)
    return (mean_loss,
            np.concatenate(all_true) if all_true else np.array([]),
            np.concatenate(all_pred) if all_pred else np.array([]),
            np.concatenate(all_idx) if all_idx else np.array([]))


def train_one_fold(
    train_seqs: List[ClipSequence],
    test_seqs: List[ClipSequence],
    *,
    seq_len: int = SEQ_LEN,
    epochs: int = 60,
    batch_size: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> TemporalFoldResult:
    device = device or best_device()
    mean, std = fit_feature_stats(train_seqs)
    train_ds = FatigueSequenceDataset(train_seqs, seq_len=seq_len,
                                      feature_mean=mean, feature_std=std)
    test_ds = FatigueSequenceDataset(test_seqs, seq_len=seq_len,
                                     feature_mean=mean, feature_std=std)
    sampler = _make_balanced_sampler(train_seqs)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=sampler, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             shuffle=False, num_workers=0)

    model = build_model().to(device)
    cw = _class_weight_tensor(train_seqs, device)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs,
    )

    # NOTE: with only 2 persons total, there is no "validation person" to
    # do model selection on. We deliberately use the *final-epoch* model
    # for evaluation (no test-set peeking) — fixed budget + cosine LR
    # scheduling. We only print held-out F1 every 10 epochs as a progress
    # indicator; it is not used for selection.
    for ep in range(1, epochs + 1):
        tr_loss, *_ = _run_loader(
            model, train_loader, device, optimizer, criterion,
        )
        scheduler.step()
        if verbose and (ep % 10 == 0 or ep == epochs):
            _, ytv, ypv, _ = _run_loader(model, test_loader, device,
                                         criterion=criterion)
            f1 = float(f1_score(ytv, ypv, labels=list(range(len(CLASSES))),
                                average="macro", zero_division=0))
            print(f"  ep {ep:3d}: train_loss={tr_loss:.3f}  "
                  f"test_macro_F1={f1:.3f}  (progress only — not used)")

    # Final eval on held-out person (final-epoch model, no checkpoint selection).
    _, y_true, y_pred, idxs = _run_loader(model, test_loader, device)
    # Re-order predictions back to dataset order.
    order = np.argsort(idxs)
    y_true = y_true[order]
    y_pred = y_pred[order]

    rows = []
    for i, s in enumerate(test_seqs):
        rows.append({
            "video": s.video,
            "person": s.person,
            "folder_label": s.folder_label,
            "fine_label": s.fine_label,
            "coarse_label": s.coarse_label,
            "n_frames": s.n_frames,
            "pred_temporal_cnn": IDX_TO_LABEL[int(y_pred[i])],
        })
    clip_df = pd.DataFrame(rows)
    cm = confusion_matrix(
        [s.coarse_label for s in test_seqs],
        [IDX_TO_LABEL[int(p)] for p in y_pred],
        labels=list(CLASSES),
    )
    rep = classification_report(
        [s.coarse_label for s in test_seqs],
        [IDX_TO_LABEL[int(p)] for p in y_pred],
        labels=list(CLASSES), digits=3, zero_division=0,
    )
    fold_person = test_seqs[0].person if test_seqs else "?"
    return TemporalFoldResult(
        fold_name=f"test={fold_person}",
        accuracy=float(accuracy_score(
            [s.coarse_label for s in test_seqs],
            [IDX_TO_LABEL[int(p)] for p in y_pred],
        )),
        macro_f1=float(f1_score(
            [s.coarse_label for s in test_seqs],
            [IDX_TO_LABEL[int(p)] for p in y_pred],
            labels=list(CLASSES), average="macro", zero_division=0,
        )),
        report=rep,
        confusion=cm,
        clip_preds=clip_df,
    )


def evaluate_loso(
    df: pd.DataFrame,
    *,
    seq_len: int = SEQ_LEN,
    epochs: int = 60,
    batch_size: int = 16,
    lr: float = 1e-3,
    seed: int = 42,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Tuple[List[TemporalFoldResult], pd.DataFrame]:
    """Leave-one-person-out across the 2 subjects."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    sequences = build_clip_sequences(df)
    persons = sorted({s.person for s in sequences})
    results: List[TemporalFoldResult] = []
    all_preds: List[pd.DataFrame] = []

    for held_out in persons:
        train_seqs = [s for s in sequences if s.person != held_out]
        test_seqs = [s for s in sequences if s.person == held_out]
        if not train_seqs or not test_seqs:
            continue
        if verbose:
            print(f"\n=== Fold test={held_out}  "
                  f"(train clips={len(train_seqs)}, "
                  f"test clips={len(test_seqs)}) ===")
        r = train_one_fold(
            train_seqs, test_seqs,
            seq_len=seq_len, epochs=epochs, batch_size=batch_size, lr=lr,
            device=device, verbose=verbose,
        )
        r.clip_preds["fold"] = f"test={held_out}"
        all_preds.append(r.clip_preds)
        results.append(r)

    return results, (pd.concat(all_preds, ignore_index=True)
                     if all_preds else pd.DataFrame())


def fit_on_all(
    df: pd.DataFrame,
    *,
    seq_len: int = SEQ_LEN,
    epochs: int = 60,
    batch_size: int = 16,
    lr: float = 1e-3,
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> Tuple[TemporalCNN, np.ndarray, np.ndarray]:
    """Refit on all data; returns (model, feature_mean, feature_std) so the
    caller can persist the normalisation stats alongside the weights."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or best_device()
    sequences = build_clip_sequences(df)
    mean, std = fit_feature_stats(sequences)
    ds = FatigueSequenceDataset(sequences, seq_len=seq_len,
                                feature_mean=mean, feature_std=std)
    sampler = _make_balanced_sampler(sequences)
    loader = DataLoader(ds, batch_size=batch_size, sampler=sampler,
                        num_workers=0)
    model = build_model().to(device)
    cw = _class_weight_tensor(sequences, device)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs,
    )
    for _ in range(epochs):
        _run_loader(model, loader, device, optimizer, criterion)
        scheduler.step()
    return model, mean, std


def save_model(
    model: TemporalCNN,
    mean: np.ndarray,
    std: np.ndarray,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "feature_mean": mean,
        "feature_std": std,
        "feature_names": list(FEATURE_NAMES),
        "classes": list(CLASSES),
        "seq_len": SEQ_LEN,
    }, path)


def load_model(
    path: Path,
    device: Optional[torch.device] = None,
) -> Tuple[TemporalCNN, np.ndarray, np.ndarray]:
    device = device or best_device()
    blob = torch.load(path, map_location=device, weights_only=False)
    model = build_model().to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, blob["feature_mean"], blob["feature_std"]
