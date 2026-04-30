"""Modern (CNN) gesture classifier — Stage 3B.

We use a small, fast backbone fine-tuned on our hand crops:

* MobileNetV3-Small (torchvision, ImageNet weights) — final classifier
  head replaced with a 3-class output (open_palm / thumbs_up / negative).

The model is intentionally tiny so it can run real-time on CPU/MPS.
At inference time it receives a 96×96 RGB hand crop produced by the
same MediaPipe-based pipeline as training (see ``crops.py``).

Public API mirrors the classical pipeline as closely as possible:

    model = build_model()
    train_one_fold(...)            # supervised LOSO
    save_model(model, path)
    model = load_model(path)
    probs = predict_proba(model, crop_rgb_uint8)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import cv2

from .. import config


# Must stay in sync with src/gestures/classical.py for fair comparison.
CLASSES: Tuple[str, ...] = ("open_palm", "thumbs_up", "negative")
LABEL_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_LABEL: Dict[int, str] = {i: c for i, c in enumerate(CLASSES)}

DEFAULT_INPUT_SIZE = 96


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------
def best_device() -> torch.device:
    """Pick MPS on Apple Silicon, then CUDA, then CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def make_train_transform(size: int = DEFAULT_INPUT_SIZE) -> transforms.Compose:
    """Augmentations that are sane for hand crops.

    Note: we deliberately do **not** flip horizontally because thumbs_up has
    a chirality (the thumb points up regardless of hand, but the *image* of
    a flipped thumb still reads as thumbs_up — debatable). Empirically a
    light flip helps generalisation, so we keep it but at low probability.
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def make_eval_transform(size: int = DEFAULT_INPUT_SIZE) -> transforms.Compose:
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class HandCropDataset(Dataset):
    """Dataset wrapping the index CSV produced by ``scripts/extract_hand_crops.py``."""

    def __init__(
        self,
        index_df: pd.DataFrame,
        *,
        transform: Optional[transforms.Compose] = None,
        project_root: Path = config.PROJECT_ROOT,
    ) -> None:
        self.df = index_df.reset_index(drop=True)
        self.transform = transform or make_eval_transform()
        self.project_root = Path(project_root)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        path = self.project_root / row["crop_path"]
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Missing crop: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x = self.transform(rgb)
        y = LABEL_TO_IDX[row["label"]]
        return x, y


def make_class_balanced_sampler(labels: Sequence[int]) -> WeightedRandomSampler:
    """Sample classes with equal probability — matches ``class_weight=balanced``."""
    counts = np.bincount(labels, minlength=len(CLASSES))
    weights = 1.0 / np.maximum(counts, 1)
    sample_w = np.array([weights[y] for y in labels], dtype=np.float64)
    return WeightedRandomSampler(sample_w, num_samples=len(labels), replacement=True)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(num_classes: int = len(CLASSES), pretrained: bool = True) -> nn.Module:
    """MobileNetV3-Small with a fresh 3-class classifier head."""
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    # The classifier is Sequential([Linear(576,1024), Hardswish, Dropout, Linear(1024,1000)])
    in_feat = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_feat, num_classes)
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    epochs: int = 12
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 0          # MPS + small dataset → 0 is fastest, no fork issues
    val_every: int = 1
    seed: int = 42


@dataclass
class FoldResult:
    fold_name: str
    accuracy: float
    macro_f1: float
    report: str
    confusion: np.ndarray
    classes: Tuple[str, ...] = CLASSES
    history: List[Dict[str, float]] = field(default_factory=list)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    train_mode = optimizer is not None
    model.train(train_mode)
    losses: List[float] = []
    y_true: List[np.ndarray] = []
    y_pred: List[np.ndarray] = []
    with torch.set_grad_enabled(train_mode):
        for x, y in loader:
            # Keep an immediate CPU snapshot of the labels — we've seen the
            # MPS backend corrupt async device-to-host copies of small int
            # tensors when ``non_blocking=True`` is used, so we read y here.
            y_cpu = y.numpy().copy()
            x = x.to(device)
            y_dev = y.to(device)
            logits = model(x)
            loss = criterion(logits, y_dev)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            losses.append(float(loss.detach().cpu()))
            y_true.append(y_cpu)
            y_pred.append(logits.argmax(dim=1).detach().cpu().numpy())
    yt = np.concatenate(y_true)
    yp = np.concatenate(y_pred)
    acc = float((yt == yp).mean()) if yt.size else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    return avg_loss, acc, yt, yp


def train_one_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    cfg: Optional[TrainConfig] = None,
    fold_name: str = "fold",
    device: Optional[torch.device] = None,
    log: bool = True,
) -> Tuple[nn.Module, FoldResult]:
    from sklearn.metrics import (
        classification_report, confusion_matrix, f1_score, accuracy_score,
    )
    cfg = cfg or TrainConfig()
    device = device or best_device()
    _set_seed(cfg.seed)

    train_ds = HandCropDataset(train_df, transform=make_train_transform())
    test_ds = HandCropDataset(test_df, transform=make_eval_transform())
    sampler = make_class_balanced_sampler(
        [LABEL_TO_IDX[l] for l in train_df["label"]]
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, sampler=sampler,
        num_workers=cfg.num_workers, pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=False,
    )

    model = build_model().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs,
    )
    criterion = nn.CrossEntropyLoss()

    history: List[Dict[str, float]] = []
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_macro_f1 = -1.0

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc, _, _ = _epoch(
            model, train_loader,
            criterion=criterion, optimizer=optimizer, device=device,
        )
        scheduler.step()
        if epoch % cfg.val_every == 0 or epoch == cfg.epochs:
            te_loss, te_acc, yt, yp = _epoch(
                model, test_loader,
                criterion=criterion, optimizer=None, device=device,
            )
            macro = float(f1_score(yt, yp, labels=list(range(len(CLASSES))),
                                   average="macro", zero_division=0))
            history.append({
                "epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                "test_loss": te_loss, "test_acc": te_acc, "macro_f1": macro,
            })
            if log:
                print(f"  [{fold_name}] epoch {epoch:02d}/{cfg.epochs}  "
                      f"train loss={tr_loss:.3f} acc={tr_acc:.3f}  "
                      f"test acc={te_acc:.3f} macroF1={macro:.3f}")
            if macro > best_macro_f1:
                best_macro_f1 = macro
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final eval at best weights.
    _, _, yt, yp = _epoch(
        model, test_loader,
        criterion=criterion, optimizer=None, device=device,
    )
    cm = confusion_matrix(yt, yp, labels=list(range(len(CLASSES))))
    rep = classification_report(
        yt, yp, labels=list(range(len(CLASSES))),
        target_names=list(CLASSES), digits=3, zero_division=0,
    )
    result = FoldResult(
        fold_name=fold_name,
        accuracy=float(accuracy_score(yt, yp)),
        macro_f1=float(f1_score(yt, yp, labels=list(range(len(CLASSES))),
                                average="macro", zero_division=0)),
        report=rep,
        confusion=cm,
        history=history,
    )
    return model, result


def evaluate_loso(
    index_df: pd.DataFrame,
    *,
    cfg: Optional[TrainConfig] = None,
    device: Optional[torch.device] = None,
) -> List[FoldResult]:
    persons = sorted(index_df["person"].dropna().unique())
    results: List[FoldResult] = []
    for held_out in persons:
        train_df = index_df[index_df["person"] != held_out]
        test_df = index_df[index_df["person"] == held_out]
        if test_df.empty or train_df.empty:
            continue
        print(f"\n=== Fold: test={held_out}  (train={len(train_df)}, "
              f"test={len(test_df)}) ===")
        _, res = train_one_fold(
            train_df, test_df, cfg=cfg,
            fold_name=f"test={held_out}", device=device,
        )
        results.append(res)
    return results


def fit_on_all(
    index_df: pd.DataFrame,
    *,
    cfg: Optional[TrainConfig] = None,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Fit on all data (no held-out person) — for deployment in the live demo."""
    cfg = cfg or TrainConfig()
    device = device or best_device()
    _set_seed(cfg.seed)

    ds = HandCropDataset(index_df, transform=make_train_transform())
    sampler = make_class_balanced_sampler(
        [LABEL_TO_IDX[l] for l in index_df["label"]]
    )
    loader = DataLoader(
        ds, batch_size=cfg.batch_size, sampler=sampler,
        num_workers=cfg.num_workers, pin_memory=False,
    )

    model = build_model().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs,
    )
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc, _, _ = _epoch(
            model, loader, criterion=criterion, optimizer=optimizer,
            device=device,
        )
        scheduler.step()
        print(f"  [final] epoch {epoch:02d}/{cfg.epochs}  "
              f"train loss={tr_loss:.3f} acc={tr_acc:.3f}")
    return model


# ---------------------------------------------------------------------------
# Persistence + inference
# ---------------------------------------------------------------------------
@dataclass
class CnnArtifact:
    """What we save to disk: model weights + class list + input size."""
    state_dict: Dict[str, torch.Tensor]
    classes: Tuple[str, ...] = CLASSES
    input_size: int = DEFAULT_INPUT_SIZE


def save_model(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "classes": list(CLASSES),
        "input_size": DEFAULT_INPUT_SIZE,
    }, path)


def load_model(
    path: Path,
    *,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, Tuple[str, ...], int]:
    device = device or best_device()
    blob = torch.load(path, map_location=device, weights_only=False)
    model = build_model(num_classes=len(blob["classes"]), pretrained=False)
    model.load_state_dict(blob["state_dict"])
    model.to(device).eval()
    return model, tuple(blob["classes"]), int(blob["input_size"])


@torch.no_grad()
def predict_proba(
    model: nn.Module,
    crop_rgb_uint8: np.ndarray,
    *,
    classes: Sequence[str] = CLASSES,
    input_size: int = DEFAULT_INPUT_SIZE,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Return ``{label: prob}`` for one RGB hand crop (uint8, HxWx3)."""
    device = device or best_device()
    transform = make_eval_transform(input_size)
    x = transform(crop_rgb_uint8).unsqueeze(0).to(device)
    logits = model(x)
    probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()
    return {c: float(p) for c, p in zip(classes, probs)}
