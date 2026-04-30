"""Reusable plotting helpers used by analysis / report scripts."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=10)


def plot_confusion(
    cm: np.ndarray,
    classes: Sequence[str],
    title: str,
    out_path: Path,
    *,
    cmap: str = "Blues",
    normalize: bool = False,
    figsize=(4.5, 4.5),
) -> None:
    """Save a confusion-matrix heatmap to ``out_path``."""
    cm_disp = cm.astype(float)
    if normalize:
        row_sums = cm_disp.sum(axis=1, keepdims=True)
        cm_disp = np.divide(
            cm_disp, row_sums, out=np.zeros_like(cm_disp), where=row_sums > 0,
        )

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_disp, cmap=cmap, vmin=0,
                   vmax=cm_disp.max() if cm_disp.size else 1)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    threshold = cm_disp.max() / 2.0 if cm_disp.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if normalize:
                txt = f"{cm_disp[i, j]:.2f}\n({int(cm[i, j])})"
            else:
                txt = str(int(cm[i, j]))
            ax.text(j, i, txt,
                    ha="center", va="center", fontsize=10,
                    color="white" if cm_disp[i, j] > threshold else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_grouped_bars(
    categories: Sequence[str],
    series: dict,
    title: str,
    ylabel: str,
    out_path: Path,
    *,
    ylim: Optional[tuple] = None,
    figsize=(8, 4.5),
    annotate: bool = True,
) -> None:
    """Grouped bar chart. ``series`` maps series-name → list of values per category."""
    n_series = len(series)
    n_cat = len(categories)
    width = 0.8 / max(n_series, 1)
    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.get_cmap("tab10")
    for i, (name, values) in enumerate(series.items()):
        x = np.arange(n_cat) + (i - (n_series - 1) / 2) * width
        bars = ax.bar(x, values, width=width, label=name, color=cmap(i))
        if annotate:
            for b, v in zip(bars, values):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                        f"{v:.2f}" if isinstance(v, float) else str(v),
                        ha="center", va="bottom", fontsize=8)

    ax.set_xticks(range(n_cat))
    ax.set_xticklabels(categories, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if n_series > 1:
        ax.legend(loc="best")
    style_axes(ax)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_horizontal_bars(
    labels: Sequence[str],
    values: Sequence[float],
    title: str,
    xlabel: str,
    out_path: Path,
    *,
    figsize=(8, 5),
    color: str = "#3b78b8",
    annotate: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(len(labels))
    bars = ax.barh(y, values, color=color)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    if annotate:
        for b, v in zip(bars, values):
            ax.text(b.get_width(), b.get_y() + b.get_height() / 2,
                    f" {v:.2f}" if isinstance(v, float) else f" {v}",
                    va="center", fontsize=9)
    style_axes(ax)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
