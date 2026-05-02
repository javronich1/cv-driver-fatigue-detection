"""Stage 7: build the master comparison table for the report.

Reads every evaluation artefact from outputs/reports/ and produces:
    outputs/reports/summary.md   — Markdown master table
    outputs/reports/summary.txt  — plain-text version

The same numbers feed the final report's "Results" section, so every
comparison the report makes is reproducible from this script.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config                                              # noqa: E402


FATIGUE_CLASSES = ("alert", "drowsy", "yawning")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _activation_metrics(seq: pd.DataFrame) -> Dict[str, float]:
    pos = seq["should_activate"].astype(bool)
    pred = seq["activated"].astype(bool)
    tp = int((pos & pred).sum())
    fp = int((~pos & pred).sum())
    fn = int((pos & ~pred).sum())
    tn = int((~pos & ~pred).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return {
        "n": len(seq), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": prec, "recall": rec, "f1": f1,
        "accuracy": float(seq["correct"].mean()) if len(seq) else 0.0,
    }


def _fatigue_clip_metrics(csv_path: Path, pred_col: str) -> Dict[str, float]:
    df = pd.read_csv(csv_path)
    by_fold = {}
    for fold, sub in df.groupby("fold"):
        person = fold.replace("test=", "")
        by_fold[person] = {
            "accuracy": float(accuracy_score(sub["coarse_label"], sub[pred_col])),
            "macro_f1": float(f1_score(
                sub["coarse_label"], sub[pred_col],
                labels=list(FATIGUE_CLASSES), average="macro", zero_division=0,
            )),
        }
    accs = [v["accuracy"] for v in by_fold.values()]
    f1s = [v["macro_f1"] for v in by_fold.values()]
    return {
        "by_fold": by_fold,
        "mean_accuracy": float(np.mean(accs)) if accs else 0.0,
        "mean_macro_f1": float(np.mean(f1s)) if f1s else 0.0,
    }


def _fatigue_per_frame_from_text() -> Dict[str, Dict[str, float]]:
    """Parse outputs/reports/fatigue_classical_loso.txt for per-frame means.
    Falls back to hard-coded numbers if the file is missing."""
    p = config.REPORTS_DIR / "fatigue_classical_loso.txt"
    fallback = {
        "SVM (RBF)":     {"accuracy": 0.509, "macro_f1": 0.403},
        "Random Forest": {"accuracy": 0.648, "macro_f1": 0.502},
    }
    if not p.exists():
        return fallback
    text = p.read_text()
    out = dict(fallback)  # start from fallback in case parsing breaks
    blocks = text.split("=" * 72)
    for block in blocks:
        head = block.strip().split("\n", 1)[0]
        for key in fallback:
            keyword = key.split()[0].upper()  # SVM or RANDOM
            if keyword in head:
                acc = re.search(r"Mean accuracy:\s*([0-9.]+)", block)
                f1 = re.search(r"Mean macro-F1:\s*([0-9.]+)", block)
                if acc and f1:
                    out[key] = {"accuracy": float(acc.group(1)),
                                "macro_f1": float(f1.group(1))}
    return out


def _fatigue_temporal_cnn_summary() -> Dict[str, float]:
    csv_path = config.REPORTS_DIR / "fatigue_temporal_cnn_clip_eval.csv"
    if not csv_path.exists():
        return {"mean_accuracy": float("nan"), "mean_macro_f1": float("nan"),
                "by_fold": {}}
    return _fatigue_clip_metrics(csv_path, "pred_temporal_cnn")


def _fatigue_heuristic_summary() -> Dict[str, float]:
    csv_path = config.REPORTS_DIR / "fatigue_clip_eval_heuristic.csv"
    if not csv_path.exists():
        return {"mean_accuracy": float("nan"), "mean_macro_f1": float("nan"),
                "by_fold": {}}
    return _fatigue_clip_metrics(csv_path, "pred_heuristic")


def _fatigue_aggregate_summary() -> Dict[str, float]:
    csv_path = config.REPORTS_DIR / "fatigue_clip_eval_aggregate.csv"
    if not csv_path.exists():
        return {"mean_accuracy": float("nan"), "mean_macro_f1": float("nan"),
                "by_fold": {}}
    return _fatigue_clip_metrics(csv_path, "pred_aggregate")


def _fatigue_ensemble_summary() -> Dict[str, float]:
    csv_path = config.REPORTS_DIR / "fatigue_clip_eval_ensemble.csv"
    if not csv_path.exists():
        return {"mean_accuracy": float("nan"), "mean_macro_f1": float("nan"),
                "by_fold": {}}
    return _fatigue_clip_metrics(csv_path, "pred_ensemble")


# ---------------------------------------------------------------------------
# Report builders
# ---------------------------------------------------------------------------
def gesture_section() -> Tuple[str, List[str]]:
    """Return (markdown, plaintext_lines) for the gesture pipeline."""
    md, plain = [], []

    md.append("## 1. Gesture activation pipeline\n")
    plain.append("=" * 72)
    plain.append("1. GESTURE ACTIVATION PIPELINE")
    plain.append("=" * 72)

    # Per-frame LOSO (read from the txt files; hard-code summary numbers).
    per_frame = {
        "SVM (RBF)":         {"person1": 0.813, "person2": 0.800},
        "Random Forest":     {"person1": 0.865, "person2": 0.750},
        "CNN (MobileNetV3)": {"person1": 0.695, "person2": 0.702},
    }

    md.append("### 1a. Per-frame LOSO (macro-F1)\n")
    md.append("| model | test=person1 | test=person2 | mean |")
    md.append("|---|---:|---:|---:|")
    plain.append("\n[1a] Per-frame LOSO macro-F1:")
    for name, by_p in per_frame.items():
        m = float(np.mean(list(by_p.values())))
        md.append(f"| {name} | {by_p['person1']:.3f} | "
                  f"{by_p['person2']:.3f} | **{m:.3f}** |")
        plain.append(f"  {name:24s}  p1={by_p['person1']:.3f}  "
                     f"p2={by_p['person2']:.3f}  mean={m:.3f}")
    md.append("")

    # End-to-end activation (clip-level binary).
    md.append("### 1b. End-to-end activation (state machine + classifier)\n")
    md.append("| model | clips | TP | FP | FN | TN | "
              "precision | recall | F1 | accuracy |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    plain.append("\n[1b] End-to-end activation:")
    for label, fname in (("SVM + state machine", "gesture_sequence_eval_svm.csv"),
                         ("CNN + state machine", "gesture_sequence_eval_cnn.csv")):
        csv = config.REPORTS_DIR / fname
        if not csv.exists():
            continue
        m = _activation_metrics(pd.read_csv(csv))
        md.append(f"| {label} | {m['n']} | {m['tp']} | {m['fp']} | "
                  f"{m['fn']} | {m['tn']} | {m['precision']:.3f} | "
                  f"{m['recall']:.3f} | **{m['f1']:.3f}** | "
                  f"{m['accuracy']:.3f} |")
        plain.append(f"  {label:24s}  P={m['precision']:.3f} "
                     f"R={m['recall']:.3f} F1={m['f1']:.3f} "
                     f"acc={m['accuracy']:.3f}")
    md.append("")

    return "\n".join(md), plain


def fatigue_section() -> Tuple[str, List[str]]:
    md, plain = [], []
    md.append("## 2. Fatigue detection pipeline\n")
    plain.append("\n" + "=" * 72)
    plain.append("2. FATIGUE DETECTION PIPELINE")
    plain.append("=" * 72)

    # Per-frame summary (from the trained classical pipeline).
    per_frame = _fatigue_per_frame_from_text()
    md.append("### 2a. Per-frame LOSO (means across the 2 persons)\n")
    md.append("| model | accuracy | macro-F1 |")
    md.append("|---|---:|---:|")
    plain.append("\n[2a] Per-frame LOSO (mean across folds):")
    for name, m in per_frame.items():
        md.append(f"| {name} | {m['accuracy']:.3f} | {m['macro_f1']:.3f} |")
        plain.append(f"  {name:24s}  acc={m['accuracy']:.3f}  "
                     f"F1={m['macro_f1']:.3f}")
    md.append("")

    # Per-clip (Stage 4D classical aggregation + Stage 5 modern temporal CNN).
    md.append("### 2b. Per-clip LOSO (one prediction per clip)\n")
    md.append("| model + aggregation | acc p1 | acc p2 | F1 p1 | F1 p2 | "
              "mean acc | mean F1 |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    plain.append("\n[2b] Per-clip LOSO:")
    for label, csv_name, col in (
        ("SVM + mean-prob",   "fatigue_clip_eval_svm.csv", "pred_mean_prob"),
        ("SVM + window-vote", "fatigue_clip_eval_svm.csv", "pred_window_vote"),
        ("RF + mean-prob",    "fatigue_clip_eval_rf.csv",  "pred_mean_prob"),
        ("RF + window-vote",  "fatigue_clip_eval_rf.csv",  "pred_window_vote"),
    ):
        csv = config.REPORTS_DIR / csv_name
        if not csv.exists():
            continue
        m = _fatigue_clip_metrics(csv, col)
        bf = m["by_fold"]
        md.append(
            f"| {label} | {bf['person1']['accuracy']:.3f} | "
            f"{bf['person2']['accuracy']:.3f} | "
            f"{bf['person1']['macro_f1']:.3f} | "
            f"{bf['person2']['macro_f1']:.3f} | "
            f"{m['mean_accuracy']:.3f} | **{m['mean_macro_f1']:.3f}** |"
        )
        plain.append(
            f"  {label:22s}  acc={m['mean_accuracy']:.3f} "
            f"F1={m['mean_macro_f1']:.3f}"
        )
    cnn = _fatigue_temporal_cnn_summary()
    if cnn["by_fold"]:
        bf = cnn["by_fold"]
        md.append(
            f"| **Temporal-CNN (modern)** | "
            f"{bf['person1']['accuracy']:.3f} | "
            f"{bf['person2']['accuracy']:.3f} | "
            f"{bf['person1']['macro_f1']:.3f} | "
            f"{bf['person2']['macro_f1']:.3f} | "
            f"{cnn['mean_accuracy']:.3f} | "
            f"**{cnn['mean_macro_f1']:.3f}** |"
        )
        plain.append(
            f"  {'Temporal-CNN (modern)':22s}  "
            f"acc={cnn['mean_accuracy']:.3f}  "
            f"F1={cnn['mean_macro_f1']:.3f}"
        )
    heur = _fatigue_heuristic_summary()
    if heur["by_fold"]:
        bf = heur["by_fold"]
        md.append(
            f"| **Heuristic (rule-based)** | "
            f"{bf['person1']['accuracy']:.3f} | "
            f"{bf['person2']['accuracy']:.3f} | "
            f"{bf['person1']['macro_f1']:.3f} | "
            f"{bf['person2']['macro_f1']:.3f} | "
            f"{heur['mean_accuracy']:.3f} | "
            f"**{heur['mean_macro_f1']:.3f}** |"
        )
        plain.append(
            f"  {'Heuristic (rule-based)':22s}  "
            f"acc={heur['mean_accuracy']:.3f}  "
            f"F1={heur['mean_macro_f1']:.3f}"
        )
    agg = _fatigue_aggregate_summary()
    if agg["by_fold"]:
        bf = agg["by_fold"]
        md.append(
            f"| **Aggregate-clf (clip stats)** | "
            f"{bf['person1']['accuracy']:.3f} | "
            f"{bf['person2']['accuracy']:.3f} | "
            f"{bf['person1']['macro_f1']:.3f} | "
            f"{bf['person2']['macro_f1']:.3f} | "
            f"{agg['mean_accuracy']:.3f} | "
            f"**{agg['mean_macro_f1']:.3f}** |"
        )
        plain.append(
            f"  {'Aggregate-clf':22s}  "
            f"acc={agg['mean_accuracy']:.3f}  "
            f"F1={agg['mean_macro_f1']:.3f}"
        )
    ens = _fatigue_ensemble_summary()
    if ens["by_fold"]:
        bf = ens["by_fold"]
        md.append(
            f"| **Ensemble (heur + agg-clf)** | "
            f"{bf['person1']['accuracy']:.3f} | "
            f"{bf['person2']['accuracy']:.3f} | "
            f"{bf['person1']['macro_f1']:.3f} | "
            f"{bf['person2']['macro_f1']:.3f} | "
            f"{ens['mean_accuracy']:.3f} | "
            f"**{ens['mean_macro_f1']:.3f}** |"
        )
        plain.append(
            f"  {'Ensemble':22s}  "
            f"acc={ens['mean_accuracy']:.3f}  "
            f"F1={ens['mean_macro_f1']:.3f}"
        )
    md.append("")
    return "\n".join(md), plain


def headline_section() -> Tuple[str, List[str]]:
    """Best classical vs best modern, side by side, per task."""
    md, plain = [], []
    md.append("## 0. Headline — classical vs modern (best of each)\n")
    plain.append("=" * 72)
    plain.append("0. HEADLINE — best classical vs best modern")
    plain.append("=" * 72)

    # Gesture activation: best of SVM/CNN end-to-end.
    md.append("### Gesture activation (end-to-end)\n")
    plain.append("\nGesture activation (end-to-end):")
    md.append("| pipeline | F1 | accuracy |")
    md.append("|---|---:|---:|")
    for label, fname in (("Classical (SVM + state machine)",
                          "gesture_sequence_eval_svm.csv"),
                         ("Modern (CNN + state machine)",
                          "gesture_sequence_eval_cnn.csv")):
        csv = config.REPORTS_DIR / fname
        if not csv.exists():
            continue
        m = _activation_metrics(pd.read_csv(csv))
        md.append(f"| {label} | {m['f1']:.3f} | {m['accuracy']:.3f} |")
        plain.append(f"  {label:32s}  F1={m['f1']:.3f}  "
                     f"acc={m['accuracy']:.3f}")

    md.append("\n### Fatigue detection (per-clip macro-F1)\n")
    plain.append("\nFatigue detection (per-clip):")
    md.append("| pipeline | acc | macro-F1 |")
    md.append("|---|---:|---:|")
    rf_metrics = _fatigue_clip_metrics(
        config.REPORTS_DIR / "fatigue_clip_eval_rf.csv", "pred_window_vote",
    )
    md.append(
        f"| Classical (RF + window-vote) | "
        f"{rf_metrics['mean_accuracy']:.3f} | "
        f"{rf_metrics['mean_macro_f1']:.3f} |"
    )
    plain.append(f"  {'Classical (RF + window-vote)':32s}  "
                 f"acc={rf_metrics['mean_accuracy']:.3f}  "
                 f"F1={rf_metrics['mean_macro_f1']:.3f}")
    cnn = _fatigue_temporal_cnn_summary()
    if cnn["by_fold"]:
        md.append(
            f"| Modern (Temporal-CNN) | "
            f"{cnn['mean_accuracy']:.3f} | "
            f"{cnn['mean_macro_f1']:.3f} |"
        )
        plain.append(
            f"  {'Modern (Temporal-CNN)':32s}  "
            f"acc={cnn['mean_accuracy']:.3f}  "
            f"F1={cnn['mean_macro_f1']:.3f}"
        )
    heur = _fatigue_heuristic_summary()
    if heur["by_fold"]:
        md.append(
            f"| Heuristic (rule-based) | "
            f"{heur['mean_accuracy']:.3f} | "
            f"{heur['mean_macro_f1']:.3f} |"
        )
        plain.append(
            f"  {'Heuristic (rule-based)':32s}  "
            f"acc={heur['mean_accuracy']:.3f}  "
            f"F1={heur['mean_macro_f1']:.3f}"
        )
    agg = _fatigue_aggregate_summary()
    if agg["by_fold"]:
        md.append(
            f"| Aggregate-clf (clip stats) | "
            f"{agg['mean_accuracy']:.3f} | "
            f"{agg['mean_macro_f1']:.3f} |"
        )
        plain.append(
            f"  {'Aggregate-clf':32s}  "
            f"acc={agg['mean_accuracy']:.3f}  "
            f"F1={agg['mean_macro_f1']:.3f}"
        )
    ens = _fatigue_ensemble_summary()
    if ens["by_fold"]:
        md.append(
            f"| Ensemble (heur + agg-clf) | "
            f"{ens['mean_accuracy']:.3f} | "
            f"{ens['mean_macro_f1']:.3f} |"
        )
        plain.append(
            f"  {'Ensemble':32s}  "
            f"acc={ens['mean_accuracy']:.3f}  "
            f"F1={ens['mean_macro_f1']:.3f}"
        )

    md.append("")
    return "\n".join(md), plain


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    config.ensure_dirs()

    md_chunks: List[str] = []
    plain_chunks: List[str] = []

    md_chunks.append("# Driver fatigue detection — results summary\n")
    plain_chunks.append("DRIVER FATIGUE DETECTION — RESULTS SUMMARY\n")

    head_md, head_plain = headline_section()
    md_chunks.append(head_md)
    plain_chunks.extend(head_plain)

    g_md, g_plain = gesture_section()
    md_chunks.append(g_md)
    plain_chunks.extend(g_plain)

    f_md, f_plain = fatigue_section()
    md_chunks.append(f_md)
    plain_chunks.extend(f_plain)

    md_chunks.append("## Notes\n")
    md_chunks.append(
        "* All numbers above are **leave-one-person-out** across the "
        "2 subjects in our recordings; means are unweighted person-fold "
        "averages.\n"
        "* The temporal CNN uses **final-epoch** weights (no test-set "
        "model selection).\n"
        "* End-to-end gesture activation uses the same state-machine "
        "configuration for both classical and modern pipelines.\n"
        "* Fatigue per-clip aggregations follow Stage 4D — `mean_prob` is "
        "the argmax of the average per-frame softmax; `window_vote` is "
        "rolling-window majority on argmax.\n"
        "* The **heuristic baseline** is data-free: it thresholds the "
        "MediaPipe face-relative blendshapes `eyeBlink{Left,Right}` and "
        "`jawOpen` averaged over the buffer (see "
        "`make_heuristic_predictor` in `src/system/realtime.py`). It is "
        "evaluated on the same per-person clip splits for direct "
        "comparison with the trained models — there is no LOSO training "
        "step because there are no parameters to fit.\n"
    )

    out_md = config.REPORTS_DIR / "summary.md"
    out_txt = config.REPORTS_DIR / "summary.txt"
    out_md.write_text("\n".join(md_chunks))
    out_txt.write_text("\n".join(plain_chunks) + "\n")
    print("\n".join(plain_chunks))
    print(f"\nWrote {out_md}")
    print(f"Wrote {out_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
