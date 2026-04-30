"""Stage 6: end-to-end realtime fatigue-detection demo.

Runs MediaPipe hand + face landmarkers, the trained gesture classifier,
the gesture-sequence state machine, and the fatigue model on a webcam
feed (or a pre-recorded video file). Draws a HUD overlay with the
current system state and per-frame predictions.

Usage:
    # Live webcam (default index 0).
    python scripts/run_realtime.py

    # Webcam + use the modern (CNN) gesture pipeline.
    python scripts/run_realtime.py --gesture-model cnn

    # Use the classical (SVM) fatigue pipeline instead of the temporal CNN.
    python scripts/run_realtime.py --fatigue-model svm

    # Pre-recorded clip + write the annotated result to disk.
    python scripts/run_realtime.py --video data/raw/something.mp4 \
        --output outputs/realtime_demo.mp4
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config                                                  # noqa: E402
from src.gestures.classical import load_model as load_gesture_classical # noqa: E402
from src.gestures.evaluate_sequences import (                           # noqa: E402
    make_classical_predictor, make_cnn_predictor,
)
from src.gestures.state_machine import State, StateMachineConfig        # noqa: E402
from src.fatigue.classical import load_model as load_fatigue_classical  # noqa: E402
from src.system.realtime import (                                       # noqa: E402
    FrameOutcome, RealtimeConfig, RealtimeFatigueSystem, SystemState,
    make_classical_aggregator_predictor, make_temporal_cnn_predictor,
)


# ---------------------------------------------------------------------------
# HUD rendering
# ---------------------------------------------------------------------------
GREEN = (40, 220, 100)
RED = (60, 60, 240)
YELLOW = (40, 220, 240)
WHITE = (245, 245, 245)
BLACK = (10, 10, 10)


def _put(text: str, img: np.ndarray, org: Tuple[int, int],
         color=WHITE, scale=0.6, thickness=1) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                BLACK, thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)


def render_hud(bgr: np.ndarray, outcome: FrameOutcome) -> np.ndarray:
    """Draw the system state, gesture prediction, and fatigue probs."""
    h, w = bgr.shape[:2]
    overlay = bgr.copy()

    # Top banner
    banner_color = {
        SystemState.GESTURE_WAITING: (60, 60, 60),
        SystemState.FATIGUE_MONITOR: (50, 110, 50),
        SystemState.ALERT:           (40, 40, 200),
    }[outcome.system_state]
    cv2.rectangle(overlay, (0, 0), (w, 40), banner_color, -1)
    state_txt = {
        SystemState.GESTURE_WAITING:
            "WAITING — show OPEN PALM, then THUMBS UP",
        SystemState.FATIGUE_MONITOR: "MONITORING FATIGUE",
        SystemState.ALERT:           "DROWSINESS ALERT!",
    }[outcome.system_state]
    _put(state_txt, overlay, (12, 27), color=WHITE, scale=0.7, thickness=2)

    # Gesture line
    if outcome.system_state == SystemState.GESTURE_WAITING:
        sm = outcome.sm_state
        sm_txt = {
            State.IDLE: "step 1/2: OPEN PALM",
            State.GOT_GESTURE_1: "step 2/2: THUMBS UP",
            State.ACTIVATED: "ACTIVATED",
        }[sm]
        _put(sm_txt, overlay, (12, 70), color=YELLOW, scale=0.7, thickness=2)
        if outcome.gesture.label is not None:
            _put(
                f"hand: {outcome.gesture.label}  "
                f"({outcome.gesture.confidence:.2f})",
                overlay, (12, 100), color=WHITE, scale=0.55,
            )
        else:
            _put("hand: —", overlay, (12, 100), color=WHITE, scale=0.55)

    # Fatigue panel (bottom-left).
    if outcome.system_state in (SystemState.FATIGUE_MONITOR, SystemState.ALERT):
        probs = outcome.fatigue.fatigue_probs
        if probs is not None:
            y0 = h - 110
            cv2.rectangle(overlay, (8, y0 - 22), (260, h - 8),
                          (30, 30, 30), -1)
            _put("fatigue:", overlay, (16, y0 - 4), color=WHITE, scale=0.55)
            for i, (cls, p) in enumerate(probs.items()):
                bar_x0 = 16 + 90
                bar_y = y0 + 18 + i * 22
                cv2.rectangle(overlay, (bar_x0, bar_y - 12),
                              (bar_x0 + int(140 * p), bar_y),
                              (90, 200, 100) if cls == "alert"
                              else (60, 130, 240) if cls == "drowsy"
                              else (50, 200, 240), -1)
                _put(f"{cls}", overlay, (16, bar_y - 1),
                     color=WHITE, scale=0.5)
                _put(f"{p:.2f}", overlay, (bar_x0 + 145, bar_y - 1),
                     color=WHITE, scale=0.5)
        else:
            _put("fatigue: warming up…",
                 overlay, (12, h - 16), color=WHITE, scale=0.55)

        if outcome.alert_on:
            _put(f"!! ALERT ({outcome.alert_age_s:.1f}s) !!",
                 overlay, (w - 260, 70),
                 color=RED, scale=0.8, thickness=2)

    # Timestamp (top-right).
    _put(f"t={outcome.timestamp_s:5.1f}s", overlay, (w - 110, 27),
         color=WHITE, scale=0.55)

    return overlay


# ---------------------------------------------------------------------------
# Source helpers
# ---------------------------------------------------------------------------
def open_source(video_path: Optional[str], cam_index: int) -> cv2.VideoCapture:
    if video_path:
        cap = cv2.VideoCapture(video_path)
    else:
        cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open {'video' if video_path else 'camera'} "
            f"({video_path or cam_index})"
        )
    return cap


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def build_gesture_predictor(kind: str):
    if kind == "svm":
        path = config.MODELS_DIR / "gesture_svm.joblib"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run scripts/train_gesture_classical.py."
            )
        return make_classical_predictor(load_gesture_classical(path))
    if kind == "cnn":
        from src.gestures.cnn import (
            CLASSES, DEFAULT_INPUT_SIZE, load_model as load_gesture_cnn,
        )
        path = config.MODELS_DIR / "gesture_cnn.pt"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run scripts/train_gesture_cnn.py."
            )
        model = load_gesture_cnn(path)
        return make_cnn_predictor(model, classes=CLASSES,
                                  input_size=DEFAULT_INPUT_SIZE)
    raise ValueError(f"Unknown --gesture-model: {kind!r}")


def build_fatigue_predictor(kind: str):
    if kind == "temporal_cnn":
        path = config.MODELS_DIR / "fatigue_temporal_cnn.pt"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. "
                "Run scripts/train_fatigue_temporal_cnn.py."
            )
        return make_temporal_cnn_predictor(path)
    if kind in ("svm", "rf"):
        path = config.MODELS_DIR / f"fatigue_{kind}.joblib"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run scripts/train_fatigue_classical.py."
            )
        return make_classical_aggregator_predictor(
            load_fatigue_classical(path),
        )
    raise ValueError(f"Unknown --fatigue-model: {kind!r}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=None,
                        help="Path to a video file. If omitted, use webcam.")
    parser.add_argument("--cam", type=int, default=0,
                        help="Webcam device index (default 0).")
    parser.add_argument("--output", default=None,
                        help="Optional: write annotated MP4 to this path.")
    parser.add_argument("--gesture-model", choices=("svm", "cnn"),
                        default="svm")
    parser.add_argument("--fatigue-model",
                        choices=("temporal_cnn", "svm", "rf"),
                        default="temporal_cnn")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Stop after N processed frames (0 = no cap).")
    parser.add_argument("--no-display", action="store_true",
                        help="Skip the OpenCV window (still writes --output).")
    parser.add_argument("--resize-width", type=int, default=720,
                        help="Resize frames to this width (default 720).")
    args = parser.parse_args(argv)

    config.ensure_dirs()
    print(f"Gesture model : {args.gesture_model}")
    print(f"Fatigue model : {args.fatigue_model}")

    gesture_pred = build_gesture_predictor(args.gesture_model)
    fatigue_pred = build_fatigue_predictor(args.fatigue_model)

    cap = open_source(args.video, args.cam)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"Source FPS    : {src_fps:.1f}    "
          f"input={'video' if args.video else f'cam {args.cam}'}")

    rt_cfg = RealtimeConfig(
        target_fps=src_fps if args.video else 30.0,
        sm_config=StateMachineConfig(),
    )

    writer: Optional[cv2.VideoWriter] = None
    t0 = time.time()
    n_processed = 0
    processed_fps_window = []

    with RealtimeFatigueSystem(
        gesture_predictor=gesture_pred,
        fatigue_predictor=fatigue_pred,
        config=rt_cfg,
    ) as system:
        try:
            while True:
                ok, bgr = cap.read()
                if not ok or bgr is None:
                    break
                # Resize to a sensible width for speed + display.
                if args.resize_width and bgr.shape[1] != args.resize_width:
                    h0, w0 = bgr.shape[:2]
                    new_w = args.resize_width
                    new_h = int(h0 * (new_w / w0))
                    bgr = cv2.resize(bgr, (new_w, new_h))

                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                ts = (n_processed / src_fps) if args.video else (time.time() - t0)
                outcome = system.step(rgb, ts)
                hud = render_hud(bgr, outcome)

                # Live FPS in the corner.
                tick = time.time()
                processed_fps_window.append(tick)
                processed_fps_window = processed_fps_window[-30:]
                if len(processed_fps_window) >= 2:
                    fps = (len(processed_fps_window) - 1) / (
                        processed_fps_window[-1] - processed_fps_window[0] + 1e-9
                    )
                    cv2.putText(hud, f"{fps:4.1f} FPS",
                                (hud.shape[1] - 110, hud.shape[0] - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, BLACK, 3,
                                cv2.LINE_AA)
                    cv2.putText(hud, f"{fps:4.1f} FPS",
                                (hud.shape[1] - 110, hud.shape[0] - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1,
                                cv2.LINE_AA)

                # Optional: write annotated video.
                if args.output:
                    if writer is None:
                        out_path = Path(args.output)
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(
                            str(out_path), fourcc, src_fps,
                            (hud.shape[1], hud.shape[0]),
                        )
                    writer.write(hud)

                if not args.no_display:
                    cv2.imshow("Driver fatigue — gesture-activated", hud)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break

                n_processed += 1
                if args.max_frames and n_processed >= args.max_frames:
                    break
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            cv2.destroyAllWindows()

    print(f"\nProcessed {n_processed} frames "
          f"in {time.time() - t0:.1f}s.")
    if args.output:
        print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
