# Driver Fatigue Detection with Gesture-Based Activation

Computer vision system that monitors driver fatigue, but only after the driver
performs a predefined gesture sequence to activate it. Built for the final
Computer Vision course project.

## What this project does

1. Camera watches the driver. System is **inactive** by default.
2. Driver performs gesture sequence: **open palm → thumbs up** within a time window.
3. Once activated, the system monitors for fatigue cues:
   - Eye closure / blink duration (EAR)
   - Yawning (MAR)
   - Head pose (nodding, tilting)
   - General fatigue facial expression
4. When fatigue persists, an alert is triggered.

## Approach: Hybrid pipeline

We implement and compare **two** pipelines:

| Component        | Classical                                    | Modern                                  |
|------------------|----------------------------------------------|-----------------------------------------|
| Hand detection   | MediaPipe Hands                              | MediaPipe Hands                         |
| Gesture classify | Hand-landmark features + SVM / Random Forest | CNN on hand crops                       |
| Face landmarks   | MediaPipe FaceMesh                           | MediaPipe FaceMesh                      |
| Fatigue cues     | EAR + MAR + head-pose thresholds + RF        | CNN-LSTM on frame sequences             |
| Sequence logic   | State machine (same for both)                | State machine (same for both)           |

Both pipelines run on the same dataset and are compared head-to-head in the report.

## Project structure

```
CV_FINAL_PROJECT/
├── dataset/                 # raw videos (gitignored - download separately)
│   ├── fatigue/
│   └── gestures/
├── external_datasets/       # for pretraining only (gitignored)
├── data_processed/          # extracted frames, splits (gitignored, regenerable)
├── outputs/                 # trained models, results (gitignored)
├── src/                     # source modules
│   ├── config.py
│   ├── data/                # data loading, splits, frame extraction
│   ├── gestures/            # gesture pipelines (classical + modern)
│   ├── fatigue/             # fatigue pipelines (classical + modern)
│   ├── system/              # state machine, integration
│   └── utils/
├── scripts/                 # runnable scripts (inventory, train, eval, demo)
├── notebooks/               # exploratory notebooks
├── tests/
├── requirements.txt
└── README.md
```

## Setup

Requires **Python 3.12** and macOS (tested on Apple Silicon M3).

```bash
# Clone the repo
git clone https://github.com/javronich1/cv-driver-fatigue-detection.git
cd cv-driver-fatigue-detection

# Create virtual environment (use python3.12)
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Place the dataset
# Copy / symlink your `dataset/` folder into the project root (with `fatigue/`
# and `gestures/` subfolders).
```

## Running

```bash
# Stage 1: dataset inventory (sanity-check that all videos load)
python scripts/inventory.py
```

(More commands will be added as later stages land.)

## Dataset

The dataset was captured by the project members in a parked car, simulating an
in-car driver-monitoring camera. Two subjects, single session, single lighting
condition. External datasets (UTA-RLDD, NTHU-DDD, MRL Eye, YawDD) are used
**only for pretraining**; all final evaluation is on our own captured data.

### Folder structure

```
dataset/
├── fatigue/
│   ├── alert/  alert looking/  calibration/  eyes closed/  fatigue face/
│   ├── head drooping/  head tilting/  microsleep/  phone/  slow blinks/
│   ├── talking/  transition/  yawning/
└── gestures/
    ├── correct sequence/  incomplete/  no hands/  open palm/
    ├── random hands/  thumbs up/  too slow/  wrong sequence/
```

## Authors

CV course final project, group of 3 (videos recorded by 2 of the 3 members).
