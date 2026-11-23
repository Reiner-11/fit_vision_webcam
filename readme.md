# Fitness Motion App

Real-time exercise tracking from your webcam. A YOLO pose model locates your
joints, the app measures the joint angle that drives each movement, counts
clean reps, and shows a form cue when the angle or your posture drifts.

---

## Features

- Six movements from the training dataset: **squat**, **push-up**, **knee push-up**,
  **reverse lunge**, **bird dog** and a timed **plank** hold
- Rep counting for the five rep movements, seconds held for the plank
- Joint-angle state machine with a dead band, so bouncing or jitter near a
  threshold never inflates the count
- Form cues: squat depth, forward lean, hip sag, torso lean, raised knee, plank line
- HUD with the rep count or hold time, live joint angle, cue line, the model's
  own class guess and FPS
- Hotkeys to switch movement, reset the set, or quit
- Plug-in custom weights: drop a `.pt` in `models/`. The 13-point dataset skeleton
  and the 17-point COCO skeleton are both mapped onto the joints the counters use

---

## Requirements

- Python 3.10-3.14 (tested on 3.12.8)
- A webcam
- `ultralytics` pulls in PyTorch, so the install is roughly 1.1 GB of packages

---

## Quick start

These are the exact steps that were run end to end on Windows 11 with
ultralytics 8.4.208 and torch 2.15.0 (CPU).

### 1. Create the environment

```bash
# Windows: see which interpreters exist, then use 3.12
py -0p
py -3.12 -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3.12 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Avoid Python 3.15: PyYAML has no wheel for it and the install stops with
`Microsoft Visual C++ 14.0 or greater is required`.

### 2. Run the app

```bash
python src/main.py --exercise squat --model yolo11n-pose.pt
```

A window opens with the camera feed, the skeleton and the HUD, and the terminal
prints `Squat: press q in the window to quit.` once the camera is open. The first
start is slow - importing PyTorch alone takes ~50 s on a slow CPU - and the stock
weights (6 MB) download once.

Without `--model` the app loads `models/best.pt`, which only exists after you
train your own model; if that file is missing the app stops with an error.

### 3. Check the setup without a camera

```bash
python -c "from ultralytics import YOLO; print(YOLO('yolo11n-pose.pt').model.kpt_shape)"
```

It should print `[17, 3]` - 17 COCO keypoints, which the app reduces to the 13
joints its counters use.

Your own weights go in `models/best.pt`, which is the default. Any other path
works with `--model`.

| Option | Default | Meaning |
| --- | --- | --- |
| `--exercise` | `squat` | `squat`, `pushup`, `knee_pushup`, `lunge`, `plank` or `birddog` |
| `--model` | `models/best.pt` | pose weights; a stock name like `yolo11n-pose.pt` auto-downloads |
| `--camera` | `0` | camera index |
| `--conf` | `0.25` | person detection confidence |

## Controls

| Key | Action |
| --- | --- |
| `1` ... `6` | switch to squat / push-up / knee push-up / reverse lunge / plank / bird dog (resets the set) |
| `r` | reset the rep count (or the plank hold) |
| `q` or `Esc` | quit |

---

## How reps are counted

Every exercise measures one joint angle and runs the same state machine:

1. The limb folds to `down_angle` or below - this **arms** the rep.
2. The limb opens back up to `up_angle` or beyond - the rep is **counted**.

A rep only counts if the bottom was actually reached, and the gap between the
two thresholds absorbs noisy keypoints.

| Exercise | Measured angle | Counted when | Arm at | Count at |
| --- | --- | --- | --- | --- |
| Squat | hip-knee-ankle | you stand back up | 100 deg | 160 deg |
| Push-up | shoulder-elbow-wrist | arms straighten | 95 deg | 155 deg |
| Knee push-up | shoulder-elbow-wrist | arms straighten | 95 deg | 150 deg |
| Reverse lunge | knee of the leg that bends deepest | you stand back up | 110 deg | 160 deg |
| Bird dog | hip of the extended leg (shoulder-hip-knee) | the leg comes back down | 110 deg | 150 deg |
| Plank | shoulder-hip-ankle body line | time held while the line is at least 160 deg | - | - |

The lunge and bird dog counters follow whichever side is clearer, so one side
out of view does not stop the count. The plank accumulates seconds only while
your body line is straight; the hold pauses - it does not reset - when form
breaks.

## Form cues

| Exercise | Cue | Trigger |
| --- | --- | --- |
| Squat | `Sit deeper - thighs toward parallel` | descending, knee angle 100-145 deg |
| Squat | `Chest up - stop leaning forward` | torso more than 40 deg from vertical |
| Push-up | `Lower your chest closer to the floor` | descending, elbow angle 95-150 deg |
| Push-up / Knee push-up | `Hips sagging - brace your core` | shoulder-hip-ankle line below 160 deg |
| Knee push-up | `Lower your chest closer to the floor` | descending, elbow angle 95-145 deg |
| Reverse lunge | `Step back further - back knee toward the floor` | descending, knee angle 110-150 deg |
| Reverse lunge | `Chest up - stay tall` | torso more than 25 deg from vertical |
| Plank | `Brace your core - body in a straight line` | body line below 160 deg |
| Bird dog | `Straighten the raised knee` | raised knee below 155 deg |
| Bird dog | `Reach the raised leg long` | descending, hip angle 110-150 deg |

Cues are advisory: the count still increments while a cue is showing, so you
can review the set afterwards.

---

## Project structure

```text
body-motion-app/
|
+-- models/                 # pose weights (best.pt is the default path)
|
+-- src/
|   +-- main.py             # entry point: webcam loop, HUD, hotkeys
|   +-- inference.py        # model loading, keypoint extraction
|   +-- exercises.py        # joint angles, rep counters, form cues
|
+-- download_dataset.py     # optional: fetch a Roboflow pose dataset for training
+-- requirements.txt
+-- readme.md
```

## Tips

- Keep your whole body (or the tracked limb) in frame; the HUD asks you to step
  back when keypoints disappear.
- A side-on camera gives the most reliable squat and push-up angles.
- Only the largest person in frame is tracked, so bystanders are ignored.
- If detection drops out, lower `--conf` slightly, or brighten the scene.
## Performance

Inference runs on the CPU unless a CUDA GPU is available. Measured on live
640x480 webcam frames with `yolo11n-pose` on an i5-6200U (2 cores / 4 threads):

| Input size | Per frame | FPS |
| --- | --- | --- |
| 640 (the app's default) | 2.3 s | 0.4 |
| 320 | 1.0 s | 1.0 |
| 256 | 0.7 s | 1.3 |

Reps are still counted at those rates, but the overlay trails your movement by
seconds. `src/inference.py` always predicts at 640 px; for a responsive window
run on a CUDA GPU.

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `Could not open camera 0.` | Camera is busy or not index 0 - try `--camera 1` |
| Error loading the model at start | Add `--model yolo11n-pose.pt`, or train `models/best.pt` first |
| Window is seconds behind you | CPU inference, see [Performance](#performance) |
| Skeleton jumps to another person | Only the largest person is tracked - keep others out of frame |
| `pip install` fails on PyYAML | You are on Python 3.15 - recreate the venv with 3.12 |

---

## Training your own model

`download_dataset.py` fetches a public Roboflow pose dataset. Put the API key in
the environment rather than in the file:

```bash
# Windows
set ROBOFLOW_API_KEY=your_key
# macOS / Linux
export ROBOFLOW_API_KEY=your_key
```

```bash
python download_dataset.py                        # squat, push-up, plank, lunge
python download_dataset.py --dataset rep-counting
python download_dataset.py --dataset coco-pose
python download_dataset.py --dataset https://universe.roboflow.com/jaew4/exercise-7efha/dataset/4
```

All presets were checked against the Roboflow API (exports downloaded and
inspected):

| Preset | Source | Images | Content | Keypoints |
| --- | --- | --- | --- | --- |
| `exercise` | `jaew4/exercise-7efha` v4 | 1,324 | squat, knee push-up, push-up, plank, reverse lunge, bird dog (CC BY 4.0) | 13 |
| `rep-counting` | `lftnnisa/pose-counting-repetition` v3 | 1,100 | squat, half-squat, plank, stand (CC BY 4.0) | 19 |
| `coco-pose` | `microsoft/coco-pose-detection` v1 | 5,083 | general people (project marked license: Private) | 17 |

The `exercise` preset is the dataset the UI is built for: its six classes and
13 joints are what `src/exercises.py` and `src/inference.py` expect. `coco-pose`
weights also work - the app reduces their 17 joints to the same 13 - but they
know nothing about the six exercise classes. The `rep-counting` skeleton has 19
points and needs `KEYPOINT_NAMES`/`KEYPOINT_INDEX` edited first.

Downloads land in `datasets/<project>-v<version>` in the Ultralytics pose
layout. The script prints the dataset's `kpt_shape`; check it against
`src/inference.py` before training.

```bash
yolo pose train model=yolo11n-pose.pt data=datasets/exercise-7efha-4/data.yaml epochs=50 imgsz=640
cp runs/pose/train/weights/best.pt models/best.pt
```

The app then picks the weights up automatically through the default
`models/best.pt` path.

Train on a GPU. Measured on an i5-6200U (2 cores / 4 threads, CPU only, 256 px):
about 1.8 s per image per epoch, so a single pass over the full dataset takes
~30 minutes and 640 px scales that up roughly 6x. A 50-epoch run is days of CPU
time, so use a CUDA GPU or a Colab/Kaggle notebook.
"# fit_vision_webcam" 
