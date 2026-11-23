# Model evaluation

Snapshot of the trained pose weights on the validation split. Produced on
2026-11-12 with the commands at the bottom of this file.

## What was evaluated

| | |
| --- | --- |
| Weights | `runs/pose/runs/pose/smoke13/weights/best.pt` (3-epoch smoke run) |
| Dataset | `datasets/exercise-7efha-4/data.yaml` - 6 exercise classes, 13 keypoints |
| Split | `val`, 264 images (1 image has no person) |
| Inference size | 256 px, matching training (`args.yaml: imgsz 256`) |
| Hardware | CPU, torch 2.15.0+cpu, ultralytics 8.4.208 |

## Official metrics (Ultralytics `yolo pose val`, confidence 0.001)

| Class | Images | Instances | Box P | Box R | Box mAP50 | Box mAP50-95 | Pose P | Pose R | Pose mAP50 | Pose mAP50-95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 264 | 264 | 0.626 | 0.239 | 0.233 | 0.0964 | 0 | 0 | 0 | 0 |
| Bird Dog | 54 | 55 | 1.000 | 0.000 | 0.192 | 0.0667 | 0 | 0 | 0 | 0 |
| Knee Push up | 36 | 36 | 0.000 | 0.000 | 0.201 | 0.0737 | 0 | 0 | 0 | 0 |
| Plank | 48 | 48 | 1.000 | 0.000 | 0.0252 | 0.0081 | 0 | 0 | 0 | 0 |
| Push up | 18 | 18 | 1.000 | 0.000 | 0.0732 | 0.0258 | 0 | 0 | 0 | 0 |
| Revese Lunge | 53 | 53 | 0.362 | 0.509 | 0.442 | 0.192 | 0 | 0 | 0 | 0 |
| Squat | 54 | 54 | 0.393 | 0.925 | 0.465 | 0.212 | 0 | 0 | 0 | 0 |

Precision and recall are the values at the best-F1 confidence. Pose mAP is 0 for
every class: no predicted skeleton ever clears OKS 0.5, so the amount of box
overlap does not help the keypoints.

## Confusion matrix (confidence 0.001, IoU 0.5, class-agnostic matching)

Rows are the label, columns the prediction. Same numbers as
`class_confusion_matrix.csv`; percentages are per row in `class_confusion_matrix.png`.

| label \ prediction | Bird Dog | Knee Push up | Plank | Push up | Revese Lunge | Squat | background |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Bird Dog | 0 | 1 | 0 | 1 | 11 | 41 | 1 |
| Knee Push up | 0 | 0 | 0 | 0 | 1 | 35 | 0 |
| Plank | 0 | 0 | 0 | 0 | 5 | 43 | 0 |
| Push up | 1 | 0 | 0 | 1 | 5 | 11 | 0 |
| Revese Lunge | 0 | 0 | 0 | 0 | 26 | 27 | 0 |
| Squat | 0 | 0 | 0 | 0 | 18 | 36 | 0 |
| background | 519 | 1115 | 5264 | 1469 | 2587 | 3213 | 0 |

Reading: 63 of 264 labels (23.9%) get a box of the correct class - 36 of 54 for
Squat and 26 of 53 for Reverse Lunge, and 0-1 for the other four classes. Almost
every class collapses into "Squat", and 14,167 boxes land on background across
the 264 images (54 per image), which is why precision is around 0.01 at this
confidence. At the app's default `--conf 0.25` only 15 boxes survive on the whole
split, all Squat or Reverse Lunge.

Ultralytics draws its own `confusion_matrix.png` next to this one from the same
0.001 confidence but its internal matching, so the absolute counts differ (about
77,000 background boxes there against 14,000 here). Both agree on the shape of
the failure: nothing but Squat and Reverse Lunge is ever recognised, and the rest
lands on background.

## Keypoint accuracy

Distance from the predicted joint to its label, divided by the label box
diagonal; `PCK@t` counts a joint as correct within `t` of that diagonal and
counts an undetected joint as a miss. Matched boxes at confidence 0.001.

| keypoint | matched | detection rate | mean error | median error | PCK@0.1 | PCK@0.2 |
| --- | --- | --- | --- | --- | --- | --- |
| kpt_0 | 67 | 0.255 | 0.190 | 0.182 | 0.019 | 0.171 |
| kpt_1 | 217 | 0.825 | 0.391 | 0.400 | 0.046 | 0.095 |
| kpt_2 | 68 | 0.259 | 0.235 | 0.250 | 0.030 | 0.072 |
| kpt_3 | 230 | 0.875 | 0.363 | 0.395 | 0.129 | 0.243 |
| kpt_4 | 66 | 0.251 | 0.163 | 0.156 | 0.038 | 0.186 |
| kpt_5 | 13 | 0.049 | 0.215 | 0.203 | 0.000 | 0.023 |
| kpt_6 | 231 | 0.878 | 0.313 | 0.243 | 0.099 | 0.327 |
| kpt_7 | 65 | 0.247 | 0.155 | 0.129 | 0.095 | 0.198 |
| kpt_8 | 227 | 0.863 | 0.310 | 0.301 | 0.057 | 0.183 |
| kpt_9 | 81 | 0.308 | 0.261 | 0.238 | 0.004 | 0.099 |
| kpt_10 | 15 | 0.057 | 0.471 | 0.474 | 0.000 | 0.000 |
| kpt_11 | 96 | 0.365 | 0.330 | 0.298 | 0.000 | 0.091 |
| kpt_12 | 91 | 0.346 | 0.423 | 0.374 | 0.000 | 0.011 |

Mean over joints: error 0.29 of the box diagonal, PCK@0.2 0.131, detection rate
0.429. A joint angle that is off by a third of the person's box diagonal cannot
drive a rep counter: the 100-160 degree thresholds would fire on noise.

## What this says about the weights

- This is the 3-epoch smoke run (`epochs 3`, `fraction 0.1` = 93 training images,
  `imgsz 256`, `val false`), and `results.csv` logged only two epochs with all
  metrics still zero. Near-zero numbers are the expected outcome, not a bug in
  the evaluation: box mAP50 is 0.23, pose mAP50 is 0.
- The model has **13 keypoints**, which `src/inference.py` now reads directly
  (17-point COCO weights are reduced to the same skeleton), so these weights can
  be dropped in as `models/best.pt`. Their accuracy is still far too low for the
  rep counters.
- To get a model worth evaluating, train longer on the full split at 640 px
  (`yolo pose train model=yolo11n-pose.pt data=datasets/exercise-7efha-4/data.yaml epochs=50 imgsz=640`).
  Then rerun the two commands below; the numbers here should move a long way.

## Reproduce

```bash
# Official metrics, confusion matrix and PR/F1 curves
python -c "from ultralytics import YOLO; YOLO('runs/pose/runs/pose/smoke13/weights/best.pt').val(data='datasets/exercise-7efha-4/data.yaml', split='val', imgsz=256, plots=True, project='runs/pose', name='eval_smoke13', exist_ok=True)"

# Class confusion matrix with counts plus per-keypoint accuracy
python evaluate.py --model runs/pose/runs/pose/smoke13/weights/best.pt \
    --data datasets/exercise-7efha-4/data.yaml --split val --imgsz 256 --conf 0.001
```

## Result files

All in `runs/pose/runs/pose/eval_smoke13/`:

| File | Content |
| --- | --- |
| `confusion_matrix.png` | Ultralytics matrix, raw counts |
| `confusion_matrix_normalized.png` | Ultralytics matrix, column-normalised |
| `class_confusion_matrix.png` | `evaluate.py` matrix, percentages per label row |
| `class_confusion_matrix.csv` | counts behind that image |
| `keypoint_metrics.csv` | per-joint error, detection rate, PCK |
| `metrics.json` | everything above plus per-class precision/recall/F1 |
| `BoxPR_curve.png`, `BoxP_curve.png`, `BoxR_curve.png`, `BoxF1_curve.png` | box metric curves over confidence |
| `PosePR_curve.png`, `PoseP_curve.png`, `PoseR_curve.png`, `PoseF1_curve.png` | pose metric curves over confidence |
| `val_batch*_pred.jpg`, `val_batch*_labels.jpg` | predictions and labels side by side |
