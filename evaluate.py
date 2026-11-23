"""Evaluate a trained pose model: class confusion matrix and keypoint accuracy.

``yolo pose val`` already reports box/pose mAP and draws a confusion matrix for
the exercise classes. This script adds what those plots do not show: how far each
predicted joint lands from its label, which is what decides whether the joint
angle - and therefore the rep count - is usable.

Usage:

    python evaluate.py --model runs/pose/runs/pose/smoke13/weights/best.pt \
        --data datasets/exercise-7efha-4/data.yaml --split val --imgsz 256

Writes into ``--out`` (default ``runs/pose/eval_<weights stem>``):

    class_confusion_matrix.csv  counts, rows = label, cols = prediction
    class_confusion_matrix.png  the same matrix, annotated
    keypoint_metrics.csv        per-joint error and PCK
    metrics.json                everything above plus per-class precision/recall/F1

Every prediction is matched to a label greedily by IoU at 0.5, ignoring class, so
a wrongly classified exercise lands off the diagonal instead of being counted as
a miss. Joint errors are divided by the label box diagonal, so they are    resolution independent, and PCK@0.1 / PCK@0.2 use the same normalisation over
    every labelled joint of a matched box - a joint the model missed counts as a
    failure, so PCK is not flattered by dropping hard points.

"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset

matplotlib.use("Agg")  # draw without a display, we only save files
import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BACKGROUND = "background"
PCK_THRESHOLDS = (0.1, 0.2)
MIN_KPT_CONF = 0.5  # a joint below this is treated as "not detected"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Class confusion matrix and per-keypoint accuracy for a pose model."
    )
    parser.add_argument("--model", required=True, help="trained pose weights (.pt)")
    parser.add_argument("--data", required=True, help="dataset data.yaml")
    parser.add_argument("--split", default="val", help="split to evaluate (default: val)")
    parser.add_argument("--imgsz", type=int, default=256, help="inference size (default: 256)")
    parser.add_argument("--conf", type=float, default=0.25, help="prediction confidence (default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.5, help="matching IoU (default: 0.5)")
    parser.add_argument("--device", default="cpu", help="torch device, e.g. cpu or 0 (default: cpu)")
    parser.add_argument("--out", default=None, help="output folder (default: runs/pose/eval_<stem>)")
    return parser.parse_args()


def load_split(data_yaml: str, split: str) -> tuple[list[Path], dict[int, str]]:
    """Image paths of one split and the class names, resolved like Ultralytics does."""
    data = check_det_dataset(data_yaml, autodownload=False)
    if split not in ("train", "val", "test"):
        raise SystemExit(f"Unknown split {split!r}; use train, val or test.")
    images = Path(data[split])
    if not images.is_dir():
        raise SystemExit(f"{split} images not found at {images}")
    paths = sorted(p for p in images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    names = {int(i): str(n) for i, n in data["names"].items()}
    return paths, names


def label_path_for(image: Path) -> Path:
    """``.../images/x.jpg`` -> ``.../labels/x.txt``."""
    return image.parent.parent / "labels" / f"{image.stem}.txt"


def load_labels(path: Path, width: int, height: int):
    """(classes, xyxy boxes in pixels, keypoints (n, k, 3) in pixels) of one label file."""
    if not path.exists():
        return np.zeros(0, int), np.zeros((0, 4), float), np.zeros((0, 0, 3), float)

    rows = [line.split() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        return np.zeros(0, int), np.zeros((0, 4), float), np.zeros((0, 0, 3), float)

    keypoints = (len(rows[0]) - 5) // 3
    classes = np.array([int(row[0]) for row in rows], int)
    boxes = np.zeros((len(rows), 4), float)
    points = np.zeros((len(rows), keypoints, 3), float)

    for i, row in enumerate(rows):
        values = np.array(row[5:], float).reshape(keypoints, 3)
        cx, cy, bw, bh = (float(v) for v in row[1:5])
        boxes[i] = (
            (cx - bw / 2) * width,
            (cy - bh / 2) * height,
            (cx + bw / 2) * width,
            (cy + bh / 2) * height,
        )
        points[i, :, 0] = values[:, 0] * width
        points[i, :, 1] = values[:, 1] * height
        points[i, :, 2] = values[:, 2]

    return classes, boxes, points


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between boxes ``a`` (N,4) and ``b`` (M,4), both xyxy."""
    area_a = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    area_b = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)
    left = np.maximum(a[:, None, 0], b[None, :, 0])
    top = np.maximum(a[:, None, 1], b[None, :, 1])
    right = np.minimum(a[:, None, 2], b[None, :, 2])
    bottom = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = (right - left).clip(0) * (bottom - top).clip(0)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def match_boxes(boxes_gt: np.ndarray, boxes_pred: np.ndarray, scores: np.ndarray, threshold: float):
    """Greedy confidence-ordered IoU matching, class-agnostic like the Ultralytics matrix."""
    pairs: list[tuple[int, int]] = []
    if len(boxes_gt) and len(boxes_pred):
        iou = box_iou(boxes_gt, boxes_pred)
        taken = np.zeros(len(boxes_gt), bool)
        for pi in np.argsort(-scores):
            candidates = np.where(taken, -1.0, iou[:, pi])
            gi = int(candidates.argmax())
            if candidates[gi] >= threshold:
                taken[gi] = True
                pairs.append((gi, int(pi)))
        unmatched_gt = np.flatnonzero(~taken)
    else:
        unmatched_gt = np.arange(len(boxes_gt))

    matched_pred = {pi for _, pi in pairs}
    unmatched_pred = np.array([i for i in range(len(boxes_pred)) if i not in matched_pred], int)
    return pairs, unmatched_gt, unmatched_pred


def plot_confusion_matrix(matrix: np.ndarray, names: list[str], out: Path) -> None:
    """Annotated heat map, each row scaled to 100% of that class' labels."""
    normalised = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)  # row-normalised
    fig, ax = plt.subplots(figsize=(1.1 * len(names) + 2.5, 1.1 * len(names) + 1.5))
    ax.imshow(normalised, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("prediction")
    ax.set_ylabel("label")
    ax.set_title("Confusion matrix (row = 1.0 at full colour)")

    for (row, col), count in np.ndenumerate(matrix):
        share = normalised[row, col]
        colour = "white" if share > 0.5 else "black"
        ax.text(col, row, f"{count}\n{share:.0%}", ha="center", va="center", color=colour, fontsize=8)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> int:
    args = parse_args()

    out = Path(args.out) if args.out else Path("runs/pose") / f"eval_{Path(args.model).stem}"
    out.mkdir(parents=True, exist_ok=True)

    images, names = load_split(args.data, args.split)
    class_names = [names[i] for i in sorted(names)]
    labels = class_names + [BACKGROUND]

    model = YOLO(args.model)
    keypoint_count = int(model.model.kpt_shape[0])

    matrix = np.zeros((len(labels), len(labels)), int)
    errors: list[list[float]] = [[] for _ in range(keypoint_count)]  # normalised per joint
    visible = [0] * keypoint_count  # joints labelled visible
    detected = [0] * keypoint_count  # joints labelled visible the model also found

    skipped = 0
    for index, image in enumerate(images, 1):
        if index % 25 == 0 or index == len(images):
            print(f"  {index}/{len(images)} images", flush=True)

        frame = cv2.imread(str(image))
        if frame is None:
            skipped += 1
            continue
        height, width = frame.shape[:2]
        gt_classes, gt_boxes, gt_points = load_labels(label_path_for(image), width, height)

        result = model.predict(
            frame, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False
        )[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            pred_boxes = np.zeros((0, 4), float)
            pred_classes = np.zeros(0, int)
            scores = np.zeros(0, float)
            pred_points = np.zeros((0, keypoint_count, 3), float)
        else:
            pred_boxes = boxes.xyxy.cpu().numpy()
            pred_classes = boxes.cls.cpu().numpy().astype(int)
            scores = boxes.conf.cpu().numpy()
            pred_points = result.keypoints.data.cpu().numpy()

        pairs, unmatched_gt, unmatched_pred = match_boxes(gt_boxes, pred_boxes, scores, args.iou)

        for gi, pi in pairs:
            matrix[gt_classes[gi], pred_classes[pi]] += 1
        for gi in unmatched_gt:
            matrix[gt_classes[gi], len(class_names)] += 1
        for pi in unmatched_pred:
            matrix[len(class_names), pred_classes[pi]] += 1

        for gi, pi in pairs:
            if gt_points.shape[1] != keypoint_count:
                raise SystemExit(
                    f"Label {image.stem} has {gt_points.shape[1]} keypoints but the model predicts "
                    f"{keypoint_count}."
                )
            diagonal = float(
                np.hypot(gt_boxes[gi, 2] - gt_boxes[gi, 0], gt_boxes[gi, 3] - gt_boxes[gi, 1])
            )
            for joint in range(keypoint_count):
                if gt_points[gi, joint, 2] <= 0:
                    continue  # joint not labelled
                visible[joint] += 1
                if pred_points[pi, joint, 2] < MIN_KPT_CONF:
                    continue
                detected[joint] += 1
                if diagonal > 0:
                    distance = float(
                        np.hypot(
                            pred_points[pi, joint, 0] - gt_points[gi, joint, 0],
                            pred_points[pi, joint, 1] - gt_points[gi, joint, 1],
                        )
                    )
                    errors[joint].append(distance / diagonal)

    if skipped:
        print(f"Skipped {skipped} unreadable images.")

    # Per-class precision / recall / F1 straight from the matrix (background excluded).
    per_class = []
    for i, name in enumerate(class_names):
        predicted = int(matrix[:, i].sum())
        labelled = int(matrix[i, :].sum())
        hit = int(matrix[i, i])
        precision = hit / predicted if predicted else 0.0
        recall = hit / labelled if labelled else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append(
            {"class": name, "labels": labelled, "predictions": predicted,
             "correct": hit, "precision": precision, "recall": recall, "f1": f1}
        )

    joint_rows = []
    for joint in range(keypoint_count):
        values = np.array(errors[joint], float)
        row = {
            "keypoint": f"kpt_{joint}",
            "matched": int(len(values)),
            "detection_rate": detected[joint] / visible[joint] if visible[joint] else 0.0,
            "mean_error": float(values.mean()) if len(values) else float("nan"),
            "median_error": float(np.median(values)) if len(values) else float("nan"),
        }
        for threshold in PCK_THRESHOLDS:
            # A joint the model never detected counts as a PCK failure, not as excluded.
            hits = int((values <= threshold).sum())
            row[f"pck_{threshold}"] = hits / visible[joint] if visible[joint] else float("nan")
        joint_rows.append(row)

    plot_confusion_matrix(matrix, labels, out / "class_confusion_matrix.png")
    write_csv(
        out / "class_confusion_matrix.csv",
        ["label"] + labels,
        [[name] + [int(v) for v in row] for name, row in zip(labels, matrix)],
    )
    write_csv(
        out / "keypoint_metrics.csv",
        ["keypoint", "matched", "detection_rate", "mean_error", "median_error"]
        + [f"pck_{t}" for t in PCK_THRESHOLDS],
        [
            [row["keypoint"], row["matched"], f"{row['detection_rate']:.3f}",
             f"{row['mean_error']:.4f}", f"{row['median_error']:.4f}"]
            + [f"{row[f'pck_{t}']:.3f}" for t in PCK_THRESHOLDS]
            for row in joint_rows
        ],
    )

    labelled = int(matrix[: len(class_names), :].sum())
    correct = int(np.trace(matrix[: len(class_names), : len(class_names)]))
    summary = {
        "model": str(args.model),
        "data": str(args.data),
        "split": args.split,
        "images": len(images) - skipped,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "match_iou": args.iou,
        "classes": class_names,
        "confusion_matrix": matrix.tolist(),
        "overall": {
            "matched": labelled,
            "correct": correct,
            "accuracy": correct / labelled if labelled else 0.0,
            "macro_precision": float(np.mean([row["precision"] for row in per_class])),
            "macro_recall": float(np.mean([row["recall"] for row in per_class])),
            "macro_f1": float(np.mean([row["f1"] for row in per_class])),
        },
        "per_class": per_class,
        "keypoints": joint_rows,
        "keypoint_mean_error": float(
            np.nanmean([row["mean_error"] for row in joint_rows])
        ),
        "keypoint_pck_0.2": float(np.nanmean([row["pck_0.2"] for row in joint_rows])),
        "keypoint_detection_rate": float(
            np.mean([row["detection_rate"] for row in joint_rows])
        ),
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nSplit {args.split}: {summary['images']} images, {labelled} labelled objects")
    print(f"{'class':<16}{'labels':>8}{'preds':>8}{'hits':>7}{'P':>9}{'R':>9}{'F1':>9}")
    for row in per_class:
        print(
            f"{row['class'][:15]:<16}{row['labels']:>8}{row['predictions']:>8}{row['correct']:>7}"
            f"{row['precision']:>9.3f}{row['recall']:>9.3f}{row['f1']:>9.3f}"
        )
    print(
        f"\nkeypoints: mean error {summary['keypoint_mean_error']:.3f} of box diagonal, "
        f"PCK@0.2 {summary['keypoint_pck_0.2']:.3f}, "
        f"detection rate {summary['keypoint_detection_rate']:.3f}"
    )
    print(
        f"\nWrote {out / 'class_confusion_matrix.csv'}, .png, keypoint_metrics.csv, metrics.json"
    )
    return 0


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    raise SystemExit(main())
