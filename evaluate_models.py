"""Compare every model in the project on the same validation split.

``yolo pose val`` gives official box/pose precision, recall and mAP for one set
of weights. This script runs it for every ``.pt`` file it finds, adds the
correct/incorrect breakdown a presentation needs, and writes one comparable
summary plus slide-ready charts.

Usage:

    python evaluate_models.py --data datasets/exercise-7efha-4/data.yaml --split val
    python evaluate_models.py --models runs/pose/runs/pose/smoke13/weights/best.pt
    python evaluate_models.py --conf 0.001      # max-recall operating point

Writes into ``--out`` (default ``evaluation``)::

    summary.csv / summary.md        one row per model, all metrics
    model_comparison.png            grouped bars: P, R, mAP50, accuracy
    detection_outcomes.png          correct class / wrong class / missed, per model
    <model>/confusion_matrix.png    label rows against prediction columns
    <model>/confusion_matrix.csv    the same counts
    <model>/per_class.csv           correct, wrong, missed, extra, P, R, F1
    <model>/keypoint_metrics.csv    per-joint error, detection rate, PCK
    <model>/metrics.json            everything above in one file

Two kinds of numbers are reported and they answer different questions:

* **mAP** (from ``model.val``, confidence 0.001) is confidence-free and lets
  models with different score calibration be compared. Pose mAP needs the same
  skeleton as the labels; a model trained on another keypoint layout gets
  ``n/a`` there and its box mAP is still reported.
* **Detection accuracy / precision / recall** (from the pass at ``--conf``) is
  the operating point the app actually runs at. A label counts as *detected*
  when a prediction overlaps it at IoU ``--iou``, regardless of class;
  *correct* means that prediction also has the right exercise class. So a box
  around the right person with the wrong label is detected-but-wrong, which is
  exactly what separates a detector problem from a classifier problem.

Models with identical file bytes (a 3-epoch smoke run where ``best.pt`` and
``last.pt`` end up the same) are evaluated once and repeated in the table with a
``duplicate_of`` note, instead of being run twice.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
from ultralytics import YOLO

matplotlib.use("Agg")  # draw without a display, we only save files
import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)

from evaluate import (  # noqa: E402  reuse the single-model helpers
    BACKGROUND,
    IMAGE_SUFFIXES,
    MIN_KPT_CONF,
    PCK_THRESHOLDS,
    load_labels,
    load_split,
    match_boxes,
    write_csv,
)

SKIP_DIRS = {".venv", "venv", "node_modules", "datasets", ".git", "__pycache__"}
VAL_CONF = 0.001  # Ultralytics' default for mAP: every candidate box is kept


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate and compare every pose/detection model in the project."
    )
    parser.add_argument("--data", default="datasets/exercise-7efha-4/data.yaml", help="dataset data.yaml")
    parser.add_argument("--split", default="val", help="split to evaluate (default: val)")
    parser.add_argument("--imgsz", type=int, default=256, help="inference size (default: 256)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="confidence for accuracy/precision/recall (default: 0.25, the app default)")
    parser.add_argument("--iou", type=float, default=0.5, help="matching IoU (default: 0.5)")
    parser.add_argument("--device", default="cpu", help="torch device, e.g. cpu or 0 (default: cpu)")
    parser.add_argument("--models", nargs="*", default=None,
                        help="model paths or globs (default: every .pt in the project)")
    parser.add_argument("--out", default="evaluation", help="output folder (default: evaluation)")
    parser.add_argument("--plots", action="store_true",
                        help="also let Ultralytics save its PR/F1 curves and val batches")
    return parser.parse_args()


# --------------------------------------------------------------------------- models


def find_models(root: Path, out: Path, patterns: list[str] | None) -> list[Path]:
    """Every ``.pt`` under ``root`` unless explicit paths/globs were given."""
    if patterns:
        found: list[Path] = []
        for pattern in patterns:
            matches = sorted(root.glob(pattern)) if any(c in pattern for c in "*?[") else [Path(pattern)]
            found.extend(m for m in matches if m.is_file() and m.suffix == ".pt")
    else:
        found = []
        for path in root.rglob("*.pt"):
            relative = path.relative_to(root)
            if any(part in SKIP_DIRS for part in relative.parts) or out in path.parents:
                continue
            found.append(path)

    unique = {p.resolve() for p in found}
    return sorted(unique, key=lambda p: str(p).lower())


def file_digest(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def label_for(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


# ------------------------------------------------------------------- official val


def run_val(path: Path, args: argparse.Namespace, out: Path) -> dict:
    """Ultralytics val: confidence-free box and pose metrics, per class."""
    summary = {
        "box_precision": float("nan"), "box_recall": float("nan"),
        "box_map50": float("nan"), "box_map50_95": float("nan"),
        "pose_precision": float("nan"), "pose_recall": float("nan"),
        "pose_map50": float("nan"), "pose_map50_95": float("nan"),
        "per_class_val": {}, "error": None,
    }
    name = path.stem
    try:
        result = YOLO(str(path)).val(
            data=args.data, split=args.split, imgsz=args.imgsz, device=args.device,
            conf=VAL_CONF, iou=0.7, plots=args.plots, verbose=False,
            project=str(out / "val"), name=name, exist_ok=True,
        )
    except Exception as error:  # a 17-point model cannot be scored on 13-point labels
        summary["error"] = f"{type(error).__name__}: {error}"
        print(f"    val: skipped ({summary['error']})", flush=True)
        return summary

    metrics = result.results_dict
    summary.update(
        box_precision=float(metrics.get("metrics/precision(B)", float("nan"))),
        box_recall=float(metrics.get("metrics/recall(B)", float("nan"))),
        box_map50=float(metrics.get("metrics/mAP50(B)", float("nan"))),
        box_map50_95=float(metrics.get("metrics/mAP50-95(B)", float("nan"))),
    )
    for prefix, head in (("box", result.box), ("pose", getattr(result, "pose", None))):
        if head is None:
            continue
        summary.update({
            f"{prefix}_precision": float(metrics.get(f"metrics/precision({prefix[0].upper()})", float("nan"))),
            f"{prefix}_recall": float(metrics.get(f"metrics/recall({prefix[0].upper()})", float("nan"))),
            f"{prefix}_map50": float(metrics.get(f"metrics/mAP50({prefix[0].upper()})", float("nan"))),
            f"{prefix}_map50_95": float(metrics.get(f"metrics/mAP50-95({prefix[0].upper()})", float("nan"))),
            "per_class_val": per_class_val(head, prefix),
        })
    return summary


def per_class_val(head, prefix: str) -> dict[int, dict[str, float]]:
    """``{class index: {precision, recall, map50, map50_95}}`` from a val result."""
    table: dict[int, dict[str, float]] = {}
    try:
        indices = np.asarray(head.ap_class_index, dtype=int).reshape(-1)
        arrays = {
            "precision": np.asarray(head.p, dtype=float).reshape(-1),
            "recall": np.asarray(head.r, dtype=float).reshape(-1),
            "map50": np.asarray(head.ap50, dtype=float).reshape(-1),
            "map50_95": np.asarray(head.ap, dtype=float).reshape(-1),
        }
    except (AttributeError, TypeError, ValueError):
        return table
    for position, class_index in enumerate(indices):
        row = {}
        for key, values in arrays.items():
            row[key] = float(values[position]) if position < len(values) else 0.0
        table[int(class_index)] = row
    return table


# ------------------------------------------------------------- confusion / accuracy


def matrix_columns(class_names: list[str], pred_names: list[str]) -> tuple[list[str], dict[int, int]]:
    """Prediction columns for the confusion matrix; unknown model classes get appended."""
    columns = list(class_names)
    mapping: dict[int, int] = {}
    for class_id, name in enumerate(pred_names):
        if name not in columns:
            columns.append(name)
        mapping[class_id] = columns.index(name)
    return columns, mapping


def plot_confusion(matrix: np.ndarray, rows: list[str], columns: list[str], title: str, out: Path) -> None:
    """Annotated heat map, each row scaled to 100% of that class' labels."""
    normalised = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    width = max(1.05 * len(columns) + 2.8, 6.0)
    height = max(1.0 * len(rows) + 1.8, 4.5)
    fig, ax = plt.subplots(figsize=(width, height))
    ax.imshow(normalised, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(columns)), columns, rotation=40, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    ax.set_xlabel("prediction")
    ax.set_ylabel("label (ground truth)")
    ax.set_title(title)

    for (row, col), count in np.ndenumerate(matrix):
        share = normalised[row, col]
        colour = "white" if share > 0.5 else ("black" if share > 0 else "#9aa4b2")
        ax.text(col, row, f"{count}\n{share:.0%}", ha="center", va="center", color=colour, fontsize=8)

    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def evaluate_model(path: Path, root: Path, args: argparse.Namespace,
                   class_names: list[str], out: Path) -> dict:
    """One model: official val metrics plus a per-image correct/incorrect pass."""
    print(f"\n=== {label_for(root, path)}", flush=True)
    model = YOLO(str(path))
    pred_names = [str(name) for _, name in sorted(model.names.items())]
    model_kpts = int(getattr(model.model, "kpt_shape", [0, 0])[0]) if getattr(model, "task", "") == "pose" else 0
    print(f"    classes: {pred_names} | keypoints: {model_kpts or 'none'}", flush=True)

    official = run_val(path, args, out)

    columns, class_column = matrix_columns(class_names, pred_names)
    rows = class_names + [BACKGROUND]
    matrix = np.zeros((len(rows), len(columns) + 1), int)  # last column = background
    background_column = len(columns)

    images, _ = load_split(args.data, args.split)
    errors: list[list[float]] = [[] for _ in range(model_kpts)]
    visible = [0] * model_kpts
    detected = [0] * model_kpts
    keypoint_note = ""

    skipped = 0
    for index, image in enumerate(images, 1):
        if index % 100 == 0 or index == len(images):
            print(f"    {index}/{len(images)} images", flush=True)

        frame = cv2.imread(str(image))
        if frame is None:
            skipped += 1
            continue
        height, width = frame.shape[:2]
        gt_classes, gt_boxes, gt_points = load_labels(image.parent.parent / "labels" / f"{image.stem}.txt", width, height)

        result = model.predict(frame, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            pred_boxes = np.zeros((0, 4), float)
            pred_classes = np.zeros(0, int)
            scores = np.zeros(0, float)
        else:
            pred_boxes = boxes.xyxy.cpu().numpy()
            pred_classes = boxes.cls.cpu().numpy().astype(int)
            scores = boxes.conf.cpu().numpy()

        pairs, unmatched_gt, unmatched_pred = match_boxes(gt_boxes, pred_boxes, scores, args.iou)
        for gi, pi in pairs:
            matrix[gt_classes[gi], class_column[int(pred_classes[pi])]] += 1
        for gi in unmatched_gt:
            matrix[gt_classes[gi], background_column] += 1
        for pi in unmatched_pred:
            matrix[len(class_names), class_column[int(pred_classes[pi])]] += 1

        if model_kpts and gt_points.shape[1] == model_kpts:
            for gi, pi in pairs:
                diagonal = float(np.hypot(gt_boxes[gi, 2] - gt_boxes[gi, 0], gt_boxes[gi, 3] - gt_boxes[gi, 1]))
                for joint in range(model_kpts):
                    if gt_points[gi, joint, 2] <= 0:
                        continue
                    visible[joint] += 1
                    point = result.keypoints.data[pi, joint].cpu().numpy()
                    if point[2] < MIN_KPT_CONF:
                        continue
                    detected[joint] += 1
                    if diagonal > 0:
                        errors[joint].append(float(np.hypot(point[0] - gt_points[gi, joint, 0],
                                                            point[1] - gt_points[gi, joint, 1]) / diagonal))
        elif model_kpts and not keypoint_note:
            keypoint_note = (f"model has {model_kpts} keypoints, labels have {gt_points.shape[1]}: "
                             "joint accuracy not comparable")

    if skipped:
        print(f"    skipped {skipped} unreadable images", flush=True)

    # ---- correct / incorrect breakdown ------------------------------------
    labelled = int(matrix[:len(class_names), :].sum())
    correct = int(np.trace(matrix[:len(class_names), :len(class_names)]))
    matched = correct + int(matrix[:len(class_names), len(class_names):background_column].sum())
    predictions = int(matrix[:, :background_column].sum())
    extra = int(matrix[len(class_names), :background_column].sum())

    per_class = []
    for i, name in enumerate(class_names):
        predicted = int(matrix[:, i].sum())
        labels_i = int(matrix[i, :].sum())
        hit = int(matrix[i, i])
        wrong = int(matrix[i, len(class_names):].sum()) - int(matrix[i, background_column])
        precision = hit / predicted if predicted else 0.0
        recall = hit / labels_i if labels_i else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        confusions = [
            {"predicted_as": columns[j], "count": int(matrix[i, j])}
            for j in range(len(columns)) if j != i and matrix[i, j] > 0
        ]
        confusions.sort(key=lambda item: -item["count"])
        per_class.append({
            "class": name, "labels": labels_i, "correct": hit,
            "wrong_class": wrong,
            "missed": int(matrix[i, background_column]),
            "extra_predictions": int(matrix[len(class_names), i]),
            "precision": precision, "recall": recall, "f1": f1,
            "top_confusion": confusions[0] if confusions else None,
            "confusions": confusions,
        })

    joint_rows = []
    for joint in range(model_kpts):
        values = np.array(errors[joint], float)
        row = {
            "keypoint": f"kpt_{joint}",
            "visible": visible[joint],
            "detected": detected[joint],
            "detection_rate": detected[joint] / visible[joint] if visible[joint] else float("nan"),
            "mean_error": float(values.mean()) if len(values) else float("nan"),
            "median_error": float(np.median(values)) if len(values) else float("nan"),
        }
        for threshold in PCK_THRESHOLDS:
            hits = int((values <= threshold).sum())
            row[f"pck_{threshold}"] = hits / visible[joint] if visible[joint] else float("nan")
        joint_rows.append(row)

    model_dir = out / path.stem
    model_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion(matrix, rows, columns + [BACKGROUND],
                   f"{label_for(root, path)} - conf {args.conf:g}, IoU {args.iou:g}",
                   model_dir / "confusion_matrix.png")
    write_csv(model_dir / "confusion_matrix.csv", ["label"] + columns + [BACKGROUND],
              [[name] + [int(v) for v in row] for name, row in zip(rows, matrix)])
    write_csv(
        model_dir / "per_class.csv",
        ["class", "labels", "correct", "wrong_class", "missed", "extra_predictions",
         "precision", "recall", "f1", "top_confusion"],
        [[row["class"], row["labels"], row["correct"], row["wrong_class"], row["missed"],
          row["extra_predictions"], f"{row['precision']:.3f}", f"{row['recall']:.3f}",
          f"{row['f1']:.3f}",
          f"{row['top_confusion']['predicted_as']} ({row['top_confusion']['count']})"
          if row["top_confusion"] else "-"] for row in per_class],
    )
    if joint_rows:
        write_csv(
            model_dir / "keypoint_metrics.csv",
            ["keypoint", "visible", "detected", "detection_rate", "mean_error", "median_error"]
            + [f"pck_{t}" for t in PCK_THRESHOLDS],
            [[row["keypoint"], row["visible"], row["detected"],
              f"{row['detection_rate']:.3f}", f"{row['mean_error']:.4f}", f"{row['median_error']:.4f}"]
             + [f"{row[f'pck_{t}']:.3f}" for t in PCK_THRESHOLDS] for row in joint_rows],
        )

    keypoint_error = float(np.nanmean([row["mean_error"] for row in joint_rows])) if joint_rows else float("nan")
    pck = float(np.nanmean([row["pck_0.2"] for row in joint_rows])) if joint_rows else float("nan")
    joint_detection = float(np.nanmean([row["detection_rate"] for row in joint_rows])) if joint_rows else float("nan")

    metrics = {
        "model": label_for(root, path),
        "short": f"{path.parent.name}/{path.name}",
        "out_dir": model_dir.relative_to(root).as_posix(),
        "weights": path.name,
        "md5": file_digest(path),
        "size_mb": round(path.stat().st_size / 1e6, 1),
        "task": str(getattr(model, "task", "")),
        "keypoints": model_kpts,
        "model_classes": pred_names,
        "images": len(images) - skipped,
        "labels": labelled,
        "predictions": predictions,
        "detected": matched,
        "detection_accuracy": matched / labelled if labelled else 0.0,
        "detection_precision": matched / predictions if predictions else 0.0,
        "detection_recall": matched / labelled if labelled else 0.0,
        "correct": correct,
        "classification_accuracy": correct / matched if matched else 0.0,
        "overall_accuracy": correct / labelled if labelled else 0.0,
        "wrong_class": matched - correct,
        "missed": labelled - matched,
        "extra_predictions": extra,
        "macro_precision": float(np.mean([row["precision"] for row in per_class])),
        "macro_recall": float(np.mean([row["recall"] for row in per_class])),
        "macro_f1": float(np.mean([row["f1"] for row in per_class])),
        "box_precision": official["box_precision"],
        "box_recall": official["box_recall"],
        "box_map50": official["box_map50"],
        "box_map50_95": official["box_map50_95"],
        "pose_precision": official["pose_precision"],
        "pose_recall": official["pose_recall"],
        "pose_map50": official["pose_map50"],
        "pose_map50_95": official["pose_map50_95"],
        "val_error": official["error"],
        "keypoint_mean_error": keypoint_error,
        "keypoint_pck_0.2": pck,
        "keypoint_detection_rate": joint_detection,
        "keypoint_note": keypoint_note,
        "confusion_label": rows,
        "confusion_columns": columns + [BACKGROUND],
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
        "per_class_val": official["per_class_val"],
        "keypoints_detail": joint_rows,
        "duplicate_of": None,
    }
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"    detected {matched}/{labelled} labels ({metrics['detection_accuracy']:.1%}), "
          f"correct class {correct}/{matched or 0} ({metrics['classification_accuracy']:.1%}), "
          f"box mAP50 {metrics['box_map50']:.3f}, pose mAP50 {metrics['pose_map50']:.3f}", flush=True)
    return metrics


# ------------------------------------------------------------------------ reporting


def plot_comparison(models: list[dict], out: Path) -> None:
    """Grouped bars for the metrics a presentation usually shows side by side."""
    series = [
        ("Box P", "box_precision"), ("Box R", "box_recall"),
        ("Box mAP50", "box_map50"), ("Pose mAP50", "pose_map50"),
        ("Detection acc", "detection_accuracy"), ("Class acc", "classification_accuracy"),
    ]
    labels = [row["short"] for row in models]
    x = np.arange(len(labels))
    width = 0.8 / len(series)

    fig, ax = plt.subplots(figsize=(max(9.0, 2.6 * len(labels) + 4.0), 6.0))
    for offset, (title, key) in enumerate(series):
        values = [row[key] for row in models]
        plotted = [0.0 if not np.isfinite(v) else v for v in values]
        bars = ax.bar(x + offset * width, plotted, width, label=title)
        for bar, value, raw in zip(bars, plotted, values):
            text = "n/a" if not np.isfinite(raw) else f"{value:.2f}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015, text,
                    ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_xticks(x + width * (len(series) - 1) / 2, labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("score (1.0 = perfect)")
    ax.set_title("Model comparison on the validation split")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=min(len(series), 3), fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "model_comparison.png", dpi=170)
    plt.close(fig)


def plot_outcomes(models: list[dict], out: Path) -> None:
    """Where each label ends up: right class, wrong class, or not detected."""
    labels = [row["short"] for row in models]
    correct = np.array([row["correct"] for row in models], float)
    wrong = np.array([row["wrong_class"] for row in models], float)
    missed = np.array([row["missed"] for row in models], float)
    totals = np.maximum(correct + wrong + missed, 1)

    fig, ax = plt.subplots(figsize=(10.0, max(2.8, 0.9 * len(labels) + 2.0)))
    left = np.zeros(len(labels))
    for values, colour, title in ((correct, "#2e7d32", "correct class"),
                                  (wrong, "#ef6c00", "detected, wrong class"),
                                  (missed, "#c62828", "missed")):
        shares = values / totals * 100
        ax.barh(labels, shares, left=left, color=colour, label=title, height=0.55)
        for y, (share, value) in enumerate(zip(shares, values)):
            if share > 7:
                ax.text(left[y] + share / 2, y, f"{int(value)}\n{share:.0f}%",
                        ha="center", va="center", color="white", fontsize=8)
        left += shares
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of labelled objects")
    ax.set_title("Correct vs incorrect predictions per model")
    ax.legend(ncol=3, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out / "detection_outcomes.png", dpi=170)
    plt.close(fig)


def fmt(value: float, digits: int = 3) -> str:
    return "n/a" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def write_summary(models: list[dict], args: argparse.Namespace, out: Path) -> None:
    header = ["model", "task", "keypoints", "classes", "images", "labels", "detection_accuracy",
              "detection_precision", "detection_recall", "classification_accuracy",
              "macro_precision", "macro_recall", "macro_f1",
              "box_precision", "box_recall", "box_map50", "box_map50_95",
              "pose_precision", "pose_recall", "pose_map50", "pose_map50_95",
              "keypoint_pck_0.2", "keypoint_mean_error", "duplicate_of"]
    rows = []
    for row in models:
        rows.append([
            row["model"], row["task"], row["keypoints"], len(row["model_classes"]),
            row["images"], row["labels"],
            f"{row['detection_accuracy']:.4f}", f"{row['detection_precision']:.4f}",
            f"{row['detection_recall']:.4f}", f"{row['classification_accuracy']:.4f}",
            f"{row['macro_precision']:.4f}", f"{row['macro_recall']:.4f}", f"{row['macro_f1']:.4f}",
            fmt(row["box_precision"], 4), fmt(row["box_recall"], 4),
            fmt(row["box_map50"], 4), fmt(row["box_map50_95"], 4),
            fmt(row["pose_precision"], 4), fmt(row["pose_recall"], 4),
            fmt(row["pose_map50"], 4), fmt(row["pose_map50_95"], 4),
            fmt(row["keypoint_pck_0.2"], 4), fmt(row["keypoint_mean_error"], 4),
            row["duplicate_of"] or "",
        ])
    write_csv(out / "summary.csv", header, rows)

    lines = [
        "# Model evaluation",
        "",
        f"Dataset `{args.data}`, split `{args.split}`, inference {args.imgsz} px on {args.device}.",
        f"Accuracy, precision and recall at confidence {args.conf:g} and IoU {args.iou:g}; "
        f"mAP from `yolo pose val` at confidence {VAL_CONF:g}.",
        "",
        "`detection` scores ignore the exercise class (was the person found at all), "
        "`classification` scores require the right class on top of that, "
        "and `macro` averages the six per-class values.",
        "",
        "## Summary",
        "",
        "| model | det. acc | det. P | det. R | class acc | macro P | macro R | macro F1 | "
        "box mAP50 | box mAP50-95 | pose mAP50 | pose mAP50-95 | PCK@0.2 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in models:
        name = row["model"] + ("*" if row["duplicate_of"] else "")
        lines.append(
            f"| {name} | {row['detection_accuracy']:.3f} | {row['detection_precision']:.3f} | "
            f"{row['detection_recall']:.3f} | {row['classification_accuracy']:.3f} | "
            f"{row['macro_precision']:.3f} | {row['macro_recall']:.3f} | {row['macro_f1']:.3f} | "
            f"{fmt(row['box_map50'])} | {fmt(row['box_map50_95'])} | {fmt(row['pose_map50'])} | "
            f"{fmt(row['pose_map50_95'])} | {fmt(row['keypoint_pck_0.2'])} |"
        )

    duplicates = [row for row in models if row["duplicate_of"]]
    if duplicates:
        lines += ["", "\\* identical file bytes to `" + duplicates[0]["duplicate_of"]
                  + "`, so the metrics are the same and the weights were only run once."]

    for row in models:
        lines += ["", f"## {row['model']}", "",
                  f"{row['size_mb']} MB, task `{row['task']}`, {row['keypoints'] or 'no'} keypoints, "
                  f"classes {row['model_classes']}.",
                  ""]
        if row["val_error"]:
            lines += [f"mAP not available: `{row['val_error']}` - the model cannot be scored against "
                      "this keypoint layout/class set, so trust the detection numbers below.", ""]
        lines += [
            "| class | labels | correct | wrong class | missed | extra | P | R | F1 | most common error |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for item in row["per_class"]:
            top = (f"{item['top_confusion']['predicted_as']} ({item['top_confusion']['count']})"
                   if item["top_confusion"] else "-")
            lines.append(
                f"| {item['class']} | {item['labels']} | {item['correct']} | {item['wrong_class']} | "
                f"{item['missed']} | {item['extra_predictions']} | {item['precision']:.3f} | "
                f"{item['recall']:.3f} | {item['f1']:.3f} | {top} |"
            )
        lines += [
            "",
            f"Total: {row['labels']} labels, {row['detected']} detected, {row['correct']} with the right "
            f"class, {row['wrong_class']} with the wrong class, {row['missed']} missed, "
            f"{row['extra_predictions']} extra boxes on background/distractor regions.",
        ]
        if row["keypoint_note"]:
            lines += ["", f"Keypoints: {row['keypoint_note']}."]
        elif row["keypoints_detail"]:
            lines += ["", f"Keypoints: mean error {row['keypoint_mean_error']:.3f} of the box diagonal, "
                          f"PCK@0.2 {row['keypoint_pck_0.2']:.3f}, "
                          f"joint detection rate {row['keypoint_detection_rate']:.3f}."]
        lines += ["", f"Confusion matrix: `{row['out_dir']}/confusion_matrix.png`."]

    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_table(models: list[dict]) -> None:
    print("\n" + "=" * 108)
    print(f"{'model':<28}{'det acc':>9}{'det P':>8}{'det R':>8}{'cls acc':>9}"
          f"{'box mAP50':>11}{'pose mAP50':>12}{'PCK@0.2':>9}")
    print("-" * 108)
    for row in models:
        name = row["short"][:27]
        print(f"{name:<28}{row['detection_accuracy']:>9.3f}{row['detection_precision']:>8.3f}"
              f"{row['detection_recall']:>8.3f}{row['classification_accuracy']:>9.3f}"
              f"{fmt(row['box_map50']):>11}{fmt(row['pose_map50']):>12}"
              f"{fmt(row['keypoint_pck_0.2']):>9}")
    print("=" * 108)


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent
    out = (root / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    _, names = load_split(args.data, args.split)
    class_names = [names[i] for i in sorted(names)]

    paths = find_models(root, out, args.models)
    if not paths:
        raise SystemExit("No .pt models found. Pass --models <path> explicitly.")
    print(f"Evaluating {len(paths)} model(s) on {args.split}: "
          f"{', '.join(label_for(root, p) for p in paths)}", flush=True)

    models: list[dict] = []
    by_digest: dict[str, dict] = {}
    for path in paths:
        digest = file_digest(path)
        if digest in by_digest:
            metrics = dict(by_digest[digest])
            metrics["model"] = label_for(root, path)
            metrics["short"] = f"{path.parent.name}/{path.name}"
            metrics["out_dir"] = (out / path.stem).relative_to(root).as_posix()
            metrics["weights"] = path.name
            metrics["duplicate_of"] = by_digest[digest]["model"]
            print(f"\n=== {metrics['model']}: identical weights to {metrics['duplicate_of']}, "
                  "reusing its metrics", flush=True)
        else:
            metrics = evaluate_model(path, root, args, class_names, out)
            by_digest[digest] = metrics
        models.append(metrics)

    plot_comparison(models, out)
    plot_outcomes(models, out)
    write_summary(models, args, out)
    print_table(models)
    print(f"\nWrote {out}/summary.csv, summary.md, model_comparison.png, detection_outcomes.png "
          f"and one folder per model.")
    return 0


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    raise SystemExit(main())
