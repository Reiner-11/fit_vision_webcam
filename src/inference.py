"""Model loading and pose keypoint extraction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

# 13-keypoint layout of the exercise dataset (datasets/exercise-7efha-4): the
# COCO skeleton without the eyes and ears. Every keypoint the rep counters use
# is present, in the order the labels store them.
KEYPOINT_NAMES = (
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "left_wrist",
    "right_elbow",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "left_ankle",
    "right_knee",
    "right_ankle",
)
KEYPOINT_INDEX = {name: i for i, name in enumerate(KEYPOINT_NAMES)}

# COCO-17 order of the stock Ultralytics weights, reduced to those same 13 joints.
COCO17_TO_13 = (0, 5, 6, 7, 9, 8, 10, 11, 12, 13, 15, 14, 16)


@dataclass
class Keypoints:
    """Keypoints of one person: pixel coordinates plus per-point confidence."""

    xy: np.ndarray  # (13, 2)
    conf: np.ndarray  # (13,)

    def point(self, name: str) -> np.ndarray:
        return self.xy[KEYPOINT_INDEX[name]]

    def score(self, name: str) -> float:
        return float(self.conf[KEYPOINT_INDEX[name]])

    def has(self, *names: str, min_conf: float = 0.3) -> bool:
        return all(self.score(name) >= min_conf for name in names)


def _layout_indices(keypoint_count: int) -> tuple[int, ...]:
    """Dataset-order indices for a model with ``keypoint_count`` joints.

    The exercise dataset stores 13 joints; the stock weights predict 17 COCO
    joints. Both are reduced to ``KEYPOINT_NAMES`` here so the counters see one
    layout no matter which weights are loaded.
    """
    if keypoint_count == len(KEYPOINT_NAMES):
        return tuple(range(len(KEYPOINT_NAMES)))
    if keypoint_count == 17:
        return COCO17_TO_13
    raise ValueError(
        f"The model predicts {keypoint_count} keypoints; this app reads "
        f"{len(KEYPOINT_NAMES)}-joint (exercise dataset) or 17-joint COCO skeletons."
    )


def extract_keypoints(result) -> Keypoints | None:
    """Keypoints of the largest person in a YOLO result, or None if nobody is found."""
    if result.keypoints is None or len(result.keypoints.data) == 0:
        return None

    data = result.keypoints.data.cpu().numpy()
    indices = _layout_indices(data.shape[1])

    person = 0
    boxes = result.boxes
    if boxes is not None and len(boxes.xyxy) == len(data):
        xyxy = boxes.xyxy.cpu().numpy()
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        person = int(areas.argmax())

    points = data[person][list(indices)]
    return Keypoints(xy=points[:, :2], conf=points[:, 2])


def top_prediction(result) -> tuple[str, float] | None:
    """Class name and confidence of the largest person box, or None.

    The exercise dataset labels six movements, so this is what the trained
    weights think you are doing - shown in the HUD next to the selected one.
    """
    boxes = result.boxes
    if boxes is None or len(boxes.xyxy) == 0:
        return None

    xyxy = boxes.xyxy.cpu().numpy()
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    best = int(areas.argmax())
    class_id = int(boxes.cls[best].item())
    names = result.names or {}
    return str(names.get(class_id, class_id)), float(boxes.conf[best].item())


class PoseModel:
    """Thin wrapper around a YOLO pose model."""

    def __init__(self, model_path: str, conf: float = 0.25):
        self.model = YOLO(model_path)
        self.conf = conf

    def predict(self, frame):
        """Run pose inference on one BGR frame and return the first result."""
        results = self.model(frame, conf=self.conf, verbose=False)
        return results[0]
