"""Retrain the pose model on the exercise dataset.

The previous run (``runs/pose/runs/pose/smoke13``) was 3 epochs on 10% of the
data at 256 px, which is why every exercise collapsed onto Squat. This trains on
the full split at 384 px for 12 epochs - as far as a 4-core CPU gets overnight
(measured ~2 s/image/epoch, so roughly 45 min per epoch including validation).

Usage::

    python train_pose.py

Resume after an interruption with::

    python -c "from ultralytics import YOLO; YOLO('runs/pose/runs/pose/retrain384/weights/last.pt').train(resume=True)"
"""

from ultralytics import YOLO

if __name__ == "__main__":
    YOLO("yolo11n-pose.pt").train(
        data="datasets/exercise-7efha-4/data.yaml",
        epochs=12,
        imgsz=384,
        batch=8,
        device="cpu",
        workers=0,  # Windows spawn overhead outweighs the benefit at this size
        cache="ram",  # ~60 MB of images, avoids the slow-disk warnings
        close_mosaic=2,  # default 10 would disable mosaic after epoch 2 of 12
        project="runs/pose",
        name="retrain384",
        exist_ok=True,
    )
