"""Download a Roboflow pose dataset for training the fitness app's model.

The API key comes from the environment (never commit it):

    Windows:        set ROBOFLOW_API_KEY=your_key
    macOS / Linux:  export ROBOFLOW_API_KEY=your_key

Then download one of the presets into ``datasets/``:

    python download_dataset.py                     # exercise: squat, push-up, plank, lunge
    python download_dataset.py --dataset rep-counting
    python download_dataset.py --dataset coco-pose
    python download_dataset.py --dataset https://universe.roboflow.com/jaew4/exercise-7efha/dataset/4

Exports use the ``yolov8`` format, which is the Ultralytics pose layout, so the
result can be fed straight to ``yolo pose train``.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

# name -> (workspace, project, version). All public, all verified against the API.
PRESETS: dict[str, tuple[str, str, int]] = {
    "exercise": ("jaew4", "exercise-7efha", 4),
    "rep-counting": ("lftnnisa", "pose-counting-repetition", 3),
    "coco-pose": ("microsoft", "coco-pose-detection", 1),
}
DEFAULT_PRESET = "exercise"
DEFAULT_FORMAT = "yolov8"
DEFAULT_OUT = "datasets"

PRESET_HELP = """\
presets (checked against the API on 2026-11-10):
  exercise      jaew4/exercise-7efha v4              1,324 images  squat, knee push-up, push-up, plank, lunge, bird dog - 13 keypoints
  rep-counting  lftnnisa/pose-counting-repetition v3 1,100 images  squat, half-squat, plank, stand - 19 keypoints
  coco-pose     microsoft/coco-pose-detection v1     5,083 images  general people - 17 COCO keypoints

The app is built for the `exercise` preset: its six classes and 13 joints are
what src/exercises.py and src/inference.py expect. COCO-17 weights are reduced
to the same 13 joints on load. A dataset with any other layout needs
KEYPOINT_NAMES/KEYPOINT_INDEX in src/inference.py edited first.
"""

URL_PATTERN = re.compile(r"universe\.roboflow\.com/([\w.-]+)/([\w.-]+)(?:/dataset/(\d+))?")
KPT_SHAPE_PATTERN = re.compile(r"kpt_shape:\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]")


def parse_dataset(value: str) -> tuple[str, str, int]:
    """Resolve a preset name or a Universe dataset URL to (workspace, project, version)."""
    if value in PRESETS:
        return PRESETS[value]

    match = URL_PATTERN.search(value)
    if match is None:
        raise SystemExit(
            f"Unknown dataset {value!r}.\nUse a preset ({', '.join(PRESETS)}) or a URL like\n"
            "  https://universe.roboflow.com/jaew4/exercise-7efha/dataset/4"
        )

    workspace, project, version = match.groups()
    if version is None:
        raise SystemExit("Add the dataset version to the URL, for example .../exercise-7efha/dataset/4")
    return workspace, project, int(version)


def download(workspace: str, project_id: str, version_number: int, model_format: str, out: str):
    """Download one dataset version and return the Roboflow Dataset object."""
    api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not api_key:
        raise SystemExit(
            "ROBOFLOW_API_KEY is not set.\n"
            "Create a key at https://app.roboflow.com/settings/api then run:\n"
            "  Windows:        set ROBOFLOW_API_KEY=your_key\n"
            "  macOS / Linux:  export ROBOFLOW_API_KEY=your_key"
        )

    try:
        from roboflow import Roboflow  # imported here so --help works without the package
    except ImportError:
        raise SystemExit("The roboflow package is missing. Install it with:\n  pip install roboflow")

    project = Roboflow(api_key=api_key).workspace(workspace).project(project_id)
    return project.version(version_number).download(model_format, location=out)


def report_keypoints(data_yaml: Path) -> None:
    """Warn when a dataset's keypoint layout does not match what the app expects."""
    if not data_yaml.exists():
        print(f"Note: {data_yaml} was not found; cannot verify the keypoint layout.")
        return

    text = data_yaml.read_text(encoding="utf-8", errors="replace")
    match = KPT_SHAPE_PATTERN.search(text)
    if match is None:
        print("Note: data.yaml has no kpt_shape entry - confirm this is a pose export.")
        return

    count = int(match.group(1))
    if count == 13:
        print(f"Keypoints: {count} - matches the exercise-dataset layout in src/inference.py.")
    elif count == 17:
        print(
            f"Keypoints: {count} - COCO layout; src/inference.py reduces these to 13 joints,\n"
            "but the model will not know the six exercise classes."
        )
    else:
        print(
            f"Keypoints: {count} - src/inference.py reads 13 (exercise dataset) or 17 (COCO).\n"
            "Update KEYPOINT_NAMES and KEYPOINT_INDEX there before using this model."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a pose dataset for training the fitness app model.",
        epilog=PRESET_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset", default=DEFAULT_PRESET, help=f"preset name or Universe URL (default: {DEFAULT_PRESET})"
    )
    parser.add_argument(
        "--format", default=DEFAULT_FORMAT, help=f"export format (default: {DEFAULT_FORMAT} = Ultralytics pose)"
    )
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"folder to download into (default: {DEFAULT_OUT})")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workspace, project_id, version_number = parse_dataset(args.dataset)

    print(f"Downloading {workspace}/{project_id} v{version_number} as {args.format} ...")
    dataset = download(workspace, project_id, version_number, args.format, args.out)

    location = Path(dataset.location)
    data_yaml = location / "data.yaml"
    print(f"\nSUCCESS! Dataset downloaded to: {location}")
    report_keypoints(data_yaml)

    print("\nTrain a pose model on it, then copy the weights into models/:")
    print(f"  yolo pose train model=yolo11n-pose.pt data={data_yaml.as_posix()} epochs=50 imgsz=640")
    print("  cp runs/pose/train/weights/best.pt models/best.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
