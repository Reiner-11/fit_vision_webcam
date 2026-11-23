"""Fitness exercise motion app: live webcam pose tracking with rep counting.

The movements and keypoints match ``datasets/exercise-7efha-4``. Run from the
project root:

    python src/main.py --exercise squat
    python src/main.py --exercise plank --model models/best.pt
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import cv2

from exercises import EXERCISES, Exercise, ExerciseState
from inference import PoseModel, extract_keypoints, top_prediction

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "best.pt"
WINDOW = "Fitness Motion"
MOVEMENT_KEYS = ("squat", "pushup", "knee_pushup", "lunge", "plank", "birddog")
MOVEMENT_LABELS = (
    "squat",
    "push-up",
    "knee push-up",
    "reverse lunge",
    "plank",
    "bird dog",
)
CONTROLS = "  ".join(f"{i} {label}" for i, label in enumerate(MOVEMENT_LABELS, 1))
ACTIONS = "r reset   q quit"
SWITCH_KEYS = {ord(str(i)): name for i, name in enumerate(MOVEMENT_KEYS, 1)}

WHITE = (255, 255, 255)
GREY = (170, 170, 170)
GREEN = (80, 255, 120)
YELLOW = (60, 200, 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live rep counting and form feedback.")
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help="pose weights (.pt). A stock name like yolo11n-pose.pt is downloaded on first use.",
    )
    parser.add_argument(
        "--exercise",
        choices=MOVEMENT_KEYS,
        default="squat",
        help="movement to track (default: squat)",
    )
    parser.add_argument("--camera", type=int, default=0, help="camera index (default: 0)")
    parser.add_argument(
        "--conf", type=float, default=0.25, help="person detection confidence (default: 0.25)"
    )
    return parser.parse_args()


def put_text(frame, text, org, scale=0.6, color=WHITE, thickness=2) -> None:
    """Draw readable text with a dark outline."""
    cv2.putText(
        frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA
    )
    cv2.putText(
        frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA
    )


def put_text_right(frame, text, right, y, scale=0.6, color=WHITE, thickness=2) -> None:
    """Draw text whose right edge sits at ``right``."""
    width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]
    put_text(frame, text, (max(right - width, 0), y), scale, color, thickness)


def draw_hud(
    frame,
    exercise: Exercise,
    state: ExerciseState,
    fps: float,
    prediction: tuple[str, float] | None = None,
) -> None:
    """Top bar with the selected movement, reps/hold and angle, plus cues and controls."""
    height, width = frame.shape[:2]
    panel_height = 92

    shade = frame.copy()
    cv2.rectangle(shade, (0, 0), (width, panel_height), (0, 0, 0), -1)
    frame[:] = cv2.addWeighted(shade, 0.55, frame, 0.45, 0)

    put_text(frame, exercise.name.upper(), (18, 26), 0.6, GREY, 1)

    value = str(int(state.hold)) if exercise.timed else str(state.reps)
    unit = "sec" if exercise.timed else "reps"
    put_text(frame, value, (18, 82), 1.8, GREEN, 3)
    value_width = cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 3)[0][0]
    put_text(frame, unit, (26 + value_width, 80), 0.6, GREY, 1)

    if prediction is not None:
        name, confidence = prediction
        put_text_right(frame, f"model: {name} {confidence:.0%}", width - 18, 28, 0.55, YELLOW, 1)

    angle = f"{state.angle:.0f} deg" if math.isfinite(state.angle) else "-- deg"
    put_text_right(frame, angle, width - 18, 72, 0.7, WHITE, 2)

    if state.cue:
        put_text(frame, state.cue, (18, panel_height + 34), 0.65, YELLOW, 2)

    put_text(frame, CONTROLS, (18, height - 34), 0.45, GREY, 1)
    put_text(frame, ACTIONS, (18, height - 14), 0.5, GREY, 1)
    put_text_right(frame, f"{fps:4.1f} fps", width - 18, height - 14, 0.55, GREY, 1)


def main() -> int:
    args = parse_args()

    model_path = Path(args.model).expanduser()
    if not model_path.exists() and len(model_path.parts) > 1:
        print(f"Model not found: {model_path}", file=sys.stderr)
        print("Drop your trained weights in models/ or use a stock model:", file=sys.stderr)
        print("  python src/main.py --model yolo11n-pose.pt", file=sys.stderr)
        return 2

    exercise = EXERCISES[args.exercise]()
    model = PoseModel(str(model_path), conf=args.conf)

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        print(f"Could not open camera {args.camera}.", file=sys.stderr)
        return 1

    print(f"{exercise.name}: press q in the window to quit.")
    print(f"Movements: {CONTROLS}")
    fps = 0.0
    last = time.perf_counter()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Camera stream ended.", file=sys.stderr)
                break

            frame = cv2.flip(frame, 1)  # mirror so the movement feels natural
            result = model.predict(frame)
            keypoints = extract_keypoints(result)
            prediction = top_prediction(result)

            dt = time.perf_counter() - last
            last += dt
            state = exercise.update(keypoints, dt)

            if dt > 0:
                fps = 0.9 * fps + 0.1 / dt if fps else 1.0 / dt

            annotated = result.plot()
            draw_hud(annotated, exercise, state, fps, prediction)
            cv2.imshow(WINDOW, annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("r"):
                exercise.reset()
            elif key in SWITCH_KEYS:
                print(f"Switched to {EXERCISES[SWITCH_KEYS[key]].name}.")
                exercise = EXERCISES[SWITCH_KEYS[key]]()
    except KeyboardInterrupt:
        pass
    finally:
        capture.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
