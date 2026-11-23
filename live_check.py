"""Self-check for the live app: environment, display, camera pipeline and rep logic.

Run it after changing Python packages or the camera setup:

    python live_check.py
    python live_check.py --exercise pushup --frames 30
    python live_check.py --no-window --frames 10
    python live_check.py --save-dir livecheck_out

It exercises the same code path as ``python src/main.py``: the same model wrapper,
the same keypoint extraction, the same rep counter and the same HUD drawing.
Nothing is stubbed, so a pass here means the app runs on this machine.

Exit status is 0 when the environment checks pass, 1 otherwise. Checks that
depend on what is in front of the camera are reported as warnings, because an
empty room is not a broken install.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import main as app  # noqa: E402  (same helpers the app uses)
from inference import KEYPOINT_INDEX, Keypoints  # noqa: E402

MARK_PASS = "PASS"
MARK_FAIL = "FAIL"
MARK_WARN = "WARN"
MARK_SKIP = "SKIP"


class Report:
    """Collects check results and decides the exit status."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def add(self, status: str, name: str, detail: str = "") -> None:
        self.rows.append((status, name, detail))
        print(f"[{status}] {name}" + (f" - {detail}" if detail else ""))

    def failed(self) -> bool:
        return any(status == MARK_FAIL for status, _, _ in self.rows)

    def summary(self) -> None:
        counts = {mark: 0 for mark in (MARK_PASS, MARK_FAIL, MARK_WARN, MARK_SKIP)}
        for status, _, _ in self.rows:
            counts[status] = counts.get(status, 0) + 1
        print()
        print(
            f"{counts[MARK_PASS]} passed, {counts[MARK_FAIL]} failed, "
            f"{counts[MARK_WARN]} warnings, {counts[MARK_SKIP]} skipped"
        )


def check_gui(report: Report) -> bool:
    """True when cv2 can open a window, which cv2.imshow in the app needs."""
    import cv2 as cv

    gui = ""
    for line in cv.getBuildInformation().splitlines():
        if line.strip().startswith("GUI:"):
            gui = line.split(":", 1)[1].strip()
            break

    if not gui or gui.upper() == "NONE":
        report.add(
            MARK_FAIL,
            "OpenCV display support",
            "no GUI in this build; cv2.imshow cannot work. Install the GUI package: "
            "python -m pip install --force-reinstall --no-deps opencv-python",
        )
        return False

    report.add(MARK_PASS, "OpenCV display support", f"GUI: {gui}")
    return True


def check_backend(report: Report) -> None:
    import ultralytics

    report.add(
        MARK_PASS,
        "Packages import",
        f"cv2 {cv2.__version__}, ultralytics {ultralytics.__version__}, "
        f"python {sys.version.split()[0]}",
    )


def synthetic_pose(exercise, angle: float, conf: float = 0.9) -> Keypoints:
    """A fake skeleton whose measured joint (elbow or knee) is bent to ``angle``."""
    xy = np.full((len(KEYPOINT_INDEX), 2), 320.0, dtype=float)
    scores = np.full(len(KEYPOINT_INDEX), conf, dtype=float)

    # Chain is a -> b -> c with the angle measured at b: hip-knee-ankle or
    # shoulder-elbow-wrist. ``b`` sits at a fixed point, ``a`` straight above it
    # and ``c`` rotated so the angle at ``b`` is exactly ``angle``.
    for side, offset in (("left", -20.0), ("right", 20.0)):
        first, middle, last = (f"{side}_{name}" for name in exercise.parts)
        b = np.array([320.0 + offset, 320.0])
        a = b + np.array([0.0, -100.0])
        theta = math.radians(180.0 - angle)
        c = b + np.array([100.0 * math.sin(theta), 100.0 * math.cos(theta)])
        for name, point in ((first, a), (middle, b), (last, c)):
            xy[KEYPOINT_INDEX[name]] = point
            scores[KEYPOINT_INDEX[name]] = conf

    return Keypoints(xy=xy, conf=scores)


def check_hold_logic(report: Report, exercise) -> None:
    """Drive a timed exercise (plank) through hold and rest frames without a camera."""
    straight = synthetic_pose(exercise, exercise.hold_angle + 5.0)
    state = exercise.update(straight, 1 / 30)
    for _ in range(30):
        state = exercise.update(straight, 1 / 30)

    held = state.hold > 0.5 and state.phase == "hold"
    report.add(
        MARK_PASS if held else MARK_FAIL,
        f"{exercise.name} hold accumulates on a straight line",
        f"{state.hold:.2f}s after 31 frames, phase {state.phase!r}, expected > 0.5s",
    )
    report.add(
        MARK_PASS if math.isfinite(state.angle) else MARK_FAIL,
        "Angle measured on a full skeleton",
        f"{state.angle:.0f} deg, phase {state.phase!r}",
    )

    before = state.hold
    rested = exercise.update(synthetic_pose(exercise, exercise.hold_angle - 30.0), 1 / 30)
    report.add(
        MARK_PASS if rested.phase == "rest" and abs(rested.hold - before) < 1e-9 else MARK_FAIL,
        "Hold pauses when the line breaks",
        f"{rested.hold:.2f}s, phase {rested.phase!r}",
    )

    empty = exercise.update(None, 1 / 30)
    report.add(
        MARK_PASS if empty.hold == before and empty.cue else MARK_FAIL,
        "Empty frame keeps the hold",
        f"hold {before:.2f}s -> {empty.hold:.2f}s, cue {empty.cue!r}",
    )

    hidden = exercise.update(synthetic_pose(exercise, exercise.hold_angle - 30.0, conf=0.1), 1 / 30)
    report.add(
        MARK_PASS if hidden.hold == before and "whole body" in hidden.cue else MARK_FAIL,
        "Hidden limbs are refused",
        f"hold {before:.2f}s -> {hidden.hold:.2f}s, cue {hidden.cue!r}",
    )

    exercise.reset()
    report.add(
        MARK_PASS if exercise.hold == 0.0 else MARK_FAIL,
        "Reset clears the hold",
        f"hold {exercise.hold:.2f}s",
    )


def check_rep_logic(report: Report, exercise_name: str) -> None:
    """Drive the counter through one full rep without a camera."""
    exercise = app.EXERCISES[exercise_name]()
    if exercise.timed:
        check_hold_logic(report, exercise)
        return

    up = exercise.up_angle + 5.0
    down = exercise.down_angle - 5.0
    mid = (exercise.up_angle + exercise.down_angle) / 2.0

    states = [
        exercise.update(synthetic_pose(exercise, angle))
        for angle in (up, mid, down, mid, up)
    ]

    counted = exercise.reps == 1
    report.add(
        MARK_PASS if counted else MARK_FAIL,
        f"Rep counted from one {exercise.name} cycle",
        f"up -> bottom -> up gave {exercise.reps} rep(s), expected 1",
    )
    report.add(
        MARK_PASS if math.isfinite(states[-1].angle) else MARK_FAIL,
        "Angle measured on a full skeleton",
        f"{states[-1].angle:.0f} deg, phase {states[-1].phase!r}",
    )

    before = exercise.reps
    empty = exercise.update(None)
    report.add(
        MARK_PASS if empty.reps == before and empty.cue else MARK_FAIL,
        "Empty frame keeps the count",
        f"reps {before} -> {empty.reps}, cue {empty.cue!r}",
    )

    hidden = exercise.update(synthetic_pose(exercise, mid, conf=0.1))
    report.add(
        MARK_PASS if hidden.reps == before and "whole body" in hidden.cue else MARK_FAIL,
        "Hidden limbs are refused",
        f"reps {before} -> {hidden.reps}, cue {hidden.cue!r}",
    )

    exercise.reset()
    report.add(
        MARK_PASS if exercise.reps == 0 else MARK_FAIL,
        "Reset clears the count",
        f"reps {exercise.reps}",
    )


def check_camera(report: Report, args: argparse.Namespace, gui_ok: bool) -> None:
    """Run the app's per-frame pipeline on the camera and report what it saw."""
    model_path = Path(args.model).expanduser()
    if len(model_path.parts) > 1 and not model_path.exists():
        report.add(MARK_FAIL, "Model file", f"not found: {model_path}")
        return
    report.add(MARK_PASS, "Model file", str(args.model))

    model = app.PoseModel(str(args.model), conf=args.conf)
    # Warm up at camera resolution: the first inference pays the predictor setup cost.
    model.predict(np.zeros((480, 640, 3), dtype=np.uint8))
    report.add(MARK_PASS, "Model loaded")

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        report.add(MARK_FAIL, "Camera opened", f"index {args.camera} unavailable")
        return
    width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    report.add(MARK_PASS, "Camera opened", f"index {args.camera}, {width:.0f}x{height:.0f}")

    exercise = app.EXERCISES[args.exercise]()
    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    frame_times: list[float] = []
    detections = 0
    measurable = 0
    cues: dict[str, int] = {}
    shown = 0

    try:
        for index in range(args.frames):
            started = time.perf_counter()
            ok, frame = capture.read()
            if not ok:
                report.add(MARK_FAIL, "Camera frame", f"read failed at frame {index}")
                break

            frame = cv2.flip(frame, 1)
            result = model.predict(frame)
            keypoints = app.extract_keypoints(result)
            state = exercise.update(keypoints, time.perf_counter() - started)
            annotated = result.plot()

            fps = 1.0 / frame_times[-1] if frame_times else 0.0
            app.draw_hud(annotated, exercise, state, fps)

            if keypoints is not None:
                detections += 1
            if math.isfinite(state.angle):
                measurable += 1
            if state.cue:
                cues[state.cue] = cues.get(state.cue, 0) + 1

            if save_dir and index in (0, args.frames // 2, args.frames - 1):
                path = save_dir / f"frame_{index:03d}.jpg"
                cv2.imwrite(str(path), annotated)
                report.add(MARK_PASS, "Annotated frame saved", str(path))

            if gui_ok and not args.no_window:
                cv2.imshow(app.WINDOW, annotated)
                cv2.waitKey(1)
                shown += 1

            frame_times.append(time.perf_counter() - started)
    finally:
        capture.release()
        if shown:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    if not frame_times:
        report.add(MARK_FAIL, "Camera pipeline", "no frames processed")
        return

    steady = frame_times[1:] or frame_times  # first frame is slower, see warm-up
    mean_ms = sum(steady) / len(steady) * 1000
    report.add(MARK_PASS, "Camera pipeline", f"{len(frame_times)} frames, HUD drawn")
    report.add(
        MARK_PASS if mean_ms <= 500 else MARK_WARN,
        "Frame rate",
        f"{mean_ms:.0f} ms/frame ({1000 / mean_ms:.1f} fps, first frame excluded)"
        + ("" if mean_ms <= 500 else "; too slow for smooth rep counting"),
    )
    if shown:
        report.add(MARK_PASS, "Window displayed", f"{shown} frames via cv2.imshow")

    report.add(
        MARK_PASS if detections else MARK_WARN,
        "Person detected",
        f"{detections}/{len(frame_times)} frames",
    )
    report.add(
        MARK_PASS if measurable else MARK_WARN,
        f"{exercise.name} angle measurable",
        f"{measurable}/{len(frame_times)} frames"
        + ("" if measurable else "; whole body must be in view"),
    )
    for cue, count in sorted(cues.items(), key=lambda item: -item[1]):
        report.add(MARK_WARN, "Cue shown", f"{cue!r} on {count} frame(s)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-check the live fitness app.")
    parser.add_argument("--model", default="yolo11n-pose.pt", help="pose weights to test")
    parser.add_argument(
        "--exercise",
        choices=sorted(app.EXERCISES),
        default="squat",
        help="movement to test (default: squat)",
    )
    parser.add_argument("--camera", type=int, default=0, help="camera index (default: 0)")
    parser.add_argument("--frames", type=int, default=20, help="camera frames to process")
    parser.add_argument("--conf", type=float, default=0.25, help="detection confidence")
    parser.add_argument(
        "--no-window", action="store_true", help="skip cv2.imshow (headless machines)"
    )
    parser.add_argument(
        "--save-dir", default=None, help="save annotated frames here (e.g. livecheck_out)"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.frames < 1:
        print("--frames must be at least 1", file=sys.stderr)
        return 2

    print(f"live check: {args.exercise} | model {args.model} | camera {args.camera}")
    print()

    report = Report()
    check_backend(report)
    gui_ok = check_gui(report)
    check_rep_logic(report, args.exercise)
    check_camera(report, args, gui_ok)
    report.summary()

    if report.failed():
        print("\nFAILED: fix the lines marked FAIL above, then run this script again.")
        return 1
    print("\nOK: the app can run on this machine.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
