"""Exercise definitions: joint angle measurement, rep counting and form cues.

The movements match the classes of ``datasets/exercise-7efha-4`` and the
counters read the dataset's 13-joint skeleton (``inference.KEYPOINT_NAMES``):

    Bird Dog, Knee Push up, Plank, Push up, Reverse Lunge, Squat

Each rep exercise measures one joint angle and runs the same small state machine:

    angle <= down_angle              -> bottom of the rep (limb bent)
    angle >= up_angle while armed    -> rep counted, back at the top

The gap between the two thresholds leaves room for jitter, so a noisy angle
hovering near a single value cannot inflate the count. A rep is only counted
after the bottom was actually reached. ``Plank`` is isometric, so it builds a
hold timer instead of a rep count.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:  # imported for type hints only, keeps this module importable alone
    from inference import Keypoints

MIN_CONFIDENCE = 0.3
SIDES = ("left", "right")


def joint_angle(a, b, c) -> float:
    """Angle at joint ``b`` in degrees, between the segments ``b->a`` and ``b->c``."""
    ba = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    bc = np.asarray(c, dtype=float) - np.asarray(b, dtype=float)
    denom = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom < 1e-6:
        return math.nan
    cosine = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def segment_tilt(a, b) -> float:
    """Tilt of the segment ``a->b`` away from vertical, in degrees (0 = upright)."""
    dx = float(b[0]) - float(a[0])
    dy = float(b[1]) - float(a[1])
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return math.nan
    return math.degrees(math.atan2(abs(dx), abs(dy)))


def torso_lean(kp, side: str) -> float:
    """Tilt of the torso (shoulder to hip) away from vertical, in degrees."""
    return segment_tilt(kp.point(f"{side}_shoulder"), kp.point(f"{side}_hip"))


def knee_angle(kp, side: str) -> float:
    """Angle at the knee (hip-knee-ankle)."""
    return joint_angle(
        kp.point(f"{side}_hip"), kp.point(f"{side}_knee"), kp.point(f"{side}_ankle")
    )


def hip_angle(kp, side: str) -> float:
    """Angle at the hip (shoulder-hip-knee)."""
    return joint_angle(
        kp.point(f"{side}_shoulder"), kp.point(f"{side}_hip"), kp.point(f"{side}_knee")
    )


def elbow_angle(kp, side: str) -> float:
    """Angle at the elbow (shoulder-elbow-wrist)."""
    return joint_angle(
        kp.point(f"{side}_shoulder"), kp.point(f"{side}_elbow"), kp.point(f"{side}_wrist")
    )


def side_values(kp, measure: Callable[[object, str], float]) -> list[float]:
    """``measure`` for every side whose joints are clearly visible."""
    values: list[float] = []
    for side in SIDES:
        value = measure(kp, side)
        if math.isfinite(value):
            values.append(value)
    return values


def pick_side(kp, parts: tuple[str, ...]) -> str | None:
    """Most confident side whose required joints are all clearly visible."""
    best: str | None = None
    best_score = 0.0
    for side in SIDES:
        names = tuple(f"{side}_{part}" for part in parts)
        if not kp.has(*names, min_conf=MIN_CONFIDENCE):
            continue
        score = min(kp.score(name) for name in names)
        if score > best_score:
            best, best_score = side, score
    return best


@dataclass
class ExerciseState:
    """Snapshot of one exercise at one frame."""

    reps: int
    phase: str  # "up"/"down" for reps, "hold"/"rest" for timed holds
    angle: float  # measured joint angle in degrees, NaN when not measurable
    cue: str = ""  # form feedback, empty when the movement looks fine
    hold: float = 0.0  # seconds held so far, only used by timed exercises


class Exercise:
    """Rep counter for a single movement. Subclasses measure one joint angle."""

    name = "exercise"
    parts: tuple[str, ...] = ()  # keypoint suffixes needed on one side
    down_angle = 90.0
    up_angle = 150.0
    timed = False  # isometric hold instead of reps
    hold_angle = 160.0  # angle above which a timed exercise counts as held

    def __init__(self) -> None:
        self.reps = 0
        self.hold = 0.0
        self.phase = "unknown"
        self._armed = False
        self._previous_angle = math.nan
        self._delta = 0.0

    def reset(self) -> None:
        self.reps = 0
        self.hold = 0.0
        self.phase = "unknown"
        self._armed = False
        self._previous_angle = math.nan
        self._delta = 0.0

    @property
    def descending(self) -> bool:
        """True while the measured angle is getting smaller (limb folding)."""
        return self._delta <= 0.0

    def measure(self, kp, side: str) -> float:
        """Joint angle that drives the rep count."""
        raise NotImplementedError

    def form_cue(self, kp, side: str, angle: float) -> str:
        """Form feedback for the current frame, or an empty string."""
        return ""

    def _unknown(self, cue: str) -> ExerciseState:
        """Unreadable frame: keep the count, forget the angle history."""
        self._previous_angle = math.nan
        return ExerciseState(self.reps, "unknown", math.nan, cue, self.hold)

    def update(self, kp, dt: float = 0.0) -> ExerciseState:
        """Feed one frame of keypoints and return the updated state.

        ``dt`` is the seconds since the previous frame; only timed exercises use it.
        """
        if kp is None:
            return self._unknown("Step into the frame")

        side = pick_side(kp, self.parts)
        if side is None:
            return self._unknown("Step back - keep your whole body in view")

        angle = self.measure(kp, side)
        if not math.isfinite(angle):
            return self._unknown("Hold still")

        self._delta = angle - self._previous_angle if math.isfinite(self._previous_angle) else 0.0
        self._previous_angle = angle

        if self.timed:
            if angle >= self.hold_angle:
                self.hold += max(dt, 0.0)
                self.phase = "hold"
            else:
                self.phase = "rest"
            return ExerciseState(
                self.reps, self.phase, angle, self.form_cue(kp, side, angle), self.hold
            )

        if angle <= self.down_angle:
            self.phase = "down"
            self._armed = True
        elif angle >= self.up_angle:
            if self._armed:
                self.reps += 1
                self._armed = False
            self.phase = "up"

        return ExerciseState(
            self.reps, self.phase, angle, self.form_cue(kp, side, angle), self.hold
        )


class Squat(Exercise):
    """Standing squat, counted from the knee angle (hip-knee-ankle)."""

    name = "Squat"
    parts = ("hip", "knee", "ankle")
    down_angle = 100.0
    up_angle = 160.0

    def measure(self, kp, side: str) -> float:
        return knee_angle(kp, side)

    def form_cue(self, kp, side: str, angle: float) -> str:
        if self.descending and self.down_angle < angle < 145.0:
            return "Sit deeper - thighs toward parallel"
        lean = torso_lean(kp, side)
        if math.isfinite(lean) and lean > 40.0:
            return "Chest up - stop leaning forward"
        return ""


class PushUp(Exercise):
    """Full push-up, counted from the elbow angle (shoulder-elbow-wrist)."""

    name = "Push-up"
    parts = ("shoulder", "elbow", "wrist")
    down_angle = 95.0
    up_angle = 155.0

    def measure(self, kp, side: str) -> float:
        return elbow_angle(kp, side)

    def form_cue(self, kp, side: str, angle: float) -> str:
        if self.descending and self.down_angle < angle < 150.0:
            return "Lower your chest closer to the floor"
        return hips_cue(kp, side)


class KneePushUp(Exercise):
    """Push-up from the knees, so the range is slightly shorter than a full push-up."""

    name = "Knee push-up"
    parts = ("shoulder", "elbow", "wrist")
    down_angle = 95.0
    up_angle = 150.0

    def measure(self, kp, side: str) -> float:
        return elbow_angle(kp, side)

    def form_cue(self, kp, side: str, angle: float) -> str:
        if self.descending and self.down_angle < angle < 145.0:
            return "Lower your chest closer to the floor"
        return hips_cue(kp, side)


def hips_cue(kp, side: str) -> str:
    """Cue when the shoulder-hip-ankle line sags, else an empty string."""
    if not kp.has(f"{side}_shoulder", f"{side}_hip", f"{side}_ankle", min_conf=MIN_CONFIDENCE):
        return ""
    body_line = joint_angle(
        kp.point(f"{side}_shoulder"), kp.point(f"{side}_hip"), kp.point(f"{side}_ankle")
    )
    return "Hips sagging - brace your core" if math.isfinite(body_line) and body_line < 160.0 else ""


class ReverseLunge(Exercise):
    """Reverse lunge, counted on the knee of whichever leg bends deepest."""

    name = "Reverse lunge"
    parts = ("hip", "knee", "ankle")
    down_angle = 110.0
    up_angle = 160.0

    def measure(self, kp, side: str) -> float:
        angles = side_values(kp, knee_angle)
        return min(angles) if angles else math.nan

    def form_cue(self, kp, side: str, angle: float) -> str:
        if self.descending and self.down_angle < angle < 150.0:
            return "Step back further - back knee toward the floor"
        lean = torso_lean(kp, side)
        if math.isfinite(lean) and lean > 25.0:
            return "Chest up - stay tall"
        return ""


class Plank(Exercise):
    """Isometric plank: the body line (shoulder-hip-ankle) is held, time accumulates."""

    name = "Plank"
    parts = ("shoulder", "hip", "ankle")
    timed = True
    hold_angle = 160.0

    def measure(self, kp, side: str) -> float:
        angles = side_values(kp, body_line)
        return max(angles) if angles else math.nan

    def form_cue(self, kp, side: str, angle: float) -> str:
        if angle < self.hold_angle:
            return "Brace your core - body in a straight line"
        return ""


def body_line(kp, side: str) -> float:
    """Straightness of shoulder-hip-ankle; 180 is a perfect line."""
    return joint_angle(
        kp.point(f"{side}_shoulder"), kp.point(f"{side}_hip"), kp.point(f"{side}_ankle")
    )


class BirdDog(Exercise):
    """Bird dog, counted on the hip of the extended leg (shoulder-hip-knee)."""

    name = "Bird dog"
    parts = ("shoulder", "hip", "knee")
    down_angle = 110.0
    up_angle = 150.0

    def measure(self, kp, side: str) -> float:
        angles = side_values(kp, hip_angle)
        return max(angles) if angles else math.nan

    def form_cue(self, kp, side: str, angle: float) -> str:
        knees = side_values(kp, knee_angle)
        if knees and max(knees) < 155.0 and angle >= self.down_angle:
            return "Straighten the raised knee"
        if self.descending and self.down_angle < angle < 150.0:
            return "Reach the raised leg long"
        return ""


EXERCISES: dict[str, type[Exercise]] = {
    "squat": Squat,
    "pushup": PushUp,
    "knee_pushup": KneePushUp,
    "lunge": ReverseLunge,
    "plank": Plank,
    "birddog": BirdDog,
}
