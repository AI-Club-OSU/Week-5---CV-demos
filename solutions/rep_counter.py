"""Demo 3, finished: push-up and squat counter with the Pose Landmarker.

Run from the repo root:  python solutions/rep_counter.py
Keys: S switch exercise, R reset, Q quit. Stand side-on, whole body in frame.

Starting thresholds come from real projects: 160 and 90 degrees at the elbow
for push-ups (Labellerr's push-up counter) and 170 and 90 degrees at the hip
for squats (the liveTrain squat code on Hugging Face). Tune them on yourself.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # lets us import common.py

import cv2
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from common import (ORANGE, FrameClock, draw_pose, draw_text, model_path, open_camera,
                    should_quit, to_mp_image, to_px)

WINDOW = "Rep counter (S switch, R reset, Q quit)"

# Three landmarks per side; the angle is measured at the middle one.
EXERCISES = {
    "push-up": {"joint": "elbow", "left": (11, 13, 15), "right": (12, 14, 16), "up": 160, "down": 90},
    "squat":   {"joint": "hip",   "left": (11, 23, 25), "right": (12, 24, 26), "up": 170, "down": 90},
}
MIN_VISIBILITY = 0.5  # skip frames where the camera can't really see the joints


def joint_angle(a, b, c) -> float:
    """Angle at b, in degrees from 0 to 180, between the lines b->a and b->c.
    Points are (x, y) in pixels."""
    angle = abs(math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    ))
    return 360 - angle if angle > 180 else angle


class RepCounter:
    """Above `up` degrees the stage is "up", below `down` it is "down".
    One rep = the stage going from "down" back to "up"."""

    def __init__(self, up: float, down: float) -> None:
        self.up, self.down = up, down
        self.stage: str | None = None
        self.count = 0

    def update(self, angle: float) -> int:
        if angle > self.up:
            if self.stage == "down":
                self.count += 1
            self.stage = "up"
        elif angle < self.down:
            self.stage = "down"
        return self.count


def visibility(landmark) -> float:
    return landmark.visibility if landmark.visibility is not None else 1.0


def visible_side(pose, exercise: dict) -> str:
    """Side-on, one side of the body hides the other. Use the side the camera sees better."""
    left = sum(visibility(pose[i]) for i in exercise["left"])
    right = sum(visibility(pose[i]) for i in exercise["right"])
    return "left" if left >= right else "right"


def main() -> None:
    base = mp_python.BaseOptions(model_asset_path=model_path("pose_landmarker_lite.task"))
    options = vision.PoseLandmarkerOptions(
        base_options=base,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )
    cap, clock = open_camera(), FrameClock()
    names = list(EXERCISES)
    current = 0

    def new_counter() -> RepCounter:
        exercise = EXERCISES[names[current]]
        return RepCounter(exercise["up"], exercise["down"])

    counter = new_counter()

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)

            result = landmarker.detect_for_video(to_mp_image(frame), clock.now_ms())
            name = names[current]
            exercise = EXERCISES[name]

            if result.pose_landmarks:
                pose = result.pose_landmarks[0]
                draw_pose(frame, pose)
                side = visible_side(pose, exercise)
                ids = exercise[side]
                if min(visibility(pose[i]) for i in ids) < MIN_VISIBILITY:
                    draw_text(frame, f"Turn side-on so the camera sees your {exercise['joint']}", (30, 110))
                else:
                    a, b, c = (to_px(pose[i], frame) for i in ids)
                    angle = joint_angle(a, b, c)
                    counter.update(angle)
                    cv2.line(frame, a, b, ORANGE, 6, cv2.LINE_AA)
                    cv2.line(frame, b, c, ORANGE, 6, cv2.LINE_AA)
                    cv2.circle(frame, b, 12, ORANGE, -1, cv2.LINE_AA)
                    draw_text(frame, f"{angle:.0f} deg", (b[0] + 20, b[1]))
                    draw_text(frame, f"{side} {exercise['joint']}   stage: {counter.stage or '-'}", (30, 110))
            else:
                draw_text(frame, "Step back so your whole body is in frame", (30, 110))

            draw_text(frame, f"{name}s: {counter.count}", (30, 50), scale=1.3)
            up, down = exercise["up"], exercise["down"]
            draw_text(frame, f"up > {up} deg, down < {down} deg   S switch   R reset   Q quit",
                      (30, frame.shape[0] - 30), scale=0.7)
            cv2.imshow(WINDOW, frame)

            key = cv2.waitKey(1) & 0xFF
            if should_quit(key):
                break
            if key in (ord("s"), ord("S")):
                current = (current + 1) % len(names)
                counter = new_counter()
            if key in (ord("r"), ord("R")):
                counter = new_counter()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
