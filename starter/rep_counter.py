"""Demo 3 starter: push-up and squat counter with the Pose Landmarker.

Run from the repo root:  python starter/rep_counter.py
Keys: S switch exercise, R reset, Q quit. Stand side-on to the camera.

Already working: the webcam, the MediaPipe setup, drawing your body, and
turning landmarks into pixel points.
Your job (or your AI's): the two TODOs.
Prompts: prompts/03-rep-counter.md   Finished version: solutions/rep_counter.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # lets us import common.py

import cv2
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from common import (FrameClock, draw_pose, draw_text, model_path, open_camera,
                    should_quit, to_mp_image, to_px)

WINDOW = "Rep counter (S switch, R reset, Q quit)"

# Three landmarks per exercise; the angle is measured at the middle one.
# Left side of the body (the right side is 12, 14, 16 and 12, 24, 26).
EXERCISES = {
    "push-up": {"points": (11, 13, 15), "up": 160, "down": 90},  # shoulder, elbow, wrist
    "squat":   {"points": (11, 23, 25), "up": 170, "down": 90},  # shoulder, hip, knee
}


def joint_angle(a, b, c) -> float:
    """TODO 1: the angle at b, in degrees from 0 to 180, between the lines b->a and b->c.

    a, b and c are (x, y) points in pixels. Hint: math.atan2, and if the
    result is over 180, use 360 minus it.
    """
    return 180.0


class RepCounter:
    """TODO 2: above `up` degrees the stage is "up"; below `down` it is "down".
    Count one rep each time the stage goes from "down" to "up"."""

    def __init__(self, up: float, down: float) -> None:
        self.up, self.down = up, down
        self.stage: str | None = None
        self.count = 0

    def update(self, angle: float) -> int:
        return self.count


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
    counter = RepCounter(EXERCISES[names[current]]["up"], EXERCISES[names[current]]["down"])

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)

            result = landmarker.detect_for_video(to_mp_image(frame), clock.now_ms())
            name = names[current]
            if result.pose_landmarks:
                pose = result.pose_landmarks[0]
                draw_pose(frame, pose)
                a, b, c = (to_px(pose[i], frame) for i in EXERCISES[name]["points"])
                angle = joint_angle(a, b, c)
                counter.update(angle)
                draw_text(frame, f"angle: {angle:.0f} deg   stage: {counter.stage}", (30, 100))
            draw_text(frame, f"{name}s: {counter.count}", (30, 50), scale=1.2)

            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF
            if should_quit(key):
                break
            if key in (ord("s"), ord("S")):
                current = (current + 1) % len(names)
                exercise = EXERCISES[names[current]]
                counter = RepCounter(exercise["up"], exercise["down"])
            if key in (ord("r"), ord("R")):
                exercise = EXERCISES[names[current]]
                counter = RepCounter(exercise["up"], exercise["down"])

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
