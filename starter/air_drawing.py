"""Demo 1 starter: air drawing with the Hand Landmarker.

Run from the repo root:  python starter/air_drawing.py      (Q to quit)

Already working: the webcam, the MediaPipe setup, drawing your hand, and
laying the canvas over the video.
Your job (or your AI's): the three TODOs.
Prompts: prompts/01-air-drawing.md   Finished version: solutions/air_drawing.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # lets us import common.py

import cv2
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from common import (ORANGE, FrameClock, draw_hand, draw_text, model_path, open_camera,
                    should_quit, to_mp_image, to_px)

WINDOW = "Air drawing (Q to quit)"
TIPS = [8, 12, 16, 20]    # index, middle, ring and pinky fingertips
JOINTS = [6, 10, 14, 18]  # the joint two landmarks below each tip
PEN_COLOR = ORANGE
PEN_THICKNESS = 8


def fingers_up(hand) -> list[bool]:
    """Return [index, middle, ring, pinky], True when that finger is up.

    TODO 1: a finger is up when its tip is higher on screen than its joint.
    Higher on screen means a SMALLER y, because y grows downward in images.
    """
    return [False, False, False, False]


def choose_mode(up: list[bool]) -> str:
    """TODO 2: return "draw" when only the index finger is up,
    "clear" when no fingers are up, and "lift" for anything else."""
    return "lift"


def overlay(frame, canvas):
    """Paint the canvas's non-black pixels on top of the camera frame."""
    mask = canvas.any(axis=2)
    frame[mask] = canvas[mask]
    return frame


def main() -> None:
    base = mp_python.BaseOptions(model_asset_path=model_path("hand_landmarker.task"))
    options = vision.HandLandmarkerOptions(
        base_options=base,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
    )
    cap, clock = open_camera(), FrameClock()
    canvas = None      # the drawing, kept between frames
    last_point = None  # where the pen was on the previous frame

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # mirror before detecting
            if canvas is None:
                canvas = np.zeros_like(frame)

            result = landmarker.detect_for_video(to_mp_image(frame), clock.now_ms())
            if result.hand_landmarks:
                hand = result.hand_landmarks[0]
                mode = choose_mode(fingers_up(hand))
                tip = to_px(hand[8], frame)  # index fingertip in pixels

                # TODO 3: if mode is "draw", draw a line on `canvas` from last_point
                # to tip (cv2.line, PEN_COLOR, PEN_THICKNESS), then set last_point = tip.
                # Otherwise set last_point = None so the next stroke doesn't jump.
                # If mode is "clear", wipe the canvas:  canvas[:] = 0

                frame = overlay(frame, canvas)
                draw_hand(frame, hand)
                draw_text(frame, f"mode: {mode}", (30, 50))
            else:
                last_point = None
                frame = overlay(frame, canvas)
                draw_text(frame, "Show your hand", (30, 50))

            cv2.imshow(WINDOW, frame)
            if should_quit(cv2.waitKey(1) & 0xFF):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
