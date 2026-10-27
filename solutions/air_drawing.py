"""Demo 1, finished: air drawing with the Hand Landmarker.

Run from the repo root:  python solutions/air_drawing.py

  Index finger up ......... draw
  Index + middle up ....... lift the pen
  Fist (hold it) .......... clear
  Keys: 1-4 change color, C clear, Q quit

Finger logic adapted from Educative's "Create a virtual paint app using
MediaPipe" (it uses the legacy API; the landmark numbers are the same).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # lets us import common.py

import cv2
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from common import (BLUE, GREEN, ORANGE, WHITE, FrameClock, draw_hand, draw_text,
                    model_path, open_camera, should_quit, to_mp_image, to_px)

WINDOW = "Air drawing (Q to quit)"
TIPS = [8, 12, 16, 20]    # index, middle, ring and pinky fingertips
JOINTS = [6, 10, 14, 18]  # the joint two landmarks below each tip
CLEAR_HOLD_FRAMES = 10    # hold a fist about a third of a second to clear
PEN_THICKNESS = 8
COLORS = {ord("1"): ORANGE, ord("2"): BLUE, ord("3"): GREEN, ord("4"): WHITE}


def fingers_up(hand) -> list[bool]:
    """[index, middle, ring, pinky]: a finger is up when its tip is higher on
    screen (smaller y) than the joint two landmarks below it."""
    return [hand[tip].y < hand[joint].y for tip, joint in zip(TIPS, JOINTS)]


def choose_mode(up: list[bool]) -> str:
    index, middle, ring, pinky = up
    if index and not (middle or ring or pinky):
        return "draw"
    if not any(up):
        return "clear"
    return "lift"  # index + middle, or any other combination


class Painter:
    """Keeps the drawing between frames."""

    def __init__(self, frame_shape) -> None:
        self.canvas = np.zeros(frame_shape, dtype=np.uint8)
        self.last_point = None
        self.fist_frames = 0
        self.color = ORANGE

    def update(self, mode: str, point) -> None:
        if mode == "draw":
            if self.last_point is not None:
                cv2.line(self.canvas, self.last_point, point, self.color, PEN_THICKNESS)
            self.last_point = point
        else:
            self.last_point = None  # lift the pen so the next stroke starts fresh

        # A quick flick through a fist shouldn't wipe everything: require a hold.
        self.fist_frames = self.fist_frames + 1 if mode == "clear" else 0
        if self.fist_frames == CLEAR_HOLD_FRAMES:
            self.clear()

    def hand_lost(self) -> None:
        self.last_point = None
        self.fist_frames = 0

    def clear(self) -> None:
        self.canvas[:] = 0
        self.last_point = None

    def draw_on(self, frame):
        """Paint the canvas's non-black pixels on top of the camera frame."""
        mask = self.canvas.any(axis=2)
        frame[mask] = self.canvas[mask]
        return frame


def main() -> None:
    base = mp_python.BaseOptions(model_asset_path=model_path("hand_landmarker.task"))
    options = vision.HandLandmarkerOptions(
        base_options=base,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
    )
    cap, clock = open_camera(), FrameClock()
    painter = None

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # mirror before detecting
            if painter is None:
                painter = Painter(frame.shape)

            result = landmarker.detect_for_video(to_mp_image(frame), clock.now_ms())
            if result.hand_landmarks:
                hand = result.hand_landmarks[0]
                mode = choose_mode(fingers_up(hand))
                tip = to_px(hand[8], frame)
                painter.update(mode, tip)

                frame = painter.draw_on(frame)
                draw_hand(frame, hand)
                if mode == "draw":
                    cv2.circle(frame, tip, PEN_THICKNESS, painter.color, -1, cv2.LINE_AA)
                draw_text(frame, f"mode: {mode}", (30, 50))
            else:
                painter.hand_lost()
                frame = painter.draw_on(frame)
                draw_text(frame, "Show your hand", (30, 50))

            draw_text(frame, "1-4 color   C clear   Q quit", (30, frame.shape[0] - 30), scale=0.7)
            cv2.imshow(WINDOW, frame)

            key = cv2.waitKey(1) & 0xFF
            if should_quit(key):
                break
            if key in COLORS:
                painter.color = COLORS[key]
            if key in (ord("c"), ord("C")):
                painter.clear()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
