"""Shared helpers for the workshop demos.

This file is plain OpenCV: opening the webcam, timestamps, and drawing.
The MediaPipe setup stays in each demo file so you can see the four-step
Tasks API pattern there.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp

REPO_ROOT = Path(__file__).resolve().parent
MODELS_DIR = REPO_ROOT / "models"

# Lines between the 21 hand landmarks. Same pairs as
# mediapipe.tasks.python.vision.HandLandmarksConnections.HAND_CONNECTIONS.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index finger
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring finger
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # edge of the palm
]

# Body lines between the 33 pose landmarks (the face points are left out).
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # shoulders and arms
    (11, 23), (12, 24), (23, 24),                      # torso
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),  # left leg
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),  # right leg
]

# OpenCV colors are (blue, green, red).
ORANGE = (26, 122, 255)
BLUE = (255, 155, 91)
GREEN = (120, 200, 80)
WHITE = (240, 242, 244)
DARK = (31, 24, 20)


def model_path(filename: str) -> str:
    """Full path to a model in ./models, with a clear message if it's missing."""
    path = MODELS_DIR / filename
    if not path.exists():
        sys.exit(
            f"Missing model file: {path}\n"
            "Download the models first:  python download_models.py"
        )
    return str(path)


def camera_index() -> int:
    """Camera number from the command line, e.g.  --camera 1  (default 0)."""
    if "--camera" in sys.argv:
        i = sys.argv.index("--camera")
        if i + 1 < len(sys.argv) and sys.argv[i + 1].isdigit():
            return int(sys.argv[i + 1])
    return 0


def open_camera(width: int = 1280, height: int = 720) -> cv2.VideoCapture:
    """Open the webcam, or exit with tips if it won't open."""
    cap = cv2.VideoCapture(camera_index())
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        sys.exit(
            "Couldn't open the webcam.\n"
            "- Close other apps that use it (Zoom, Teams, FaceTime).\n"
            "- Mac: allow Camera access for your terminal app in\n"
            "  System Settings > Privacy & Security > Camera, then restart the terminal.\n"
            "- External webcam? Try:  --camera 1"
        )
    return cap


class FrameClock:
    """Millisecond timestamps that always increase, which VIDEO mode requires."""

    def __init__(self) -> None:
        self._last = -1

    def now_ms(self) -> int:
        ts = int(time.monotonic() * 1000)
        if ts <= self._last:
            ts = self._last + 1
        self._last = ts
        return ts


def to_mp_image(frame_bgr) -> mp.Image:
    """OpenCV frames are BGR. MediaPipe wants RGB."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def to_px(landmark, frame) -> tuple[int, int]:
    """Landmarks are 0-1 across the frame. Convert one to pixel coordinates."""
    height, width = frame.shape[:2]
    return int(landmark.x * width), int(landmark.y * height)


def draw_landmarks(frame, landmarks, connections, point_ids=None,
                   line_color=BLUE, dot_color=ORANGE) -> None:
    """Draw lines between landmarks, then a dot on each one."""
    points = [to_px(lm, frame) for lm in landmarks]
    for start, end in connections:
        cv2.line(frame, points[start], points[end], line_color, 3, cv2.LINE_AA)
    for i in point_ids if point_ids is not None else range(len(points)):
        cv2.circle(frame, points[i], 5, dot_color, -1, cv2.LINE_AA)


def draw_hand(frame, hand_landmarks) -> None:
    draw_landmarks(frame, hand_landmarks, HAND_CONNECTIONS)


def draw_pose(frame, pose_landmarks) -> None:
    draw_landmarks(frame, pose_landmarks, POSE_CONNECTIONS, point_ids=range(11, 33))


def draw_text(frame, text: str, origin, scale: float = 0.9,
              color=WHITE, background=DARK, thickness: int = 2) -> None:
    """Text on a dark box so it's readable on any background. ASCII only."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = int(origin[0]), int(origin[1])
    cv2.rectangle(frame, (x - 10, y - h - 12), (x + w + 10, y + baseline + 8), background, -1)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_big_text(frame, text: str, scale: float = 4.0, color=WHITE) -> None:
    """Large outlined text in the middle of the frame (countdowns, results)."""
    font = cv2.FONT_HERSHEY_DUPLEX
    thickness = max(2, int(scale * 2))
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
    frame_h, frame_w = frame.shape[:2]
    origin = ((frame_w - w) // 2, (frame_h + h) // 2)
    cv2.putText(frame, text, origin, font, scale, DARK, thickness + 8, cv2.LINE_AA)
    cv2.putText(frame, text, origin, font, scale, color, thickness, cv2.LINE_AA)


def should_quit(key: int) -> bool:
    """True for Q or Esc."""
    return key in (ord("q"), ord("Q"), 27)
