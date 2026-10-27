"""Demo 2 starter: rock-paper-scissors vs. the computer with the Gesture Recognizer.

Run from the repo root:  python starter/rock_paper_scissors.py      (Q to quit)

Already working: the webcam, the MediaPipe setup, and reading the top
gesture every frame.
Your job (or your AI's): the three TODOs.
Prompts: prompts/02-rock-paper-scissors.md   Finished version: solutions/rock_paper_scissors.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # lets us import common.py

import cv2
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from common import (FrameClock, draw_hand, draw_text, model_path, open_camera,
                    should_quit, to_mp_image)

WINDOW = "Rock-paper-scissors (Q to quit)"

# TODO 1: map MediaPipe's built-in gesture labels to moves, e.g. "Closed_Fist": "rock".
# Built-in labels (exact spelling): None, Closed_Fist, Open_Palm, Pointing_Up,
# Thumb_Down, Thumb_Up, Victory, ILoveYou
GESTURE_TO_MOVE: dict[str, str] = {}

BEATS = {"rock": "scissors", "scissors": "paper", "paper": "rock"}  # key beats value


def round_winner(player: str, computer: str) -> str:
    """TODO 2: return "player", "computer" or "tie"."""
    return "tie"


def top_gesture(result, min_score: float = 0.6) -> str | None:
    """The most likely gesture label for the first hand, or None."""
    if result.gestures and result.gestures[0]:
        best = result.gestures[0][0]
        if best.score >= min_score:
            return best.category_name
    return None


def main() -> None:
    base = mp_python.BaseOptions(model_asset_path=model_path("gesture_recognizer.task"))
    options = vision.GestureRecognizerOptions(
        base_options=base,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
    )
    cap, clock = open_camera(), FrameClock()

    with vision.GestureRecognizer.create_from_options(options) as recognizer:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # mirror before detecting

            result = recognizer.recognize_for_video(to_mp_image(frame), clock.now_ms())
            gesture = top_gesture(result)
            move = GESTURE_TO_MOVE.get(gesture)

            if result.hand_landmarks:
                draw_hand(frame, result.hand_landmarks[0])
            draw_text(frame, f"gesture: {gesture}   move: {move}", (30, 50))

            # TODO 3: the game. SPACE starts it. Each round: a 3-2-1 countdown that
            # uses time.monotonic() (never time.sleep, so the video keeps running),
            # the computer picks random.choice(list(BEATS)), then the player's move
            # locks in once the same move shows up in 3 frames in a row.
            # Show both moves, the winner and the score. First to 3 wins.

            cv2.imshow(WINDOW, frame)
            if should_quit(cv2.waitKey(1) & 0xFF):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
