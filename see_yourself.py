"""Setup check: see your hand as 21 MediaPipe landmarks.

Run:  python see_yourself.py        (Q or Esc to quit)

If you see dots and lines on your fingers, your camera, MediaPipe and the
models all work and you're ready for the demos.
"""

from __future__ import annotations

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from common import (FrameClock, draw_hand, draw_text, model_path, open_camera,
                    should_quit, to_mp_image, to_px)

WINDOW = "See yourself (Q to quit)"


def main() -> None:
    print(f"MediaPipe {mp.__version__} | OpenCV {cv2.__version__}")

    # The four-step MediaPipe Tasks pattern. Every demo uses the same shape.
    base = mp_python.BaseOptions(model_asset_path=model_path("hand_landmarker.task"))  # 1. which model
    options = vision.HandLandmarkerOptions(                                            # 2. how to run it
        base_options=base,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )
    cap, clock = open_camera(), FrameClock()

    with vision.HandLandmarker.create_from_options(options) as landmarker:            # 3. build once
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # mirror before detecting, like a selfie camera

            result = landmarker.detect_for_video(to_mp_image(frame), clock.now_ms())  # 4. every frame

            for hand, handedness in zip(result.hand_landmarks, result.handedness):
                draw_hand(frame, hand)
                x, y = to_px(hand[8], frame)  # landmark 8 = index fingertip
                draw_text(frame, f"{handedness[0].category_name} index tip ({x}, {y})", (x + 20, y))
            if not result.hand_landmarks:
                draw_text(frame, "Show a hand to the camera", (30, 50))

            cv2.imshow(WINDOW, frame)
            if should_quit(cv2.waitKey(1) & 0xFF):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
