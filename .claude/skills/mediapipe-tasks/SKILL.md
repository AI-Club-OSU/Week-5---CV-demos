---
name: mediapipe-tasks
description: Use when writing, fixing or explaining Python webcam code that uses MediaPipe hand landmarks, gestures or pose in this repo, including the air drawing, rock-paper-scissors and rep counter demos.
---

# MediaPipe Tasks in this repo

Follow `AGENTS.md`. This skill adds the details that are easy to get wrong.

## The loop every demo uses

```python
import cv2
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from common import FrameClock, model_path, open_camera, should_quit, to_mp_image, to_px

base = mp_python.BaseOptions(model_asset_path=model_path("hand_landmarker.task"))
options = vision.HandLandmarkerOptions(base_options=base, running_mode=vision.RunningMode.VIDEO, num_hands=1)
cap, clock = open_camera(), FrameClock()
with vision.HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)                      # mirror BEFORE detecting
        result = landmarker.detect_for_video(to_mp_image(frame), clock.now_ms())
        ...                                             # logic + drawing here
        cv2.imshow("demo", frame)
        if should_quit(cv2.waitKey(1) & 0xFF):
            break
cap.release()
cv2.destroyAllWindows()
```

For gestures swap in `GestureRecognizerOptions`, `GestureRecognizer` and
`recognize_for_video` (not `detect_for_video`). For pose use `PoseLandmarkerOptions`
with `num_poses=1`, `PoseLandmarker` and `pose_landmarker_lite.task`.

## Landmark numbers you'll need

Full lists are in [landmarks.md](landmarks.md).

- Hand: 0 wrist, 4 thumb tip, 8 index tip, 12 middle tip, 16 ring tip, 20 pinky tip.
  The joint two landmarks below each fingertip is 6, 10, 14, 18.
- Pose: 11/12 shoulders, 13/14 elbows, 15/16 wrists, 23/24 hips, 25/26 knees,
  27/28 ankles. Odd numbers are the person's left side.
- In image coordinates y grows downward, so "higher on screen" means smaller y.

## Recipes for the three demos

- **Air drawing (Hand Landmarker):** a finger is up when `hand[tip].y < hand[joint].y`.
  Only index up means draw from the last point to landmark 8 on a separate canvas;
  index + middle means lift the pen (reset the last point); no fingers means clear,
  after holding for several frames. Overlay the canvas on the video.
- **Rock-paper-scissors (Gesture Recognizer):** `Closed_Fist` is rock, `Open_Palm` is paper,
  `Victory` is scissors, and anything else is ignored. Run a 3-2-1 countdown on
  `time.monotonic()`, pick the computer's move with `random.choice`, and lock the
  player's move after 3 matching frames in a row.
- **Rep counter (Pose Landmarker):** the angle at the middle joint is
  `abs(atan2(c - b) - atan2(a - b))` in degrees, folded to 0-180, computed in pixels.
  Push-up: 11, 13, 15 with up above 160 and down below 90. Squat: 11, 23, 25 with up
  above 170 and down below 90. Count one rep each time the stage goes down to up.

## Bugs to check for before running

| You see | Why | Fix |
|---|---|---|
| `AttributeError: ... no attribute 'solutions'` | Legacy API, removed in 0.10.31 | Rewrite with `mediapipe.tasks.python.vision` |
| Dots stuck in the top-left corner | Landmarks are 0-1, not pixels | `to_px(landmark, frame)` |
| Left and right swapped | Frame not mirrored before detection | `cv2.flip(frame, 1)` first |
| Timestamp error in VIDEO mode | Timestamps didn't increase | `FrameClock().now_ms()` |
| Wrong finger reacts | Guessed landmark index | Check landmarks.md |
| Gesture flickers | Raw per-frame decision | Require N matching frames in a row |
| `drawing_utils` not found | Removed with the legacy API | `draw_hand()` / `draw_pose()` in common.py |
| Count races up while still | Counting every frame | Count only on the down-to-up change |
| `??` drawn on screen | Non-ASCII text in `cv2.putText` | Write `deg`, `->` instead |
