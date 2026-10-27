# Rules for AI coding assistants in this repo

Webcam computer-vision demos in Python, built with MediaPipe and OpenCV for an
OSU AI Club workshop. Codex, Cursor and GitHub Copilot read this file;
Claude Code reads it through `CLAUDE.md`.

## MediaPipe: use the Tasks API only

- Import it like this:
  ```python
  from mediapipe.tasks import python as mp_python
  from mediapipe.tasks.python import vision
  ```
- Never use `mediapipe.solutions` (`mp.solutions.hands`, `mp.solutions.drawing_utils`
  and so on). It was removed in mediapipe 0.10.31, so that code crashes with
  `AttributeError: module 'mediapipe' has no attribute 'solutions'`.
- Every task follows the same four steps:
  ```python
  base = mp_python.BaseOptions(model_asset_path=model_path("hand_landmarker.task"))   # 1. which model
  options = vision.HandLandmarkerOptions(base_options=base,                            # 2. how to run it
                                         running_mode=vision.RunningMode.VIDEO, num_hands=1)
  with vision.HandLandmarker.create_from_options(options) as landmarker:               # 3. build once
      result = landmarker.detect_for_video(to_mp_image(frame), clock.now_ms())         # 4. every frame
  ```
- VIDEO-mode calls: `HandLandmarker.detect_for_video`, `PoseLandmarker.detect_for_video`,
  `GestureRecognizer.recognize_for_video`. Each takes an `mp.Image` and an int
  timestamp in milliseconds that must increase every frame.
- Results: hands are `result.hand_landmarks` and `result.handedness`; gestures are
  `result.gestures[0][0].category_name` and `.score` (plus `result.hand_landmarks`);
  poses are `result.pose_landmarks`.
- Built-in gesture labels (exact, case-sensitive): `None`, `Closed_Fist`, `Open_Palm`,
  `Pointing_Up`, `Thumb_Down`, `Thumb_Up`, `Victory`, `ILoveYou`.

## Project conventions

- Models live in `models/`: `hand_landmarker.task`, `gesture_recognizer.task`,
  `pose_landmarker_lite.task`. `python download_models.py` fetches them. Load them
  with `model_path()` from `common.py`.
- Reuse `common.py`: `open_camera()`, `FrameClock` (timestamps that always increase),
  `to_mp_image()` (BGR frame to RGB `mp.Image`), `to_px()` (0-1 landmark to pixels),
  `draw_hand()`, `draw_pose()`, `draw_text()`, `draw_big_text()`, `should_quit()`.
- Mirror the frame with `cv2.flip(frame, 1)` before detecting. Handedness assumes
  a mirrored, selfie-style image.
- Landmarks are normalized 0-1. Convert with `to_px()` before drawing or measuring
  angles; raw 0-1 values stretch x and y differently.
- Draw with OpenCV (`cv2.line`, `cv2.circle`) or the `common.py` helpers.
  `drawing_utils` doesn't exist anymore.
- On-screen text must be plain ASCII. OpenCV's fonts can't draw a degree sign or arrows.
- Time things with `time.monotonic()`. Never call `time.sleep()` inside the camera loop.
- Don't add `opencv-python` to the requirements. MediaPipe already installs
  `opencv-contrib-python`, and two copies of cv2 conflict.
- Landmark numbers are in `.claude/skills/mediapipe-tasks/landmarks.md`.

## Before you say it works

- `python -m pytest tests` passes.
- Say which landmark numbers the logic uses and why.
