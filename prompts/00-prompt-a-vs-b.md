# Live demo: same request, two prompts

Try both before the workshop, since results vary by tool and model. Use a fresh
environment with current MediaPipe (`pip install mediapipe`, version 0.10.31 or
newer) so legacy code fails the same way it will for members.

## Prompt A: what everyone types

> Make a hand tracker in Python with MediaPipe that shows my hand on the webcam.

This often comes back as `mp.solutions.hands` code, which crashes on MediaPipe 0.10.31 and newer:

```text
AttributeError: module 'mediapipe' has no attribute 'solutions'
```

## Prompt B: with context

> Use the MediaPipe Tasks API, not mediapipe.solutions. HandLandmarker in VIDEO mode, model at ./models/hand_landmarker.task. Read the webcam with OpenCV, mirror it, and draw the 21 landmarks.

If Prompt A happens to work, ask the AI which API it used and check the installed version:

```bash
python -c "import mediapipe; print(mediapipe.__version__)"
```

## Why it happens

Most tutorials, and so most AI training data, use the legacy `solutions` API. A
MediaPipe maintainer confirmed it was removed in 0.10.31
([google-ai-edge/mediapipe#6204](https://github.com/google-ai-edge/mediapipe/issues/6204)).
The fix isn't clever wording, it's context: name the API, the model file and the
running mode.
