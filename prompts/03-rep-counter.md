# Demo 3: Squat or push-up counter (Pose Landmarker)

One angle and two thresholds.

| Exercise | Angle at | Landmarks (left side) | Up | Down |
|---|---|---|---|---|
| Push-up | Elbow | 11, 13, 15 | above 160° | below 90° |
| Squat | Hip | 11, 23, 25 | above 170° | below 90° |

The right side is 12, 14, 16 and 12, 24, 26. Count one rep each time the stage
goes from down to up. Film from the side with your whole body in frame.

## Prompt: from scratch

Paste into any AI chat.

> Using the MediaPipe Tasks API (not mediapipe.solutions), build a Python rep counter with PoseLandmarker in VIDEO mode. The model is at ./models/pose_landmarker_lite.task. Read the webcam with OpenCV. Convert landmarks to pixels, then compute the angle at the elbow from landmarks 11, 13 and 15. Above 160° the stage is up; below 90° it is down. Count one rep each time the stage goes from down to up. Press S to switch to squats: the hip angle from landmarks 11, 23 and 25, with 170° and 90°. Show the angle, stage and count on screen. Use strictly increasing millisecond timestamps.

## Prompt: with the starter

Use it in an AI coding tool opened on this repo.

> Open starter/rep_counter.py and finish the two TODOs. Keep the MediaPipe Tasks setup (PoseLandmarker in VIDEO mode) and the helpers in common.py. joint_angle(a, b, c) returns the angle at b in degrees, from 0 to 180, from pixel points, using math.atan2. RepCounter.update(angle): above `up` the stage is "up", below `down` it is "down", and count one rep each time the stage goes from down to up. Use the landmark numbers and thresholds in EXERCISES. Write "deg" on screen instead of the degree symbol.

## Before you run it: three checks

1. Does the count go up once per rep, not once per frame?
2. Are angles computed from pixel coordinates, not raw 0-1 values?
3. Are you side-on to the camera with your whole body in frame?

## If it breaks, paste one of these

| You see | Paste back |
|---|---|
| The count races up while you hold still | Only count when the stage changes from down to up, not every frame. |
| The angle jumps around | Skip frames where the joints' visibility is below 0.5, and use the side of the body the camera sees better. |
| It never counts | Print the angle every frame so I can tune the thresholds. |
| `??` on screen where the degree sign should be | OpenCV's fonts can't draw the degree symbol. Write "deg" instead. |

## Finished version

```bash
python solutions/rep_counter.py    # S switch exercise, R reset, Q quit
```

It also picks whichever side of your body the camera sees better.

## Where the numbers come from

- Landmark numbers: [MediaPipe Pose Landmarker guide](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker).
- Push-up thresholds (elbow above 160° up, below 90° down): [Labellerr's AI push-up counter](https://www.labellerr.com/blog/ai-powered-pushup-counter-form-corrector/).
- Squat thresholds (hip above 170° standing, below 90° at the bottom): [liveTrain's MediaPipe squat code on Hugging Face](https://huggingface.co/spaces/pjxcharya/liveTrain/commit/14e399f9849262825b303a03a4152511168afea5).
- Filming from the side: [LearnOpenCV's AI Fitness Trainer](https://learnopencv.com/ai-fitness-trainer-using-mediapipe), which measures the angles in the image.

These are starting points. Tune them on yourself.
