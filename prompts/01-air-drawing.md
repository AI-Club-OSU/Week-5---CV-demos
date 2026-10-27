# Demo 1: Air drawing (Hand Landmarker)

Your index finger is the pen.

| Fingers up | What happens |
|---|---|
| Index only | Draw a line from the last point to tip 8 |
| Index + middle | Lift the pen |
| None (hold a fist) | Clear the canvas |

A finger is up when its tip is higher on screen (smaller y) than the joint two
landmarks below it: 8 vs 6, 12 vs 10, 16 vs 14, 20 vs 18.

## Prompt: from scratch

Paste into any AI chat.

> Using the MediaPipe Tasks API (not mediapipe.solutions), build a Python air-drawing app with HandLandmarker in VIDEO mode, one hand. The model is at ./models/hand_landmarker.task. Read the webcam with OpenCV and mirror it before detecting. A finger is up when its tip is higher on screen than the joint two landmarks below it (8 vs 6, 12 vs 10, 16 vs 14, 20 vs 18). Index up only: draw on a separate canvas from the last index-tip point to landmark 8. Index and middle up: lift the pen. No fingers up: clear the canvas. Overlay the canvas on the video. Use strictly increasing millisecond timestamps.

## Prompt: with the starter

Use it in an AI coding tool opened on this repo.

> Open starter/air_drawing.py and finish the three TODOs. Keep the MediaPipe Tasks API setup that's already there (never use mediapipe.solutions) and reuse the helpers in common.py. fingers_up() returns [index, middle, ring, pinky]: a finger is up when its tip has a smaller y than the joint two landmarks below it (8 vs 6, 12 vs 10, 16 vs 14, 20 vs 18). choose_mode() returns "draw" when only the index finger is up, "clear" when no fingers are up, and "lift" otherwise. In the loop: on "draw", draw a line on canvas from last_point to the index tip and update last_point; otherwise set last_point to None; on "clear", wipe the canvas. Then tell me which landmark numbers you used and why.

## Before you run it: three checks

1. Does it multiply x and y by the frame size before drawing?
2. Does the last point reset when the pen lifts? If not, lines jump across the screen.
3. Is the frame mirrored before detection, so the dots line up?

## If it breaks, paste one of these

| You see | Paste back |
|---|---|
| `AttributeError: module 'mediapipe' has no attribute 'solutions'` | Rewrite this with the MediaPipe Tasks API (from mediapipe.tasks.python import vision). mediapipe.solutions was removed in 0.10.31. |
| A line jumps across the screen when you start drawing again | Reset the last point to None whenever the pen lifts or the hand disappears. |
| The canvas clears when you just move your hand | Only clear after a fist is held for 10 frames in a row. |
| The dots are in the top-left corner | Landmarks are 0-1. Multiply x by the frame width and y by the frame height. |

## Finished version

```bash
python solutions/air_drawing.py    # 1-4 change color, C clear, Q quit
```

## Where the logic comes from

Educative's [Create a virtual paint app using MediaPipe](https://www.educative.io/answers/create-a-virtual-paint-app-using-mediapipe):
index finger only draws, index and middle select, all fingers down clears. It uses
the legacy API, but the landmark numbers are the same in the Tasks API.
