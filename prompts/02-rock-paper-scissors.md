# Demo 2: Rock-paper-scissors vs. the computer (Gesture Recognizer)

No training needed: the Gesture Recognizer's built-in labels already cover the three moves.

| MediaPipe label | Move |
|---|---|
| `Closed_Fist` | Rock |
| `Open_Palm` | Paper |
| `Victory` | Scissors |
| anything else | Show again |

## Prompt: from scratch

Paste into any AI chat.

> Using the MediaPipe Tasks API (not mediapipe.solutions), build a Python rock-paper-scissors game with GestureRecognizer in VIDEO mode. The model is at ./models/gesture_recognizer.task. Read the webcam with OpenCV and mirror it. Map the built-in gestures Closed_Fist → rock, Open_Palm → paper and Victory → scissors; ignore everything else. Each round: run a 3-2-1 countdown without pausing the video, pick the computer's move with random.choice, then lock in my move once the same gesture shows up in 3 frames in a row. Show both moves, the winner and the score. First to 3 wins. Use strictly increasing millisecond timestamps.

## Prompt: with the starter

Use it in an AI coding tool opened on this repo.

> Open starter/rock_paper_scissors.py and finish the three TODOs. Keep the MediaPipe Tasks setup (GestureRecognizer in VIDEO mode with recognize_for_video) and the helpers in common.py. Fill GESTURE_TO_MOVE with Closed_Fist → rock, Open_Palm → paper and Victory → scissors. Write round_winner(). Then add the game: SPACE starts it; each round shows a 3-2-1 countdown timed with time.monotonic() (no time.sleep), the computer picks random.choice(list(BEATS)) when the countdown starts, and after "Shoot!" my move locks in once the same move appears in 3 frames in a row (give up after 3 seconds). Show both moves, the winner and the score; first to 3 wins. Keep on-screen text ASCII.

## Before you run it: three checks

1. Does it call `recognize_for_video`? `detect_for_video` is the Hand Landmarker's method.
2. Do the labels match exactly? `Closed_Fist`, not "fist".
3. Does the countdown use a timer instead of `time.sleep()`, so the video keeps running?

## If it breaks, paste one of these

| You see | Paste back |
|---|---|
| `AttributeError: ... has no attribute 'detect_for_video'` | GestureRecognizer uses recognize_for_video, not detect_for_video. |
| Your move never registers | Print the gesture label and score every frame. Check the label strings match exactly and the score threshold isn't too high. |
| The video freezes during the countdown | Replace time.sleep with a timer based on time.monotonic() so the camera loop keeps running. |
| The move flickers between two answers | Only lock in a move after it's the same for 3 frames in a row. |

## Finished version

```bash
python solutions/rock_paper_scissors.py    # SPACE start, Q quit
```

## Where the ideas come from

- The built-in labels: [MediaPipe Gesture Recognizer guide](https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer).
- Waiting for the same gesture in several frames in a row: [andypotato's rock-paper-scissors post on DEV](https://dev.to/andypotato/rock-paper-scissors-with-hand-gesture-recognition-465g).
- Want your own gestures? Google's [Model Maker gesture tutorial](https://developers.google.com/edge/mediapipe/solutions/customization/gesture_recognizer) retrains the model on a rock-paper-scissors dataset.
