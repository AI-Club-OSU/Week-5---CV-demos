# MediaPipe landmark numbers

Generated from the enums in the `mediapipe` package (`HandLandmark`, `PoseLandmark`),
so the names match the library exactly.

## Hand (21 landmarks, Hand Landmarker and Gesture Recognizer)

| # | Name | # | Name | # | Name |
|---|---|---|---|---|---|
| 0 | WRIST | 7 | INDEX_FINGER_DIP | 14 | RING_FINGER_PIP |
| 1 | THUMB_CMC | 8 | INDEX_FINGER_TIP | 15 | RING_FINGER_DIP |
| 2 | THUMB_MCP | 9 | MIDDLE_FINGER_MCP | 16 | RING_FINGER_TIP |
| 3 | THUMB_IP | 10 | MIDDLE_FINGER_PIP | 17 | PINKY_MCP |
| 4 | THUMB_TIP | 11 | MIDDLE_FINGER_DIP | 18 | PINKY_PIP |
| 5 | INDEX_FINGER_MCP | 12 | MIDDLE_FINGER_TIP | 19 | PINKY_DIP |
| 6 | INDEX_FINGER_PIP | 13 | RING_FINGER_MCP | 20 | PINKY_TIP |

Fingertips: 4, 8, 12, 16, 20. The joint two landmarks below each of the four
fingertips (used for "is this finger up?"): 6, 10, 14, 18.

## Pose (33 landmarks, Pose Landmarker)

Odd numbers are the person's left side, even numbers their right.

| # | Name | # | Name | # | Name |
|---|---|---|---|---|---|
| 0 | NOSE | 11 | LEFT_SHOULDER | 22 | RIGHT_THUMB |
| 1 | LEFT_EYE_INNER | 12 | RIGHT_SHOULDER | 23 | LEFT_HIP |
| 2 | LEFT_EYE | 13 | LEFT_ELBOW | 24 | RIGHT_HIP |
| 3 | LEFT_EYE_OUTER | 14 | RIGHT_ELBOW | 25 | LEFT_KNEE |
| 4 | RIGHT_EYE_INNER | 15 | LEFT_WRIST | 26 | RIGHT_KNEE |
| 5 | RIGHT_EYE | 16 | RIGHT_WRIST | 27 | LEFT_ANKLE |
| 6 | RIGHT_EYE_OUTER | 17 | LEFT_PINKY | 28 | RIGHT_ANKLE |
| 7 | LEFT_EAR | 18 | RIGHT_PINKY | 29 | LEFT_HEEL |
| 8 | RIGHT_EAR | 19 | LEFT_INDEX | 30 | RIGHT_HEEL |
| 9 | MOUTH_LEFT | 20 | RIGHT_INDEX | 31 | LEFT_FOOT_INDEX |
| 10 | MOUTH_RIGHT | 21 | LEFT_THUMB | 32 | RIGHT_FOOT_INDEX |

Pose landmarks also have a `visibility` score from 0 to 1. Use it to skip joints
the camera can't see.

## Built-in gestures (Gesture Recognizer)

`None`, `Closed_Fist`, `Open_Palm`, `Pointing_Up`, `Thumb_Down`, `Thumb_Up`,
`Victory`, `ILoveYou`. Exact spelling, case-sensitive.

## Coordinates

`x` and `y` are normalized to 0-1 by the frame's width and height. `z` is depth,
with the wrist as the origin for hands. Multiply x by the frame width and y by the
frame height to get pixels.
