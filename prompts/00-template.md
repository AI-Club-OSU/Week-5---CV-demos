# The template

Fill in the brackets and paste.

> Using the MediaPipe Tasks API (not mediapipe.solutions), build a Python app with **[TASK]** in VIDEO mode. The model file is at **./models/[file].task**. Read the webcam with OpenCV and mirror it. Draw the landmarks with cv2.circle. When **[LANDMARK LOGIC]**, **[ACTION]**. Use strictly increasing millisecond timestamps.

| Bracket | Fill it with |
|---|---|
| [TASK] | `HandLandmarker`, `GestureRecognizer` or `PoseLandmarker` |
| [file] | `hand_landmarker`, `gesture_recognizer` or `pose_landmarker_lite` |
| [LANDMARK LOGIC] | Something measurable, like "the index fingertip (8) is above its joint (6)", "the gesture is Thumb_Up" or "the elbow angle drops below 90 degrees" |
| [ACTION] | What happens then: draw, show text, count, play a sound |

The three demo prompts are this template filled in.

## Three habits

1. When it breaks, paste the full error back word for word.
2. Ask it to explain its landmark math before you run it.
3. Point it at sources: the MediaPipe docs, this repo's `AGENTS.md`, or add
   "use context7" if your tool has the Context7 MCP server.
