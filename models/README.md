# Models

The MediaPipe model files go here. From the repo root, run:

```bash
python download_models.py
```

It downloads these from Google's official MediaPipe model storage:

| File | Used by |
|---|---|
| `hand_landmarker.task` | `see_yourself.py`, demo 1 (air drawing) |
| `gesture_recognizer.task` | demo 2 (rock-paper-scissors) |
| `pose_landmarker_lite.task` | demo 3 (rep counter) |

If the script can't connect, it prints the links so you can download the files in
a browser and save them in this folder.
