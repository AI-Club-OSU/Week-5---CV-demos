"""Checks that the MediaPipe Tasks API calls in this repo exist in the installed
version, and that the shared helpers behave. No webcam or model files needed."""

from __future__ import annotations

import re
import sys
from types import SimpleNamespace as Fake

import numpy as np
import pytest
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import hand_landmarker

import common
import download_models
from conftest import ROOT


def test_video_mode_methods_exist():
    assert callable(vision.HandLandmarker.detect_for_video)
    assert callable(vision.PoseLandmarker.detect_for_video)
    assert callable(vision.GestureRecognizer.recognize_for_video)


@pytest.mark.parametrize("options_class, count_field", [
    (vision.HandLandmarkerOptions, "num_hands"),
    (vision.GestureRecognizerOptions, "num_hands"),
    (vision.PoseLandmarkerOptions, "num_poses"),
])
def test_options_build(options_class, count_field):
    base = mp_python.BaseOptions(model_asset_path="models/example.task")
    options = options_class(base_options=base, running_mode=vision.RunningMode.VIDEO, **{count_field: 1})
    assert options.running_mode == vision.RunningMode.VIDEO


def test_missing_model_file_fails_loudly():
    base = mp_python.BaseOptions(model_asset_path=str(ROOT / "models" / "does_not_exist.task"))
    options = vision.HandLandmarkerOptions(base_options=base, running_mode=vision.RunningMode.VIDEO)
    with pytest.raises(Exception):
        vision.HandLandmarker.create_from_options(options)


def test_hand_connections_match_mediapipe():
    ours = {tuple(sorted(pair)) for pair in common.HAND_CONNECTIONS}
    theirs = {tuple(sorted((c.start, c.end))) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS}
    # Older releases join the palm at 1-5 instead of 0-5; everything else matches.
    assert len(ours ^ theirs) <= 2


def test_pose_connections_are_real_mediapipe_pose_lines():
    theirs = {tuple(sorted((c.start, c.end))) for c in vision.PoseLandmarksConnections.POSE_LANDMARKS}
    assert {tuple(sorted(pair)) for pair in common.POSE_CONNECTIONS} <= theirs


def test_hand_landmark_numbers_used_in_demos():
    names = {int(m): m.name for m in hand_landmarker.HandLandmark}
    assert [names[i] for i in (8, 12, 16, 20)] == [
        "INDEX_FINGER_TIP", "MIDDLE_FINGER_TIP", "RING_FINGER_TIP", "PINKY_TIP"]
    assert [names[i] for i in (6, 10, 14, 18)] == [
        "INDEX_FINGER_PIP", "MIDDLE_FINGER_PIP", "RING_FINGER_PIP", "PINKY_PIP"]


def test_to_mp_image_converts_bgr_to_rgb():
    frame = np.zeros((480, 640, 3), np.uint8)
    frame[..., 0] = 255  # pure blue in OpenCV's BGR order
    image = common.to_mp_image(frame)
    assert (image.width, image.height) == (640, 480)
    assert image.numpy_view()[0, 0].tolist() == [0, 0, 255]  # blue is last in RGB


def test_frame_clock_always_increases():
    clock = common.FrameClock()
    stamps = [clock.now_ms() for _ in range(2000)]
    assert all(later > earlier for earlier, later in zip(stamps, stamps[1:]))


def test_to_px():
    frame = np.zeros((720, 1280, 3), np.uint8)
    assert common.to_px(Fake(x=0.5, y=0.25), frame) == (640, 180)


def test_drawing_helpers_draw():
    hand = [Fake(x=0.3 + 0.01 * i, y=0.3 + 0.01 * i) for i in range(21)]
    pose = [Fake(x=0.2 + 0.015 * i, y=0.2 + 0.01 * i) for i in range(33)]
    for draw, points in ((common.draw_hand, hand), (common.draw_pose, pose)):
        frame = np.zeros((720, 1280, 3), np.uint8)
        draw(frame, points)
        assert frame.any()
    frame = np.zeros((720, 1280, 3), np.uint8)
    common.draw_text(frame, "hello", (30, 50))
    common.draw_big_text(frame, "3")
    assert frame.any()


def test_missing_model_exits_with_download_tip(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "MODELS_DIR", tmp_path)
    with pytest.raises(SystemExit) as info:
        common.model_path("hand_landmarker.task")
    assert "download_models.py" in str(info.value)


def test_camera_index(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["demo.py", "--camera", "2"])
    assert common.camera_index() == 2
    monkeypatch.setattr(sys, "argv", ["demo.py"])
    assert common.camera_index() == 0


def test_should_quit():
    assert common.should_quit(ord("q")) and common.should_quit(27)
    assert not common.should_quit(ord(" ")) and not common.should_quit(255)


def test_every_model_used_is_downloadable():
    used = set()
    for path in ROOT.rglob("*.py"):
        if ".venv" in path.parts or "tests" in path.parts:
            continue
        used |= set(re.findall(r'model_path\("([\w.]+\.task)"\)', path.read_text(encoding="utf-8")))
    assert used == set(download_models.MODELS)
    for url in download_models.MODELS.values():
        assert url.startswith("https://storage.googleapis.com/mediapipe-models/")


def test_screen_text_is_ascii():
    """OpenCV's built-in fonts can't draw characters like a degree sign or arrows."""
    pattern = re.compile(r'draw_(?:big_)?text\(\s*frame,\s*(f?"[^"]*")')
    for path in ROOT.rglob("*.py"):
        if ".venv" in path.parts:
            continue
        for literal in pattern.findall(path.read_text(encoding="utf-8")):
            assert literal.isascii(), f"{path.name}: {literal}"
