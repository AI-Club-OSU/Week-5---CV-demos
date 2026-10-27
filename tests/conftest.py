"""Shared test helpers: load the demo scripts as modules, and fake landmarks."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace as Fake

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_module(relative_path: str):
    """Import a script like solutions/air_drawing.py without making packages."""
    path = ROOT / relative_path
    name = relative_path.replace("/", "_").removesuffix(".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def air():
    return load_module("solutions/air_drawing.py")


@pytest.fixture(scope="session")
def rps():
    return load_module("solutions/rock_paper_scissors.py")


@pytest.fixture(scope="session")
def rep():
    return load_module("solutions/rep_counter.py")


def fake_hand(up=(False, False, False, False), tip_x: float = 0.5):
    """21 hand landmarks. Each finger's tip sits above (up) or below its joint."""
    hand = [Fake(x=0.5, y=0.7, z=0.0, visibility=None) for _ in range(21)]
    for is_up, tip, joint in zip(up, (8, 12, 16, 20), (6, 10, 14, 18)):
        hand[joint] = Fake(x=0.5, y=0.5, z=0.0, visibility=None)
        hand[tip] = Fake(x=0.5, y=0.3 if is_up else 0.6, z=0.0, visibility=None)
    hand[8] = Fake(x=tip_x, y=hand[8].y, z=0.0, visibility=None)
    return hand


def fake_pose(left_visibility: float = 0.99, right_visibility: float = 0.2):
    """33 pose landmarks with the left side clearly visible by default."""
    pose = [Fake(x=0.5, y=0.5, z=0.0, visibility=left_visibility) for _ in range(33)]
    for i in range(12, 33, 2):  # even ids from 12 up are the right side of the body
        pose[i] = Fake(x=0.5, y=0.5, z=0.0, visibility=right_visibility)
    return pose
