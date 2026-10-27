"""Logic tests for the three demos. No webcam or model files needed.

Run from the repo root:  python -m pytest tests
"""

from __future__ import annotations

from types import SimpleNamespace as Fake

import numpy as np
import pytest

from conftest import fake_hand, fake_pose

# --- Demo 1: air drawing -----------------------------------------------------


def test_fingers_up(air):
    assert air.fingers_up(fake_hand((True, False, False, False))) == [True, False, False, False]
    assert air.fingers_up(fake_hand((True, True, True, True))) == [True, True, True, True]
    assert air.fingers_up(fake_hand((False, False, False, False))) == [False, False, False, False]


@pytest.mark.parametrize("up, mode", [
    ([True, False, False, False], "draw"),
    ([True, True, False, False], "lift"),
    ([False, False, False, False], "clear"),
    ([True, True, True, True], "lift"),
    ([False, True, False, False], "lift"),
])
def test_choose_mode(air, up, mode):
    assert air.choose_mode(up) == mode


def test_painter_draws_and_lifting_prevents_jumps(air):
    painter = air.Painter((100, 200, 3))
    painter.update("draw", (10, 10))
    assert not painter.canvas.any()             # first point: nothing to connect yet
    painter.update("draw", (50, 10))
    assert painter.canvas[10, 30].any()         # a line was drawn
    painter.update("lift", (60, 60))
    assert painter.last_point is None
    painter.update("draw", (150, 90))           # a new stroke starts here
    assert not painter.canvas[50, 100].any()    # no line jumped from (50, 10)


def test_clear_needs_a_held_fist(air):
    painter = air.Painter((100, 200, 3))
    painter.update("draw", (10, 10))
    painter.update("draw", (80, 10))
    for _ in range(air.CLEAR_HOLD_FRAMES - 1):
        painter.update("clear", (0, 0))
    assert painter.canvas.any()                 # a quick fist doesn't wipe it
    painter.update("clear", (0, 0))
    assert not painter.canvas.any()


def test_canvas_overlay_only_replaces_drawn_pixels(air):
    painter = air.Painter((10, 10, 3))
    painter.canvas[2, 3] = (1, 2, 3)
    frame = np.full((10, 10, 3), 200, np.uint8)
    out = painter.draw_on(frame)
    assert out[2, 3].tolist() == [1, 2, 3]
    assert out[0, 0].tolist() == [200, 200, 200]


# --- Demo 2: rock-paper-scissors ---------------------------------------------


@pytest.mark.parametrize("player, computer, winner", [
    ("rock", "scissors", "player"), ("scissors", "paper", "player"), ("paper", "rock", "player"),
    ("scissors", "rock", "computer"), ("paper", "scissors", "computer"), ("rock", "paper", "computer"),
    ("rock", "rock", "tie"), ("paper", "paper", "tie"), ("scissors", "scissors", "tie"),
])
def test_round_winner(rps, player, computer, winner):
    assert rps.round_winner(player, computer) == winner


def test_gesture_mapping_uses_mediapipe_labels(rps):
    assert rps.GESTURE_TO_MOVE == {"Closed_Fist": "rock", "Open_Palm": "paper", "Victory": "scissors"}
    assert rps.GESTURE_TO_MOVE.get("Thumb_Up") is None
    assert rps.GESTURE_TO_MOVE.get(None) is None


def test_top_gesture(rps):
    sure = Fake(gestures=[[Fake(category_name="Victory", score=0.9)]])
    unsure = Fake(gestures=[[Fake(category_name="Victory", score=0.3)]])
    assert rps.top_gesture(sure) == "Victory"
    assert rps.top_gesture(unsure) is None
    assert rps.top_gesture(Fake(gestures=[])) is None


def test_move_lock_needs_three_in_a_row(rps):
    lock = rps.MoveLock(3)
    assert [lock.update("rock") for _ in range(3)] == [None, None, "rock"]
    lock.reset()
    lock.update("rock")
    lock.update("paper")                         # switching restarts the count
    assert lock.update("paper") is None
    assert lock.update("paper") == "paper"
    lock.reset()
    lock.update("rock")
    lock.update(None)                            # losing the hand breaks the streak
    lock.update("rock")
    assert lock.update("rock") is None


class FixedChoice:
    """Stands in for `random` so the computer always picks the same move."""

    def __init__(self, move):
        self.move = move

    def choice(self, options):
        assert self.move in options
        return self.move


def play_round(rps, game, now, move):
    game.update(now, None)                       # still counting down
    now += rps.COUNTDOWN_S
    game.update(now, None)
    assert game.state == "shoot"
    for _ in range(rps.LOCK_FRAMES):
        now += 0.03
        game.update(now, move)
    assert game.state == "result"
    return now


def test_full_game_first_to_three(rps):
    game = rps.Game(rng=FixedChoice("scissors"))
    assert game.state == "ready"
    game.press_space(0.0)
    assert game.state == "countdown" and game.computer_move == "scissors"
    now = 0.0
    for round_number in (1, 2, 3):
        now = play_round(rps, game, now, "rock")
        assert game.outcome == "player"
        assert game.score["player"] == round_number
        now += rps.RESULT_S
        game.update(now, None)
    assert game.state == "over"
    game.press_space(now + 1)
    assert game.state == "countdown"
    assert game.score == {"player": 0, "computer": 0}


def test_no_move_times_out_without_points(rps):
    game = rps.Game(rng=FixedChoice("rock"))
    game.press_space(0.0)
    game.update(rps.COUNTDOWN_S, None)
    game.update(rps.COUNTDOWN_S + rps.SHOOT_TIMEOUT_S, None)
    assert game.state == "result"
    assert game.outcome == "no move"
    assert game.score == {"player": 0, "computer": 0}


def test_countdown_numbers(rps):
    game = rps.Game(rng=FixedChoice("rock"))
    game.press_space(10.0)
    assert [game.countdown_number(10.0 + s) for s in (0.0, 0.99, 1.0, 2.5)] == [3, 3, 2, 1]


# --- Demo 3: rep counter -----------------------------------------------------


def test_joint_angle(rep):
    assert rep.joint_angle((1, 0), (0, 0), (0, 1)) == pytest.approx(90)
    assert rep.joint_angle((-1, 0), (0, 0), (1, 0)) == pytest.approx(180)
    assert rep.joint_angle((1, 0), (0, 0), (1, 1)) == pytest.approx(45)
    assert rep.joint_angle((1, 1), (0, 0), (1, 0)) == pytest.approx(45)
    # Two points either side of the +/-180 degree seam are 1.15 degrees apart, not 358.85.
    assert rep.joint_angle((-1, 0.01), (0, 0), (-1, -0.01)) == pytest.approx(1.146, abs=0.01)


def test_rep_counter_counts_only_full_reps(rep):
    counter = rep.RepCounter(up=160, down=90)
    for angle in (175, 150, 120, 95):             # never went below 90: no rep
        counter.update(angle)
    assert counter.count == 0
    for angle in (85, 120, 165):                  # down, then back up: one rep
        counter.update(angle)
    assert counter.count == 1
    for angle in (100, 165, 170):                 # didn't go down far enough
        counter.update(angle)
    assert counter.count == 1
    for angle in (80, 70, 170, 80, 170):
        counter.update(angle)
    assert counter.count == 3


def test_starting_in_the_up_position_is_not_a_rep(rep):
    counter = rep.RepCounter(160, 90)
    counter.update(170)
    assert counter.count == 0 and counter.stage == "up"


def test_visible_side(rep):
    pushup = rep.EXERCISES["push-up"]
    assert rep.visible_side(fake_pose(left_visibility=0.9, right_visibility=0.2), pushup) == "left"
    assert rep.visible_side(fake_pose(left_visibility=0.1, right_visibility=0.95), pushup) == "right"


def test_missing_visibility_counts_as_visible(rep):
    assert rep.visibility(Fake(visibility=None)) == 1.0
    assert rep.visibility(Fake(visibility=0.3)) == 0.3


def test_exercise_landmarks_match_mediapipe_names(rep):
    vision = pytest.importorskip("mediapipe.tasks.python.vision")
    if not hasattr(vision, "PoseLandmark"):
        pytest.skip("this mediapipe version has no PoseLandmark enum")
    names = {int(m): m.name for m in vision.PoseLandmark}
    pushup, squat = rep.EXERCISES["push-up"], rep.EXERCISES["squat"]
    assert [names[i] for i in pushup["left"]] == ["LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"]
    assert [names[i] for i in pushup["right"]] == ["RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"]
    assert [names[i] for i in squat["left"]] == ["LEFT_SHOULDER", "LEFT_HIP", "LEFT_KNEE"]
    assert [names[i] for i in squat["right"]] == ["RIGHT_SHOULDER", "RIGHT_HIP", "RIGHT_KNEE"]
