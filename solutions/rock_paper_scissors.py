"""Demo 2, finished: rock-paper-scissors vs. the computer with the Gesture Recognizer.

Run from the repo root:  python solutions/rock_paper_scissors.py
Keys: SPACE start / play again, Q quit. First to 3 wins.

No training needed: MediaPipe's built-in gestures Closed_Fist, Open_Palm and
Victory map to rock, paper and scissors. A move only locks in after it shows
up in 3 frames in a row, which stops flicker between gestures (the idea comes
from andypotato's rock-paper-scissors post on DEV).
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # lets us import common.py

import cv2
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from common import (FrameClock, draw_big_text, draw_hand, draw_text, model_path,
                    open_camera, should_quit, to_mp_image)

WINDOW = "Rock-paper-scissors (SPACE to start, Q to quit)"
GESTURE_TO_MOVE = {"Closed_Fist": "rock", "Open_Palm": "paper", "Victory": "scissors"}
BEATS = {"rock": "scissors", "scissors": "paper", "paper": "rock"}  # key beats value
MIN_SCORE = 0.6         # ignore gestures the model isn't sure about
LOCK_FRAMES = 3         # same move in 3 frames in a row = locked in
COUNTDOWN_S = 3.0
SHOOT_TIMEOUT_S = 3.0   # how long to wait for a move after "Shoot!"
RESULT_S = 2.5
WIN_SCORE = 3


def top_gesture(result, min_score: float = MIN_SCORE) -> str | None:
    """The most likely gesture label for the first hand, or None."""
    if result.gestures and result.gestures[0]:
        best = result.gestures[0][0]
        if best.score >= min_score:
            return best.category_name
    return None


def round_winner(player: str, computer: str) -> str:
    if player == computer:
        return "tie"
    return "player" if BEATS[player] == computer else "computer"


class MoveLock:
    """Returns a move only once it has shown up in `frames` frames in a row."""

    def __init__(self, frames: int = LOCK_FRAMES) -> None:
        self.frames = frames
        self.reset()

    def reset(self) -> None:
        self.candidate: str | None = None
        self.count = 0

    def update(self, move: str | None) -> str | None:
        if move is None:
            self.reset()
            return None
        if move == self.candidate:
            self.count += 1
        else:
            self.candidate, self.count = move, 1
        return move if self.count >= self.frames else None


class Game:
    """States: ready -> countdown -> shoot -> result -> (next round or over)."""

    def __init__(self, rng=random) -> None:
        self.rng = rng
        self.lock = MoveLock()
        self.reset()

    def reset(self) -> None:
        self.state = "ready"
        self.score = {"player": 0, "computer": 0}
        self.player_move: str | None = None
        self.computer_move: str | None = None
        self.outcome: str | None = None
        self.started = 0.0

    def press_space(self, now: float) -> None:
        if self.state in ("ready", "over"):
            self.reset()
            self.start_round(now)

    def start_round(self, now: float) -> None:
        self.state, self.started = "countdown", now
        self.computer_move = self.rng.choice(list(BEATS))  # picked now, revealed later
        self.player_move = self.outcome = None
        self.lock.reset()

    def countdown_number(self, now: float) -> int:
        return max(1, int(COUNTDOWN_S) - int(now - self.started))

    def update(self, now: float, move: str | None) -> None:
        elapsed = now - self.started
        if self.state == "countdown" and elapsed >= COUNTDOWN_S:
            self.state, self.started = "shoot", now
        elif self.state == "shoot":
            locked = self.lock.update(move)
            if locked:
                self.finish_round(now, locked)
            elif elapsed >= SHOOT_TIMEOUT_S:
                self.finish_round(now, None)
        elif self.state == "result" and elapsed >= RESULT_S:
            if max(self.score.values()) >= WIN_SCORE:
                self.state = "over"
            else:
                self.start_round(now)

    def finish_round(self, now: float, player_move: str | None) -> None:
        self.player_move = player_move
        if player_move is None:
            self.outcome = "no move"  # nobody scores; the next round starts
        else:
            self.outcome = round_winner(player_move, self.computer_move)
            if self.outcome != "tie":
                self.score[self.outcome] += 1
        self.state, self.started = "result", now


RESULT_TEXT = {
    "player": "You win the round!",
    "computer": "Computer wins the round",
    "tie": "Tie!",
    "no move": "No move seen - again!",
}


def draw_game(frame, game: Game, now: float, gesture: str | None) -> None:
    draw_text(frame, f"You {game.score['player']} - {game.score['computer']} Computer", (30, 50), scale=1.1)
    draw_text(frame, f"seeing: {gesture or 'no gesture'}", (30, frame.shape[0] - 30), scale=0.7)

    if game.state == "ready":
        draw_big_text(frame, "SPACE to start", scale=2.0)
    elif game.state == "countdown":
        draw_big_text(frame, str(game.countdown_number(now)), scale=6.0)
    elif game.state == "shoot":
        draw_big_text(frame, "Shoot!", scale=4.0)
    elif game.state == "result":
        draw_text(frame, f"You: {game.player_move or 'nothing'}   Computer: {game.computer_move}", (30, 110))
        draw_big_text(frame, RESULT_TEXT[game.outcome], scale=1.8)
    elif game.state == "over":
        winner = "You win!" if game.score["player"] > game.score["computer"] else "Computer wins!"
        draw_big_text(frame, winner, scale=3.0)
        draw_text(frame, "SPACE to play again", (30, 110))


def main() -> None:
    base = mp_python.BaseOptions(model_asset_path=model_path("gesture_recognizer.task"))
    options = vision.GestureRecognizerOptions(
        base_options=base,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
    )
    cap, clock = open_camera(), FrameClock()
    game = Game()

    with vision.GestureRecognizer.create_from_options(options) as recognizer:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # mirror before detecting

            result = recognizer.recognize_for_video(to_mp_image(frame), clock.now_ms())
            gesture = top_gesture(result)
            now = time.monotonic()
            game.update(now, GESTURE_TO_MOVE.get(gesture))

            if result.hand_landmarks:
                draw_hand(frame, result.hand_landmarks[0])
            draw_game(frame, game, now, gesture)
            cv2.imshow(WINDOW, frame)

            key = cv2.waitKey(1) & 0xFF
            if should_quit(key):
                break
            if key == ord(" "):
                game.press_space(time.monotonic())

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
