"""Randomised challenge-response liveness.

Blink detection alone does not prove liveness. It proves a blink happened, and
a replayed video of someone blinking satisfies it perfectly. The original
system this project grew out of named exactly that as its limitation.

The fix is to make the thing being asked for unpredictable. A challenge is
drawn at random *at the moment it is issued*, from a pool of actions, in a
random order, with a randomised repeat count. Footage recorded beforehand
cannot anticipate the sequence, so satisfying it requires a person present and
responding now.

Three rules give the randomness its teeth:

1. Evidence only counts after the challenge is issued. A blink two frames
   before the prompt appeared proves nothing about the prompt.
2. The face must return to neutral between challenges, so one held pose cannot
   satisfy two consecutive challenges.
3. Each challenge has a deadline. Without one, an attacker could cycle through
   poses until they happened to match.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum

from config import (
    CHALLENGE_COUNT,
    CHALLENGE_RETRY_S,
    CHALLENGE_TIMEOUT_S,
    MAX_BLINK_REPEATS,
)
from engine.liveness import Baseline, FaceMetrics


class ChallengeKind(str, Enum):
    BLINK = "blink"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    SMILE = "smile"


class LivenessState(str, Enum):
    BASELINE = "baseline"    # learning the person's resting geometry
    ACTIVE = "active"        # a challenge is on screen awaiting a response
    PASSED = "passed"
    FAILED = "failed"


@dataclass
class Challenge:
    kind: ChallengeKind
    target: int = 1          # how many times the action must be performed
    progress: int = 0
    issued_at: float = 0.0

    @property
    def prompt(self) -> str:
        if self.kind is ChallengeKind.BLINK:
            if self.target == 1:
                return "Blink"
            return f"Blink {self.target} times ({self.progress}/{self.target})"
        return {
            ChallengeKind.TURN_LEFT: "Turn your head left",
            ChallengeKind.TURN_RIGHT: "Turn your head right",
            ChallengeKind.SMILE: "Smile",
        }[self.kind]

    @property
    def satisfied(self) -> bool:
        return self.progress >= self.target

    def deadline(self) -> float:
        return self.issued_at + CHALLENGE_TIMEOUT_S


def _draw_sequence(rng: random.Random, count: int) -> list[Challenge]:
    """Pick a random, non-repeating sequence of challenges.

    Sampling without replacement avoids asking for the same action twice in a
    row, which would be both irritating and weaker evidence - a single held
    smile should never clear two smile challenges.
    """
    kinds = rng.sample(list(ChallengeKind), k=min(count, len(ChallengeKind)))
    return [
        Challenge(
            kind=kind,
            target=rng.randint(1, MAX_BLINK_REPEATS) if kind is ChallengeKind.BLINK else 1,
        )
        for kind in kinds
    ]


class ChallengeSession:
    """Runs one person through a liveness challenge sequence.

    One session belongs to one track. It is created when that track's identity
    becomes stable and discarded with the track, so a person who walks away and
    returns is challenged afresh rather than inheriting old progress.
    """

    def __init__(self, rng: random.Random | None = None, count: int = CHALLENGE_COUNT) -> None:
        self._rng = rng or random.Random()
        self._count = count
        self._sequence = _draw_sequence(self._rng, count)
        self._index = 0
        self.baseline = Baseline()
        self.state = LivenessState.BASELINE

        # Blink counting is edge-triggered: a blink is the eyes closing and
        # then reopening. Level-triggering on "eyes are closed" would count one
        # long blink many times over.
        self._eyes_were_closed = False

        # Set after a challenge is satisfied and cleared once the face returns
        # to neutral, gating the next challenge behind a reset.
        self._awaiting_neutral = False

        self.failure_reason: str | None = None
        self._failed_at: float | None = None
        self.attempts = 1

    # --- inspection ------------------------------------------------------

    @property
    def current(self) -> Challenge | None:
        if self.state is not LivenessState.ACTIVE or self._index >= len(self._sequence):
            return None
        return self._sequence[self._index]

    @property
    def completed(self) -> int:
        return self._index

    @property
    def total(self) -> int:
        return len(self._sequence)

    @property
    def prompt(self) -> str:
        if self.state is LivenessState.BASELINE:
            return "Hold still"
        if self.state is LivenessState.PASSED:
            return "Verified"
        if self.state is LivenessState.FAILED:
            return self.failure_reason or "Verification failed"
        if self._awaiting_neutral:
            return "Face the camera"
        current = self.current
        return current.prompt if current else ""

    # --- driving ---------------------------------------------------------

    def update(self, metrics: FaceMetrics, now: float) -> LivenessState:
        """Advance the session by one frame."""
        if self.state is LivenessState.PASSED:
            return self.state

        if self.state is LivenessState.FAILED:
            self._maybe_retry(now)
            if self.state is LivenessState.FAILED:
                return self.state

        if not metrics.valid:
            return self.state

        if self.state is LivenessState.BASELINE:
            self._collect_baseline(metrics, now)
            return self.state

        self._track_blink_edges(metrics)

        # Returning to neutral gates the next challenge, so a single sustained
        # pose cannot be reused to satisfy the one that follows.
        if self._awaiting_neutral:
            if self._is_neutral(metrics):
                self._awaiting_neutral = False
                self._issue_next(now)
            return self.state

        challenge = self.current
        if challenge is None:
            return self.state

        if now > challenge.deadline():
            self.state = LivenessState.FAILED
            self.failure_reason = "Timed out - try again"
            self._failed_at = now
            return self.state

        if self._is_satisfied_this_frame(challenge, metrics):
            challenge.progress += 1

        if challenge.satisfied:
            self._index += 1
            if self._index >= len(self._sequence):
                self.state = LivenessState.PASSED
            else:
                self._awaiting_neutral = True

        return self.state

    # --- internals -------------------------------------------------------

    def _maybe_retry(self, now: float) -> None:
        """Restart a failed attempt once the cooldown has elapsed.

        Someone who misread a prompt or glanced away should not have to walk
        out of frame and return to try again. The retry draws an entirely new
        sequence, so repeated attempts leak nothing about what comes next.

        The baseline is kept: it describes this person's resting geometry,
        which a failed challenge says nothing about, and re-measuring it would
        make every retry slower for no gain.
        """
        if self._failed_at is None or now - self._failed_at < CHALLENGE_RETRY_S:
            return

        self._sequence = _draw_sequence(self._rng, self._count)
        self._index = 0
        self._eyes_were_closed = False
        self._awaiting_neutral = False
        self.failure_reason = None
        self._failed_at = None
        self.attempts += 1
        self.state = LivenessState.ACTIVE
        self._issue_next(now)

    def _collect_baseline(self, metrics: FaceMetrics, now: float) -> None:
        # Only neutral frames belong in a resting baseline; sampling while the
        # head is turned would bias it.
        if Baseline.is_facing_forward(metrics):
            self.baseline.observe(metrics)
        if self.baseline.ready:
            self.state = LivenessState.ACTIVE
            self._issue_next(now)

    def _issue_next(self, now: float) -> None:
        """Stamp the upcoming challenge with the time it became visible.

        This stamp is what makes evidence count only from the moment the person
        was actually asked.

        Half-finished blink state is discarded at the same moment. Without
        this, eyes that closed while the person was returning to neutral would
        still be "closed" when the next challenge appeared, and the very first
        frame of them reopening would score a blink they were never asked for.
        """
        self._eyes_were_closed = False
        if self._index < len(self._sequence):
            self._sequence[self._index].issued_at = now

    def _track_blink_edges(self, metrics: FaceMetrics) -> None:
        if self.baseline.eyes_closed(metrics):
            self._eyes_were_closed = True

    def _consume_blink(self, metrics: FaceMetrics) -> bool:
        """A completed blink: eyes were closed and have now reopened."""
        if self._eyes_were_closed and self.baseline.eyes_open(metrics):
            self._eyes_were_closed = False
            return True
        return False

    def _is_satisfied_this_frame(self, challenge: Challenge, metrics: FaceMetrics) -> bool:
        if challenge.kind is ChallengeKind.BLINK:
            return self._consume_blink(metrics)
        if challenge.kind is ChallengeKind.TURN_LEFT:
            return Baseline.is_turned(metrics, sign=-1)
        if challenge.kind is ChallengeKind.TURN_RIGHT:
            return Baseline.is_turned(metrics, sign=+1)
        if challenge.kind is ChallengeKind.SMILE:
            return self.baseline.is_smiling(metrics)
        return False

    def _is_neutral(self, metrics: FaceMetrics) -> bool:
        return (
            Baseline.is_facing_forward(metrics)
            and not self.baseline.is_smiling(metrics)
            and self.baseline.eyes_open(metrics)
        )
