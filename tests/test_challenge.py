"""Tests for randomised challenge-response liveness.

This is the security-critical part of the system, so the tests are written
around the properties an attacker would try to violate rather than around the
happy path alone.
"""

import random

import pytest

from config import BASELINE_SAMPLES, CHALLENGE_TIMEOUT_S
from engine.challenge import Challenge, ChallengeKind, ChallengeSession, LivenessState
from engine.liveness import FaceMetrics

NEUTRAL = FaceMetrics(eye_openness=0.30, smile_ratio=0.90, yaw=0.0, pitch=0.0)
CLOSED = FaceMetrics(eye_openness=0.12, smile_ratio=0.90, yaw=0.0, pitch=0.0)
SMILING = FaceMetrics(eye_openness=0.30, smile_ratio=1.20, yaw=0.0, pitch=0.0)
LEFT = FaceMetrics(eye_openness=0.30, smile_ratio=0.90, yaw=-30.0, pitch=0.0)
RIGHT = FaceMetrics(eye_openness=0.30, smile_ratio=0.90, yaw=30.0, pitch=0.0)


def session_with(*kinds: ChallengeKind, blink_target: int = 1) -> ChallengeSession:
    """A session with a pinned sequence, so tests are deterministic."""
    s = ChallengeSession(rng=random.Random(0), count=len(kinds))
    s._sequence = [
        Challenge(kind=k, target=blink_target if k is ChallengeKind.BLINK else 1)
        for k in kinds
    ]
    return s


def establish_baseline(s: ChallengeSession, t: float = 0.0) -> float:
    for _ in range(BASELINE_SAMPLES + 1):
        s.update(NEUTRAL, t)
        t += 0.1
    assert s.state is LivenessState.ACTIVE
    return t


def blink(s: ChallengeSession, t: float) -> float:
    """Drive a full close-then-open blink."""
    s.update(CLOSED, t)
    s.update(NEUTRAL, t + 0.1)
    return t + 0.2


def test_baseline_must_be_established_before_challenges_begin():
    s = session_with(ChallengeKind.SMILE)

    assert s.state is LivenessState.BASELINE
    assert s.prompt == "Hold still"

    establish_baseline(s)
    assert s.current.kind is ChallengeKind.SMILE


def test_baseline_ignores_frames_where_the_head_is_turned():
    """A baseline sampled mid-turn would be a biased resting pose."""
    s = session_with(ChallengeKind.SMILE)

    for _ in range(BASELINE_SAMPLES * 2):
        s.update(LEFT, 0.0)

    assert s.state is LivenessState.BASELINE


def test_a_full_sequence_passes():
    s = session_with(ChallengeKind.SMILE, ChallengeKind.TURN_LEFT)
    t = establish_baseline(s)

    s.update(SMILING, t)
    assert s.state is LivenessState.ACTIVE       # gated on returning to neutral

    s.update(NEUTRAL, t + 0.1)                   # neutral issues the next one
    assert s.current.kind is ChallengeKind.TURN_LEFT

    s.update(LEFT, t + 0.2)
    assert s.state is LivenessState.PASSED
    assert s.prompt == "Verified"


def test_a_challenge_not_answered_in_time_fails():
    s = session_with(ChallengeKind.SMILE)
    t = establish_baseline(s)

    s.update(NEUTRAL, t + CHALLENGE_TIMEOUT_S + 1.0)

    assert s.state is LivenessState.FAILED
    assert s.failure_reason == "Timed out"


def test_a_passed_session_is_terminal():
    s = session_with(ChallengeKind.SMILE)
    t = establish_baseline(s)
    s.update(SMILING, t)
    assert s.state is LivenessState.PASSED

    s.update(NEUTRAL, t + CHALLENGE_TIMEOUT_S * 10)
    assert s.state is LivenessState.PASSED


# --- anti-replay properties ---------------------------------------------


def test_a_blink_before_the_prompt_does_not_count():
    """Evidence must postdate the request, or randomisation buys nothing.

    Eyes that close while the person is returning to neutral must not be
    carried into the next challenge and scored the instant they reopen.
    """
    s = session_with(ChallengeKind.SMILE, ChallengeKind.BLINK)
    t = establish_baseline(s)

    s.update(SMILING, t)                  # first challenge satisfied
    s.update(CLOSED, t + 0.1)             # blink during the return to neutral
    s.update(NEUTRAL, t + 0.2)            # neutral reached, blink challenge issued
    assert s.current.kind is ChallengeKind.BLINK

    # The frame that issues a challenge returns before scoring it, so the
    # stale blink would only be consumed here - on the first frame the new
    # challenge is actually evaluated.
    s.update(NEUTRAL, t + 0.3)

    assert s.state is LivenessState.ACTIVE, "a blink before the prompt passed the challenge"
    assert s.current.progress == 0, "a blink before the prompt was counted"


def test_one_long_blink_counts_once():
    """Edge-triggered, not level-triggered."""
    s = session_with(ChallengeKind.BLINK, blink_target=1)
    t = establish_baseline(s)

    for i in range(10):                   # eyes held shut
        s.update(CLOSED, t + i * 0.1)
    assert s.state is LivenessState.ACTIVE

    s.update(NEUTRAL, t + 2.0)            # reopening completes exactly one blink
    assert s.state is LivenessState.PASSED


def test_a_held_pose_cannot_satisfy_two_challenges():
    """Returning to neutral between challenges is mandatory."""
    s = session_with(ChallengeKind.TURN_LEFT, ChallengeKind.TURN_LEFT)
    t = establish_baseline(s)

    for i in range(20):                   # head held left throughout
        s.update(LEFT, t + i * 0.1)

    assert s.state is not LivenessState.PASSED
    assert s.completed == 1


def test_repeated_blinks_are_counted_individually():
    s = session_with(ChallengeKind.BLINK, blink_target=3)
    t = establish_baseline(s)

    t = blink(s, t)
    assert s.current.progress == 1
    t = blink(s, t)
    assert s.current.progress == 2
    blink(s, t)

    assert s.state is LivenessState.PASSED


def test_eye_openness_hovering_at_the_boundary_does_not_manufacture_blinks():
    """Hysteresis: the close and reopen thresholds are deliberately apart."""
    s = session_with(ChallengeKind.BLINK, blink_target=1)
    t = establish_baseline(s)

    borderline = FaceMetrics(eye_openness=0.30 * 0.75, smile_ratio=0.90, yaw=0.0, pitch=0.0)
    for i in range(30):
        s.update(borderline, t + i * 0.1)

    assert s.state is LivenessState.ACTIVE
    assert s.current.progress == 0


def test_the_wrong_action_does_not_satisfy_a_challenge():
    s = session_with(ChallengeKind.TURN_RIGHT)
    t = establish_baseline(s)

    s.update(LEFT, t)
    s.update(SMILING, t + 0.1)

    assert s.state is LivenessState.ACTIVE
    assert s.current.progress == 0


def test_sequences_differ_between_sessions():
    """The unpredictability is the whole security property."""
    sequences = set()
    for seed in range(40):
        s = ChallengeSession(rng=random.Random(seed))
        sequences.add(tuple((c.kind, c.target) for c in s._sequence))

    assert len(sequences) > 5, f"only {len(sequences)} distinct sequences in 40 draws"


def test_a_sequence_never_repeats_a_challenge():
    for seed in range(40):
        s = ChallengeSession(rng=random.Random(seed))
        kinds = [c.kind for c in s._sequence]
        assert len(kinds) == len(set(kinds))


def test_invalid_geometry_is_ignored_rather_than_advancing_state():
    """A frame with no usable landmarks must not count as evidence either way."""
    s = session_with(ChallengeKind.SMILE)
    t = establish_baseline(s)

    blank = FaceMetrics(0.0, 0.0, 0.0, 0.0)
    for i in range(20):
        s.update(blank, t + i * 0.1)

    assert s.state is LivenessState.ACTIVE
