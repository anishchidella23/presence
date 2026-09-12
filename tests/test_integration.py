"""End-to-end test against real photographs.

The unit tests drive the liveness machine with synthetic geometry. This one
runs the whole pipeline - detection, embedding, matching, tracking, gating and
liveness - over actual frames of a face, and simulates a cooperative user by
responding to whichever challenge the system happens to ask for.

It needs real photographs of one person in several states, which are personal
data and are deliberately not committed. Point PRESENCE_TEST_FRAMES at a
directory containing:

    neutral/   facing the camera, eyes open
    closed/    mid-blink
    left/      head turned left past the yaw threshold
    right/     head turned right past the yaw threshold
    smiling/   smiling

Only `neutral` is required. The test discovers which of the others are present
and constrains the challenge sequence to actions it can actually perform, so a
partial fixture set still exercises the pipeline. Every frame must be sharp
enough to clear the blur gate, or the pipeline will correctly refuse to look at
it.

The test skips when that directory is absent, so the suite still runs anywhere.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import cv2
import pytest

from engine.challenge import ChallengeKind, ChallengeSession
from engine.gallery import Gallery
from engine.types import FaceState

FRAMES = os.environ.get("PRESENCE_TEST_FRAMES")

pytestmark = pytest.mark.skipif(
    not FRAMES or not Path(FRAMES).is_dir(),
    reason="PRESENCE_TEST_FRAMES not set; see this module's docstring",
)


POSE_DIRECTORIES = {
    ChallengeKind.BLINK: "closed",
    ChallengeKind.TURN_LEFT: "left",
    ChallengeKind.TURN_RIGHT: "right",
    ChallengeKind.SMILE: "smiling",
}


def load(kind: str) -> list:
    paths = sorted(Path(FRAMES).joinpath(kind).glob("*.jpg"))
    return [image for image in (cv2.imread(str(p)) for p in paths) if image is not None]


@pytest.fixture(scope="module")
def detector():
    from engine.detector import FaceDetector

    return FaceDetector()


@pytest.fixture(scope="module")
def frames():
    available = {
        name: load(name)
        for name in ("neutral", *POSE_DIRECTORIES.values())
    }
    assert available["neutral"], "neutral/ must contain at least one frame"
    return {name: images for name, images in available.items() if images}


def answerable_kinds(frames: dict) -> set:
    return {
        kind
        for kind, directory in POSE_DIRECTORIES.items()
        if frames.get(directory)
    }


def seed_answerable_with(frames: dict) -> int:
    """Find a challenge sequence the available frames can actually perform.

    Fixtures rarely cover every action - a head turn fast enough to be
    convincing is often too motion-blurred to clear the quality gate - so the
    test constrains the sequence to what it can answer rather than failing on
    a challenge it was never equipped for.
    """
    answerable = answerable_kinds(frames)
    assert answerable, "no pose directories present; cannot answer any challenge"

    for seed in range(2000):
        session = ChallengeSession(rng=random.Random(seed))
        if all(c.kind in answerable for c in session._sequence):
            return seed
    raise AssertionError(f"no sequence drawn from {answerable} in 2000 attempts")


def respond_to(prompt: str, frames: dict):
    """Pick the frame a cooperative person would produce for this prompt."""
    if "Blink" in prompt:
        return frames["closed"][0]
    if "left" in prompt:
        return frames["left"][0]
    if "right" in prompt:
        return frames["right"][0]
    if "Smile" in prompt:
        return frames["smiling"][0]
    return frames["neutral"][0]          # hold still, or return to neutral


def test_a_cooperative_person_is_verified_and_logged(detector, frames):
    from engine.pipeline import PresencePipeline

    embedding = detector.embed_crop(frames["neutral"][0])
    assert embedding is not None, "no face found in the reference frame"

    gallery = Gallery()
    gallery.add("SUBJECT", embedding)

    pipeline = PresencePipeline(
        detector=detector, gallery=gallery, rng=random.Random(seed_answerable_with(frames))
    )

    verified: list[str] = []
    pipeline.on_verified = lambda name, result: verified.append(name)

    prompt, state = "", None
    for step in range(300):
        frame = respond_to(prompt, frames)
        result = pipeline.process(frame, now=step * 0.1)
        assert result.faces, f"lost the face at step {step}"
        face = result.faces[0]
        prompt, state = face.prompt, face.state

        if state is FaceState.VERIFIED:
            break

        # A blink needs the eyes to reopen; holding them shut counts once.
        if "Blink" in prompt:
            pipeline.process(frames["neutral"][0], now=step * 0.1 + 0.05)

    assert state is FaceState.VERIFIED, f"never verified; stuck at {state} / {prompt!r}"
    assert verified == ["SUBJECT"], "the verification callback did not fire exactly once"


def test_a_static_photo_never_passes(detector, frames):
    """The property the whole design exists for.

    A single unchanging image is recognised perfectly well - identity is not
    the question - but it cannot answer a challenge, so it is never logged.
    """
    from engine.pipeline import PresencePipeline

    embedding = detector.embed_crop(frames["neutral"][0])
    gallery = Gallery()
    gallery.add("SUBJECT", embedding)

    pipeline = PresencePipeline(detector=detector, gallery=gallery, rng=random.Random(0))

    verified: list[str] = []
    pipeline.on_verified = lambda name, result: verified.append(name)

    still = frames["neutral"][0]
    states = set()
    for step in range(300):
        result = pipeline.process(still, now=step * 0.1)
        states.add(result.faces[0].state)

    assert not verified, "a static photo was logged as present"
    assert FaceState.VERIFIED not in states
    # It should still be identified - the failure must come from liveness, not
    # from the recogniser quietly failing to place the face.
    assert states & {FaceState.CHALLENGED, FaceState.SPOOF_SUSPECTED}
