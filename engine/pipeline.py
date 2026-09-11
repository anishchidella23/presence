"""The verification pipeline: frame in, per-face state out.

Stage order is deliberate. Each gate is cheaper than the one after it and
rejects work the later stages would otherwise waste effort on, and each one
answers a question the next stage depends on:

    detect -> quality -> identify -> smooth over time -> liveness -> log

Liveness runs last because challenging someone is pointless until the system
knows who it is challenging, and because it is the only stage that asks
anything of the user. Everything cheap and invisible happens first.
"""

from __future__ import annotations

import random
import time

import numpy as np

from config import BLUR_THRESHOLD, MIN_FACE_PIXELS, REQUIRED_MATCHES
from engine import liveness
from engine.challenge import ChallengeSession, LivenessState
from engine.detector import FaceDetector, blur_score
from engine.gallery import Gallery
from engine.tracker import FaceTracker, Track
from engine.types import Detection, FaceResult, FaceState, FrameResult


class PresencePipeline:
    """Runs detection, quality gating, identification, smoothing and liveness."""

    def __init__(
        self,
        detector: FaceDetector | None = None,
        gallery: Gallery | None = None,
        blur_threshold: float = BLUR_THRESHOLD,
        required_matches: int = REQUIRED_MATCHES,
        rng: random.Random | None = None,
    ) -> None:
        self.detector = detector or FaceDetector()
        self.gallery = gallery if gallery is not None else Gallery.from_directory(detector=self.detector)
        self.tracker = FaceTracker()
        self.blur_threshold = blur_threshold
        self.required_matches = required_matches

        # Injectable so tests can pin the challenge sequence. In production the
        # unpredictability is the entire security property, so this must stay
        # seeded from the system source.
        self._rng = rng or random.Random()

        # Called with (name, FaceResult) the moment a person passes liveness.
        # Persistence subscribes here rather than the pipeline knowing about
        # storage.
        self.on_verified = None

    def process(self, frame_bgr: np.ndarray, now: float | None = None) -> FrameResult:
        """Process one frame and return the state of every face in it."""
        started = time.perf_counter()
        now = time.monotonic() if now is None else now

        detections = self.detector.detect(frame_bgr)
        results: list[FaceResult] = []

        for track, detection in self.tracker.update(detections):
            results.append(self._process_face(frame_bgr, track, detection, now))

        height, width = frame_bgr.shape[:2]
        return FrameResult(
            faces=results,
            frame_width=width,
            frame_height=height,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )

    # --- per-face stages -----------------------------------------------

    def _process_face(
        self, frame_bgr: np.ndarray, track: Track, detection: Detection, now: float
    ) -> FaceResult:
        sharpness = blur_score(frame_bgr, detection.bbox)

        result = FaceResult(
            track_id=track.track_id,
            bbox=detection.bbox,
            state=FaceState.UNKNOWN,
            blur_score=sharpness,
            det_score=detection.det_score,
            required_matches=self.required_matches,
        )

        # --- quality gates ------------------------------------------------
        # A face failing these is reported but not identified: the user should
        # learn that the system saw them and why it could not act, rather than
        # watching a box sit there with no explanation.
        if detection.size < MIN_FACE_PIXELS:
            result.state = FaceState.TOO_SMALL
            result.detail = "Move closer"
            track.observe(detection, None)
            return result

        if sharpness < self.blur_threshold:
            result.state = FaceState.TOO_BLURRY
            result.detail = f"Too blurry ({sharpness:.0f})"
            track.observe(detection, None)
            return result

        # --- identity -----------------------------------------------------
        match = self.gallery.match(detection.embedding)
        track.observe(detection, match.name)

        result.similarity = match.similarity
        result.margin = match.margin

        stable_name, count = track.stable_identity(self.required_matches)
        result.matches_in_window = count

        if not match.accepted:
            result.state = FaceState.UNKNOWN
            result.detail = match.rejected_reason or "Unknown"
            return result

        result.name = match.name
        result.confidence = match.confidence

        if stable_name != match.name:
            result.state = FaceState.CONFIRMING
            result.detail = f"Confirming {match.name} ({count}/{self.required_matches})"
            return result

        # --- liveness -----------------------------------------------------
        return self._run_liveness(track, detection, result, now)

    def _run_liveness(
        self, track: Track, detection: Detection, result: FaceResult, now: float
    ) -> FaceResult:
        """Challenge a stably-identified face and log it once it passes."""
        if track.challenge is None:
            track.challenge = ChallengeSession(rng=self._rng)

        session = track.challenge
        metrics = liveness.measure(detection.landmarks_2d, detection.keypoints, detection.pose)
        state = session.update(metrics, now)

        result.eye_openness = metrics.eye_openness
        result.yaw = metrics.yaw
        result.prompt = session.prompt
        result.challenges_passed = session.completed
        result.challenges_total = session.total

        if state is LivenessState.PASSED:
            result.state = FaceState.VERIFIED
            result.detail = f"{result.name} verified"
            # Logging is edge-triggered on the track, so presence is recorded
            # once per visit however long the person lingers in frame.
            if not track.logged:
                track.logged = True
                if self.on_verified is not None:
                    self.on_verified(result.name, result)
        elif state is LivenessState.FAILED:
            result.state = FaceState.SPOOF_SUSPECTED
            result.detail = session.failure_reason or "Liveness failed"
        else:
            result.state = FaceState.CHALLENGED
            result.detail = session.prompt

        return result
