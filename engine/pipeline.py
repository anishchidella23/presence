"""The verification pipeline: frame in, per-face state out.

Stage order is deliberate. Each gate is cheaper than the one after it and
rejects work the later stages would otherwise waste effort on, and each one
answers a question the next stage depends on:

    detect -> quality gate -> identify -> smooth over time -> (liveness)

Liveness arrives in a later phase and slots in after smoothing, since there is
no point challenging someone whose identity is not yet stable.
"""

from __future__ import annotations

import time

import numpy as np

from config import BLUR_THRESHOLD, MIN_FACE_PIXELS, REQUIRED_MATCHES
from engine.detector import FaceDetector, blur_score
from engine.gallery import Gallery
from engine.tracker import FaceTracker
from engine.types import FaceResult, FaceState, FrameResult


class PresencePipeline:
    """Runs detection, quality gating, identification and smoothing."""

    def __init__(
        self,
        detector: FaceDetector | None = None,
        gallery: Gallery | None = None,
        blur_threshold: float = BLUR_THRESHOLD,
        required_matches: int = REQUIRED_MATCHES,
    ) -> None:
        self.detector = detector or FaceDetector()
        self.gallery = gallery if gallery is not None else Gallery.from_directory(detector=self.detector)
        self.tracker = FaceTracker()
        self.blur_threshold = blur_threshold
        self.required_matches = required_matches

    def process(self, frame_bgr: np.ndarray) -> FrameResult:
        """Process one frame and return the state of every face in it."""
        started = time.perf_counter()

        detections = self.detector.detect(frame_bgr)
        results: list[FaceResult] = []

        for track, detection in self.tracker.update(detections):
            sharpness = blur_score(frame_bgr, detection.bbox)

            result = FaceResult(
                track_id=track.track_id,
                bbox=detection.bbox,
                state=FaceState.UNKNOWN,
                blur_score=sharpness,
                det_score=detection.det_score,
                required_matches=self.required_matches,
            )

            # --- quality gates -------------------------------------------
            # A face that fails these is reported but not identified: the user
            # should learn that the system saw them and why it could not act,
            # instead of watching a box sit there with no explanation.
            if detection.size < MIN_FACE_PIXELS:
                result.state = FaceState.TOO_SMALL
                result.detail = "Move closer"
                track.observe(detection, None)
                results.append(result)
                continue

            if sharpness < self.blur_threshold:
                result.state = FaceState.TOO_BLURRY
                result.detail = f"Too blurry ({sharpness:.0f})"
                track.observe(detection, None)
                results.append(result)
                continue

            # --- identity -------------------------------------------------
            match = self.gallery.match(detection.embedding)
            track.observe(detection, match.name)

            result.similarity = match.similarity
            result.margin = match.margin

            stable_name, count = track.stable_identity(self.required_matches)
            result.matches_in_window = count

            if not match.accepted:
                result.state = FaceState.UNKNOWN
                result.detail = match.rejected_reason or "Unknown"
                results.append(result)
                continue

            result.name = match.name
            result.confidence = match.confidence

            if stable_name == match.name:
                # Identity is stable. Liveness verification lands here in the
                # next phase; until it exists a stable identity is as far as
                # the pipeline goes and nothing is logged.
                result.state = FaceState.RECOGNISED
                result.detail = f"{match.name} ({result.confidence:.0f}%)"
            else:
                result.state = FaceState.CONFIRMING
                result.detail = f"Confirming {match.name} ({count}/{self.required_matches})"

            results.append(result)

        height, width = frame_bgr.shape[:2]
        return FrameResult(
            faces=results,
            frame_width=width,
            frame_height=height,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )
