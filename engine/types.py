"""Structured types passed between engine stages and out to the UI.

The engine never draws anything. Every stage produces plain data, which keeps
the pipeline testable against still images with no camera or browser involved,
and lets the web layer render however it likes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class FaceState(str, Enum):
    """What the system currently believes about one tracked face."""

    TOO_BLURRY = "too_blurry"      # quality gate rejected the crop
    TOO_SMALL = "too_small"        # face is too far away to embed reliably
    UNKNOWN = "unknown"            # embedded fine, matched nobody
    CONFIRMING = "confirming"      # candidate identity, not yet stable
    RECOGNISED = "recognised"      # identity stable, liveness not yet proven
    VERIFIED = "verified"          # passed liveness, logged


@dataclass
class Detection:
    """One face found in one frame, before any identity reasoning."""

    bbox: tuple[int, int, int, int]   # (x1, y1, x2, y2) in full-frame pixels
    det_score: float
    keypoints: np.ndarray             # (5, 2) eyes, nose, mouth corners
    embedding: np.ndarray | None = None   # (512,) L2-normalised, None if gated out

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def size(self) -> int:
        """Longest bbox edge, used as the face-too-small proxy."""
        return max(self.width, self.height)

    @property
    def centre(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass
class MatchResult:
    """Outcome of comparing one embedding against the enrolled gallery.

    Carries the runner-up as well as the winner, because the decision rule
    depends on the gap between them, not just the top score.
    """

    name: str | None                  # None when rejected as unknown
    similarity: float                 # cosine similarity to the best candidate
    margin: float                     # best similarity minus runner-up
    runner_up: str | None = None
    rejected_reason: str | None = None   # why a candidate was refused, if any

    @property
    def accepted(self) -> bool:
        return self.name is not None

    @property
    def confidence(self) -> float:
        """Similarity rescaled to 0-100 for display.

        This is a monotone rescale of cosine similarity above the acceptance
        threshold, NOT a probability. It is shown to users because a raw
        cosine value is meaningless to them, but nothing downstream should
        treat it as calibrated.
        """
        from config import RECOGNITION_THRESHOLD

        if self.similarity <= RECOGNITION_THRESHOLD:
            return 0.0
        span = 1.0 - RECOGNITION_THRESHOLD
        return round(min((self.similarity - RECOGNITION_THRESHOLD) / span, 1.0) * 100, 1)


@dataclass
class FaceResult:
    """Everything the UI needs to render one tracked face for one frame."""

    track_id: int
    bbox: tuple[int, int, int, int]
    state: FaceState
    name: str | None = None
    confidence: float = 0.0
    similarity: float = 0.0
    margin: float = 0.0
    blur_score: float = 0.0
    det_score: float = 0.0
    matches_in_window: int = 0
    required_matches: int = 0
    detail: str = ""                  # short human-readable status line

    def to_dict(self) -> dict:
        """JSON-safe form sent over the websocket to the browser."""
        return {
            "track_id": self.track_id,
            "bbox": list(self.bbox),
            "state": self.state.value,
            "name": self.name,
            "confidence": self.confidence,
            "similarity": round(self.similarity, 4),
            "margin": round(self.margin, 4),
            "blur_score": round(self.blur_score, 1),
            "det_score": round(self.det_score, 3),
            "matches_in_window": self.matches_in_window,
            "required_matches": self.required_matches,
            "detail": self.detail,
        }


@dataclass
class FrameResult:
    """All faces in one processed frame, plus timing."""

    faces: list[FaceResult] = field(default_factory=list)
    frame_width: int = 0
    frame_height: int = 0
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "faces": [f.to_dict() for f in self.faces],
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "elapsed_ms": round(self.elapsed_ms, 1),
        }
