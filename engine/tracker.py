"""Multi-face tracking with stable identifiers.

Tracking exists because every reliability layer in this system is stateful.
Temporal smoothing needs a history of who a particular face looked like over
the last N frames, and the liveness challenge needs to know that the person
being asked to look left is the same person who looks left a moment later.
None of that is expressible without a per-face identity that survives between
frames, so detections must be associated across time before any gating runs.

Association is greedy IoU matching. With a handful of faces at webcam frame
rates, movement between frames is small relative to face size, so boxes
overlap heavily frame to frame and a global assignment algorithm would buy
nothing for the extra complexity.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from config import (
    SMOOTHING_WINDOW,
    TRACK_IOU_THRESHOLD,
    TRACK_MAX_AGE,
)
from engine.types import Detection


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Intersection over union of two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    intersection = iw * ih
    if intersection == 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


@dataclass
class Track:
    """One face followed across frames, with the state that accumulates on it."""

    track_id: int
    bbox: tuple[int, int, int, int]
    frames_seen: int = 1
    frames_missing: int = 0

    # Rolling window of per-frame identity guesses. None means "seen but
    # matched nobody", which must occupy a slot like any other observation so
    # that a face drifting out of recognition decays out of stability.
    recent_names: deque[str | None] = field(
        default_factory=lambda: deque(maxlen=SMOOTHING_WINDOW)
    )

    # Set once this track has been logged, so presence is recorded once per
    # visit rather than once per frame.
    logged: bool = False

    def observe(self, detection: Detection, name: str | None) -> None:
        self.bbox = detection.bbox
        self.frames_seen += 1
        self.frames_missing = 0
        self.recent_names.append(name)

    def mark_missing(self) -> None:
        self.frames_missing += 1

    @property
    def expired(self) -> bool:
        return self.frames_missing > TRACK_MAX_AGE

    def stable_identity(self, required: int) -> tuple[str | None, int]:
        """The identity holding a majority of the recent window, if any.

        Returns the winning name (or None) alongside its count, so the UI can
        show progress towards confirmation rather than a bare yes/no.
        """
        if not self.recent_names:
            return None, 0

        counts: dict[str, int] = {}
        for name in self.recent_names:
            if name is not None:
                counts[name] = counts.get(name, 0) + 1

        if not counts:
            return None, 0

        best_name = max(counts, key=lambda n: counts[n])
        best_count = counts[best_name]
        return (best_name, best_count) if best_count >= required else (None, best_count)


class FaceTracker:
    """Assigns detections to tracks, keeping identifiers stable over time."""

    def __init__(self, iou_threshold: float = TRACK_IOU_THRESHOLD) -> None:
        self._tracks: dict[int, Track] = {}
        self._next_id = 0
        self._iou_threshold = iou_threshold

    @property
    def tracks(self) -> dict[int, Track]:
        return self._tracks

    def update(self, detections: list[Detection]) -> list[tuple[Track, Detection]]:
        """Associate this frame's detections with existing tracks.

        Returns (track, detection) pairs in detection order. Tracks that go
        unmatched for too long are dropped, taking their accumulated smoothing
        and liveness state with them.
        """
        unmatched_track_ids = set(self._tracks)
        pairs: list[tuple[Track, Detection]] = []

        # Largest faces first: they are the closest to the camera and the most
        # likely to be the subject actually checking in, so they get first
        # claim on an existing track when two detections contest one.
        for detection in sorted(detections, key=lambda d: d.size, reverse=True):
            best_id, best_iou = None, self._iou_threshold
            for track_id in unmatched_track_ids:
                score = iou(self._tracks[track_id].bbox, detection.bbox)
                if score >= best_iou:
                    best_id, best_iou = track_id, score

            if best_id is None:
                track = Track(track_id=self._next_id, bbox=detection.bbox)
                self._tracks[self._next_id] = track
                self._next_id += 1
            else:
                track = self._tracks[best_id]
                unmatched_track_ids.discard(best_id)

            pairs.append((track, detection))

        for track_id in unmatched_track_ids:
            self._tracks[track_id].mark_missing()

        for track_id in [t for t, tr in self._tracks.items() if tr.expired]:
            del self._tracks[track_id]

        # Restore the caller's detection order so results line up with input.
        order = {id(d): i for i, d in enumerate(detections)}
        pairs.sort(key=lambda pair: order[id(pair[1])])
        return pairs
