"""Facial geometry used to prove a face is a live person rather than an image.

All of it is derived from landmarks and head pose that buffalo_l already
produces alongside the embedding, so liveness costs no extra inference.

Every measurement here is deliberately *scale invariant* - expressed as a ratio
against another distance on the same face - so that moving closer to or further
from the camera does not change the value.

The thresholds that matter are relative to a per-person baseline rather than
fixed constants. Resting eye openness and neutral mouth width vary enough
between people that any single global cutoff mislabels someone: a fixed eye
threshold that works for wide eyes will read a narrow-eyed person as
permanently mid-blink. Baselines are captured while the face is being
identified, which is dead time the system already spends anyway.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from config import (
    BASELINE_SAMPLES,
    BLINK_CLOSE_RATIO,
    BLINK_OPEN_RATIO,
    EYE_LANDMARK_COUNT,
    SMILE_RISE_RATIO,
    YAW_TURN_DEGREES,
)


@dataclass
class FaceMetrics:
    """Scale-invariant geometry for one face in one frame."""

    eye_openness: float   # mean eye height/width; falls sharply during a blink
    smile_ratio: float    # mouth width / inter-ocular distance
    yaw: float            # degrees; negative and positive are opposite turns
    pitch: float

    @property
    def valid(self) -> bool:
        return self.eye_openness > 0.0 and self.smile_ratio > 0.0


def _eye_openness(landmarks: np.ndarray, eye_point: np.ndarray) -> float:
    """Openness of the eye nearest a given keypoint.

    Takes the landmarks closest to the detector's eye keypoint and measures the
    vertical extent of that cluster against its horizontal extent. Using the
    cluster's bounding extents rather than named landmark indices keeps this
    from depending on a published index map that varies between model packs.

    This is the eye aspect ratio in spirit: eyelids closing collapses the
    vertical extent while the corners stay put, so the ratio drops sharply.
    """
    distances = np.linalg.norm(landmarks - eye_point, axis=1)
    cluster = landmarks[np.argsort(distances)[:EYE_LANDMARK_COUNT]]

    height = float(cluster[:, 1].max() - cluster[:, 1].min())
    width = float(cluster[:, 0].max() - cluster[:, 0].min())
    return height / width if width > 0 else 0.0


def measure(
    landmarks_2d: np.ndarray | None,
    keypoints: np.ndarray,
    pose: np.ndarray | None,
) -> FaceMetrics:
    """Derive liveness geometry from one detection's landmarks.

    `keypoints` is the detector's 5-point set: left eye, right eye, nose,
    left mouth corner, right mouth corner.
    """
    if landmarks_2d is None or keypoints is None or len(keypoints) < 5:
        return FaceMetrics(0.0, 0.0, 0.0, 0.0)

    left_eye, right_eye = keypoints[0], keypoints[1]
    left_mouth, right_mouth = keypoints[3], keypoints[4]

    openness = (
        _eye_openness(landmarks_2d, left_eye) + _eye_openness(landmarks_2d, right_eye)
    ) / 2.0

    # Inter-ocular distance is the standard normaliser for facial measurement:
    # it is stable under expression, which mouth and jaw distances are not.
    interocular = float(np.linalg.norm(left_eye - right_eye))
    mouth_width = float(np.linalg.norm(left_mouth - right_mouth))
    smile = mouth_width / interocular if interocular > 0 else 0.0

    pitch, yaw = (float(pose[0]), float(pose[1])) if pose is not None else (0.0, 0.0)
    return FaceMetrics(eye_openness=openness, smile_ratio=smile, yaw=yaw, pitch=pitch)


@dataclass
class Baseline:
    """A person's resting geometry, learned while their identity is confirmed.

    Collected from frames the pipeline is already processing during
    identification, so establishing it costs the user no extra time.
    """

    eye_samples: list[float] = field(default_factory=list)
    smile_samples: list[float] = field(default_factory=list)

    def observe(self, metrics: FaceMetrics) -> None:
        if not metrics.valid or self.ready:
            return
        self.eye_samples.append(metrics.eye_openness)
        self.smile_samples.append(metrics.smile_ratio)

    @property
    def ready(self) -> bool:
        return len(self.eye_samples) >= BASELINE_SAMPLES

    @property
    def eye_open(self) -> float:
        """Resting eye openness.

        The median resists the blinks that will inevitably occur during
        collection; a mean would be dragged down by them.
        """
        return float(np.median(self.eye_samples)) if self.eye_samples else 0.0

    @property
    def smile_neutral(self) -> float:
        return float(np.median(self.smile_samples)) if self.smile_samples else 0.0

    # --- baseline-relative tests -----------------------------------------

    def eyes_closed(self, metrics: FaceMetrics) -> bool:
        return metrics.eye_openness < self.eye_open * BLINK_CLOSE_RATIO

    def eyes_open(self, metrics: FaceMetrics) -> bool:
        """Deliberately a higher bar than `eyes_closed` is low.

        The gap between the two makes blink counting hysteretic, so a value
        hovering near the boundary cannot rattle between states and register a
        burst of phantom blinks.
        """
        return metrics.eye_openness > self.eye_open * BLINK_OPEN_RATIO

    def is_smiling(self, metrics: FaceMetrics) -> bool:
        return metrics.smile_ratio > self.smile_neutral * SMILE_RISE_RATIO

    @staticmethod
    def is_turned(metrics: FaceMetrics, sign: int) -> bool:
        """True when the head is turned far enough in the given direction."""
        return metrics.yaw * sign > YAW_TURN_DEGREES

    @staticmethod
    def is_facing_forward(metrics: FaceMetrics) -> bool:
        return abs(metrics.yaw) < YAW_TURN_DEGREES / 2.0
