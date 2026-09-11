"""Face detection and embedding, wrapping InsightFace's buffalo_l pack.

Isolated behind a small interface so the recogniser can be swapped without
touching the rest of the pipeline.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from config import (
    DET_SIZE,
    MIN_DETECTION_SCORE,
    MIN_FACE_PIXELS,
    MODEL_PACK,
    MODEL_ROOT,
)
from engine.types import Detection

log = logging.getLogger(__name__)


class FaceDetector:
    """Detects faces and produces L2-normalised 512-d ArcFace embeddings."""

    def __init__(self, det_size: tuple[int, int] = DET_SIZE) -> None:
        # Imported lazily: loading InsightFace pulls in ONNX Runtime and costs
        # a few seconds, which we do not want to pay just to import this module
        # (the test suite and type checks never need the real model).
        from insightface.app import FaceAnalysis

        MODEL_ROOT.mkdir(parents=True, exist_ok=True)
        log.info("loading %s (first run downloads ~300MB)", MODEL_PACK)

        self._app = FaceAnalysis(
            name=MODEL_PACK,
            root=str(MODEL_ROOT),
            providers=["CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=0, det_size=det_size)
        log.info("detector ready")

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        """Find faces in a BGR frame and embed the ones worth embedding.

        Faces failing the score or size gates are still returned - the UI
        should show that something was seen and why it was not identified,
        rather than silently dropping it - but carry no embedding.
        """
        detections: list[Detection] = []

        for face in self._app.get(frame_bgr):
            x1, y1, x2, y2 = (int(v) for v in face.bbox)
            # The detector can return boxes that run past the frame edge.
            h, w = frame_bgr.shape[:2]
            bbox = (max(0, x1), max(0, y1), min(w, x2), min(h, y2))

            det = Detection(
                bbox=bbox,
                det_score=float(face.det_score),
                keypoints=np.asarray(face.kps, dtype=np.float32),
            )

            if det.det_score < MIN_DETECTION_SCORE or det.size < MIN_FACE_PIXELS:
                detections.append(det)
                continue

            det.embedding = np.asarray(face.normed_embedding, dtype=np.float32)
            detections.append(det)

        return detections

    def embed_crop(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        """Embed the single most prominent face in an image.

        Used during enrolment, where we want one embedding from one reference
        photo and should ignore bystanders in the background.
        """
        faces = self._app.get(frame_bgr)
        if not faces:
            return None
        largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return np.asarray(largest.normed_embedding, dtype=np.float32)


def blur_score(frame_bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
    """Variance of the Laplacian over the face crop.

    Sharp faces show strong second-derivative variation across edges; a
    motion-blurred or out-of-focus crop does not. The absolute value depends
    on crop resolution and camera, so it is only meaningful against a
    threshold calibrated on the same setup.
    """
    x1, y1, x2, y2 = bbox
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
