"""The enrolled-identity gallery and the open-set matching rule.

The matching decision here is deliberately stricter than a plain threshold.
Comparing an embedding against a gallery is an *open-set* problem: the person
in front of the camera may well not be enrolled at all. A lone similarity
cutoff handles that badly, because the correct cutoff drifts with lighting,
camera and distance, so any fixed value is either too loose in good conditions
or too tight in poor ones.

Requiring the winner to beat the runner-up by a margin adds a second, largely
independent piece of evidence: if the best candidate is not clearly better
than the next best, the embedding is not landing distinctly on anyone and the
honest answer is Unknown.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from config import FACES_DIR, MIN_MATCH_MARGIN, RECOGNITION_THRESHOLD
from engine.types import MatchResult

log = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class Person:
    """One enrolled identity and every reference embedding held for them.

    Multiple embeddings per person is the single cheapest accuracy win
    available: one reference photo pins an identity to one pose and one
    lighting condition, and everything that deviates from it scores lower.
    """

    name: str
    embeddings: list[np.ndarray] = field(default_factory=list)

    @property
    def matrix(self) -> np.ndarray:
        """Reference embeddings stacked as (n, 512) for vectorised comparison."""
        return np.vstack(self.embeddings)

    def best_similarity(self, embedding: np.ndarray) -> float:
        """Highest cosine similarity across this person's references.

        Max rather than mean: the reference closest to the current pose is the
        relevant evidence, and averaging lets unrelated poses drag it down.
        """
        return float(np.max(self.matrix @ embedding))


class Gallery:
    """Holds enrolled people and answers 'who is this?' for an embedding."""

    def __init__(self, people: dict[str, Person] | None = None) -> None:
        self._people: dict[str, Person] = people or {}

    # --- construction ----------------------------------------------------

    @classmethod
    def from_directory(cls, root: Path = FACES_DIR, detector=None) -> "Gallery":
        """Build a gallery from reference images on disk.

        Two layouts are accepted:
            data/faces/Anish/front.jpg, left.jpg, ...   (preferred, multi-shot)
            data/faces/Anish.jpg                        (single reference)
        """
        if detector is None:
            from engine.detector import FaceDetector

            detector = FaceDetector()

        people: dict[str, Person] = {}
        if not root.exists():
            log.warning("gallery directory %s does not exist", root)
            return cls(people)

        for entry in sorted(root.iterdir()):
            if entry.name.startswith("."):
                continue

            if entry.is_dir():
                name, images = entry.name, sorted(
                    p for p in entry.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
                )
            elif entry.suffix.lower() in IMAGE_SUFFIXES:
                name, images = entry.stem, [entry]
            else:
                continue

            person = Person(name=name)
            for path in images:
                image = cv2.imread(str(path))
                if image is None:
                    log.warning("could not read %s", path)
                    continue
                embedding = detector.embed_crop(image)
                if embedding is None:
                    log.warning("no face found in %s - skipping", path)
                    continue
                person.embeddings.append(embedding)

            if person.embeddings:
                people[name] = person
                log.info("enrolled %s with %d reference(s)", name, len(person.embeddings))
            else:
                log.warning("no usable references for %s", name)

        return cls(people)

    # --- access ----------------------------------------------------------

    @property
    def names(self) -> list[str]:
        return sorted(self._people)

    def __len__(self) -> int:
        return len(self._people)

    def add(self, name: str, embedding: np.ndarray) -> None:
        self._people.setdefault(name, Person(name=name)).embeddings.append(embedding)

    # --- matching --------------------------------------------------------

    def match(self, embedding: np.ndarray | None) -> MatchResult:
        """Identify an embedding, or refuse to.

        Returns the winner and runner-up scores either way, so the caller can
        show *why* someone was rejected rather than only that they were.
        """
        if embedding is None:
            return MatchResult(None, 0.0, 0.0, rejected_reason="no embedding")

        if not self._people:
            return MatchResult(None, 0.0, 0.0, rejected_reason="gallery is empty")

        scores = sorted(
            ((person.best_similarity(embedding), name) for name, person in self._people.items()),
            reverse=True,
        )

        best_score, best_name = scores[0]
        runner_score, runner_name = scores[1] if len(scores) > 1 else (0.0, None)
        margin = best_score - runner_score

        # A single enrolled person has no runner-up to compare against, so the
        # margin rule cannot apply and the threshold carries the decision alone.
        margin_required = len(scores) > 1

        if best_score < RECOGNITION_THRESHOLD:
            return MatchResult(
                None,
                best_score,
                margin,
                runner_up=runner_name,
                rejected_reason=f"similarity {best_score:.3f} < {RECOGNITION_THRESHOLD}",
            )

        if margin_required and margin < MIN_MATCH_MARGIN:
            return MatchResult(
                None,
                best_score,
                margin,
                runner_up=runner_name,
                rejected_reason=(
                    f"ambiguous: {best_name} {best_score:.3f} vs "
                    f"{runner_name} {runner_score:.3f} (margin {margin:.3f})"
                ),
            )

        return MatchResult(best_name, best_score, margin, runner_up=runner_name)
