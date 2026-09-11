"""Tests for the open-set matching rule.

These use synthetic unit vectors rather than real faces: the decision logic is
what is under test, and constructing embeddings by hand makes the similarity
values exact instead of approximate.
"""

import numpy as np
import pytest

from config import MIN_MATCH_MARGIN, RECOGNITION_THRESHOLD
from engine.gallery import Gallery, Person


def unit(*components: float) -> np.ndarray:
    """A normalised embedding, so dot products are cosine similarities."""
    vector = np.zeros(512, dtype=np.float32)
    for index, value in enumerate(components):
        vector[index] = value
    return vector / np.linalg.norm(vector)


def gallery_with(**people: np.ndarray) -> Gallery:
    return Gallery({name: Person(name, [vec]) for name, vec in people.items()})


def test_confident_match_is_accepted():
    target = unit(1, 0, 0)
    gallery = gallery_with(alice=target, bob=unit(0, 1, 0))

    result = gallery.match(target)

    assert result.accepted
    assert result.name == "alice"
    assert result.similarity == pytest.approx(1.0, abs=1e-5)


def test_stranger_is_rejected_not_forced_onto_nearest_identity():
    gallery = gallery_with(alice=unit(1, 0, 0), bob=unit(0, 1, 0))

    result = gallery.match(unit(0, 0, 1))

    assert not result.accepted
    assert result.name is None
    assert "similarity" in result.rejected_reason


def test_ambiguous_face_is_rejected_even_when_above_threshold():
    """The margin rule's reason for existing.

    A probe sitting between two enrolled people can clear the similarity
    threshold against both. The top score alone would hand back a confident
    name; the margin check notices the runner-up is just as close and refuses.
    """
    alice, bob = unit(1, 0.02, 0), unit(1, -0.02, 0)
    gallery = gallery_with(alice=alice, bob=bob)

    probe = unit(1, 0, 0)
    result = gallery.match(probe)

    assert result.similarity > RECOGNITION_THRESHOLD, "probe should clear the threshold"
    assert result.margin < MIN_MATCH_MARGIN, "probe should be ambiguous between the two"
    assert not result.accepted
    assert "ambiguous" in result.rejected_reason


def test_margin_rule_does_not_apply_to_a_single_enrolled_person():
    """With nobody to be runner-up, the threshold has to decide alone."""
    target = unit(1, 0, 0)
    gallery = gallery_with(only=target)

    result = gallery.match(target)

    assert result.accepted
    assert result.name == "only"


def test_empty_gallery_rejects_cleanly():
    result = Gallery({}).match(unit(1, 0, 0))

    assert not result.accepted
    assert result.rejected_reason == "gallery is empty"


def test_missing_embedding_rejects_cleanly():
    """Faces gated out for quality reach the matcher with no embedding."""
    result = gallery_with(alice=unit(1, 0, 0)).match(None)

    assert not result.accepted
    assert result.rejected_reason == "no embedding"


def test_multiple_references_take_the_closest_pose():
    """Extra references should only ever help, never dilute a good match."""
    front, side = unit(1, 0, 0), unit(0, 1, 0)
    person = Person("alice", [front, side])

    assert person.best_similarity(side) == pytest.approx(1.0, abs=1e-5)
    assert person.best_similarity(front) == pytest.approx(1.0, abs=1e-5)


def test_confidence_is_zero_at_threshold_and_rises_with_similarity():
    gallery = gallery_with(alice=unit(1, 0, 0))

    weak = gallery.match(unit(1, 0, 0) * 0 + unit(1, 1, 0))
    strong = gallery.match(unit(1, 0, 0))

    assert strong.confidence > weak.confidence
    assert 0.0 <= weak.confidence <= 100.0
    assert strong.confidence == pytest.approx(100.0, abs=0.1)
