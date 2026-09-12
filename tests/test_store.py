"""Tests for persistence and the deduplication window."""

import numpy as np
import pytest

from engine.gallery import Gallery
from engine.store import Store


@pytest.fixture
def store(tmp_path):
    s = Store(path=tmp_path / "test.db")
    yield s
    s.close()


def vector(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=512).astype(np.float32)
    return v / np.linalg.norm(v)


def test_enrolling_stores_multiple_references_for_one_person(store):
    store.add_embedding("alice", vector(1), source="front.jpg")
    store.add_embedding("alice", vector(2), source="left.jpg")

    loaded = store.load_embeddings()

    assert list(loaded) == ["alice"]
    assert len(loaded["alice"]) == 2


def test_embeddings_survive_the_round_trip_through_the_database(store):
    original = vector(7)
    store.add_embedding("alice", original)

    restored = store.load_embeddings()["alice"][0]

    assert restored.shape == (512,)
    np.testing.assert_allclose(restored, original, rtol=1e-6)


def test_a_gallery_can_be_built_from_the_database(store):
    target = vector(3)
    store.add_embedding("alice", target)
    store.add_embedding("bob", vector(4))

    gallery = Gallery.from_store(store)

    assert gallery.names == ["alice", "bob"]
    assert gallery.match(target).name == "alice"


def test_presence_is_recorded(store):
    assert store.log_presence("alice", 91.0, 0.82, 2) is True

    events = store.recent_events()
    assert len(events) == 1
    assert events[0].name == "alice"
    assert events[0].challenges_passed == 2


def test_the_same_person_is_not_logged_twice_inside_the_window(store):
    """A kiosk sees someone for many frames and often twice a day."""
    assert store.log_presence("alice", 91.0, 0.82, 2) is True
    assert store.log_presence("alice", 88.0, 0.79, 2) is False

    assert len(store.recent_events()) == 1


def test_a_new_arrival_is_logged_once_the_window_has_passed(store):
    store.log_presence("alice", 91.0, 0.82, 2)

    assert store.log_presence("alice", 90.0, 0.81, 2, within_hours=0) is True
    assert len(store.recent_events()) == 2


def test_deduplication_is_per_person(store):
    store.log_presence("alice", 91.0, 0.82, 2)

    assert store.log_presence("bob", 90.0, 0.80, 2) is True


def test_deleting_a_person_removes_their_references_and_events(store):
    store.add_embedding("alice", vector(1))
    store.log_presence("alice", 91.0, 0.82, 2)

    store.delete_person("alice")

    assert store.load_embeddings() == {}
    assert store.recent_events() == []


def test_roster_reports_reference_counts(store):
    store.add_embedding("alice", vector(1))
    store.add_embedding("alice", vector(2))
    store.add_person("bob")

    roster = {r["name"]: r for r in store.roster()}

    assert roster["alice"]["references_held"] == 2
    assert roster["bob"]["references_held"] == 0


def test_presence_can_be_queried_by_day(store):
    from datetime import datetime, timezone

    store.log_presence("alice", 91.0, 0.82, 2)
    today = datetime.now(timezone.utc).date().isoformat()

    assert store.present_on(today) == ["alice"]
    assert store.present_on("2000-01-01") == []
