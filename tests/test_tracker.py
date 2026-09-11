"""Tests for track association and temporal smoothing."""

import numpy as np

from engine.tracker import FaceTracker, Track, iou
from engine.types import Detection


def detection(x1, y1, x2, y2) -> Detection:
    return Detection(
        bbox=(x1, y1, x2, y2),
        det_score=0.9,
        keypoints=np.zeros((5, 2), dtype=np.float32),
        embedding=np.zeros(512, dtype=np.float32),
    )


def test_iou_of_identical_boxes_is_one():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_iou_of_disjoint_boxes_is_zero():
    assert iou((0, 0, 10, 10), (50, 50, 60, 60)) == 0.0


def test_a_face_keeps_its_id_while_it_moves():
    """The point of tracking: identity survives motion between frames."""
    tracker = FaceTracker()

    (track, _), = tracker.update([detection(100, 100, 200, 200)])
    original_id = track.track_id

    for offset in range(1, 6):   # drift a few pixels per frame
        shift = offset * 5
        (track, _), = tracker.update([detection(100 + shift, 100, 200 + shift, 200)])

    assert track.track_id == original_id


def test_a_new_face_gets_a_new_id():
    tracker = FaceTracker()

    (first, _), = tracker.update([detection(0, 0, 100, 100)])
    (second, _), = tracker.update([detection(500, 500, 600, 600)])

    assert first.track_id != second.track_id


def test_two_faces_keep_separate_ids_across_frames():
    tracker = FaceTracker()

    pairs = tracker.update([detection(0, 0, 100, 100), detection(400, 0, 500, 100)])
    ids = {track.track_id for track, _ in pairs}
    assert len(ids) == 2

    again = tracker.update([detection(5, 0, 105, 100), detection(405, 0, 505, 100)])
    assert {track.track_id for track, _ in again} == ids


def test_a_track_expires_after_being_missing_too_long():
    from config import TRACK_MAX_AGE

    tracker = FaceTracker()
    tracker.update([detection(0, 0, 100, 100)])

    for _ in range(TRACK_MAX_AGE + 2):
        tracker.update([])

    assert tracker.tracks == {}


def test_results_are_returned_in_detection_order():
    """The caller indexes results against the detections it passed in."""
    tracker = FaceTracker()

    small, large = detection(0, 0, 50, 50), detection(400, 0, 600, 200)
    pairs = tracker.update([small, large])

    assert [det.bbox for _, det in pairs] == [small.bbox, large.bbox]


def test_identity_is_unstable_until_it_wins_the_window():
    track = Track(track_id=0, bbox=(0, 0, 10, 10))

    for _ in range(4):
        track.recent_names.append("alice")
    assert track.stable_identity(required=5) == (None, 4)

    track.recent_names.append("alice")
    assert track.stable_identity(required=5) == ("alice", 5)


def test_a_single_bad_frame_does_not_break_a_stable_identity():
    """Exactly the one-frame error that temporal smoothing exists to absorb."""
    track = Track(track_id=0, bbox=(0, 0, 10, 10))

    for _ in range(6):
        track.recent_names.append("alice")
    track.recent_names.append("bob")      # one bad frame

    name, count = track.stable_identity(required=5)
    assert name == "alice"
    assert count == 6


def test_unrecognised_frames_decay_a_stale_identity():
    """Walking out of frame should not leave a stale name confirmed forever."""
    from config import SMOOTHING_WINDOW

    track = Track(track_id=0, bbox=(0, 0, 10, 10))

    for _ in range(SMOOTHING_WINDOW):
        track.recent_names.append("alice")
    assert track.stable_identity(required=5)[0] == "alice"

    for _ in range(SMOOTHING_WINDOW):
        track.recent_names.append(None)
    assert track.stable_identity(required=5)[0] is None
