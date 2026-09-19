from bokasher.tracker import FaceTracker, iou, merge_boxes
from bokasher.types import FaceBox


def test_iou_identical() -> None:
    box = FaceBox(10, 10, 40, 40)
    assert iou(box, box) == 1.0


def test_iou_no_overlap() -> None:
    assert iou(FaceBox(0, 0, 10, 10), FaceBox(20, 20, 10, 10)) == 0.0


def test_tracker_matches_and_smooths() -> None:
    tracker = FaceTracker(max_missed=2)
    first = tracker.update([FaceBox(0, 0, 20, 20)])
    assert first[0].w == 20
    second = tracker.update([FaceBox(10, 0, 20, 20)])
    assert len(second) == 1
    # 低めの EMA なので、新しい検出へは少しだけ寄る
    assert 0 < second[0].x < 10


def test_tracker_keeps_boxes_when_detection_skipped() -> None:
    tracker = FaceTracker()
    first = tracker.update([FaceBox(2, 2, 12, 12)])[0]
    kept = tracker.update(None)[0]
    assert abs(kept.x - first.x) <= 1
    assert abs(kept.w - first.w) <= 1


def test_skip_frames_do_not_count_as_misses() -> None:
    tracker = FaceTracker(max_missed=2)
    tracker.update([FaceBox(10, 10, 20, 20)])
    for _ in range(8):
        boxes = tracker.update(None)
        assert boxes
    assert tracker.update([FaceBox(12, 10, 20, 20)])


def test_merge_boxes_keeps_higher_score() -> None:
    kept = merge_boxes(
        [
            FaceBox(0, 0, 20, 20, score=0.4),
            FaceBox(2, 2, 20, 20, score=0.9),
            FaceBox(80, 80, 10, 10, score=0.8),
        ]
    )
    assert len(kept) == 2
    assert kept[0].score == 0.9


def test_tracker_drops_after_misses() -> None:
    tracker = FaceTracker(max_missed=1)
    tracker.update([FaceBox(0, 0, 10, 10)])
    tracker.update([])
    assert tracker.update([]) == []


def test_tracker_holds_box_when_detection_stops() -> None:
    tracker = FaceTracker(max_missed=30)
    tracker.update([FaceBox(10, 10, 30, 30)])
    for _ in range(5):
        boxes = tracker.update([], width=100, height=100)
        assert boxes
    assert len(boxes) == 1


def test_tracker_does_not_pulse_size_on_skip() -> None:
    tracker = FaceTracker()
    first = tracker.update([FaceBox(20, 20, 40, 40)])[0]
    skipped = tracker.update(None, width=200, height=200)[0]
    assert skipped.w == first.w
    assert skipped.h == first.h


def test_tracker_caps_noisy_detections() -> None:
    tracker = FaceTracker(max_tracks=3, max_missed=24)
    for i in range(20):
        tracker.update([FaceBox(i * 15, 0, 8, 8, score=0.4)])
    assert len(tracker.update(None)) <= 3


def test_tracker_survives_mixed_detection_and_skips() -> None:
    tracker = FaceTracker(max_missed=30)
    tracker.update([FaceBox(10, 10, 30, 30)])
    tracker.update(None, 100, 100)
    tracker.update(None, 100, 100)
    boxes = tracker.update([], width=100, height=100)
    assert boxes
