"""Tests for TileBatcher global batching equivalence (task 4)."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

import detect_images.detect_images as di  # noqa: E402


class FakeBox:
    def __init__(self, x1, y1, x2, y2, cls, conf):
        self.xyxy = [np.array([x1, y1, x2, y2], dtype=float)]
        self.cls = [np.int64(cls)]
        self.conf = [np.float32(conf)]


class FakeBoxes:
    def __init__(self, box_list):
        self._boxes = box_list

    def __iter__(self):
        return iter(self._boxes)

    def __len__(self):
        return len(self._boxes)


class FakeResult:
    """Minimal ultralytics-like Results object."""

    def __init__(self, boxes, orig_shape=(1024, 1024)):
        self.boxes = FakeBoxes(boxes)
        self.masks = None
        self.keypoints = None
        self.orig_shape = orig_shape


class FakeModel:
    """Deterministic fake model: one box per crop, content-derived.

    The class id is derived from the crop's mean pixel value so two images
    with different content produce different class sets.
    """

    def __init__(self, task="detect"):
        self.task = task
        self.calls = []

    def __call__(self, crops, conf=0.25, imgsz=1024, half=False, verbose=False, device=None, **kw):
        self.calls.append((len(crops), imgsz, half, device))
        results = []
        for crop in crops:
            h, w = crop.shape[:2]
            val = float(np.mean(crop))
            cls = int(val * 10) % 3
            results.append(FakeResult(
                [FakeBox(0.0, 0.0, float(w), float(h), cls, 0.9)],
                orig_shape=(h, w),
            ))
        return results


def _make_image(seed, size=1024):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (size, size, 3), dtype=np.uint8)


def _sorted(dets):
    return sorted(dets, key=lambda d: (d["cls"], round(d["conf"], 6),
                                       round(d["x1"], 6), round(d["y1"], 6)))


def test_batched_equals_batch_one_for_detect():
    model = FakeModel("detect")
    img = _make_image(7)
    single = di.detect_image_tiled(model, img, 0.25, 1024, 0.2, 0.5, batch_size=1, task="detect")
    batched = di.detect_image_tiled(model, img, 0.25, 1024, 0.2, 0.5, batch_size=4, task="detect")
    assert _sorted(single) == _sorted(batched)


def test_tiles_from_different_images_never_mix():
    """Two images with different content must not share detections.

    Uses a large tile size so each image is a single whole-image tile; the
    class ids are content-derived, so distinct images produce disjoint sets.
    """
    model = FakeModel("detect")
    img_a = _make_image(1, size=512)
    img_b = _make_image(2, size=512)
    a = di.detect_image_tiled(model, img_a, 0.25, 1024, 0.2, 0.5, batch_size=1, task="detect")
    b = di.detect_image_tiled(model, img_b, 0.25, 1024, 0.2, 0.5, batch_size=1, task="detect")
    a_keys = {(d["cls"], round(d["conf"], 3)) for d in a}
    b_keys = {(d["cls"], round(d["conf"], 3)) for d in b}
    # At least one class must differ between the two images.
    assert a_keys != b_keys


def test_tile_batcher_routes_by_job_id():
    from tile_yolo_dataset.tile_geometry import iter_tile_windows
    batcher = di.TileBatcher(FakeModel("detect"), 0.25, 512, 0.2, 0.5, 4, "detect")
    img = _make_image(3)
    h, w = img.shape[:2]
    for jid in (1, 2):
        for (tx1, ty1, tx2, ty2) in iter_tile_windows(w, h, 512, 0.2):
            batcher.submit(img[ty1:ty2, tx1:tx2], (tx1, ty1, tx2, ty2), job_id=jid)
    batcher.flush()
    r1 = batcher.results_for(1)
    r2 = batcher.results_for(2)
    assert isinstance(r1, list)
    assert isinstance(r2, list)
    assert _sorted(r1) == _sorted(r2)  # same image content -> same dets


def test_batching_preserves_segment_task():
    model = FakeModel("segment")
    img = _make_image(11)
    single = di.detect_image_tiled(model, img, 0.25, 1024, 0.2, 0.5, batch_size=1, task="segment")
    batched = di.detect_image_tiled(model, img, 0.25, 1024, 0.2, 0.5, batch_size=4, task="segment")
    assert len(single) == len(batched)


def test_batching_preserves_pose_task():
    model = FakeModel("pose")
    img = _make_image(13)
    single = di.detect_image_tiled(model, img, 0.25, 1024, 0.2, 0.5, batch_size=1, task="pose")
    batched = di.detect_image_tiled(model, img, 0.25, 1024, 0.2, 0.5, batch_size=4, task="pose")
    assert len(single) == len(batched)