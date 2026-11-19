"""Unit tests for label-stat computation and pilot trend gate (no GPU)."""

from pathlib import Path

import train_detector.train_detector as td


def _write_labels(dir_path: Path, files: dict):
    dir_path.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (dir_path / name).write_text(content, encoding="utf-8")


def test_compute_label_stats_detect_boxes(tmp_path):
    labels = tmp_path / "labels"
    # normalized cx cy w h — two images, one empty
    _write_labels(labels, {
        "a.txt": "0 0.5 0.5 0.20 0.10\n0 0.3 0.3 0.40 0.20\n",
        "b.txt": "0 0.5 0.5 0.10 0.10\n",
        "c.txt": "",
    })
    stats = td.compute_label_stats(labels, task="detect")
    assert stats["label_files_sampled"] == 3
    assert stats["boxes_sampled"] == 3
    assert stats["median_box_w"] is not None
    assert stats["empty_label_fraction"] == round(1 / 3, 4)
    assert stats["boxes_per_image"] == 1.0


def test_compute_label_stats_segment_polygon_bbox(tmp_path):
    labels = tmp_path / "labels"
    # polygon: cls x1 y1 x2 y2 x3 y3 x4 y4 — bbox w=0.4 h=0.2
    _write_labels(labels, {
        "a.txt": "0 0.1 0.1 0.5 0.1 0.5 0.3 0.1 0.3\n",
    })
    stats = td.compute_label_stats(labels, task="segment")
    assert stats["boxes_sampled"] == 1
    assert abs(stats["median_box_w"] - 0.4) < 1e-6
    assert abs(stats["median_box_h"] - 0.2) < 1e-6


def test_prefer_imgsz_for_small_objects_sorts_descending():
    stats = {"median_box_w": 0.01, "median_box_h": 0.01, "p25_box_w": 0.005, "p25_box_h": 0.005}
    imgszs = [640, 1024, 1280]
    out = td.prefer_imgsz_for_objects(imgszs, stats, native_size=1280)
    assert out[0] == 1280
    assert out[-1] == 640


def test_prefer_imgsz_keeps_order_when_objects_large():
    stats = {"median_box_w": 0.2, "median_box_h": 0.2, "p25_box_w": 0.1, "p25_box_h": 0.1}
    imgszs = [640, 1024, 1280]
    assert td.prefer_imgsz_for_objects(imgszs, stats, native_size=1280) == imgszs


def test_pilot_trend_ok_requires_loss_down_and_score_up():
    assert td.pilot_trend_ok([
        {"loss": 2.0, "score": 0.1},
        {"loss": 1.0, "score": 0.3},
    ])
    assert not td.pilot_trend_ok([
        {"loss": 1.0, "score": 0.2},
        {"loss": 1.0, "score": 0.2},
    ])
    assert not td.pilot_trend_ok([
        {"loss": 2.0, "score": 0.3},
        {"loss": 1.0, "score": 0.2},  # loss down but score down
    ])


def test_object_px_at_imgsz():
    assert td.object_px_at_imgsz(0.05, 640) == 32.0
    assert td.object_px_at_imgsz(None, 640) is None
