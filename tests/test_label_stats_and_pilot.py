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


def test_filter_pilot_history_drops_final_eval_refire():
    """Ultralytics final_eval() re-fires on_fit_epoch_end past the scheduled epochs
    with a bogus loss spike; that must not flip a healthy pilot to FAIL (rust_Merged)."""
    history = [
        {"epoch": 1, "loss": 4.39766, "score": 0.00022},
        {"epoch": 2, "loss": 4.36458, "score": 0.00057},
        {"epoch": 3, "loss": 4.0965, "score": 0.00031},
        {"epoch": 4, "loss": 4.1126, "score": 0.00092},
        {"epoch": 5, "loss": 4.08877, "score": 0.00076},
        {"epoch": 6, "loss": 51.484, "score": 0.00076},  # final_eval re-fire
    ]
    filtered = td.filter_pilot_history(history, scheduled_epochs=5)
    assert [h["epoch"] for h in filtered] == [1, 2, 3, 4, 5]
    assert td.pilot_trend_ok(filtered)
    assert not td.pilot_trend_ok(history)  # unfiltered would false-fail


def test_metric_from_trainer_prefers_tloss_dict_over_last_batch_loss():
    """on_fit_epoch_end: trainer.metrics is val-only, trainer.loss is last-batch raw,
    trainer.tloss is the epoch-mean dict shown in the progress bar (rust_Merged false FAIL)."""
    class _Trainer:
        metrics = {"metrics/mAP50(B)": 0.00022, "val/box_loss": 4.39766}
        loss = 50.599632263183594  # last-batch raw — must NOT win for train/box_loss
        tloss = {"box_loss": 2.536, "cls_loss": 7.792, "l1_loss": 0.04277}

        def label_loss_items(self, loss_items=None, prefix="train"):
            if loss_items is None:
                return [f"{prefix}/{x}" for x in ("box_loss", "cls_loss", "l1_loss")]
            return {f"{prefix}/{k}": float(v) for k, v in loss_items.items()}

    t = _Trainer()
    assert td._metric_from_trainer(t, "train/box_loss", "train/loss") == 2.536
    assert td._metric_from_trainer(t, "metrics/mAP50(B)", "mAP50") == 0.00022
    # rust_Merged pilot numbers with correct loss → PASS
    history = [
        {"epoch": 1, "loss": 2.536, "score": 0.00022},
        {"epoch": 5, "loss": 2.436, "score": 0.00076},
    ]
    assert td.pilot_trend_ok(history)


def test_estimate_calibration_probe_budget():
    # 2 healthy × 4 rungs + 2 degraded × 4 rungs + finalize slack
    assert td.estimate_calibration_probe_budget(
        healthy=[32, 16], degraded=[8, 4], ladder_len=4, finalize_extra=4,
    ) == 2 * 4 + 2 * 4 + 4


def test_format_hardware_test_progress():
    msg = td.format_hardware_test_progress(
        7, 20, model="yolo26x.pt", imgsz=640, batch=8, workers=0,
    )
    assert msg.startswith("[Hardware test] 7/20:")
    assert "yolo26x.pt" in msg
    assert "batch 8" in msg


def test_object_px_at_imgsz():
    assert td.object_px_at_imgsz(0.05, 640) == 32.0
    assert td.object_px_at_imgsz(None, 640) is None
