"""Pure-function tests for the quality-first auto selection policy.

No GPU, no training, no probes: these lock the selection ladder semantics
(model ladder, augmentation ladder, config/selection signatures, cache schema).
"""

import argparse
from pathlib import Path

import pytest

import train_detector.train_detector as td
from train_detector.train_detector import AUG_LADDER, model_candidates


EXPECTED_GPU_CHAIN = [
    "yolo26x.pt",
    "yolo26l.pt",
    "yolo26m.pt",
    "yolo26n.pt",
]


@pytest.mark.parametrize("image_count,vram_gb", [
    (10, 8.0),        # tiny dataset on a modest GPU still starts at x
    (1000, 12.0),     # mid-size dataset: no 'm'/'l' size up-shift anymore
    (100000, 8.0),    # big dataset under 16 GB VRAM no longer gates x out
])
def test_gpu_model_candidates_are_largest_first_for_every_dataset_size(image_count, vram_gb):
    assert model_candidates(image_count, vram_gb, for_cpu=False) == EXPECTED_GPU_CHAIN


@pytest.mark.parametrize("task,suffix", [
    ("detect", ""),
    ("segment", "-seg"),
    ("pose", "-pose"),
])
def test_gpu_model_candidates_carry_task_suffixes(task, suffix):
    assert model_candidates(100000, 24.0, for_cpu=False, task=task) == [
        f"yolo26x{suffix}.pt",
        f"yolo26l{suffix}.pt",
        f"yolo26m{suffix}.pt",
        f"yolo26n{suffix}.pt",
    ]


@pytest.mark.parametrize("image_count", [10, 1000, 100000])
def test_cpu_model_candidates_start_at_l_and_exclude_x(image_count):
    assert model_candidates(image_count, None, for_cpu=True) == [
        "yolo26l.pt",
        "yolo26m.pt",
        "yolo26n.pt",
    ]


def test_aug_ladder_begins_at_heaviest_combo():
    assert AUG_LADDER[0] == (1.0, 0.3, 0.15, 0.5)


def test_aug_ladder_is_strictly_non_increasing():
    # (mosaic, copy_paste, mixup, multi_scale) intensity must strictly step down rung by rung.
    for i in range(1, len(AUG_LADDER)):
        assert AUG_LADDER[i] < AUG_LADDER[i - 1]


class FakeCalibrateLogger:
    def __init__(self):
        self.lines = []

    def log(self, msg):
        self.lines.append(msg)


def make_calibrate_args():
    return argparse.Namespace(
        model=None, imgsz=640, batch=None, workers=None,
        memory_profile="safe", no_memory_probe=False, force_unsafe=False,
    )


def run_calibrate(monkeypatch, probe_peaks):
    """Run calibrate_training_config with fake hardware + fake probes; return (chosen, calls)."""
    calls = []

    def fake_probe(model_name, device, data_yaml, imgsz, batch, multi_scale, mosaic, mixup,
                   copy_paste, fraction, cuda_index, workers=0):
        calls.append({
            "model": model_name, "imgsz": imgsz, "batch": batch,
            "multi_scale": multi_scale, "mosaic": mosaic, "mixup": mixup,
            "copy_paste": copy_paste, "workers": workers,
        })
        return probe_peaks(calls[-1])

    monkeypatch.setattr(td, "cuda_available_for", lambda device: True)
    monkeypatch.setattr(td, "cuda_index_for", lambda device: 0)
    monkeypatch.setattr(td, "probe_train_once", fake_probe)
    monkeypatch.setattr(td, "get_vram_per_image", lambda model: 0.3)
    monkeypatch.setattr(td, "save_memory_cache", lambda cache, logger: True)

    hw = {"vram_gb": 4.0, "ram_gb": 16.0, "cpu_cores": 8, "gpu_name": "FakeGPU"}
    ds_fp = {"image_count": 1000, "median_img_size": None}
    cache = {"version": 3, "hardware_profiles": {}}
    chosen, provenance = td.calibrate_training_config(
        make_calibrate_args(), FakeCalibrateLogger(), cache, {"fake": 1}, hw, "0",
        Path("fake/data.yaml"), Path("fake/train/images"), ds_fp,
    )
    return chosen, provenance, calls


def heavy_unsafe(cfg):
    # multi-scale rungs blow past the 4 GB x 0.68 = 2.72 GB safe cap.
    if cfg["multi_scale"] > 0:
        return (4.0, 5.0, 0.5)
    return (1.0, 1.5, 0.5)


def heavy_safe(cfg):
    return (1.0, 1.2, 0.5)


def test_calibrate_attempts_heaviest_aug_rung_first(monkeypatch):
    chosen, provenance, calls = run_calibrate(monkeypatch, heavy_safe)
    # The batch-sizing ladder probes run with rung (1.0, 0.3, 0.15, 0.0) first by
    # design. The first probe that attempts any multi_scale must be the heaviest
    # rung (mosaic 1.0, copy_paste 0.3, mixup 0.15, ms 0.5) — not the old
    # ms 0.5 + mosaic 0.5 combo.
    workers0 = [c for c in calls if c["workers"] == 0]
    first_ms_probe = next(c for c in workers0 if c["multi_scale"] > 0)
    assert first_ms_probe["multi_scale"] == 0.5
    assert first_ms_probe["mosaic"] == 1.0
    assert first_ms_probe["mixup"] == 0.15
    assert first_ms_probe["copy_paste"] == 0.3
    assert not any(c["multi_scale"] == 0.5 and c["mosaic"] == 0.5 for c in calls)
    # ...and when it probes safe, it is selected.
    assert provenance == "measured"
    assert chosen["multi_scale"] == 0.5
    assert chosen["mosaic"] == 1.0


def test_calibrate_steps_down_ladder_when_heaviest_unsafe(monkeypatch):
    chosen, provenance, calls = run_calibrate(monkeypatch, heavy_unsafe)
    assert provenance == "measured"
    # Heaviest rung attempted and failed; selection lands on the next safe rung
    # (mosaic 1.0, mixup 0.15, ms 0.0), not the old ms 0.5 + mosaic 0.5 combo.
    workers0 = [c for c in calls if c["workers"] == 0]
    first_ms_probe = next(c for c in workers0 if c["multi_scale"] > 0)
    assert first_ms_probe["multi_scale"] == 0.5 and first_ms_probe["mosaic"] == 1.0
    assert chosen["multi_scale"] == 0.0
    assert chosen["mosaic"] == 1.0
    assert chosen["mixup"] == 0.15
    assert chosen["copy_paste"] == 0.3
    # The old standalone "multi-scale upgrade (ms 0.5, mosaic 0.5)" probe is gone.
    assert not any(c["multi_scale"] == 0.5 and c["mosaic"] == 0.5 for c in calls)
