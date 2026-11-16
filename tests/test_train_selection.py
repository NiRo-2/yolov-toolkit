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


def test_build_chosen_carries_fixed_aug_constants():
    chosen = td.build_chosen("yolo26x.pt", 640, 8, 4, 0.5, 1.0)
    assert chosen["degrees"] == 180
    assert chosen["flipud"] == 0.5
    assert chosen["close_mosaic"] == 60
    assert chosen["cos_lr"] is True
    assert chosen["nbs"] == 64
    # splat-safe through config_signature (record_calibration/run_probe style)
    td.config_signature(**chosen)


def test_config_signature_differs_on_close_mosaic_and_degrees():
    base = td.config_signature("yolo26x.pt", 640, 8, 4, 0.0, 1.0, 0.15, 0.3)
    same = td.config_signature("yolo26x.pt", 640, 8, 4, 0.0, 1.0, 0.15, 0.3)
    diff_cm = td.config_signature("yolo26x.pt", 640, 8, 4, 0.0, 1.0, 0.15, 0.3, close_mosaic=30)
    diff_deg = td.config_signature("yolo26x.pt", 640, 8, 4, 0.0, 1.0, 0.15, 0.3, degrees=45)
    assert base == same
    assert base != diff_cm
    assert base != diff_deg


def test_eight_key_resume_signature_matches_build_chosen_signature():
    """Configs written without the fixed constants (defaults) must hash identically."""
    eight = td.config_signature("yolo26x.pt", 640, 8, 4, 0.0, 1.0, 0.15, 0.3)
    thirteen = td.config_signature(**td.build_chosen("yolo26x.pt", 640, 8, 4, 0.0, 1.0, 0.15, 0.3))
    assert eight == thirteen


def test_selection_signature_contains_quality_intent_marker():
    args = argparse.Namespace(model=None, imgsz=None, batch=None, workers=None, memory_profile="safe")
    ds_fp = {"image_count": 123, "task": "detect", "names": ["a"]}
    sig = td.selection_signature(args, ds_fp)
    assert "intent:quality-v1" in sig


def test_load_memory_cache_discards_version_two_fixture(tmp_path, monkeypatch):
    import json
    cache_file = tmp_path / "memory_profiles.json"
    cache_file.write_text(json.dumps({
        "version": 2,
        "hardware_profiles": {"hw": {"selections": {"k": {"config_key": "old"}}, "configs": {}}},
    }), encoding="utf-8")
    monkeypatch.setattr(td, "MEMORY_CACHE_PATH", cache_file)
    fresh = td.load_memory_cache(FakeCalibrateLogger())
    assert fresh["version"] == 3
    assert fresh["hardware_profiles"] == {}


def make_resume_args(force_unsafe=False, no_memory_probe=False):
    return argparse.Namespace(
        force_unsafe=force_unsafe, no_memory_probe=no_memory_probe, memory_profile="safe",
    )


RESUME_CKPT_CFG = {
    "model": "yolo26x.pt", "imgsz": 640, "batch": 4, "workers": 4,
    "multi_scale": 0.5, "mosaic": 1.0, "mixup": 0.15, "copy_paste": 0.3,
}


def run_resume_calibrate(monkeypatch, tmp_path, peaks, yaml_path="use_tmp", device="0",
                         force_unsafe=False, no_memory_probe=False):
    calls = []

    def fake_probe(model_name, device, data_yaml, imgsz, batch, multi_scale, mosaic, mixup,
                   copy_paste, fraction, cuda_index, workers=0):
        calls.append({
            "model": model_name, "imgsz": imgsz, "batch": batch, "workers": workers,
            "multi_scale": multi_scale, "mosaic": mosaic, "mixup": mixup, "copy_paste": copy_paste,
        })
        return peaks

    monkeypatch.setattr(td, "probe_train_once", fake_probe)
    monkeypatch.setattr(td, "cuda_available_for", lambda d: d != "cpu")
    monkeypatch.setattr(td, "cuda_index_for", lambda d: 0)
    monkeypatch.setattr(td, "save_memory_cache", lambda cache, logger: True)
    monkeypatch.setattr(td, "_infer_resume_architecture_model",
                        lambda run_dir, ckpt_cfg=None, task="detect": "yolo26x.pt")

    if yaml_path == "use_tmp":
        ds_dir = tmp_path / "ds"
        imgs = ds_dir / "train" / "images"
        imgs.mkdir(parents=True)
        (imgs / "a.jpg").write_bytes(b"\xff\xd8\xff")
        (ds_dir / "data.yaml").write_text(
            "train: train/images\nval: val/images\nnc: 1\nnames: [a]\n", encoding="utf-8")
        yaml_arg = ds_dir / "data.yaml"
    else:
        yaml_arg = None

    logger = FakeCalibrateLogger()
    hw = {"vram_gb": 10.0, "ram_gb": 32.0, "cpu_cores": 8, "gpu_name": "FakeGPU"}
    ds_fp = {"median_img_size": 1280, "image_count": 1000, "task": "detect"}
    cache = {"version": 3, "hardware_profiles": {}}
    chosen, provenance = td.calibrate_resume_config(
        make_resume_args(force_unsafe=force_unsafe, no_memory_probe=no_memory_probe),
        logger, cache, hw, {"fp": 1}, ds_fp, dict(RESUME_CKPT_CFG), tmp_path / "run",
        yaml_path=yaml_arg, device=device,
    )
    return chosen, provenance, calls, cache, logger


def test_resume_keeps_probe_safe_checkpoint_multi_scale(monkeypatch, tmp_path):
    # 4 GB probe peak is well under the safe cap (10 * 0.68 = 6.8 GB) -> kept.
    chosen, provenance, calls, cache, logger = run_resume_calibrate(monkeypatch, tmp_path, (2.0, 4.0, 1.0))
    assert provenance == "resume-calibrated"
    assert len(calls) == 1
    assert calls[0]["multi_scale"] == 0.5
    assert calls[0]["workers"] == 4
    assert chosen["multi_scale"] == 0.5
    assert chosen["mosaic"] == 1.0


def test_resume_resets_multi_scale_when_probe_unsafe(monkeypatch, tmp_path):
    # 9 GB reserved > 6.8 GB cap -> multi_scale reset, rest of the aug kept.
    chosen, provenance, calls, cache, logger = run_resume_calibrate(monkeypatch, tmp_path, (6.0, 9.0, 1.0))
    assert len(calls) == 1
    assert chosen["multi_scale"] == 0.0
    assert chosen["mosaic"] == 1.0
    assert chosen["mixup"] == 0.15
    assert chosen["copy_paste"] == 0.3


def test_resume_resets_multi_scale_when_probe_not_possible(monkeypatch, tmp_path):
    # No dataset yaml (or CPU device) -> unverified -> multi_scale reset to 0.
    chosen, provenance, calls, cache, logger = run_resume_calibrate(monkeypatch, tmp_path, (2.0, 4.0, 1.0),
                                                                     yaml_path=None)
    assert calls == []
    assert chosen["multi_scale"] == 0.0

    chosen_cpu, _, calls_cpu, _, _ = run_resume_calibrate(monkeypatch, tmp_path, (2.0, 4.0, 1.0), device="cpu")
    assert calls_cpu == []
    assert chosen_cpu["multi_scale"] == 0.0


def test_resume_force_unsafe_keeps_checkpoint_multi_scale_unprobed(monkeypatch, tmp_path):
    chosen, provenance, calls, cache, logger = run_resume_calibrate(
        monkeypatch, tmp_path, (2.0, 9.0, 1.0), force_unsafe=True)
    assert calls == []
    assert chosen["multi_scale"] == 0.5
    assert provenance == "resume-calibrated"


def test_resume_selection_cache_key_carries_intent_marker(monkeypatch, tmp_path):
    chosen, provenance, calls, cache, logger = run_resume_calibrate(monkeypatch, tmp_path, (2.0, 4.0, 1.0))
    profile = list(cache["hardware_profiles"].values())[0]
    keys = list(profile["resume_selections"].keys())
    assert len(keys) == 1
    assert "intent:quality-v1" in keys[0]

    # The key written on the first run must be the key the lookup reads back:
    # a second calibration against the SAME cache short-circuits to "cache".
    chosen2, provenance2 = td.calibrate_resume_config(
        make_resume_args(), FakeCalibrateLogger(), cache,
        {"vram_gb": 10.0, "ram_gb": 32.0, "cpu_cores": 8, "gpu_name": "FakeGPU"},
        {"fp": 1}, {"median_img_size": 1280, "image_count": 1000, "task": "detect"},
        dict(RESUME_CKPT_CFG), tmp_path / "run",
        yaml_path=tmp_path / "ds" / "data.yaml", device="0",
    )
    assert provenance2 == "cache"
    assert chosen2["multi_scale"] == 0.5  # the probe-kept value survives the cache round-trip


def make_cfg(mosaic, copy_paste, mixup, multi_scale, batch=8):
    cfg = {"model": "yolo26x.pt", "imgsz": 640, "batch": batch, "workers": 2,
           "mosaic": mosaic, "copy_paste": copy_paste, "mixup": mixup, "multi_scale": multi_scale}
    cfg.update(td.build_chosen("yolo26x.pt", 640, batch, 2, multi_scale, mosaic,
                               mixup=mixup, copy_paste=copy_paste))
    return cfg


def test_aug_step_up_candidate_climbs_one_rung(monkeypatch):
    # A climb moves exactly one rung up the declared AUG_LADDER.
    cand = td.aug_step_up_candidate(make_cfg(0.5, 0.0, 0.0, 0.0))
    assert (cand["mosaic"], cand["copy_paste"], cand["mixup"], cand["multi_scale"]) == (0.5, 0.3, 0.0, 0.0)
    # ms 0 rung under the top -> heaviest rung (plan example: ms 0->0.5)
    cand = td.aug_step_up_candidate(make_cfg(1.0, 0.3, 0.15, 0.0))
    assert (cand["mosaic"], cand["copy_paste"], cand["mixup"], cand["multi_scale"]) == (1.0, 0.3, 0.15, 0.5)
    # mixup climb
    cand = td.aug_step_up_candidate(make_cfg(1.0, 0.3, 0.0, 0.0))
    assert (cand["mosaic"], cand["copy_paste"], cand["mixup"], cand["multi_scale"]) == (1.0, 0.3, 0.15, 0.0)
    # carried constants survive the climb
    assert cand["degrees"] == 180 and cand["close_mosaic"] == 60
    # repeated climbs from the bottom rung walk the whole ladder to the top,
    # visiting mosaic 0.5 -> 1.0 and multi_scale 0 -> 0.5 along the way.
    cfg = make_cfg(0.0, 0.0, 0.0, 0.0)
    visited = []
    for _ in range(len(AUG_LADDER)):
        cand = td.aug_step_up_candidate(cfg)
        if cand is None:
            break
        cfg = cand
        visited.append((cfg["mosaic"], cfg["copy_paste"], cfg["mixup"], cfg["multi_scale"]))
    assert visited == AUG_LADDER[:7][::-1]
    # the top rung is the plan's example endpoint: mosaic 1.0 with ms 0.5
    assert visited[-1] == (1.0, 0.3, 0.15, 0.5)
    assert td.aug_step_up_candidate(cfg) is None  # already at the top


def test_print_auto_config_states_quality_first_policy():
    logger = FakeCalibrateLogger()
    chosen = td.build_chosen("yolo26x.pt", 1024, 8, 4, 0.5, 1.0)
    args = argparse.Namespace(batch=None, workers=None)
    hw = {"gpu_name": "RTX 4080", "vram_gb": 16.0, "cpu_cores": 12, "ram_gb": 32.0}
    ds_fp = {"image_count": 1000, "median_img_size": 1280}
    td.print_auto_config(logger, chosen, "measured", args, hw, ds_fp, "0")
    policy = [l for l in logger.lines if "[Selection policy]" in l]
    assert len(policy) == 1
    assert "quality-first" in policy[0]
    assert "speed not prioritized" in policy[0]


def test_aug_step_up_candidate_returns_none_when_maxed_or_off_ladder():
    assert td.aug_step_up_candidate(make_cfg(1.0, 0.3, 0.15, 0.5)) is None
    assert td.aug_step_up_candidate(make_cfg(0.7, 0.25, 0.1, 0.3)) is None


class FakeGuard:
    def peak_reserved(self):
        return 1.0

    def start(self):
        pass

    def stop(self):
        pass


def run_supervisor_stepup(monkeypatch, tmp_path, start_aug, unsafe_rungs):
    """Drive run_adaptive_supervisor with stubbed attempts/probes; return (attempt_cfgs, probe_cfgs, status)."""
    attempt_cfgs = []
    probe_cfgs = []

    def fake_probe(model_name, device, data_yaml, imgsz, batch, multi_scale, mosaic, mixup,
                   copy_paste, fraction, cuda_index, workers=0):
        probe_cfgs.append((mosaic, copy_paste, mixup, multi_scale, batch))
        if (mosaic, copy_paste, mixup, multi_scale) in unsafe_rungs:
            return (9.0, 9.5, 1.0)  # > safe cap 10*0.68 = 6.8 GB
        return (1.0, 2.0, 1.0)

    def fake_run_attempt(args, logger, run_dir, current_ckpt, current_cfg, target_epochs,
                         device, guard, yaml_path, ckpt_cfg, mode, attempt_no, state):
        attempt_cfgs.append((current_cfg["mosaic"], current_cfg["multi_scale"], current_cfg["batch"]))
        state["completed_epoch"] = state.get("completed_epoch", 0) + 1
        import types
        tracker = types.SimpleNamespace(
            status="completed-attempt",
            completed_epoch=state["completed_epoch"],
            adaptive_triggered=False,
            emergency_triggered=False,
            oom_auto_reduced=False,
        )
        return tracker

    monkeypatch.setattr(td, "probe_train_once", fake_probe)
    monkeypatch.setattr(td, "cuda_available_for", lambda d: True)
    monkeypatch.setattr(td, "cuda_index_for", lambda d: 0)
    monkeypatch.setattr(td, "run_attempt", fake_run_attempt)

    args = argparse.Namespace(patience=100, memory_profile="safe", no_memory_probe=False, force_unsafe=False)
    hw = {"vram_gb": 10.0, "ram_gb": 32.0, "cpu_cores": 8, "gpu_name": "FakeGPU"}
    chosen = make_cfg(*start_aug)
    state = {"run_name": "t", "target_epochs": 12, "completed_epoch": 0,
             "provisional": True, "stable_epochs": 0, "downgrade_count": 0, "history": []}
    status = td.run_adaptive_supervisor(
        args=args, logger=FakeCalibrateLogger(), run_dir=tmp_path / "run", run_name="t",
        last_pt=None, chosen=chosen, target_epochs=12, device="0", hw=hw,
        guard=FakeGuard(), state=state, yaml_path=None, ckpt_cfg={}, mode="fresh",
    )
    return attempt_cfgs, probe_cfgs, status


def test_supervisor_steps_augmentation_up_when_stable_and_safe(monkeypatch, tmp_path):
    attempt_cfgs, probe_cfgs, status = run_supervisor_stepup(
        monkeypatch, tmp_path, start_aug=(0.5, 0.0, 0.0, 0.0), unsafe_rungs=set())
    assert status == "completed-target"
    # attempts start on the 0.5-mosaic rung and climb to mosaic 1.0 after stability
    assert attempt_cfgs[0] == (0.5, 0.0, 8)
    assert any(c[0] == 1.0 for c in attempt_cfgs)
    assert attempt_cfgs[-1][0] == 1.0
    # the candidate rung was probe-validated before approval
    assert (1.0, 0.0, 0.0, 0.0, 8) in probe_cfgs


def test_supervisor_keeps_aug_when_step_up_probe_rejects(monkeypatch, tmp_path):
    attempt_cfgs, probe_cfgs, status = run_supervisor_stepup(
        monkeypatch, tmp_path, start_aug=(0.5, 0.0, 0.0, 0.0),
        unsafe_rungs={(1.0, 0.0, 0.0, 0.0)})
    assert (1.0, 0.0, 0.0, 0.0, 8) in probe_cfgs  # candidate was attempted
    assert all(c[0] == 0.5 for c in attempt_cfgs)  # ...and never adopted
