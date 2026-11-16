"""Tests for the shared common/hardware.py module."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.hardware import (  # noqa: E402
    BATCH_LADDER,
    MEMORY_PROFILES,
    PSUTIL_AVAILABLE,
    detect_hardware,
    looks_like_oom,
    process_rss_bytes,
    resolve_profile,
    clamp_to_ladder,
)


def test_memory_profiles_match_documented_fractions():
    assert MEMORY_PROFILES == {
        "safe":       {"gpu": 0.68, "cpu": 0.38},
        "balanced":   {"gpu": 0.75, "cpu": 0.45},
        "aggressive": {"gpu": 0.85, "cpu": 0.55},
    }


def test_batch_ladder_value():
    assert BATCH_LADDER == [2, 4, 8, 16, 32, 64]


def test_resolve_profile_known_and_unknown():
    assert resolve_profile("safe")["gpu"] == 0.68
    assert resolve_profile("BALANCED")["cpu"] == 0.45
    assert resolve_profile("nonsense") == MEMORY_PROFILES["balanced"]


def test_clamp_to_ladder():
    assert clamp_to_ladder(1) == 2
    assert clamp_to_ladder(2) == 2
    assert clamp_to_ladder(5) == 4
    assert clamp_to_ladder(8) == 8
    assert clamp_to_ladder(100) == 64


def test_detect_hardware_keys():
    hw = detect_hardware()
    assert set(hw.keys()) == {"vram_gb", "cpu_cores", "ram_gb", "gpu_name"}
    assert hw["cpu_cores"] >= 1
    assert isinstance(hw["gpu_name"], str)


def test_looks_like_oom_returns_false_when_torch_none(monkeypatch):
    import common.hardware as h
    monkeypatch.setattr(h, "torch", None)
    assert h.looks_like_oom(RuntimeError("out of memory")) is False


def test_looks_like_oom_text_matches(monkeypatch):
    import common.hardware as h

    class FakeCuda:
        class OutOfMemoryError(Exception):
            pass

    class FakeTorch:
        cuda = FakeCuda

    monkeypatch.setattr(h, "torch", FakeTorch)
    assert h.looks_like_oom(FakeTorch.cuda.OutOfMemoryError()) is True
    assert h.looks_like_oom(RuntimeError("CUDA out of memory")) is True
    assert h.looks_like_oom(RuntimeError("cannot allocate memory")) is True
    assert h.looks_like_oom(RuntimeError("some other error")) is False


def test_process_rss_bytes_returns_int_or_none():
    rss = process_rss_bytes()
    assert rss is None or isinstance(rss, int)


def test_train_detector_reexports_moved_names():
    """train_detector.py must still expose every moved name (re-export)."""
    import importlib

    td_path = REPO_ROOT / "train_detector" / "train_detector.py"
    spec = importlib.util.spec_from_file_location(
        "td_reexport_check", str(td_path)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert mod.BATCH_LADDER == BATCH_LADDER
    assert mod.MEMORY_PROFILES == MEMORY_PROFILES
    assert mod.PSUTIL_AVAILABLE == PSUTIL_AVAILABLE
    assert mod.detect_hardware is detect_hardware
    assert mod.process_rss_bytes is process_rss_bytes
    assert mod.looks_like_oom is looks_like_oom