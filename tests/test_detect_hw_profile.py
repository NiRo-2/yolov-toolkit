"""Tests for detect_images.py hardware-aware resolution."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import detect_images.detect_images as di  # noqa: E402


def test_resolve_device_auto_picks_cuda_when_available(monkeypatch):
    monkeypatch.setattr(di, "_cuda_available", lambda: True)
    assert di.resolve_device("auto") == ("cuda:0", "auto")
    monkeypatch.setattr(di, "_cuda_available", lambda: False)
    assert di.resolve_device("auto") == ("cpu", "auto")


def test_resolve_device_explicit_cpu(monkeypatch):
    monkeypatch.setattr(di, "_cuda_available", lambda: True)
    assert di.resolve_device("cpu") == ("cpu", "manual")
    assert di.resolve_device("0") == ("cuda:0", "manual")


def test_resolve_device_cpu_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(di, "_cuda_available", lambda: False)
    dev, src = di.resolve_device("0")
    assert dev == "cpu"
    assert "CUDA unavailable" in src


def test_resolve_half_auto(monkeypatch):
    assert di.resolve_half("auto", True) == (True, "auto")
    assert di.resolve_half("auto", False) == (False, "auto")
    assert di.resolve_half("on", False) == (True, "manual")
    assert di.resolve_half("off", True) == (False, "manual")


def test_resolve_half_invalid_exits(monkeypatch, capsys):
    monkeypatch.setattr(sys, "exit", lambda code=0: (_ for _ in ()).throw(SystemExit(code)))
    try:
        di.resolve_half("bogus", True)
    except SystemExit:
        return
    raise AssertionError("expected SystemExit for invalid --half")


def test_resolve_profile_fractions():
    assert di.resolve_profile("safe")["gpu"] == 0.68
    assert di.resolve_profile("balanced")["cpu"] == 0.45
    assert di.resolve_profile("aggressive")["gpu"] == 0.85
    assert di.resolve_profile("unknown") == di.MEMORY_PROFILES["balanced"]


def test_resolve_workers_auto_caps_at_eight():
    n, src = di.resolve_workers("auto", 24, 4, False, False)
    assert src == "auto"
    assert n <= 8
    assert n >= 1


def test_resolve_workers_manual():
    assert di.resolve_workers("3", 24, 4, False, False) == (3, "manual")
    assert di.resolve_workers("0", 24, 4, False, False) == (1, "manual")


def test_resolve_decode_workers_auto():
    n, src = di.resolve_decode_workers("auto", 24)
    assert src == "auto"
    assert 1 <= n <= 6
    assert di.resolve_decode_workers("auto", 6) == (1, "auto")


def test_resolve_batch_auto_uses_budget():
    # 100 MB budget, 10 MB per item -> 10 items, ladder clamps to 8.
    n, src = di.resolve_batch("auto", 100 * 1024 * 1024, 10 * 1024 * 1024, True, False)
    assert src == "auto"
    assert n == 8


def test_resolve_batch_auto_no_budget_falls_back():
    n, src = di.resolve_batch("auto", 0, 0, True, False)
    assert n == 8
    n2, _ = di.resolve_batch("auto", 0, 0, False, False)
    assert n2 == 1


def test_resolve_batch_manual_floor_one():
    assert di.resolve_batch("0", 0, 0, True, False) == (1, "manual")
    assert di.resolve_batch("3", 0, 0, True, False) == (3, "manual")


def test_only_detections_flag_removed():
    """--only-detections was a no-op; argparse must now reject it."""
    import argparse
    # Re-run parse_args with a fake argv; the flag should be unknown.
    parser = argparse.ArgumentParser()
    # Mirror the real parser minimally by calling parse_args with --help-like rejection.
    # Direct check: build a tiny parser with the same set of store_true flags and
    # confirm the old flag is not registered.
    import detect_images.detect_images as mod
    src = Path(mod.__file__).read_text()
    assert "--only-detections" not in src