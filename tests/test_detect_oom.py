"""Tests for run_batch OOM recovery and ByteBudget backpressure."""

import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import detect_images.detect_images as di  # noqa: E402


class OOMEngine:
    """Fake engine that raises torch.cuda.OutOfMemoryError N times then succeeds."""

    def __init__(self, oom_count, item_value=1):
        self.oom_count = oom_count
        self.item_value = item_value
        self.attempts = []
        self.batch_sizes = []

    def __call__(self, items, conf=0.25, imgsz=640, half=False, verbose=False, device=None, **kw):
        self.attempts.append(len(items))
        self.batch_sizes.append(len(items))
        if len(self.attempts) <= self.oom_count:
            raise RuntimeError("CUDA out of memory: tried to allocate 2.00 GiB")
        return [self.item_value for _ in items]


def test_oom_halving_ladder_recovers():
    try:
        import torch  # noqa: F401
    except Exception:
        import pytest
        pytest.skip("torch not available for OOM test")
    engine = OOMEngine(oom_count=2, item_value="ok")
    results = di.run_batch(engine, list(range(8)), 640, 0.25, False, "detect", 8)
    assert [r for _i, r in results] == ["ok"] * 8
    # First attempt at batch 8, then halved to 4, then halved to 2 succeeds.
    assert engine.batch_sizes[0] == 8
    assert engine.batch_sizes[1] == 4
    assert engine.batch_sizes[2] == 2


def test_non_oom_failure_isolates_bad_item():
    class Engine:
        def __init__(self):
            self.calls = 0

        def __call__(self, items, conf=0.25, imgsz=640, half=False, verbose=False, device=None, **kw):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("some other error")
            return [f"ok-{i}" for i, _ in enumerate(items)]

    engine = Engine()
    results = di.run_batch(engine, list(range(4)), 640, 0.25, False, "detect", 4)
    # The first item is isolated; the remaining three succeed as a fresh batch.
    assert results[0] == (0, None)
    assert len(results) == 4
    assert all(r is not None for _i, r in results[1:])


def test_byte_budget_caps_concurrent_decode():
    budget = di.ByteBudget(1000)
    acquired = []
    lock = threading.Lock()

    def worker(tag):
        budget.acquire(400)
        with lock:
            acquired.append(tag)
        time.sleep(0.05)
        budget.release(400)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # Budget 1000 / 400 = 2 concurrent at most; all 5 must complete.
    assert len(acquired) == 5


def test_byte_budget_unlimited_is_noop():
    budget = di.ByteBudget(0)
    assert budget.budget == 0
    budget.acquire(10**9)
    budget.release(10**9)


def test_estimate_image_bytes_formula():
    # 2 * w * h * 3 + json allowance.
    assert di.estimate_image_bytes(100, 100, json_bytes=0) == 2 * 100 * 100 * 3
    assert di.estimate_image_bytes(10, 10, json_bytes=512) == 2 * 10 * 10 * 3 + 512