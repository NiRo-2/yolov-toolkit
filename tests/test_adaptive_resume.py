"""Adaptive memory-stop/resume lifecycle: hidden .adaptive/resumable_last.pt.

Locks the fix for the "clean adaptive stop treated as full-schedule completion"
bug: the snapshot is saved at the clean epoch boundary, the supervisor continues
from it after a downgrade, and resume never claims "finished the full schedule"
unless the evidence says the target epoch count was actually reached.

No GPU, no torch checkpoints: ckpt_completed_epoch is monkeypatched.
"""

import argparse
import json
import types
from pathlib import Path

import train_detector.train_detector as td


class FakeAdaptiveLogger:
    log_path = "<test-log>"

    def __init__(self):
        self.lines = []

    def log(self, msg):
        self.lines.append(str(msg))


class FakeAdaptiveGuard:
    def __init__(self, adaptive=False, emergency=False):
        self._adaptive = adaptive
        self._emergency = emergency

    def consume_adaptive(self):
        v = self._adaptive
        self._adaptive = False
        return v

    def consume_emergency(self):
        v = self._emergency
        self._emergency = False
        return v

    def peak_reserved(self):
        return 1.0

    def start(self):
        pass

    def stop(self):
        pass


def make_tracker(run_dir, state=None, guard=None, target_epochs=600):
    logger = FakeAdaptiveLogger()
    tracker = td.AttemptTracker(
        logger=logger, guard=guard or FakeAdaptiveGuard(),
        run_dir=run_dir, target_epochs=target_epochs, intended_cfg={},
        state=state if state is not None else {},
        on_emergency=None, on_adaptive=None, on_complete=None, on_crash=None,
    )
    return tracker, logger


def make_cfg(mosaic, copy_paste, mixup, multi_scale, batch=8):
    cfg = {"model": "yolo26x.pt", "imgsz": 640, "batch": batch, "workers": 2,
           "mosaic": mosaic, "copy_paste": copy_paste, "mixup": mixup, "multi_scale": multi_scale}
    cfg.update(td.build_chosen("yolo26x.pt", 640, batch, 2, multi_scale, mosaic,
                               mixup=mixup, copy_paste=copy_paste))
    return cfg


def make_supervisor_args():
    return argparse.Namespace(patience=100, memory_profile="safe",
                              no_memory_probe=False, force_unsafe=False)


def fake_hw():
    return {"vram_gb": 10.0, "ram_gb": 32.0, "cpu_cores": 8, "gpu_name": "FakeGPU"}


def hidden_epoch_fake(finalized_epochs=-1, hidden_epoch=11):
    """ckpt_completed_epoch stand-in: hidden snapshot -> hidden_epoch, everything else finalized."""
    def fake(path):
        return hidden_epoch if "resumable_last" in str(path) else finalized_epochs
    return fake


# ---------------------------------------------------------------------------
# 1. Clean adaptive stop saves the hidden resumable snapshot
# ---------------------------------------------------------------------------

def test_adaptive_snapshot_saved_on_clean_adaptive_stop(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    (run_dir / "weights").mkdir(parents=True)
    fake_last = run_dir / "weights" / "last.pt"
    fake_last.write_bytes(b"unstripped-ckpt")
    monkeypatch.setattr(td, "ckpt_completed_epoch", lambda p: 11)

    tracker, logger = make_tracker(run_dir, guard=FakeAdaptiveGuard(adaptive=True))
    trainer = types.SimpleNamespace(epoch=10, last=fake_last)  # epoch 10 (0-based) -> completed 11
    tracker.on_fit_epoch_end(trainer)

    hidden = run_dir / ".adaptive" / "resumable_last.pt"
    assert hidden.is_file()
    assert hidden.read_bytes() == b"unstripped-ckpt"
    assert trainer.stop is True
    assert tracker.completed_epoch == 11
    assert tracker.adaptive_triggered is True
    assert tracker.resumable_ckpt == hidden
    assert tracker.state["resume_pending"] is True
    assert tracker.state["resumable_epoch"] == 11
    assert tracker.state["resumable_ckpt"] == ".adaptive/resumable_last.pt"
    # State persisted immediately so an interrupted process can still find the snapshot.
    disk = json.loads((run_dir / ".adaptive" / "state.json").read_text(encoding="utf-8"))
    assert disk["resume_pending"] is True
    assert disk["resumable_epoch"] == 11
    assert any("saved hidden resumable checkpoint" in line for line in logger.lines)


def test_adaptive_snapshot_skipped_when_source_already_finalized(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    (run_dir / "weights").mkdir(parents=True)
    fake_last = run_dir / "weights" / "last.pt"
    fake_last.write_bytes(b"stripped-ckpt")
    monkeypatch.setattr(td, "ckpt_completed_epoch", lambda p: -1)

    tracker, logger = make_tracker(run_dir, guard=FakeAdaptiveGuard(adaptive=True))
    trainer = types.SimpleNamespace(epoch=10, last=fake_last)
    tracker.on_fit_epoch_end(trainer)

    assert not (run_dir / ".adaptive" / "resumable_last.pt").exists()
    assert tracker.resumable_ckpt is None
    assert tracker.state.get("resume_pending") is not True
    assert any("already finalized" in line for line in logger.lines)


# ---------------------------------------------------------------------------
# 2. Supervisor continues from the hidden checkpoint after stopped-adaptive
# ---------------------------------------------------------------------------

def test_supervisor_continues_after_stopped_adaptive_with_hidden_checkpoint(monkeypatch, tmp_path):
    run_dir = tmp_path / "run"
    weights = run_dir / "weights"
    weights.mkdir(parents=True)
    # After Ultralytics final_eval() the public last.pt is finalized (epoch -1).
    (weights / "last.pt").write_bytes(b"stripped")
    monkeypatch.setattr(td, "ckpt_completed_epoch", hidden_epoch_fake())
    monkeypatch.setattr(td, "cuda_available_for", lambda d: True)
    monkeypatch.setattr(td, "cuda_index_for", lambda d: 0)
    monkeypatch.setattr(td, "probe_train_once", lambda *a, **k: (1.0, 2.0, 1.0))

    attempts = []

    def fake_run_attempt(args, logger, rd, resume_pt, cfg, target, device, guard,
                         yaml_path, ckpt_cfg, mode, attempt_no, state):
        seen_completed = state.get("completed_epoch")
        attempts.append({
            "no": attempt_no, "batch": cfg["batch"], "mode": mode,
            "resume_pt": Path(resume_pt) if resume_pt else None,
            "resume_pending": state.get("resume_pending"),
            "completed_before": seen_completed,
        })
        if attempt_no == 1:
            # Emulate run_attempt + save_resumable_snapshot side effects.
            state["completed_epoch"] = 11
            state.setdefault("history", []).append(
                {"attempt": 1, "status": "stopped-adaptive", "completed_epoch": 11, "config": dict(cfg)})
            adir = rd / ".adaptive"
            adir.mkdir(parents=True, exist_ok=True)
            (adir / "resumable_last.pt").write_bytes(b"unstripped-epoch-11")
            state["resumable_ckpt"] = ".adaptive/resumable_last.pt"
            state["resumable_epoch"] = 11
            state["resume_pending"] = True
            return types.SimpleNamespace(
                status="stopped-adaptive", completed_epoch=11,
                adaptive_triggered=True, emergency_triggered=False,
                oom_auto_reduced=False, resumable_ckpt=adir / "resumable_last.pt",
            )
        # Attempt 2 runs the downgraded config from the hidden snapshot and completes.
        state["completed_epoch"] = target
        state.setdefault("history", []).append(
            {"attempt": 2, "status": "completed-target", "completed_epoch": target, "config": dict(cfg)})
        return types.SimpleNamespace(
            status="completed-target", completed_epoch=target,
            adaptive_triggered=False, emergency_triggered=False,
            oom_auto_reduced=False, resumable_ckpt=None,
        )

    monkeypatch.setattr(td, "run_attempt", fake_run_attempt)

    state = {"run_name": "t", "target_epochs": 600, "completed_epoch": 11, "provisional": False,
             "stable_epochs": 1, "downgrade_count": 0, "history": []}
    logger = FakeAdaptiveLogger()
    status = td.run_adaptive_supervisor(
        args=make_supervisor_args(), logger=logger, run_dir=run_dir, run_name="t",
        last_pt=weights / "last.pt", chosen=make_cfg(1.0, 0.3, 0.15, 0.5, batch=4), target_epochs=600,
        device="0", hw=fake_hw(), guard=FakeAdaptiveGuard(), state=state,
        yaml_path=None, ckpt_cfg={}, mode="resume",
    )

    assert status == "completed-target"
    assert len(attempts) == 2
    # Attempt 1 ran batch 4 and hit adaptive pressure.
    assert attempts[0]["batch"] == 4
    # Supervisor downgraded the ladder step and resumed the SAME run from the hidden checkpoint.
    assert attempts[1]["batch"] == 2
    assert attempts[1]["mode"] == "resume"
    assert attempts[1]["resume_pt"] == run_dir / ".adaptive" / "resumable_last.pt"
    assert attempts[1]["resume_pending"] is True
    # The stopped attempt's completed_epoch was never rewritten to the target.
    assert attempts[1]["completed_before"] == 11
    assert state["downgrade_count"] == 1
    # Completed target clears the pending state and removes the hidden snapshot.
    assert state["resume_pending"] is False
    assert state["completed_epoch"] == 600
    assert not (run_dir / ".adaptive" / "resumable_last.pt").exists()
    assert any("downgrading config and resuming from hidden" in line for line in logger.lines)


# ---------------------------------------------------------------------------
# 3. No hidden checkpoint -> honest adaptive-stop reporting
# ---------------------------------------------------------------------------

def test_supervisor_reports_adaptive_stop_when_no_hidden_checkpoint(monkeypatch, tmp_path):
    run_dir = tmp_path / "run"
    weights = run_dir / "weights"
    weights.mkdir(parents=True)
    (weights / "last.pt").write_bytes(b"stripped")  # finalized by the old buggy final_eval
    monkeypatch.setattr(td, "ckpt_completed_epoch", lambda p: -1)
    monkeypatch.setattr(td, "cuda_available_for", lambda d: True)
    monkeypatch.setattr(td, "cuda_index_for", lambda d: 0)
    monkeypatch.setattr(td, "probe_train_once", lambda *a, **k: (1.0, 2.0, 1.0))

    def fake_run_attempt(args, logger, rd, resume_pt, cfg, target, device, guard,
                         yaml_path, ckpt_cfg, mode, attempt_no, state):
        state["completed_epoch"] = 11
        state.setdefault("history", []).append(
            {"attempt": attempt_no, "status": "stopped-adaptive", "completed_epoch": 11, "config": dict(cfg)})
        return types.SimpleNamespace(
            status="stopped-adaptive", completed_epoch=11,
            adaptive_triggered=True, emergency_triggered=False,
            oom_auto_reduced=False, resumable_ckpt=None,
        )

    monkeypatch.setattr(td, "run_attempt", fake_run_attempt)

    # Legacy corrupted top-level state claims the full schedule; evidence must win.
    state = {"run_name": "t", "target_epochs": 600, "completed_epoch": 600, "provisional": False,
             "stable_epochs": 1, "downgrade_count": 0,
             "history": [{"attempt": 1, "status": "stopped-adaptive", "completed_epoch": 11, "config": {}}]}
    (run_dir / "results.csv").write_text(
        "epoch,mAP\n" + "\n".join(f"{i},0.4" for i in range(1, 12)) + "\n", encoding="utf-8")

    logger = FakeAdaptiveLogger()
    status = td.run_adaptive_supervisor(
        args=make_supervisor_args(), logger=logger, run_dir=run_dir, run_name="t",
        last_pt=weights / "last.pt", chosen=make_cfg(1.0, 0.3, 0.15, 0.5, batch=4), target_epochs=600,
        device="0", hw=fake_hw(), guard=FakeAdaptiveGuard(), state=state,
        yaml_path=None, ckpt_cfg={}, mode="resume",
    )

    assert status == "stopped-adaptive"
    assert state["final_status"] == "stopped-adaptive"
    assert state["completed_epoch"] == 11  # evidence, not the legacy 600
    assert state.get("resume_pending") is False
    assert state.get("last_checkpoint_finalized") is True
    joined = "\n".join(logger.lines)
    assert "after 11/600 epochs" in joined
    assert "target schedule reached" not in joined
    disk = json.loads((run_dir / ".adaptive" / "state.json").read_text(encoding="utf-8"))
    assert disk["completed_epoch"] == 11


# ---------------------------------------------------------------------------
# 4-6. choose_resume_checkpoint behavior
# ---------------------------------------------------------------------------

def _seed_run(tmp_path, state, *, hidden_bytes=None, hidden_name="resumable_last.pt"):
    run_dir = tmp_path / "run"
    weights = run_dir / "weights"
    weights.mkdir(parents=True)
    (weights / "last.pt").write_bytes(b"stripped")
    adir = run_dir / ".adaptive"
    adir.mkdir(parents=True, exist_ok=True)
    hidden = None
    if hidden_bytes is not None:
        hidden = adir / hidden_name
        hidden.write_bytes(hidden_bytes)
    (adir / "state.json").write_text(json.dumps(state), encoding="utf-8")
    return run_dir, weights / "last.pt", hidden


def test_choose_resume_checkpoint_prefers_pending_hidden_over_finalized_last(tmp_path, monkeypatch):
    state = {"completed_epoch": 11, "final_status": "stopped-adaptive",
             "history": [{"attempt": 1, "status": "stopped-adaptive", "completed_epoch": 11}],
             "resumable_ckpt": ".adaptive/resumable_last.pt", "resumable_epoch": 11,
             "resume_pending": True}
    run_dir, last_pt, hidden = _seed_run(tmp_path, state, hidden_bytes=b"unstripped-11")
    monkeypatch.setattr(td, "ckpt_completed_epoch", hidden_epoch_fake())

    chosen, info = td.choose_resume_checkpoint(run_dir, last_pt, target_epochs=600)
    assert chosen == hidden
    assert info["finalized"] is False
    assert info["epoch"] == 11
    assert td.valid_resume_checkpoint(hidden) is True


def test_choose_resume_checkpoint_ignores_stale_hidden_after_completed_target(tmp_path, monkeypatch):
    # completed-target cleared resume_pending; an older hidden snapshot is a stale relic.
    state = {"completed_epoch": 600, "final_status": "completed-target",
             "history": [{"attempt": 2, "status": "completed-target", "completed_epoch": 600}],
             "resumable_ckpt": ".adaptive/resumable_last.pt", "resumable_epoch": 11,
             "resume_pending": False}
    run_dir, last_pt, hidden = _seed_run(tmp_path, state, hidden_bytes=b"unstripped-11")
    monkeypatch.setattr(td, "ckpt_completed_epoch", hidden_epoch_fake())

    chosen, info = td.choose_resume_checkpoint(run_dir, last_pt, target_epochs=600)
    assert chosen is None  # stale hidden is never used
    assert info["finalized"] is True
    assert info["stopped_after"] == 600  # evidence says the schedule actually finished


def test_choose_resume_checkpoint_reports_finalized_adaptive_stop_without_hidden(tmp_path, monkeypatch):
    # The rust_Merged situation: finalized last.pt, no hidden snapshot, legacy state
    # says completed_epoch 600 but history + results.csv say the run stopped at 11.
    state = {"completed_epoch": 600, "final_status": "early-stopped",
             "history": [{"attempt": 1, "status": "stopped-adaptive", "completed_epoch": 11}],
             "resume_pending": False}
    run_dir, last_pt, _ = _seed_run(tmp_path, state)
    (run_dir / "results.csv").write_text(
        "epoch,mAP\n" + "\n".join(f"{i},0.4" for i in range(1, 12)) + "\n", encoding="utf-8")
    monkeypatch.setattr(td, "ckpt_completed_epoch", lambda p: -1)

    chosen, info = td.choose_resume_checkpoint(run_dir, last_pt, target_epochs=600)
    assert chosen is None
    assert info["finalized"] is True
    assert info["stopped_after"] == 11  # evidence beats the legacy corrupted 600


def test_pending_hidden_rejected_when_snapshot_epoch_regressed(tmp_path, monkeypatch):
    # Hidden snapshot predates the recorded resumable_epoch -> not trustworthy.
    state = {"completed_epoch": 11,
             "history": [{"attempt": 1, "status": "stopped-adaptive", "completed_epoch": 11}],
             "resumable_ckpt": ".adaptive/resumable_last.pt", "resumable_epoch": 11,
             "resume_pending": True}
    run_dir, last_pt, _ = _seed_run(tmp_path, state, hidden_bytes=b"unstripped-10")
    monkeypatch.setattr(td, "ckpt_completed_epoch", hidden_epoch_fake(hidden_epoch=10))

    chosen, info = td.choose_resume_checkpoint(run_dir, last_pt, target_epochs=600)
    assert chosen is None


# ---------------------------------------------------------------------------
# 7-15. Review-findings fixes (refined plan)
# ---------------------------------------------------------------------------

def test_final_eval_second_fire_freezes_epoch_and_keeps_snapshot(tmp_path, monkeypatch):
    """Task 1a/1b/1c: the final_eval() re-fire is a strict no-op."""
    run_dir = tmp_path / "run"
    (run_dir / "weights").mkdir(parents=True)
    fake_last = run_dir / "weights" / "last.pt"
    fake_last.write_bytes(b"unstripped-ckpt")
    monkeypatch.setattr(td, "ckpt_completed_epoch", lambda p: 11)

    class AlwaysReArmedGuard(FakeAdaptiveGuard):
        # MemoryGuard re-arms on every 2-sample streak: consume always says True.
        def __init__(self):
            super().__init__(adaptive=True)
            self.consume_calls = 0

        def consume_adaptive(self):
            self.consume_calls += 1
            return True

    guard = AlwaysReArmedGuard()
    tracker, logger = make_tracker(run_dir, guard=guard)
    trainer = types.SimpleNamespace(epoch=10, last=fake_last)

    tracker.on_fit_epoch_end(trainer)
    hidden = run_dir / ".adaptive" / "resumable_last.pt"
    assert hidden.is_file()
    assert tracker.completed_epoch == 11

    # final_eval() re-fires with the epoch counter advanced (11 -> 12 would be the artifact).
    trainer.epoch = 11
    tracker.on_fit_epoch_end(trainer)

    assert tracker.completed_epoch == 11          # frozen at the real stop epoch, no 12 artifact
    assert guard.consume_calls == 1               # short-circuit: re-armed flag never consumed
    assert hidden.read_bytes() == b"unstripped-ckpt"  # snapshot never re-copied from finalized src
    assert tracker.resumable_ckpt == hidden
    assert td.pending_resumable_checkpoint(run_dir, tracker.state) == hidden

    # on_train_end must not re-inflate 11 -> 12 after the re-fire either.
    tracker.on_train_end(trainer)
    assert tracker.completed_epoch == 11
    assert tracker.status == "stopped-adaptive"


def test_completed_epoch_clamped_to_target_on_clean_run(tmp_path, monkeypatch):
    """Task 1a/1b: min(epoch + 1, target) kills the 600 -> 601 clean-run artifact."""
    run_dir = tmp_path / "run"
    (run_dir / "weights").mkdir(parents=True)
    monkeypatch.setattr(td, "ckpt_completed_epoch", lambda p: 0)
    tracker, _ = make_tracker(run_dir, guard=FakeAdaptiveGuard(), target_epochs=600)
    trainer = types.SimpleNamespace(epoch=600)  # re-fire sees 600 -> 601

    tracker.on_fit_epoch_end(trainer)
    assert tracker.completed_epoch == 600
    tracker.on_train_end(trainer)
    assert tracker.completed_epoch == 600
    assert tracker.status == "completed-target"


def test_snapshot_save_failure_marks_hidden_checkpoint_invalid(tmp_path, monkeypatch):
    """Task 2a/2b: a copy failure marks the hidden snapshot invalid; resume ignores it."""
    run_dir = tmp_path / "run"
    (run_dir / "weights").mkdir(parents=True)
    fake_last = run_dir / "weights" / "last.pt"
    fake_last.write_bytes(b"unstripped-11")
    adir = run_dir / ".adaptive"
    adir.mkdir(parents=True)
    prior_hidden = adir / "resumable_last.pt"
    prior_hidden.write_bytes(b"old-epoch-5")
    monkeypatch.setattr(td, "ckpt_completed_epoch", lambda p: 11)

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(td.shutil, "copy2", boom)

    state = {"resume_pending": True, "resumable_ckpt": ".adaptive/resumable_last.pt",
             "resumable_epoch": 5, "completed_epoch": 5, "target_epochs": 600}
    tracker, logger = make_tracker(run_dir, state=state, guard=FakeAdaptiveGuard(adaptive=True))
    trainer = types.SimpleNamespace(epoch=10, last=fake_last)
    tracker.on_fit_epoch_end(trainer)

    assert tracker.resumable_ckpt is None
    assert any("copy failed" in line for line in logger.lines)
    disk = json.loads((adir / "state.json").read_text(encoding="utf-8"))
    assert disk["hidden_checkpoint_invalid"] is True
    assert disk["resume_pending"] is True  # the invalid flag gates it; pending itself untouched
    assert prior_hidden.read_bytes() == b"old-epoch-5"  # dst never replaced by a failed copy
    assert td.pending_resumable_checkpoint(run_dir, tracker.state) is None


def test_pending_hidden_rejected_when_older_than_known_progress(tmp_path, monkeypatch):
    """Task 2b: a snapshot older than the run's known completed_epoch is stale."""
    state = {"resume_pending": True, "resumable_ckpt": ".adaptive/resumable_last.pt",
             "resumable_epoch": 11, "completed_epoch": 130, "target_epochs": 600}
    run_dir, last_pt, _ = _seed_run(tmp_path, state, hidden_bytes=b"unstripped-11")
    monkeypatch.setattr(td, "ckpt_completed_epoch", hidden_epoch_fake())

    assert td.pending_resumable_checkpoint(run_dir, state) is None
    chosen, info = td.choose_resume_checkpoint(run_dir, last_pt, target_epochs=600)
    assert chosen is None


def test_supervisor_init_re_derives_inflated_completed_epoch(monkeypatch, tmp_path):
    """Task 3: legacy completed_epoch = target is corrected from evidence before any attempt."""
    run_dir = tmp_path / "run"
    weights = run_dir / "weights"
    weights.mkdir(parents=True)
    (weights / "last.pt").write_bytes(b"stripped")
    (run_dir / "results.csv").write_text(
        "epoch,mAP\n" + "\n".join(f"{i},0.4" for i in range(1, 12)) + "\n", encoding="utf-8")
    monkeypatch.setattr(td, "ckpt_completed_epoch", lambda p: -1)
    monkeypatch.setattr(td, "cuda_available_for", lambda d: True)
    monkeypatch.setattr(td, "cuda_index_for", lambda d: 0)
    monkeypatch.setattr(td, "probe_train_once", lambda *a, **k: (1.0, 2.0, 1.0))

    seen_completed = []

    def fake_run_attempt(args, logger, rd, resume_pt, cfg, target, device, guard,
                         yaml_path, ckpt_cfg, mode, attempt_no, state):
        seen_completed.append(state.get("completed_epoch"))
        return types.SimpleNamespace(
            status="interrupted", completed_epoch=11,
            adaptive_triggered=False, emergency_triggered=False, oom_auto_reduced=False,
        )

    monkeypatch.setattr(td, "run_attempt", fake_run_attempt)

    # Legacy corrupted state: top-level completed_epoch claims the full schedule.
    state = {"run_name": "t", "completed_epoch": 600, "provisional": False, "stable_epochs": 1,
             "downgrade_count": 0,
             "history": [{"attempt": 1, "status": "stopped-adaptive", "completed_epoch": 11, "config": {}}]}
    logger = FakeAdaptiveLogger()
    status = td.run_adaptive_supervisor(
        args=make_supervisor_args(), logger=logger, run_dir=run_dir, run_name="t",
        last_pt=weights / "last.pt", chosen=make_cfg(1.0, 0.3, 0.15, 0.5, batch=4), target_epochs=600,
        device="0", hw=fake_hw(), guard=FakeAdaptiveGuard(), state=state,
        yaml_path=None, ckpt_cfg={}, mode="resume",
    )

    assert seen_completed == [11]  # corrected before the first attempt ever observed it
    assert status == "interrupted"
    assert state["completed_epoch"] == 11
    disk = json.loads((run_dir / ".adaptive" / "state.json").read_text(encoding="utf-8"))
    assert disk["completed_epoch"] == 11
    assert disk["target_epochs"] == 600  # legacy files lacking the key gain it at init
    joined = "\n".join(logger.lines)
    assert "corrected completed_epoch 600 -> 11" in joined
    assert "target schedule reached" not in joined


def test_attempt_budget_exhaustion_keeps_resume_pending(monkeypatch, tmp_path, capsys):
    """Task 4b/4c: exhaustion with a pending hidden stays resumable and says so."""
    run_dir = tmp_path / "run"
    weights = run_dir / "weights"
    weights.mkdir(parents=True)
    (weights / "last.pt").write_bytes(b"stripped")
    monkeypatch.setattr(td, "ckpt_completed_epoch", hidden_epoch_fake())
    monkeypatch.setattr(td, "cuda_available_for", lambda d: True)
    monkeypatch.setattr(td, "cuda_index_for", lambda d: 0)
    monkeypatch.setattr(td, "probe_train_once", lambda *a, **k: (1.0, 2.0, 1.0))
    monkeypatch.setattr(td, "MAX_ADAPTIVE_CONFIG_DOWNGRADES", 2)

    def fake_run_attempt(args, logger, rd, resume_pt, cfg, target, device, guard,
                         yaml_path, ckpt_cfg, mode, attempt_no, state):
        adir = rd / ".adaptive"
        adir.mkdir(parents=True, exist_ok=True)
        (adir / "resumable_last.pt").write_bytes(b"unstripped-epoch-11")
        state["completed_epoch"] = 11
        state["resumable_ckpt"] = ".adaptive/resumable_last.pt"
        state["resumable_epoch"] = 11
        state["resume_pending"] = True
        state.setdefault("history", []).append(
            {"attempt": attempt_no, "status": "stopped-adaptive", "completed_epoch": 11, "config": dict(cfg)})
        return types.SimpleNamespace(
            status="stopped-adaptive", completed_epoch=11,
            adaptive_triggered=True, emergency_triggered=False, oom_auto_reduced=False,
        )

    monkeypatch.setattr(td, "run_attempt", fake_run_attempt)

    state = {"run_name": "t", "completed_epoch": 0, "provisional": True, "stable_epochs": 0,
             "downgrade_count": 0, "history": []}
    logger = FakeAdaptiveLogger()
    status = td.run_adaptive_supervisor(
        args=make_supervisor_args(), logger=logger, run_dir=run_dir, run_name="t",
        last_pt=weights / "last.pt", chosen=make_cfg(1.0, 0.3, 0.15, 0.5, batch=4), target_epochs=600,
        device="0", hw=fake_hw(), guard=FakeAdaptiveGuard(), state=state,
        yaml_path=None, ckpt_cfg={}, mode="resume",
    )

    assert status == "stopped-adaptive"
    assert state["resume_pending"] is True
    assert state["completed_epoch"] == 11
    assert (run_dir / ".adaptive" / "resumable_last.pt").is_file()
    disk = json.loads((run_dir / ".adaptive" / "state.json").read_text(encoding="utf-8"))
    assert disk["resume_pending"] is True
    joined = "\n".join(logger.lines)
    assert "attempt budget exhausted; --resume will continue from .adaptive/resumable_last.pt" in joined
    assert "target schedule reached" not in joined

    # Shared outcome printer (Task 4c) surfaces the hidden continuation point.
    td.report_run_outcome(run_dir, state, status, 600, "t", logger)
    out = capsys.readouterr().out
    assert "Training stopped-adaptive after 11/600 epochs" in out
    assert "The run is resumable: --resume will continue from .adaptive/resumable_last.pt" in out


def test_no_safe_config_with_pending_hidden_stays_resumable(monkeypatch, tmp_path):
    """Task 4a: no safe config on current hardware is a pause, not a fake terminal."""
    run_dir, last_pt, hidden = _seed_run(
        tmp_path,
        {"completed_epoch": 11, "provisional": False, "stable_epochs": 1, "downgrade_count": 0,
         "history": [{"attempt": 1, "status": "stopped-adaptive", "completed_epoch": 11, "config": {}}],
         "resumable_ckpt": ".adaptive/resumable_last.pt", "resumable_epoch": 11,
         "resume_pending": True},
        hidden_bytes=b"unstripped-11")
    monkeypatch.setattr(td, "ckpt_completed_epoch", hidden_epoch_fake())
    monkeypatch.setattr(td, "cuda_available_for", lambda d: True)
    monkeypatch.setattr(td, "cuda_index_for", lambda d: 0)
    # Every probe is unsafe: reserved 999 GB >> any profile cap (10 GB * <= 0.85).
    monkeypatch.setattr(td, "probe_train_once", lambda *a, **k: (900.0, 999.0, 900.0))

    state = {"run_name": "t", "completed_epoch": 11, "provisional": False, "stable_epochs": 1,
             "downgrade_count": 0, "history": [
                 {"attempt": 1, "status": "stopped-adaptive", "completed_epoch": 11, "config": {}}],
             "resumable_ckpt": ".adaptive/resumable_last.pt", "resumable_epoch": 11,
             "resume_pending": True}
    logger = FakeAdaptiveLogger()
    status = td.run_adaptive_supervisor(
        args=make_supervisor_args(), logger=logger, run_dir=run_dir, run_name="t",
        last_pt=last_pt, chosen=make_cfg(1.0, 0.3, 0.15, 0.5, batch=4), target_epochs=600,
        device="0", hw=fake_hw(), guard=FakeAdaptiveGuard(), state=state,
        yaml_path=None, ckpt_cfg={}, mode="resume",
    )

    assert status == "no-safe-config"
    assert state["resume_pending"] is True  # NOT terminalized: hidden stays the continuation point
    assert state.get("last_checkpoint_finalized") is not True
    assert hidden.is_file()
    disk = json.loads((run_dir / ".adaptive" / "state.json").read_text(encoding="utf-8"))
    assert disk["resume_pending"] is True
    joined = "\n".join(logger.lines)
    assert "no safe config for current hardware; run stays resumable from .adaptive/resumable_last.pt" in joined
    assert "target schedule not reached" not in joined  # finish_terminal_run was not called

    td.report_run_outcome(run_dir, state, status, 600, "t", logger)


def test_derive_keeps_genuine_mid_run_evidence_over_older_history(tmp_path):
    """Task 7: history@11 + hidden/weights@14 must NOT collapse to 11 (14 < target)."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "results.csv").write_text(
        "epoch,mAP\n" + "\n".join(f"{i},0.4" for i in range(1, 15)) + "\n", encoding="utf-8")
    state = {"history": [{"attempt": 1, "status": "stopped-adaptive", "completed_epoch": 11},
                         {"attempt": 2, "status": "crashed", "completed_epoch": None}]}
    assert td.derive_actual_epoch(run_dir, state, tracker=None, ckpt_epoch=14, target_epochs=600) == 14


def test_derive_suppresses_legacy_target_lie(tmp_path):
    """Task 7: tracker 601 / csv 600 + history stopped-adaptive@11 -> evidence says 11."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "results.csv").write_text(
        "epoch,mAP\n" + "\n".join(f"{i},0.4" for i in range(1, 601)) + "\n", encoding="utf-8")
    state = {"history": [{"attempt": 1, "status": "stopped-adaptive", "completed_epoch": 11}]}
    tracker = types.SimpleNamespace(completed_epoch=601)
    assert td.derive_actual_epoch(run_dir, state, tracker=tracker, ckpt_epoch=600, target_epochs=600) == 11


def test_partial_emergency_rollback_keeps_hidden_resumable_after_evidence_rewind(tmp_path, monkeypatch):
    """Review finding 1: after a partial-emergency backup restore the recorded evidence is
    rewound to durable progress, so the staleness gate keeps accepting the hidden snapshot."""
    run_dir = tmp_path / "run"
    weights = run_dir / "weights"
    weights.mkdir(parents=True)
    adir = run_dir / ".adaptive"
    adir.mkdir()
    # Post-restore disk state: weights/last.pt rolled back to the finalized pre-attempt
    # backup; the hidden snapshot from the earlier clean adaptive stop is the durable
    # continuation point.
    (weights / "last.pt").write_bytes(b"restored-finalized")
    (adir / "backup_last.pt").write_bytes(b"restored-finalized")
    (adir / "resumable_last.pt").write_bytes(b"unstripped-epoch-11")

    def fake(path):
        s = str(path)
        return 11 if "resumable_last" in s else -1

    monkeypatch.setattr(td, "ckpt_completed_epoch", fake)
    state = {"resume_pending": True, "resumable_ckpt": ".adaptive/resumable_last.pt",
             "resumable_epoch": 11, "completed_epoch": 11, "target_epochs": 600}

    assert td.durable_progress_epoch(run_dir, state) == 11

    # Inflated evidence from the rolled-back attempt would reject the hidden snapshot...
    assert td.pending_resumable_checkpoint(run_dir, {**state, "completed_epoch": 24}) is None
    # ...but durable (rewound) evidence keeps it, which is what the run_attempt rewind
    # writes before the staleness gate is ever consulted again.
    assert td.pending_resumable_checkpoint(run_dir, {**state, "completed_epoch": 11}) == adir / "resumable_last.pt"

    # Unstripped restored weights win over the hidden when they are actually resumable.
    def fake_unstripped(path):
        s = str(path)
        if "backup_last" in s:
            return 5
        return 11 if "resumable_last" in s else 24

    monkeypatch.setattr(td, "ckpt_completed_epoch", fake_unstripped)
    assert td.durable_progress_epoch(run_dir, state) == 24  # live weights/last.pt is resumable at 24
