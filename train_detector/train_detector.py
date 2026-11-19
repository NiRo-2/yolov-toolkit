r"""
YOLO26 Object Detector Training Script
Auto-detects hardware, dataset size and image resolution to calculate optimal training config.
Every fresh or resumed training invocation writes a per-run text log under <run_dir>/Logs/.
Memory/model/batch selection is dynamically calibrated with real-dataset probe runs for the
selected hardware, device, dataset and augmentation config; static estimates are fallback only.

AUTO SELECTION POLICY: quality-v2. Fresh runs pick the largest task model that fits
at a healthy batch size (batch >= 16), with task-appropriate augmentation, at an
imgsz that resolves the objects. Heaviest-safe augmentation is preferred only after
the healthy-batch floor is met; lighten multi_scale/mosaic before dropping below
the floor. degrees/flipud default to 0 (opt-in). Nothing trains unverified unless
--force-unsafe or --no-memory-probe is passed explicitly. Safety guarantees are
unchanged (memory-profile safe by default). The selection cache
(weights/memory_profiles.json) is schema version 4 keyed by intent "quality-v2":
the first run after this policy change recalibrates fully.

Usage:
    # Fresh training run (auto-configures everything, runs dynamic memory calibration)
    # --input is a dataset DIRECTORY: YOLO (data.yaml present or synthesized), COCO JSON, or VOC XML.
    python train_detector/train_detector.py --input /path/to/dataset_folder --name my_detector_v1

    # Override any auto-calculated value (overrides are safety-checked unless --force-unsafe)
    python train_detector/train_detector.py --input /path/to/dataset_folder --name my_detector_v1 --model yolo26x-seg.pt --batch 8
    python train_detector/train_detector.py --input /path/to/dataset_folder --name my_detector_v1 --imgsz 1024

    # Run-name collision control
    python train_detector/train_detector.py --input /path/to/dataset_folder --name my_detector_v1 --fresh
    python train_detector/train_detector.py --input /path/to/dataset_folder --name my_detector_v1 --resume

    # Resume a crashed/interrupted run
    python train_detector/train_detector.py --resume --name my_detector_v1

    # Resume without --name: auto-finds the most recent run
    python train_detector/train_detector.py --resume

    # Test run-name handling, dataset validation, fingerprinting and logging without training
    python train_detector/train_detector.py --input /path/to/dataset_folder --name DryRunTest --device cpu --dry-run

    --input   dataset DIRECTORY (required for fresh training): auto-detects the format
              (YOLO folder with data.yaml, YOLO folder without data.yaml -> synthesized,
              COCO JSON, VOC XML) and the annotation task (detect/segment/pose);
              COCO/VOC are converted once into a reusable <folder>_yolo copy.
              A data.yaml file path is rejected: pass the dataset folder instead
              works with Windows paths: c:\Users\Ni\Desktop\project\dataset_folder
    --resume  resume from last checkpoint (fails clearly if no checkpoint exists)
    --fresh   force a new auto-suffixed run even if the run name already exists
    --name    run name to resume, or name for new run (default: detector_v1)
    --task    auto|detect|segment|pose  override the auto-derived annotation task (default: auto)
    --converted-dir  override output location for COCO/VOC -> YOLO conversion (default: <dataset>_yolo sibling)
    --yes     accept auto-detection writes into the dataset folder (synthesized data.yaml,
              added kpt_shape, label-structure repair) without an interactive [y/N] prompt
    --fix-label-structure
              normalise a segment dataset that mixes a few 5-field bbox-only label lines
              into polygons in place (originals kept under <dataset>/.label_repair_backup/,
              audit in <dataset>/.label_repair_manifest.json); refused unless segment
              dominates every split at >=95% of label lines. Run before tiling.
    --model   override auto-selected model (yolo26n/m/l/x, -seg, -pose)
    --batch   override auto-calculated batch size
    --workers override auto-calculated worker count
    --imgsz   override auto-calculated image size in pixels
    --epochs  number of training epochs (default: 600)
    --patience early stopping patience in epochs (default: 100)
    --device  0 for GPU, cpu for CPU (default: 0; falls back to CPU when CUDA is unavailable)
    --memory-profile safe|balanced|aggressive   dynamic memory safety profile (default: safe)
    --no-memory-probe  skip dynamic probe calibration; conservative static caps only
    --force-unsafe     proceed even when dynamic calibration marks the selected config unsafe
    --dry-run          select run name, validate dataset, fingerprint, write log, exit before probes/training
    --probe-vram       measure/refresh legacy per-model GB/image estimates

Output:
    train_detector/weights/<model>.pt                 <- Ultralytics pretrained backbones (cached)
    train_detector/weights/memory_profiles.json       <- measured dynamic memory cache (schema v4, quality-v2 intent)
    train_detector/weights/vram_estimates.json        <- legacy fallback estimates only
    train_detector/runs/detect/<name>/Logs/run-<localtime><offset>.log     <- fresh run log
    train_detector/runs/detect/<name>/Logs/resume-<localtime><offset>.log  <- resume run log
    train_detector/runs/detect/<name>/weights/best.pt <- use this for TensorRT export
    train_detector/runs/detect/<name>/weights/last.pt
    train_detector/runs/detect/<name>/results.png
    train_detector/runs/detect/<name>/confusion_matrix.png
"""

# Expandable CUDA segments reduce fragmentation when available. Set before torch
# import; unsupported builds/OS variants ignore the value (logged at startup).
import os as _os_for_cuda_alloc
_os_for_cuda_alloc.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import contextlib
import functools
import hashlib
import json
import logging
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import yaml
from datetime import datetime
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.segment.train import SegmentationTrainer
from ultralytics.models.yolo.pose.train import PoseTrainer

# Repo-root sys.path shim so ``common.hardware`` resolves when running this
# script directly (``python train_detector/train_detector.py``). The shared
# hardware module lives in common/ and is imported by both scripts.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from common.hardware import (  # noqa: E402  (re-export; see below)
    BATCH_LADDER,
    MEMORY_PROFILES,
    PSUTIL_AVAILABLE,
    detect_hardware,
    looks_like_oom,
    process_rss_bytes,
)

# psutil is still imported directly here because a few training-only call
# sites use psutil.virtual_memory() inline (not routed through the helper).
try:
    import psutil  # type: ignore[import-untyped]
except Exception:
    psutil = None  # type: ignore[assignment]


SCRIPT_DIR = Path(__file__).resolve().parent
RUNS_DIR = SCRIPT_DIR / "runs" / "detect"
WEIGHTS_DIR = SCRIPT_DIR / "weights"

LOGS_DIRNAME = "Logs"

# ANSI escape / control sequence stripper for log-file output (terminal output stays raw).
ANSI_RE = re.compile(
    r"\x1b\[[0-9;?]*[a-zA-Z]"                 # CSI sequences: colors, cursor moves, erases
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"     # OSC sequences: window titles etc
    r"|\x1b[()][AB012]"                        # charset designators
    r"|[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"      # remaining control chars (keeps \n and \t)
)

# Memory safety profiles (safe/balanced/aggressive) and the batch ladder live in
# common/hardware.py and are re-exported above so existing imports keep working.

MIN_HEALTHY_BATCH = 16      # prefer batch >= 16 for stable BatchNorm; nbs=64 still accumulates
MIN_AUTO_BATCH = 4          # absolute auto floor; batch 2 only with explicit --force-unsafe
HEALTHY_BATCH_CANDIDATES = [32, 16]
DEGRADED_BATCH_CANDIDATES = [8, 4]
STANDARD_IMGSZ = [320, 416, 512, 640, 768, 1024, 1280, 1536]

# Task-aware model chains (plan: detect / -seg / -pose per size).
MODEL_TASK_SUFFIXES = {"detect": "", "segment": "-seg", "pose": "-pose"}

MEMORY_CACHE_SCHEMA_VERSION = 4
SELECTION_INTENT = "quality-v2"
MEMORY_CACHE_PATH = WEIGHTS_DIR / "memory_profiles.json"

PILOT_DIRNAME = ".pilot"
PILOT_EPOCHS = 5
PILOT_TRAIN_IMAGES = 2000
PILOT_VAL_IMAGES = 200
LABEL_STATS_SAMPLE_LIMIT = 2000
MIN_RESOLVED_OBJECT_PX = 16.0
MIN_RESOLVED_OBJECT_P25_PX = 8.0
GPU_UTIL_DATALOADER_BOUND = 85.0


def model_order_for_task(task: str) -> list:
    """Quality order n<m<l<x with the task-correct weights suffix."""
    suffix = MODEL_TASK_SUFFIXES.get(task, "")
    return [f"yolo26{size}{suffix}.pt" for size in ("n", "m", "l", "x")]


# -- Adaptive safe-resume / crash-prevention constants ----------------------------
# Quality-v2: never exceed these fractions of total memory during a run.
ADAPTIVE_GPU_FRACTION = 0.75       # adaptive ceiling fraction of VRAM
EMERGENCY_GPU_FRACTION = 0.82      # emergency (mid-epoch abort) fraction of VRAM
ADAPTIVE_CPU_RSS_FRACTION = 0.50   # adaptive ceiling fraction of RAM (RSS)
EMERGENCY_CPU_RSS_FRACTION = 0.60  # emergency ceiling fraction of RAM (RSS)
ADAPTIVE_AVAIL_RAM_FRACTION = 0.25 # adaptive: available RAM must be >= this fraction of total
EMERGENCY_AVAIL_RAM_FRACTION = 0.15# emergency: available RAM must be >= this fraction of total
# Profile caps (peak reserved / RSS as fraction of total) used for guard thresholds.
GUARD_PROFILE_CAPS = {
    "safe":       {"gpu": 0.68, "cpu": 0.38},
    "balanced":   {"gpu": 0.75, "cpu": 0.45},
    "aggressive": {"gpu": 0.85, "cpu": 0.55},
}
GUARD_SAMPLE_INTERVAL_SECONDS = 2.0
GUARD_POLL_EVERY_BATCHES = 16
STABLE_EPOCHS_BEFORE_STEP_UP = 3
STABLE_STEP_UP_GPU_FRACTION = 0.55
MAX_ADAPTIVE_CONFIG_DOWNGRADES = 12

# Task-aware augmentation ladders: (mosaic, copy_paste, mixup, multi_scale).
# Heaviest rung first. Detect/pose keep copy_paste=0 (CopyPaste is a no-op without
# polygons). Segment keeps copy_paste. Main levers: multi_scale, then mosaic.
AUG_LADDER_DETECT = [
    (1.0, 0.0, 0.0, 0.5),
    (1.0, 0.0, 0.0, 0.0),
    (0.5, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
]
AUG_LADDER_SEGMENT = [
    (1.0, 0.3, 0.15, 0.5),
    (1.0, 0.3, 0.0, 0.0),
    (1.0, 0.0, 0.0, 0.0),
    (0.5, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
]
# Backward-compatible alias (detect / pose default).
AUG_LADDER = AUG_LADDER_DETECT

BATCH_LADDER_DOWN = [32, 16, 8, 4, 2, 1]
BATCH_LADDER_UP = [1, 2, 4, 8, 16, 32]
IMGSZ_LADDER_DOWN = None  # computed from native cap at runtime
ADAPTIVE_DIRNAME = ".adaptive"
ADAPTIVE_STATE_FILE = "state.json"
RESUMABLE_LAST_NAME = "resumable_last.pt"  # hidden unstripped snapshot: <run_dir>/.adaptive/resumable_last.pt


def aug_ladder_for_task(task: str):
    """Heaviest-to-lightest augmentation ladder for the annotation task."""
    if task == "segment":
        return AUG_LADDER_SEGMENT
    return AUG_LADDER_DETECT


def default_mixup_for_task(task: str) -> float:
    """CLI default when --mixup is omitted: segment keeps a small top-ladder value."""
    return 0.15 if task == "segment" else 0.0


def resolve_geometry_aug(args) -> tuple:
    """degrees/flipud from CLI (defaults 0 / 0 — opt-in geometry)."""
    degrees = float(getattr(args, "degrees", 0.0) if getattr(args, "degrees", None) is not None else 0.0)
    flipud = float(getattr(args, "flipud", 0.0) if getattr(args, "flipud", None) is not None else 0.0)
    return degrees, flipud


def resolve_mixup(args, task: str, ladder_mixup: float) -> float:
    """Explicit --mixup wins; otherwise use the ladder rung's mixup (task-aware)."""
    if getattr(args, "mixup", None) is not None:
        return float(args.mixup)
    return float(ladder_mixup)


# -- Local Timestamps ------------------------------------------------------------

def local_now() -> datetime:
    """Local wall clock with timezone info (offset-aware)."""
    return datetime.now().astimezone()


def local_stamp(dt=None) -> str:
    """20261113T135943+0200 style timestamp for log filenames."""
    return (dt or local_now()).strftime("%Y%m%dT%H%M%S%z")


def local_iso(dt=None) -> str:
    """2026-11-13T13:59:43+0200 style timestamp for log content."""
    return (dt or local_now()).strftime("%Y-%m-%dT%H:%M:%S%z")


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


# -- Path Normalization ----------------------------------------------------------

def normalize_path(raw: str) -> Path:
    """Handle Windows and Unix paths on any OS."""
    cleaned = raw.strip().strip('"').strip("'")
    cleaned = cleaned.replace("\\", "/")
    return Path(cleaned).resolve()


def resolve_pretrained_weights(model: str) -> Path:
    """Resolve YOLO backbone name or path; cache under train_detector/weights/."""
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    raw = model.strip().strip('"').strip("'")
    path = Path(raw.replace("\\", "/"))

    # For bare model names, prefer the local weights cache regardless of the current directory
    if len(path.parts) == 1:
        cached = WEIGHTS_DIR / path.name
        if cached.is_file():
            return cached

    if path.is_file():
        return path.resolve()

    if len(path.parts) > 1:
        resolved = normalize_path(raw)
        if resolved.is_file():
            return resolved

    name = path.name
    local = WEIGHTS_DIR / name
    if local.is_file():
        return local

    from ultralytics.utils import SETTINGS
    from ultralytics.utils.downloads import attempt_download_asset

    old_weights_dir = SETTINGS.get("weights_dir")
    SETTINGS.update({"weights_dir": str(WEIGHTS_DIR)})
    try:
        result = attempt_download_asset(name)
    finally:
        if old_weights_dir is not None:
            SETTINGS.update({"weights_dir": old_weights_dir})

    result_path = Path(result)
    if result_path.is_file() and result_path.resolve() != local.resolve():
        if local.is_file():
            local.unlink()
        result_path.replace(local)
        return local
    if local.is_file():
        return local
    return result_path


# -- Hardware Detection ----------------------------------------------------------
# detect_hardware / process_rss_bytes / looks_like_oom are defined in
# common/hardware.py and re-exported above; this block keeps the training-only
# device helpers (canonical_device, cuda_index_for, cuda_available_for) here.

def cuda_available_for(device: str) -> bool:
    if str(device).strip().lower() == "cpu":
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def canonical_device(device: str) -> str:
    dev = str(device).strip().lower()
    if dev in ("", "cpu"):
        return "cpu"
    m = re.search(r"\d+", dev)
    return f"cuda:{m.group()}" if m else "cuda:0"


def cuda_index_for(device: str):
    """Return CUDA device index for a device string, or None for CPU."""
    if str(device).strip().lower() == "cpu":
        return None
    m = re.search(r"\d+", str(device))
    return int(m.group()) if m else 0


# -- Dataset Detection -----------------------------------------------------------

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def get_train_images_path(yaml_path: Path) -> Path:
    """Resolve the train images directory from data.yaml."""
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    train_val = str(data.get("train", "")).replace("\\", "/")
    train_path = Path(train_val)

    if not train_path.is_absolute():
        train_path = (yaml_path.parent / train_path).resolve()
    else:
        train_path = train_path.resolve()

    if not train_path.exists():
        fallback = yaml_path.parent / "train" / "images"
        if fallback.exists():
            return fallback
        return None

    return train_path


def image_files(directory: Path):
    """List all image files under a directory."""
    if directory is None:
        return []
    return [f for f in directory.rglob("*") if f.suffix.lower() in IMAGE_EXTENSIONS]


def count_images(train_path: Path) -> int:
    """Count images in a directory."""
    return len(image_files(train_path))


def sample_image_files(files, sample_fraction: float = 0.10, max_samples: int = 100):
    """Evenly spread a sample across a file list for representative stats."""
    total = len(files)
    if total == 0:
        return []
    sample_count = min(max(1, int(total * sample_fraction)), max_samples)
    step = max(1, total // sample_count)
    return files[::step][:sample_count]


def _read_image_max_dim(img_path: Path, use_pil: bool):
    if use_pil:
        from PIL import Image as PILImage
        with PILImage.open(img_path) as img:
            return max(img.size)
    img = cv2.imread(str(img_path))
    if img is not None:
        h, w = img.shape[:2]
        return max(h, w)
    return None


def detect_image_size(train_path: Path):
    """
    Sample images spread across the dataset to detect native resolution.

    Returns the median largest dimension across sampled images.
    Median is used as a representative native resolution for auto-config.
    """
    if train_path is None:
        return None

    image_list = image_files(train_path)
    if not image_list:
        return None

    sampled = sample_image_files(image_list)

    try:
        import PIL  # noqa: F401
        use_pil = True
    except ImportError:
        use_pil = False

    sizes = []
    for img_path in sampled:
        try:
            dim = _read_image_max_dim(img_path, use_pil)
            if dim is not None:
                sizes.append(dim)
        except Exception:
            continue

    if not sizes:
        return None

    min_size = min(sizes)
    max_size = max(sizes)

    # Warn if dataset has mixed resolutions
    if min_size != max_size:
        print(f"  [NOTE] Mixed image sizes detected in sample: {min_size}px - {max_size}px")
        print(f"         Using median ({int(statistics.median(sizes))}px) as native resolution reference")

    return statistics.median(sizes)


# -- Per-Run Logging ---------------------------------------------------------------

class _TeeStream:
    """File-like wrapper: raw text goes to the terminal, sanitized lines to the run log."""

    def __init__(self, stream, logger):
        self._stream = stream
        self._logger = logger

    def write(self, data):
        try:
            self._stream.write(data)
        except Exception:
            pass
        try:
            self._logger._feed(data)
        except Exception:
            pass

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def flush(self):
        try:
            self._stream.flush()
        except Exception:
            pass

    def isatty(self):
        try:
            return self._stream.isatty()
        except Exception:
            return False

    def __getattr__(self, name):
        return getattr(self._stream, name)


class RunLogger:
    """
    Per-run text log under <run_dir>/Logs plus stdout/stderr tee.

    Filenames use LOCAL time with UTC offset:
        run-YYYYMMDDTHHMMSS+HHMM.log   (fresh)
        resume-YYYYMMDDTHHMMSS+HHMM.log (resume)
    A '-2', '-3', ... suffix is appended when the name already exists.

    Terminal output stays raw; the log file strips ANSI codes, keeps only the
    final tqdm carriage-return segment of each completed line, and flushes per line.
    """

    def __init__(self, run_dir: Path, resume: bool = False):
        self.run_dir = Path(run_dir)
        logs_dir = self.run_dir / LOGS_DIRNAME
        logs_dir.mkdir(parents=True, exist_ok=True)
        prefix = "resume" if resume else "run"
        base = f"{prefix}-{local_stamp()}"
        name = f"{base}.log"
        counter = 2
        while (logs_dir / name).exists():
            name = f"{base}-{counter}.log"
            counter += 1
        self.log_path = logs_dir / name
        self._file = open(self.log_path, "a", encoding="utf-8", errors="replace")
        self._lock = threading.Lock()
        self._buffer = ""
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr

    def start_tee(self):
        sys.stdout = _TeeStream(self._orig_stdout, self)
        sys.stderr = _TeeStream(self._orig_stderr, self)

    def close(self):
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr
        with self._lock:
            self._flush_buffer(discard=True)
        try:
            self._write_line(f"[LOG CLOSE] {local_iso()}")
        except Exception:
            pass
        try:
            self._file.close()
        except Exception:
            pass

    def log(self, text=""):
        text = str(text)
        if not text:
            self._write_line("")
            return
        for line in text.rstrip("\n").split("\n"):
            self._write_line(line)

    def log_run_startup(self, args, requested_name, run_name, run_dir, mode):
        self.log("=" * 72)
        self.log(f"TRAIN DETECTOR RUN LOG ({mode})")
        self.log(f"local start time : {local_iso()}")
        self.log(f"command line     : {subprocess.list2cmdline(sys.argv)}")
        self.log(f"working dir      : {os.getcwd()}")
        self.log(f"python           : {platform.python_version()} on {platform.system()} {platform.release()}")
        self.log(f"requested name   : {requested_name}")
        self.log(f"resolved run name: {run_name}")
        self.log(f"resolved run dir : {run_dir}")
        self.log(f"device requested : {args.device if args.device else '0'}")
        prof = MEMORY_PROFILES[args.memory_profile]
        self.log(f"memory profile   : {args.memory_profile} (gpu frac {prof['gpu']}, cpu frac {prof['cpu']})")
        self.log(f"force unsafe     : {args.force_unsafe}")
        self.log(f"memory probe     : {'off (--no-memory-probe)' if args.no_memory_probe else 'on'}")
        self.log("=" * 72)

    def _write_line(self, line: str):
        line = strip_ansi(str(line)).replace("\r", " ")
        with self._lock:
            self._file.write(line + "\n")
            self._file.flush()

    def _feed(self, data: str):
        with self._lock:
            self._buffer += str(data)
            self._flush_buffer(discard=False)

    def _flush_buffer(self, discard: bool):
        # Process newline-terminated chunks. For tqdm-style carriage-return
        # streams, only the segment after the LAST \r (the complete progress
        # line) is logged; intermediate updates are skipped.
        while True:
            if discard:
                if not self._buffer:
                    return
                line, self._buffer = self._buffer, ""
            elif "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
            else:
                return
            clean = strip_ansi(line)
            if "\r" in clean:
                clean = clean.rsplit("\r", 1)[-1]
            if clean:
                self._file.write(clean + "\n")
                self._file.flush()


# -- Runtime Memory Monitoring ----------------------------------------------------
# process_rss_bytes is defined in common/hardware.py and re-exported above.

class MemoryGuard:
    """
    Background memory sampler that ENFORCES safety during training.

    Two severity levels:
      - emergency: a single sample above the emergency ceiling -> request mid-epoch
        abort (trainer.stop=True at on_train_batch_end). Partial epoch; caller
        restores backups.
      - adaptive: two consecutive samples above the adaptive ceiling -> request
        clean epoch boundary abort (trainer.stop=True at on_fit_epoch_end). No
        restore needed because the epoch completed.
    """

    SAMPLE_INTERVAL = GUARD_SAMPLE_INTERVAL_SECONDS
    SUMMARY_INTERVAL = 60.0

    def __init__(self, logger, cuda_index, gpu_adaptive_gb, gpu_emergency_gb,
                 cpu_adaptive_gb, cpu_emergency_gb, avail_adaptive_gb, avail_emergency_gb,
                 profile_name, guard_enabled=True):
        self.logger = logger
        self.cuda_index = cuda_index
        self.gpu_adaptive_gb = gpu_adaptive_gb
        self.gpu_emergency_gb = gpu_emergency_gb
        self.cpu_adaptive_gb = cpu_adaptive_gb
        self.cpu_emergency_gb = cpu_emergency_gb
        self.avail_adaptive_gb = avail_adaptive_gb
        self.avail_emergency_gb = avail_emergency_gb
        self.profile_name = profile_name
        self.guard_enabled = guard_enabled
        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.Lock()
        self._peak_alloc = 0.0
        self._peak_reserved = 0.0
        self._peak_rss = 0.0
        self._latest = {"gpu_reserved": 0.0, "gpu_alloc": 0.0, "rss": 0.0,
                        "avail": None, "temp_c": None, "power_pct": None}
        self._adaptive_streak = 0
        self._emergency_pending = False
        self._adaptive_pending = False
        self._warned_gpu = False
        self._warned_cpu = False
        self._warned_temp = False
        self._warned_power = False
        self._temp_over_count = 0
        self._last_summary = time.monotonic()

    def cuda_active(self):
        return self.cuda_index is not None

    def reset(self):
        if self.cuda_active():
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(self.cuda_index)
            except Exception as e:
                self.logger.log(f"[MEM] Could not reset CUDA peak stats: {e}")
        with self._lock:
            self._peak_alloc = 0.0
            self._peak_reserved = 0.0
            self._peak_rss = 0.0
            self._adaptive_streak = 0
            self._emergency_pending = False
            self._adaptive_pending = False

    def _sample_once(self):
        gpu_reserved = 0.0
        gpu_alloc = 0.0
        if self.cuda_active():
            try:
                # reset per-interval peak windows to capture burst usage
                torch.cuda.reset_peak_memory_stats(self.cuda_index)
                gpu_alloc = torch.cuda.max_memory_allocated(self.cuda_index) / 2**30
                gpu_reserved = torch.cuda.max_memory_reserved(self.cuda_index) / 2**30
                # also read current device memory for live burst
                try:
                    gpu_reserved = max(gpu_reserved, torch.cuda.memory_reserved(self.cuda_index) / 2**30)
                except Exception:
                    pass
            except Exception:
                pass
            # temperature / power (optional pynvml)
            try:
                import pynvml
                pynvml.nvmlInit()
                h = pynvml.nvmlDeviceGetHandleByIndex(int(self.cuda_index))
                temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
                power = pynvml.nvmlDeviceGetPowerUsage(h)
                limit = pynvml.nvmlDeviceGetPowerLimit(h)
                pct = (power / limit * 100.0) if limit else None
                with self._lock:
                    self._latest["temp_c"] = temp
                    self._latest["power_pct"] = pct
                if temp is not None and temp >= 90:
                    self._temp_over_count += 1
                    if self._temp_over_count >= 2 and not self._warned_temp:
                        self._warned_temp = True
                        self.logger.log(f"[MEM] GPU temperature {temp} C >= 90 for 2 samples; requesting adaptive stop.")
                        with self._lock:
                            self._adaptive_pending = True
                else:
                    self._temp_over_count = 0
                if pct is not None and pct > 95 and not self._warned_power:
                    self._warned_power = True
                    self.logger.log(f"[MEM WARNING] GPU power {pct:.0f}% of limit.")
            except Exception:
                pass

        rss_gb = None
        rss_bytes = process_rss_bytes()
        if rss_bytes is not None:
            rss_gb = rss_bytes / 2**30
        avail_gb = None
        if PSUTIL_AVAILABLE:
            try:
                avail_gb = psutil.virtual_memory().available / 2**30
                total_gb = psutil.virtual_memory().total / 2**30
            except Exception:
                total_gb = None
        else:
            total_gb = None

        with self._lock:
            self._peak_alloc = max(self._peak_alloc, gpu_alloc)
            self._peak_reserved = max(self._peak_reserved, gpu_reserved)
            if rss_gb is not None:
                self._peak_rss = max(self._peak_rss, rss_gb)
            self._latest["gpu_reserved"] = gpu_reserved
            self._latest["gpu_alloc"] = gpu_alloc
            self._latest["rss"] = rss_gb if rss_gb is not None else self._latest["rss"]
            self._latest["avail"] = avail_gb

        # threshold evaluation (only if guard enabled)
        if not self.guard_enabled:
            return

        emergency = False
        adaptive = False
        reasons = []
        if self.cuda_active():
            if gpu_reserved > self.gpu_emergency_gb:
                emergency = True
                reasons.append(f"GPU reserved {gpu_reserved:.2f} > emergency {self.gpu_emergency_gb:.2f}")
            elif gpu_reserved > self.gpu_adaptive_gb:
                adaptive = True
                reasons.append(f"GPU reserved {gpu_reserved:.2f} > adaptive {self.gpu_adaptive_gb:.2f}")
        if rss_gb is not None:
            if self.cpu_emergency_gb and rss_gb > self.cpu_emergency_gb:
                emergency = True
                reasons.append(f"CPU RSS {rss_gb:.2f} > emergency {self.cpu_emergency_gb:.2f}")
            elif self.cpu_adaptive_gb and rss_gb > self.cpu_adaptive_gb:
                adaptive = True
                reasons.append(f"CPU RSS {rss_gb:.2f} > adaptive {self.cpu_adaptive_gb:.2f}")
        if avail_gb is not None and total_gb:
            if avail_gb < total_gb * EMERGENCY_AVAIL_RAM_FRACTION:
                emergency = True
                reasons.append(f"available RAM {avail_gb:.2f} < emergency {total_gb*EMERGENCY_AVAIL_RAM_FRACTION:.2f}")
            elif avail_gb < total_gb * ADAPTIVE_AVAIL_RAM_FRACTION:
                adaptive = True
                reasons.append(f"available RAM {avail_gb:.2f} < adaptive {total_gb*ADAPTIVE_AVAIL_RAM_FRACTION:.2f}")

        with self._lock:
            if emergency:
                if not self._emergency_pending:
                    self.logger.log(f"[MEM ENFORCE] EMERGENCY sample: {'; '.join(reasons)} -> aborting current epoch (partial).")
                self._emergency_pending = True
                self._adaptive_pending = False
                self._adaptive_streak = 0
            elif adaptive:
                self._adaptive_streak += 1
                if self._adaptive_streak >= 2:
                    if not self._adaptive_pending:
                        self.logger.log(f"[MEM ENFORCE] ADAPTIVE pressure for 2 samples: {'; '.join(reasons)} -> finishing epoch then stopping.")
                    self._adaptive_pending = True
            else:
                self._adaptive_streak = 0

        if gpu_reserved > (self.gpu_adaptive_gb or 1e9) and not self._warned_gpu:
            self._warned_gpu = True
            self.logger.log(f"[MEM WARNING] GPU peak reserved {gpu_reserved:.2f} GB exceeds adaptive limit {self.gpu_adaptive_gb:.2f} GB.")
        if rss_gb is not None and self.cpu_adaptive_gb and rss_gb > self.cpu_adaptive_gb and not self._warned_cpu:
            self._warned_cpu = True
            self.logger.log(f"[MEM WARNING] CPU peak RSS {rss_gb:.2f} GB exceeds adaptive limit {self.cpu_adaptive_gb:.2f} GB.")

    def sample(self):
        try:
            self._sample_once()
        except Exception:
            pass

    def summary_line(self, tag=""):
        with self._lock:
            parts = []
            if self.cuda_active():
                limit = f" (adaptive {self.gpu_adaptive_gb:.2f})" if self.gpu_adaptive_gb else ""
                parts.append(f"GPU alloc peak {self._peak_alloc:.2f} GB | GPU reserved peak {self._peak_reserved:.2f} GB{limit}")
            if PSUTIL_AVAILABLE:
                limit = f" (adaptive {self.cpu_adaptive_gb:.2f})" if self.cpu_adaptive_gb else ""
                parts.append(f"CPU RSS peak {self._peak_rss:.2f} GB{limit}")
            else:
                parts.append("CPU RSS sampling unavailable (psutil not installed)")
            if not parts:
                parts.append("no memory sampling available")
            return f"[MEM {local_now().strftime('%H:%M:%S')}] {tag + ' ' if tag else ''}" + " | ".join(parts)

    def log_summary(self, tag=""):
        self.sample()
        self.logger.log(self.summary_line(tag))

    def on_epoch_end(self, trainer=None):
        try:
            epoch = getattr(trainer, "epoch", None)
            self.log_summary(f"epoch {epoch + 1}:" if isinstance(epoch, int) else "epoch:")
        except Exception:
            pass

    def start(self):
        self.reset()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="memory-guard")
        self._thread.start()

    def _run(self):
        next_summary = time.monotonic() + self.SUMMARY_INTERVAL
        while not self._stop.wait(self.SAMPLE_INTERVAL):
            try:
                self.sample()
                if time.monotonic() >= next_summary:
                    self.logger.log(self.summary_line("60s:"))
                    next_summary = time.monotonic() + self.SUMMARY_INTERVAL
            except Exception:
                pass

    def stop(self):
        if self._thread is not None:
            self._stop.set()
            try:
                self._thread.join(timeout=15)
            except RuntimeError:
                pass
            self._thread = None
        self.log_summary("final:")

    # Enforcement API used by callbacks
    def consume_emergency(self) -> bool:
        with self._lock:
            v = self._emergency_pending
            self._emergency_pending = False
            return v

    def consume_adaptive(self) -> bool:
        with self._lock:
            v = self._adaptive_pending
            self._adaptive_pending = False
            return v

    def latest(self) -> dict:
        with self._lock:
            return dict(self._latest)

    def peak_reserved(self) -> float:
        with self._lock:
            return self._peak_reserved

    def peak_rss(self) -> float:
        with self._lock:
            return self._peak_rss


def make_guard(logger, hw, device, args, guard_enabled=True) -> MemoryGuard:
    profile = GUARD_PROFILE_CAPS.get(args.memory_profile, GUARD_PROFILE_CAPS["safe"])
    cuda_index = cuda_index_for(device) if cuda_available_for(device) else None
    vram = hw.get("vram_gb") or 0.0
    ram = hw.get("ram_gb") or 0.0
    gpu_adaptive = vram * max(profile["gpu"], ADAPTIVE_GPU_FRACTION) if (cuda_index is not None and vram) else None
    gpu_emergency = vram * max(profile["gpu"], EMERGENCY_GPU_FRACTION) if (cuda_index is not None and vram) else None
    cpu_adaptive = ram * max(profile["cpu"], ADAPTIVE_CPU_RSS_FRACTION) if ram else None
    cpu_emergency = ram * max(profile["cpu"], EMERGENCY_CPU_RSS_FRACTION) if ram else None
    avail_adaptive = ram * ADAPTIVE_AVAIL_RAM_FRACTION if ram else None
    avail_emergency = ram * EMERGENCY_AVAIL_RAM_FRACTION if ram else None
    logger.log(
        f"[Memory Guard] profile '{args.memory_profile}': "
        + (f"GPU adaptive {gpu_adaptive:.2f}/emergency {gpu_emergency:.2f} GB of {vram:.1f} | " if gpu_adaptive else "GPU monitoring off (no CUDA) | ")
        + (f"CPU adaptive {cpu_adaptive:.2f}/emergency {cpu_emergency:.2f} GB of {ram:.1f}" if cpu_adaptive else "CPU RSS limit unknown")
    )
    return MemoryGuard(logger, cuda_index, gpu_adaptive, gpu_emergency,
                       cpu_adaptive, cpu_emergency, avail_adaptive, avail_emergency,
                       args.memory_profile, guard_enabled)


# -- Guarded trainer subclasses ----------------------------------------------------
class GuardedTrainerMixin:
    """
    Memory-guard plumbing layered onto a task-specific Ultralytics trainer:
      - forces unsafe checkpoint settings to safe ones on resume (check_resume override)
      - rewrites args.yaml with the forced values after base __init__
      - allows an AttemptTracker to attach callbacks
    """

    def __init__(self, overrides=None, _callbacks=(), forced_args=None, tracker=None, logger=None, **kwargs):
        self._forced_args = forced_args or {}
        self._tracker = tracker
        self._logger = logger
        self._forced_keys = list(self._forced_args.keys())
        super().__init__(overrides=overrides, _callbacks=_callbacks, **kwargs)

    def check_resume(self, overrides):
        super().check_resume(overrides)
        # Force unsafe checkpoint settings. check_resume allow-list only covers a
        # subset; we override ALL forced keys here, before get_dataset() runs.
        for k, v in self._forced_args.items():
            try:
                setattr(self.args, k, v)
            except Exception as e:
                if self._logger is not None:
                    self._logger.log(f"[GuardedTrainer] could not force {k}={v}: {e}")
        # Rewrite args.yaml with forced values (save_dir known after super().__init__).
        try:
            self._rewrite_args_yaml()
        except Exception as e:
            if self._logger is not None:
                self._logger.log(f"[GuardedTrainer] args.yaml rewrite failed: {e}")

    def _rewrite_args_yaml(self):
        try:
            save_dir = getattr(self.args, "save_dir", None)
            if not save_dir:
                return
            p = Path(save_dir) / "args.yaml"
            if not p.exists():
                return
            try:
                data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            except Exception:
                data = {}
            for k, v in self._forced_args.items():
                data[k] = v
            # write preserving simple style
            with open(p, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, default_flow_style=True, allow_unicode=True, sort_keys=False)
        except Exception as e:
            if self._logger is not None:
                self._logger.log(f"[GuardedTrainer] _rewrite_args_yaml error: {e}")


class GuardedDetectionTrainer(GuardedTrainerMixin, DetectionTrainer):
    """Detect trainer + guard plumbing."""


class GuardedSegmentTrainer(GuardedTrainerMixin, SegmentationTrainer):
    """Segment trainer + guard plumbing (YOLO26 -seg weights need this trainer class)."""


class GuardedPoseTrainer(GuardedTrainerMixin, PoseTrainer):
    """Pose trainer + guard plumbing (YOLO26 -pose weights need this trainer class)."""


GUARDED_TRAINERS = {
    "detect": GuardedDetectionTrainer,
    "segment": GuardedSegmentTrainer,
    "pose": GuardedPoseTrainer,
}


def guarded_trainer_for_task(task):
    """Task-correct guarded trainer class. Ultralytics must train -seg/-pose weights with
    SegmentationTrainer/PoseTrainer: the detect trainer produces tuple predictions that
    crash its own val NMS."""
    return GUARDED_TRAINERS.get(task, GuardedDetectionTrainer)


def resolve_attempt_task(args, ckpt_cfg=None, chosen=None) -> str:
    """Best-effort task for a supervised attempt: resolved dataset task -> checkpoint train_args
    -> model-name suffix (-pose wins over -seg) -> detect."""
    t = getattr(args, "dataset_task", None)
    if t in GUARDED_TRAINERS:
        return t
    if isinstance(ckpt_cfg, dict):
        ct = ckpt_cfg.get("task")
        if ct in GUARDED_TRAINERS:
            return ct
    model_name = str((chosen or {}).get("model") or "").lower()
    if "-pose" in model_name:
        return "pose"
    if "-seg" in model_name:
        return "segment"
    return "detect"


# -- Fingerprints ------------------------------------------------------------------

def hardware_device_fingerprint(device: str) -> dict:
    """Identify the exact hardware+device+library combo calibration data belongs to."""
    fp = {
        "os": platform.system(),
        "device": canonical_device(device),
        "cpu_cores": os.cpu_count() or 4,
        "ram_gb": None,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda or None,
        "ultralytics": "unknown",
    }
    try:
        import ultralytics
        fp["ultralytics"] = getattr(ultralytics, "__version__", "unknown")
    except Exception:
        pass
    if PSUTIL_AVAILABLE:
        try:
            fp["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 2)
        except Exception:
            pass
    idx = cuda_index_for(device)
    if idx is not None:
        try:
            props = torch.cuda.get_device_properties(idx)
            fp["gpu_name"] = props.name
            fp["vram_gb"] = round(props.total_memory / 2**30, 2)
        except Exception:
            pass
    return fp


def _box_wh_from_label_line(parts: list, task: str):
    """Return (w, h) as fractions of image size, or None if unparseable."""
    try:
        nums = [float(x) for x in parts[1:]]
    except (TypeError, ValueError):
        return None
    if len(nums) >= 4 and (task == "detect" or len(nums) == 4 or (task == "pose" and len(nums) >= 4)):
        # detect: cls cx cy w h; pose: cls cx cy w h + kpts — first 4 after cls are bbox
        if task in ("detect", "pose") or len(nums) == 4:
            w, h = abs(nums[2]), abs(nums[3])
            if 0.0 < w <= 1.5 and 0.0 < h <= 1.5:
                return w, h
    if task == "segment" and len(nums) >= 6 and len(nums) % 2 == 0:
        xs = nums[0::2]
        ys = nums[1::2]
        if not xs or not ys:
            return None
        w = max(xs) - min(xs)
        h = max(ys) - min(ys)
        if w >= 0.0 and h >= 0.0:
            return abs(w), abs(h)
    # Ambiguous segment-looking polygon on detect task: treat as polygon bbox
    if len(nums) >= 6 and len(nums) % 2 == 0:
        xs = nums[0::2]
        ys = nums[1::2]
        return abs(max(xs) - min(xs)), abs(max(ys) - min(ys))
    if len(nums) >= 4:
        return abs(nums[2]), abs(nums[3])
    return None


def compute_label_stats(labels_dir: Path, task: str = "detect",
                        sample_limit: int = LABEL_STATS_SAMPLE_LIMIT) -> dict:
    """Cheap label-file stats (up to sample_limit files) for object-size policy."""
    empty = {
        "label_files_sampled": 0,
        "boxes_sampled": 0,
        "median_box_w": None,
        "median_box_h": None,
        "p10_box_w": None,
        "p10_box_h": None,
        "p25_box_w": None,
        "p25_box_h": None,
        "boxes_per_image": None,
        "empty_label_fraction": None,
    }
    if labels_dir is None or not Path(labels_dir).is_dir():
        return empty
    files = sorted(Path(labels_dir).glob("*.txt"))
    if not files:
        return empty
    if len(files) > sample_limit:
        # Evenly spaced sample across the split
        step = len(files) / float(sample_limit)
        files = [files[int(i * step)] for i in range(sample_limit)]
    widths, heights = [], []
    boxes_per = []
    empty_count = 0
    for lp in files:
        try:
            lines = [ln.strip() for ln in lp.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
        except OSError:
            continue
        if not lines:
            empty_count += 1
            boxes_per.append(0)
            continue
        n_boxes = 0
        for ln in lines:
            parts = ln.split()
            if len(parts) < 5:
                continue
            wh = _box_wh_from_label_line(parts, task)
            if wh is None:
                continue
            widths.append(wh[0])
            heights.append(wh[1])
            n_boxes += 1
        boxes_per.append(n_boxes)
        if n_boxes == 0:
            empty_count += 1
    def _pct(vals, p):
        if not vals:
            return None
        ordered = sorted(vals)
        idx = max(0, min(len(ordered) - 1, int(round((p / 100.0) * (len(ordered) - 1)))))
        return round(ordered[idx], 6)
    n_files = len(boxes_per)
    return {
        "label_files_sampled": n_files,
        "boxes_sampled": len(widths),
        "median_box_w": round(statistics.median(widths), 6) if widths else None,
        "median_box_h": round(statistics.median(heights), 6) if heights else None,
        "p10_box_w": _pct(widths, 10),
        "p10_box_h": _pct(heights, 10),
        "p25_box_w": _pct(widths, 25),
        "p25_box_h": _pct(heights, 25),
        "boxes_per_image": round(sum(boxes_per) / n_files, 3) if n_files else None,
        "empty_label_fraction": round(empty_count / n_files, 4) if n_files else None,
    }


def object_px_at_imgsz(box_frac, imgsz):
    if box_frac is None or imgsz is None:
        return None
    return float(box_frac) * float(imgsz)


def warn_object_size_vs_imgsz(logger, label_stats: dict, imgsz: int, native_size):
    """Loud warning when objects remain tiny at the chosen imgsz."""
    med_w = object_px_at_imgsz(label_stats.get("median_box_w"), imgsz)
    med_h = object_px_at_imgsz(label_stats.get("median_box_h"), imgsz)
    p25_w = object_px_at_imgsz(label_stats.get("p25_box_w"), imgsz)
    p25_h = object_px_at_imgsz(label_stats.get("p25_box_h"), imgsz)
    if med_w is None or med_h is None:
        return
    med = min(med_w, med_h)
    p25 = min(p25_w, p25_h) if p25_w is not None and p25_h is not None else None
    if med < MIN_RESOLVED_OBJECT_PX or (p25 is not None and p25 < MIN_RESOLVED_OBJECT_P25_PX):
        native_note = f", native median {native_size}px" if native_size else ""
        msg = (
            f"[WARNING] Objects may be under-resolved at imgsz={imgsz}px{native_note}: "
            f"median object ~{med:.1f}px (need >= {MIN_RESOLVED_OBJECT_PX:.0f}px)"
            + (f", p25 ~{p25:.1f}px (need >= {MIN_RESOLVED_OBJECT_P25_PX:.0f}px)" if p25 is not None else "")
            + ". Prefer a higher memory-safe imgsz up to native resolution."
        )
        logger.log(msg)
        print(msg)


def warn_train_val_label_stats(logger, train_stats: dict, val_stats: dict):
    """Warn when train/val label distributions diverge sharply."""
    if not train_stats or not val_stats:
        return
    tw, vw = train_stats.get("median_box_w"), val_stats.get("median_box_w")
    th, vh = train_stats.get("median_box_h"), val_stats.get("median_box_h")
    if tw and vw and min(tw, vw) > 0:
        ratio = max(tw, vw) / min(tw, vw)
        if ratio > 2.0:
            logger.log(f"[WARNING] Train/val median box width differ sharply (train={tw}, val={vw}, ratio={ratio:.2f}).")
    if th and vh and min(th, vh) > 0:
        ratio = max(th, vh) / min(th, vh)
        if ratio > 2.0:
            logger.log(f"[WARNING] Train/val median box height differ sharply (train={th}, val={vh}, ratio={ratio:.2f}).")
    te, ve = train_stats.get("empty_label_fraction"), val_stats.get("empty_label_fraction")
    if te is not None and ve is not None and abs(te - ve) > 0.25:
        logger.log(f"[WARNING] Train/val empty-label fractions differ sharply (train={te}, val={ve}).")


def prefer_imgsz_for_objects(imgsz_list, label_stats: dict, native_size):
    """
    If objects look small at the first candidate, prefer higher memory-safe imgsz
    candidates up to native (never upscale past native unless explicit).
    """
    if not imgsz_list:
        return imgsz_list
    med_w = label_stats.get("median_box_w")
    med_h = label_stats.get("median_box_h")
    p25_w = label_stats.get("p25_box_w")
    p25_h = label_stats.get("p25_box_h")
    if med_w is None or med_h is None:
        return imgsz_list
    first = imgsz_list[0]
    med_px = min(object_px_at_imgsz(med_w, first), object_px_at_imgsz(med_h, first))
    p25_px = None
    if p25_w is not None and p25_h is not None:
        p25_px = min(object_px_at_imgsz(p25_w, first), object_px_at_imgsz(p25_h, first))
    if med_px >= MIN_RESOLVED_OBJECT_PX and (p25_px is None or p25_px >= MIN_RESOLVED_OBJECT_P25_PX):
        return imgsz_list
    # Prefer larger sizes first (still capped to native by imgsz_candidates).
    return sorted(imgsz_list, reverse=True)


def dataset_fingerprint(yaml_path: Path, train_path: Path, task: str = "detect") -> dict:
    """Content stats of the dataset used for calibration and cache validity.
    task makes memory-cache / calibration entries task-specific."""
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    files = image_files(train_path)
    sampled = sample_image_files(files)

    try:
        import PIL  # noqa: F401
        use_pil = True
    except ImportError:
        use_pil = False

    labels_dir = None
    if train_path is not None:
        candidate = train_path.parent / "labels"
        labels_dir = candidate if candidate.is_dir() else train_path

    sizes = []
    label_counts = []
    for img_path in sampled:
        try:
            dim = _read_image_max_dim(img_path, use_pil)
            if dim is not None:
                sizes.append(dim)
                label_path = labels_dir / (img_path.stem + ".txt")
                count = 0
                try:
                    if label_path.is_file():
                        count = sum(1 for ln in label_path.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip())
                except OSError:
                    count = 0
                label_counts.append(count)
        except Exception:
            continue

    task_name = task if task in ("detect", "segment", "pose") else "detect"
    label_stats = compute_label_stats(labels_dir, task_name) if labels_dir else compute_label_stats(Path("."), task_name)
    # Val stats when available (for divergence warnings; also folded into fingerprint lightly)
    val_stats = {}
    try:
        val_key = data.get("val")
        if val_key:
            val_images = resolve_yaml_split_path(yaml_path, str(val_key))
            val_labels = val_images.parent / "labels" if val_images is not None else None
            if val_labels is not None and val_labels.is_dir():
                val_stats = compute_label_stats(val_labels, task_name)
    except Exception:
        val_stats = {}

    fp = {
        "data_yaml": str(yaml_path).replace("\\", "/"),
        "task": task_name,
        "image_count": len(files),
        "nc": int(data.get("nc") or 0),
        "median_img_size": int(statistics.median(sizes)) if sizes else None,
        "max_sampled_img_size": int(max(sizes)) if sizes else None,
        "avg_labels_per_sampled_image": round(sum(label_counts) / len(label_counts), 2) if label_counts else None,
        "label_stats": label_stats,
    }
    if val_stats:
        fp["val_label_stats"] = val_stats
    return fp


def fingerprint_hash(fp: dict) -> str:
    payload = json.dumps(fp, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


# -- Dynamic Memory Cache (memory_profiles.json, schema v4 / quality-v2) ------------
# Legacy vram_estimates.json is a model-only fallback heuristic; it is NOT final
# safety truth. Only measured, fingerprint-matched entries in this cache can
# skip recalibration. A probe SAFE verdict is a memory estimate only — not a
# convergence guarantee.

def load_memory_cache(logger=None) -> dict:
    if not MEMORY_CACHE_PATH.exists():
        return {"version": MEMORY_CACHE_SCHEMA_VERSION, "hardware_profiles": {}}
    try:
        with open(MEMORY_CACHE_PATH, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or data.get("version") != MEMORY_CACHE_SCHEMA_VERSION:
            if logger:
                logger.log(
                    f"[Cache] {MEMORY_CACHE_PATH} is not schema version {MEMORY_CACHE_SCHEMA_VERSION}; "
                    "starting fresh (full recalibration)."
                )
            return {"version": MEMORY_CACHE_SCHEMA_VERSION, "hardware_profiles": {}}
        data.setdefault("hardware_profiles", {})
        return data
    except (OSError, json.JSONDecodeError) as e:
        if logger:
            logger.log(f"[Cache] Ignoring corrupt memory cache {MEMORY_CACHE_PATH}: {e}")
        return {"version": MEMORY_CACHE_SCHEMA_VERSION, "hardware_profiles": {}}


def save_memory_cache(cache: dict, logger) -> bool:
    try:
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(MEMORY_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
            f.write("\n")
        return True
    except OSError as e:
        logger.log(f"[Cache WARNING] Could not write {MEMORY_CACHE_PATH}: {e}; keeping in-memory calibration result.")
        return False


def sig_num(v) -> str:
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def config_signature(model, imgsz, batch, workers, multi_scale, mosaic=1.0, mixup=0.0,
                     copy_paste=0.0, degrees=0.0, close_mosaic=60, flipud=0.0,
                     cos_lr=True, nbs=64, cache=False, **_extra) -> str:
    """
    Cache config key including every newly varying quality-v2 parameter.
    cos_lr/nbs stay splat-safe; cache= is keyed when throughput probes enable it.
    """
    cache_token = "cache0" if not cache else f"cache{cache}"
    return "|".join([
        str(model), str(imgsz), str(batch), str(workers),
        f"ms{sig_num(multi_scale)}",
        f"mosaic{sig_num(mosaic)}",
        f"mixup{sig_num(mixup)}",
        f"copy{sig_num(copy_paste)}",
        f"deg{sig_num(degrees)}",
        f"flipud{sig_num(flipud)}",
        f"cm{sig_num(close_mosaic)}",
        cache_token,
    ])


def selection_signature(args, ds_fp: dict) -> str:
    """Which calibration request a cached selection answers."""
    explicit = {k: v for k, v in {
        "model": args.model, "imgsz": args.imgsz, "batch": args.batch, "workers": args.workers,
        "degrees": getattr(args, "degrees", None),
        "flipud": getattr(args, "flipud", None),
        "mixup": getattr(args, "mixup", None),
    }.items() if v is not None}
    sig = json.dumps(explicit, sort_keys=True, separators=(",", ":")) if explicit else "auto"
    # Intent marker: selections recorded under a different auto-policy never
    # satisfy a post-change run (quality-v1 / schema v3 are discarded).
    return f"ds:{fingerprint_hash(ds_fp)}|profile:{args.memory_profile}|intent:{SELECTION_INTENT}|{sig}"


def get_hw_profile(cache: dict, hw_fp: dict):
    pid = fingerprint_hash(hw_fp)
    profile = cache["hardware_profiles"].get(pid)
    if profile is None:
        profile = {"fingerprint": hw_fp, "selections": {}, "configs": {}}
        cache["hardware_profiles"][pid] = profile
    return pid, profile


def lookup_cached_selection(cache: dict, hw_fp: dict, ds_fp: dict, sel_key: str, logger):
    """Return (selection, config entry) only when every fingerprint matches a measured-safe entry."""
    pid = fingerprint_hash(hw_fp)
    profile = cache["hardware_profiles"].get(pid)
    if not profile:
        logger.log("[Cache] No hardware/device profile for this machine yet; full calibration needed.")
        return None
    sel = profile.get("selections", {}).get(sel_key)
    if not sel:
        logger.log("[Cache] No selection for this dataset/profile/explicit-flag combination; calibration required.")
        return None
    entry = profile.get("configs", {}).get(sel.get("config_key"))
    if not entry:
        logger.log("[Cache] Selection points at a missing config entry; calibration required.")
        return None
    if fingerprint_hash(entry.get("dataset_fingerprint", {})) != fingerprint_hash(ds_fp):
        logger.log("[Cache] Dataset fingerprint changed since calibration; re-calibration required.")
        return None
    if not entry.get("measured") or not entry.get("safe"):
        logger.log("[Cache] Selection entry is not measured-safe; re-calibration required.")
        return None
    return sel, entry


def record_calibration(cache: dict, logger, hw_fp: dict, ds_fp: dict, sel_key: str,
                       chosen: dict, peaks, safe: bool, measured: bool, make_selection: bool = True):
    """Store a measured (or unsafe-forced) probe result in the memory cache."""
    _pid, profile = get_hw_profile(cache, hw_fp)
    key = config_signature(**chosen)
    alloc, res, rss = peaks
    entry = {
        "dataset_fingerprint": ds_fp,
        "peak_gpu_allocated_gb": round(alloc, 3) if alloc is not None else None,
        "peak_gpu_reserved_gb": round(res, 3) if res is not None else None,
        "peak_cpu_rss_gb": round(rss, 3) if rss is not None else None,
        "safe": bool(safe),
        "measured": bool(measured),
        "updated_at": local_iso(),
    }
    profile["configs"][key] = entry
    if make_selection:
        profile["selections"][sel_key] = {
            "config_key": key,
            "config": dict(chosen),
            "updated_at": local_iso(),
        }
    cache["updated_at"] = local_iso()
    saved = save_memory_cache(cache, logger)
    if saved and measured:
        logger.log(f"[Cache] Recorded measured config {key} (peak GPU reserved {sig_num(entry['peak_gpu_reserved_gb'])} GB).")
    return entry


# -- Attempt tracker (callbacks + status) ---------------------------------------
def _metric_from_trainer(trainer, *keys):
    """Pull a float metric from trainer.metrics / trainer.validator / loss items."""
    metrics = getattr(trainer, "metrics", None) or {}
    if isinstance(metrics, dict):
        for k in keys:
            if k in metrics:
                try:
                    return float(metrics[k])
                except (TypeError, ValueError):
                    pass
        for mk, mv in metrics.items():
            mk_l = str(mk).lower()
            for k in keys:
                if k.lower() in mk_l:
                    try:
                        return float(mv)
                    except (TypeError, ValueError):
                        pass
    loss = getattr(trainer, "loss", None)
    if loss is not None and "train/loss" in keys:
        try:
            return float(loss.detach().item() if hasattr(loss, "detach") else loss)
        except Exception:
            pass
    tloss = getattr(trainer, "tloss", None)
    if tloss is not None and any("loss" in k.lower() for k in keys):
        try:
            if hasattr(tloss, "mean"):
                return float(tloss.mean().item() if hasattr(tloss.mean(), "item") else tloss.mean())
            return float(tloss)
        except Exception:
            pass
    return None


def classify_training_condition(history: list, epoch: int, patience: int = 100) -> str:
    """
    Plain-English training condition from recent (loss, score) history.
    history entries: {"epoch", "loss", "score"} with score ~= mAP50.
    """
    if epoch <= 5:
        return "Still in early learning — scores jump around; this is normal."
    if len(history) < 2:
        return "Still gathering the first readings — check back after a few epochs."
    recent = history[-min(5, len(history)):]
    losses = [h["loss"] for h in recent if h.get("loss") is not None]
    scores = [h["score"] for h in recent if h.get("score") is not None]
    last_score = scores[-1] if scores else None
    if last_score is not None and epoch >= 10 and last_score < 0.05:
        return "Struggling — check labels, object size vs image size, or try lighter augmentation."
    loss_down = len(losses) >= 2 and losses[-1] < losses[0] - 1e-6
    score_up = len(scores) >= 2 and scores[-1] > scores[0] + 1e-4
    score_flat = len(scores) >= 2 and abs(scores[-1] - scores[0]) <= 1e-4
    loss_flat = len(losses) >= 2 and abs(losses[-1] - losses[0]) <= 1e-4
    if loss_down and score_up:
        if scores[-1] - scores[0] < 0.01:
            return "Improving slowly — still worth continuing."
        return "Learning well — getting better at finding the objects."
    if loss_down and (score_flat or (len(scores) >= 2 and scores[-1] < scores[0])):
        return "Fitting the training images, but not improving on the check set yet."
    if loss_flat and score_flat:
        return "Looking stable — may stop early if nothing improves for a while."
    if score_up:
        return "Improving slowly — still worth continuing."
    return "Looking stable — may stop early if nothing improves for a while."


def format_progress_block(epoch, target, avg_sec, patience, epochs_since_best, condition,
                          loss=None, score=None) -> str:
    pct = (100.0 * epoch / target) if target else 0.0
    lines = [
        f"[Progress] epoch {epoch}/{target} ({pct:.1f}% of scheduled epochs)",
        "  note: early stop (patience) or time budget can end sooner — percent is a schedule estimate",
    ]
    if avg_sec is not None:
        eta_full = avg_sec * max(0, target - epoch)
        lines.append(f"  avg time/epoch : {format_duration(avg_sec)}")
        lines.append(f"  ETA to schedule end : ~{format_duration(eta_full)}")
        if patience is not None and epochs_since_best is not None:
            rem = max(0, int(patience) - int(epochs_since_best))
            lines.append(
                f"  if no further gains : ~{format_duration(avg_sec * rem)} "
                f"(guess from patience remaining {rem}/{patience})"
            )
    lines.append(f"  condition : {condition}")
    extras = []
    if loss is not None:
        extras.append(f"train loss {loss:.4g}")
    if score is not None:
        extras.append(f"check-set score {score:.4g}")
    if extras:
        lines.append(f"  (details: {', '.join(extras)})")
    return "\n".join(lines)


class AttemptTracker:
    def __init__(self, logger, guard, run_dir, target_epochs, intended_cfg, state,
                 on_emergency, on_adaptive, on_complete, on_crash, patience=100):
        self.logger = logger
        self.guard = guard
        self.run_dir = Path(run_dir)
        self.target_epochs = target_epochs
        self.intended_cfg = intended_cfg
        self.state = state
        self.on_emergency = on_emergency
        self.on_adaptive = on_adaptive
        self.on_complete = on_complete
        self.on_crash = on_crash
        self.patience = patience
        self.status = "running"
        self.emergency_triggered = False
        self.adaptive_triggered = False
        self.oom_auto_reduced = False
        self.start_epoch = None
        self.completed_epoch = None
        self.actual_cfg = None
        self.crash_exc = None
        self.resumable_ckpt = None
        self._stop_handled = False
        self._epoch_t0 = None
        self._epoch_durations = []
        self._metric_history = []
        self._best_score = None
        self._epochs_since_best = 0

    def callbacks(self):
        return {
            "on_train_start": self.on_train_start,
            "on_train_epoch_start": self.on_train_epoch_start,
            "on_train_batch_end": self.on_train_batch_end,
            "on_train_epoch_end": self.on_train_epoch_end,
            "on_fit_epoch_end": self.on_fit_epoch_end,
            "on_train_end": self.on_train_end,
        }

    def on_train_start(self, trainer):
        # Verify actual trainer args vs intended
        try:
            actual = {
                "model": getattr(trainer.args, "model", None),
                "imgsz": getattr(trainer.args, "imgsz", None),
                "batch": getattr(trainer.args, "batch", None),
                "workers": getattr(trainer.args, "workers", None),
                "multi_scale": getattr(trainer.args, "multi_scale", None),
                "mosaic": getattr(trainer.args, "mosaic", None),
                "mixup": getattr(trainer.args, "mixup", None),
                "copy_paste": getattr(trainer.args, "copy_paste", None),
                "epochs": getattr(trainer.args, "epochs", None),
            }
            self.actual_cfg = actual
            mismatches = []
            for k in ("imgsz", "batch", "workers", "multi_scale", "mosaic", "mixup", "copy_paste"):
                iv = self.intended_cfg.get(k)
                av = actual.get(k)
                if iv is not None and av is not None and iv != av:
                    mismatches.append(f"{k}: intended {iv} actual {av}")
            if mismatches:
                self.logger.log("[Attempt] CONFIG MISMATCH detected before any batch:")
                for m in mismatches:
                    self.logger.log(f"  {m}")
                raise RuntimeError("AttemptConfigMismatch: " + "; ".join(mismatches))
        except RuntimeError:
            raise
        except Exception as e:
            self.logger.log(f"[Attempt] config verify error: {e}")
        self.start_epoch = getattr(trainer, "start_epoch", None)
        self.logger.log(f"[Attempt] training started; start_epoch={self.start_epoch} target_epochs={self.target_epochs}")

    def on_train_epoch_start(self, trainer):
        self._epoch_t0 = time.perf_counter()
        self.guard.reset() if False else None

    def on_train_batch_end(self, trainer):
        if self.guard.consume_emergency():
            self.emergency_triggered = True
            try:
                trainer.stop = True
            except Exception:
                pass
            self.logger.log("[Attempt] emergency stop requested at batch end (partial epoch).")

    def on_train_epoch_end(self, trainer):
        # detect OOM auto-reduce: Ultralytics sets _oom_retries; batch may change
        try:
            if getattr(trainer, "_oom_retries", 0) and getattr(trainer, "_oom_retries", 0) > 0:
                self.oom_auto_reduced = True
        except Exception:
            pass
        # sync actual batch
        try:
            b = getattr(trainer.args, "batch", None)
            if b is not None and self.intended_cfg.get("batch") != b:
                self.logger.log(f"[Attempt] Ultralytics adjusted batch to {b} (OOM auto-reduce).")
                self.intended_cfg["batch"] = b
        except Exception:
            pass
        self.guard.log_summary(f"epoch {getattr(trainer,'epoch',0)+1}:")

    def on_fit_epoch_end(self, trainer):
        epoch = getattr(trainer, "epoch", None)
        # Ultralytics fires this callback a SECOND time from final_eval() after the
        # epoch loop exits. Once an adaptive stop has fired, freeze the real epoch
        # count (kills the 11 -> 12 re-fire artifact); clamp so a clean run can never
        # overshoot the target (kills the 600 -> 601 artifact).
        if isinstance(epoch, int) and not self.adaptive_triggered:
            self.completed_epoch = min(epoch + 1, self.target_epochs)
            # Live progress (skip the final_eval re-fire via adaptive_triggered gate above)
            if self._epoch_t0 is not None:
                self._epoch_durations.append(time.perf_counter() - self._epoch_t0)
                self._epoch_t0 = None
            loss = _metric_from_trainer(trainer, "train/box_loss", "train/loss", "loss")
            if loss is None:
                try:
                    loss = float(getattr(trainer, "loss", None) or 0) or None
                except Exception:
                    loss = None
            # Prefer box mAP50; fall back to any mAP50-like key
            score = _metric_from_trainer(
                trainer,
                "metrics/mAP50(B)", "metrics/mAP50(M)", "metrics/mAP50",
                "mAP50", "metrics/fitness",
            )
            self._metric_history.append({
                "epoch": self.completed_epoch, "loss": loss, "score": score,
            })
            if score is not None:
                if self._best_score is None or score > self._best_score + 1e-6:
                    self._best_score = score
                    self._epochs_since_best = 0
                else:
                    self._epochs_since_best += 1
            avg_sec = (sum(self._epoch_durations) / len(self._epoch_durations)) if self._epoch_durations else None
            condition = classify_training_condition(
                self._metric_history, self.completed_epoch, patience=self.patience,
            )
            block = format_progress_block(
                self.completed_epoch, self.target_epochs, avg_sec, self.patience,
                self._epochs_since_best, condition, loss=loss, score=score,
            )
            self.logger.log(block)
            print(block)
        # Short-circuit on the re-fire: a re-armed guard flag is never consumed here
        # (harmless — the next attempt's guard.start() reset clears it).
        if not self.adaptive_triggered and self.guard.consume_adaptive():
            self.adaptive_triggered = True
            # Ultralytics saves this epoch's unstripped last.pt BEFORE this callback
            # fires and only finalizes it (epoch = -1) in final_eval() after the loop
            # exits. Copying here preserves a resumable snapshot of the clean epoch
            # boundary so the supervisor can downgrade and continue the same run.
            save_resumable_snapshot(self.run_dir, trainer, self, self.logger)
            try:
                trainer.stop = True
            except Exception:
                pass
            self.logger.log("[Attempt] adaptive stop requested at epoch end (clean).")

    def on_train_end(self, trainer):
        # classify
        try:
            final_epoch = getattr(trainer, "epoch", None)
            # Same freeze/clamp rule as on_fit_epoch_end: an adaptive-triggered attempt
            # keeps its real stop epoch; a clean run is clamped to the target.
            if isinstance(final_epoch, int) and not self.adaptive_triggered:
                self.completed_epoch = min(final_epoch + 1, self.target_epochs)
        except Exception:
            pass
        if self.emergency_triggered:
            self.status = "partial-emergency"
        elif self.adaptive_triggered:
            self.status = "stopped-adaptive"
        else:
            # check if reached target
            if self.completed_epoch is not None and self.completed_epoch >= self.target_epochs:
                self.status = "completed-target"
            else:
                # could be early stopping or interrupted
                es = getattr(trainer, "stopper", None)
                # EarlyStopping: if stop due to patience, trainer.stop True but no adaptive
                self.status = "completed-attempt"
        self.logger.log(f"[Attempt] finished status={self.status} completed_epoch={self.completed_epoch}")
        # sync batch from trainer
        try:
            b = getattr(trainer.args, "batch", None)
            if b is not None:
                self.intended_cfg["batch"] = b
        except Exception:
            pass


# -- Adaptive state management (.adaptive/) --------------------------------------
def adaptive_dir(run_dir: Path) -> Path:
    return Path(run_dir) / ADAPTIVE_DIRNAME


def load_adaptive_state(run_dir: Path) -> dict:
    p = adaptive_dir(run_dir) / ADAPTIVE_STATE_FILE
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_adaptive_state(run_dir: Path, state: dict) -> None:
    d = adaptive_dir(run_dir)
    d.mkdir(parents=True, exist_ok=True)
    p = d / ADAPTIVE_STATE_FILE
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, p)


def snapshot_backups(run_dir: Path, logger=None) -> None:
    w = Path(run_dir) / "weights"
    d = adaptive_dir(run_dir)
    d.mkdir(parents=True, exist_ok=True)
    for name in ("last.pt", "best.pt"):
        src = w / name
        dst = d / ("backup_" + name)
        if not src.is_file():
            continue
        # skip re-copy if mtime+size match
        if dst.is_file():
            try:
                if dst.stat().st_mtime == src.stat().st_mtime and dst.stat().st_size == src.stat().st_size:
                    continue
            except Exception:
                pass
        try:
            shutil.copy2(src, dst)
            if logger is not None:
                logger.log(f"[Adaptive] snapshot backup: {name} -> {dst.name}")
        except Exception as e:
            if logger is not None:
                logger.log(f"[Adaptive] backup copy failed for {name}: {e}")


def restore_backups(run_dir: Path, logger=None) -> bool:
    w = Path(run_dir) / "weights"
    d = adaptive_dir(run_dir)
    restored = False
    for name in ("last.pt", "best.pt"):
        src = d / ("backup_" + name)
        dst = w / name
        if not src.is_file():
            continue
        try:
            shutil.copy2(src, dst)
            restored = True
            if logger is not None:
                logger.log(f"[Adaptive] restored backup: {name}")
        except Exception as e:
            if logger is not None:
                logger.log(f"[Adaptive] restore failed for {name}: {e}")
    return restored


def durable_progress_epoch(run_dir, state) -> int | None:
    """What --resume can actually continue from, based on the on-disk checkpoints.

    Prefers weights/last.pt when it is not finalized (epoch >= 0), then a valid pending
    hidden snapshot, then the recorded resumable_epoch. Returns None when the disk gives
    no evidence. Used to rewind evidence after a partial-emergency backup restore: the
    attempt's in-memory epoch count can describe epochs that the restore just rolled
    back, and recording it would let the pending-hidden staleness gate reject a hidden
    that actually matches the durable checkpoint.
    """
    run_dir = Path(run_dir)
    cur = run_dir / "weights" / "last.pt"
    if cur.is_file():
        epoch = ckpt_completed_epoch(cur)
        if epoch >= 0:
            return epoch
    hidden = pending_resumable_checkpoint(run_dir, state)
    if hidden is not None:
        return ckpt_completed_epoch(hidden)
    re_epoch = state.get("resumable_epoch")
    if isinstance(re_epoch, int) and re_epoch >= 0:
        return re_epoch
    return None


def save_resumable_snapshot(run_dir, trainer, tracker, logger) -> Path | None:
    """Preserve the unstripped Ultralytics checkpoint before final_eval() finalizes it.

    Ultralytics writes the full resumable last.pt inside the epoch loop before
    on_fit_epoch_end fires; final_eval()/strip_optimizer() then rewrites
    weights/last.pt with epoch = -1. A copy taken at the clean epoch boundary keeps a
    hidden snapshot (.adaptive/resumable_last.pt) the supervisor can resume from after
    a memory-guard adaptive stop. The snapshot is pending-only: state.json must carry
    resume_pending for it to be used (see pending_resumable_checkpoint).
    """
    run_dir = Path(run_dir)
    # Idempotency (defense-in-depth, independent of the callback gating): final_eval()
    # re-fires on_fit_epoch_end; a snapshot already taken in this attempt must never
    # be re-copied from the file after final_eval() strips it.
    if tracker.resumable_ckpt is not None:
        return tracker.resumable_ckpt
    tracker.resumable_ckpt = None
    src = Path(getattr(trainer, "last", None) or (run_dir / "weights" / "last.pt"))
    if not src.is_file():
        if logger is not None:
            logger.log(f"[Adaptive] resumable snapshot skipped: checkpoint {src} not found.")
        tracker.state["hidden_checkpoint_invalid"] = True
        write_adaptive_state(run_dir, tracker.state)
        return None
    if ckpt_completed_epoch(src) == -1:
        # Source was already finalized by a previous final_eval; a stripped checkpoint
        # cannot resume anything. The destination is NOT touched, so the good hidden
        # snapshot from the first firing survives; resume_pending stays as-is.
        if logger is not None:
            logger.log("[Adaptive] resumable snapshot skipped: source checkpoint already finalized (epoch -1).")
        return None
    dst = adaptive_dir(run_dir) / RESUMABLE_LAST_NAME
    try:
        adaptive_dir(run_dir).mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".tmp")
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
    except Exception as e:
        if logger is not None:
            logger.log(f"[Adaptive] resumable snapshot copy failed: {e}")
        tracker.state["hidden_checkpoint_invalid"] = True
        write_adaptive_state(run_dir, tracker.state)
        return None
    if ckpt_completed_epoch(dst) == -1:
        # tmp-integrity backup only: a write that lands finalized cannot resume anything.
        if logger is not None:
            logger.log("[Adaptive] resumable snapshot skipped: copied checkpoint is finalized (epoch -1).")
        try:
            dst.unlink(missing_ok=True)
        except OSError:
            pass
        tracker.state["hidden_checkpoint_invalid"] = True
        write_adaptive_state(run_dir, tracker.state)
        return None
    tracker.resumable_ckpt = dst
    tracker.state["resumable_ckpt"] = f"{ADAPTIVE_DIRNAME}/{RESUMABLE_LAST_NAME}"
    tracker.state["resumable_epoch"] = tracker.completed_epoch
    tracker.state["resume_pending"] = True
    write_adaptive_state(run_dir, tracker.state)
    if logger is not None:
        logger.log(f"[Adaptive] saved hidden resumable checkpoint from completed_epoch={tracker.completed_epoch} -> {ADAPTIVE_DIRNAME}/{RESUMABLE_LAST_NAME}")
    return dst


def trim_results_csv(run_dir: Path, completed_epoch: int, logger=None) -> None:
    """Keep header + first `completed_epoch` rows; back up original first."""
    csv_path = Path(run_dir) / "results.csv"
    if not csv_path.is_file():
        return
    try:
        lines = csv_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return
    if len(lines) < 2:
        return
    header = lines[0]
    data_rows = lines[1:]
    if len(data_rows) <= completed_epoch:
        return
    # backup
    try:
        backup = csv_path.with_suffix(".csv.bak")
        if not backup.exists():
            shutil.copy2(csv_path, backup)
    except Exception:
        pass
    new_lines = [header] + data_rows[:completed_epoch]
    try:
        csv_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        if logger is not None:
            logger.log(f"[Adaptive] trimmed results.csv from {len(data_rows)} to {completed_epoch} rows (crash-partial rows removed).")
    except Exception as e:
        if logger is not None:
            logger.log(f"[Adaptive] csv trim failed: {e}")


def ckpt_completed_epoch(ckpt_path: Path) -> int:
    """Return completed_epoch from a .pt checkpoint. -1 means finished per Ultralytics."""
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        epoch = ckpt.get("epoch", 0)
        if isinstance(epoch, (int, float)) and epoch >= 0:
            return int(epoch) + 1
        if isinstance(epoch, (int, float)) and epoch == -1:
            return -1
        return 0
    except Exception:
        return 0


# -- Resumable checkpoint selection (.adaptive/) ------------------------------------
TERMINAL_EARLY_STATUSES = ("stopped-adaptive", "partial-emergency", "crashed", "interrupted")


def valid_resume_checkpoint(path) -> bool:
    """A checkpoint is resumable only if it exists and is not finalized (epoch >= 0)."""
    return bool(path) and Path(path).is_file() and ckpt_completed_epoch(Path(path)) >= 0


def pending_resumable_checkpoint(run_dir, state) -> Path | None:
    """Return the hidden .adaptive/resumable_last.pt only while state marks it pending.

    Hidden snapshots are pending-only: they are valid continuation points only while
    state.json explicitly carries resume_pending (set by a clean adaptive stop that
    saved the snapshot) and the snapshot epoch has not regressed below resumable_epoch.
    Completed/terminal runs clear the pending flag, so old snapshots are never reused.
    """
    if not state or not state.get("resume_pending") or state.get("hidden_checkpoint_invalid"):
        return None
    raw = state.get("resumable_ckpt")
    if not raw or Path(str(raw)).name != RESUMABLE_LAST_NAME:
        return None
    path = adaptive_dir(run_dir) / RESUMABLE_LAST_NAME
    if not path.is_file():
        return None
    epoch = ckpt_completed_epoch(path)
    if epoch < 0:
        return None
    resumable_epoch = state.get("resumable_epoch")
    if isinstance(resumable_epoch, int) and resumable_epoch >= 0 and epoch < resumable_epoch:
        return None
    # Staleness cross-check: the snapshot must not predate the run's known progress.
    # The completed < target bound keeps an uncorrected inflated legacy value from
    # rejecting a legitimate hidden (the supervisor init correction runs first anyway).
    completed = state.get("completed_epoch")
    target = state.get("target_epochs")
    if isinstance(completed, int) and completed >= 0 \
            and (target is None or completed < target) and epoch < completed:
        return None
    return path


def adaptive_history_last_entry(state) -> tuple:
    """(completed_epoch, status) of the newest history entry that recorded an epoch."""
    for entry in reversed((state or {}).get("history") or []):
        ce = entry.get("completed_epoch")
        if isinstance(ce, int) and ce >= 0:
            return ce, entry.get("status")
    return None, None


def results_csv_row_count(run_dir) -> int | None:
    """Completed-epoch count from results.csv (header + one row per completed epoch)."""
    csv_path = Path(run_dir) / "results.csv"
    if not csv_path.is_file():
        return None
    try:
        lines = csv_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    rows = sum(1 for line in lines if line.strip()) - 1
    return rows if rows > 0 else None


def derive_actual_epoch(run_dir, state, tracker=None, ckpt_epoch=None, target_epochs=None) -> int:
    """Best-supported completed-epoch count from per-attempt evidence.

    Evidence wins over the legacy top-level state['completed_epoch']: old buggy runs
    stored target_epochs there even when an adaptive/patience stop finalized the
    checkpoint early. The terminal-early override fires only against target-level
    inflation (a legacy "completed_epoch = target" lie): genuine mid-run evidence
    (e.g. a hidden checkpoint at epoch 14 with a stale history entry at 11) wins,
    because 14 < target means nothing claims the schedule finished.
    """
    evidence = []
    tracker_epoch = getattr(tracker, "completed_epoch", None) if tracker is not None else None
    if isinstance(tracker_epoch, int) and tracker_epoch >= 0:
        evidence.append(tracker_epoch)
    if isinstance(ckpt_epoch, int) and ckpt_epoch >= 0:
        evidence.append(ckpt_epoch)
    hist_epoch, hist_status = adaptive_history_last_entry(state)
    if hist_epoch is not None:
        evidence.append(hist_epoch)
    csv_rows = results_csv_row_count(run_dir)
    if csv_rows is not None:
        evidence.append(csv_rows)
    if evidence:
        actual = max(evidence)
    else:
        legacy = (state or {}).get("completed_epoch")
        actual = legacy if isinstance(legacy, int) and legacy >= 0 else 0
    if hist_epoch is not None and hist_status in TERMINAL_EARLY_STATUSES \
            and actual > hist_epoch and (target_epochs is None or actual >= target_epochs):
        # The evidence claims the target was reached, but the newest terminal-early
        # history entry says the run actually stopped there. The history wins.
        actual = hist_epoch
    return max(0, int(actual))


def choose_resume_checkpoint(run_dir, weights_last, target_epochs=None) -> tuple:
    """Pick the checkpoint a resume should load. Returns (path_or_None, info).

    Candidate order:
      1. weights/last.pt when it is not finalized (epoch >= 0)
      2. .adaptive/resumable_last.pt only while resume_pending is explicitly set
      3. .adaptive/backup_last.pt only while resume_pending is set (legacy fallback)

    When nothing is resumable, info carries finalized/stopped_after evidence so the
    caller can report an accurate "stopped after N epochs" message instead of claiming
    the full schedule already finished.
    """
    run_dir = Path(run_dir)
    if weights_last is not None and valid_resume_checkpoint(weights_last):
        return Path(weights_last), {"source": "weights/last.pt", "finalized": False,
                                    "epoch": ckpt_completed_epoch(Path(weights_last))}
    state = load_adaptive_state(run_dir)
    hidden = pending_resumable_checkpoint(run_dir, state)
    if hidden is not None:
        return hidden, {"source": f"{ADAPTIVE_DIRNAME}/{RESUMABLE_LAST_NAME}", "finalized": False,
                        "epoch": ckpt_completed_epoch(hidden)}
    if state.get("resume_pending"):
        legacy = adaptive_dir(run_dir) / "backup_last.pt"
        if valid_resume_checkpoint(legacy):
            return legacy, {"source": f"{ADAPTIVE_DIRNAME}/backup_last.pt", "finalized": False,
                            "epoch": ckpt_completed_epoch(legacy)}
    weights_epoch = ckpt_completed_epoch(Path(weights_last)) if (weights_last and Path(weights_last).is_file()) else None
    finalized = weights_epoch == -1
    info = {"source": None, "finalized": finalized, "epoch": weights_epoch}
    if finalized:
        info["stopped_after"] = derive_actual_epoch(run_dir, state, tracker=None,
                                                    ckpt_epoch=None, target_epochs=target_epochs)
    return None, info


def report_run_outcome(run_dir, state, status, target, run_name, logger) -> None:
    """Print the honest post-run outcome block (shared by fresh and resume paths).

    Branch order: a pending hidden snapshot first (the run is resumable from it), then
    a finalized/invalid weights/last.pt (report the real stop, never target), and only
    then the plain 'resume again' case.
    """
    run_dir = Path(run_dir)
    print(f"\n  Training {status} after {state.get('completed_epoch', 0)}/{target} epochs. "
          f"Checkpoint state preserved in {run_dir / 'weights'}.")
    hidden = pending_resumable_checkpoint(run_dir, state)
    if state.get("resume_pending") and hidden is not None:
        print("  The run is resumable: --resume will continue from "
              f"{ADAPTIVE_DIRNAME}/{RESUMABLE_LAST_NAME} with the last downgraded safe config.")
    elif state.get("last_checkpoint_finalized") or not valid_resume_checkpoint(run_dir / "weights" / "last.pt"):
        print("  last.pt is finalized (epoch marker -1) and not resumable;")
        print(f"  start a new run from {run_dir / 'weights' / 'best.pt'} with lower workers/batch if continuation is desired.")
    else:
        print(f"  Resume again with: python train_detector/train_detector.py --resume --name {run_name}")
    print(f"  Run log: {logger.log_path}")
    print()


def _infer_resume_architecture_model(run_dir, ckpt_cfg=None, task: str = "detect"):
    """Infer architecture .pt name (e.g. yolo26l.pt) for a resume run; static fallback is task-aware."""
    ckpt_cfg = ckpt_cfg if isinstance(ckpt_cfg, dict) else {}
    model = ckpt_cfg.get("model")
    if isinstance(model, (str, Path)):
        base = Path(model).name.lower()
        if base not in ("last.pt", "best.pt") and "weights" not in str(model).lower():
            p = Path(model)
            if p.suffix == ".pt":
                return p.name
            if p.suffix == ".yaml":
                return p.stem + ".pt"
    try:
        ckpt = torch.load(str(Path(run_dir) / "weights" / "last.pt"), map_location="cpu", weights_only=False)
        ema = ckpt.get("ema")
        yf = None
        if ema is not None:
            y = getattr(ema, "yaml", None)
            if isinstance(y, dict):
                yf = y.get("yaml_file")
        if yf:
            return Path(yf).stem + ".pt"
    except Exception:
        pass
    return f"yolo26l{MODEL_TASK_SUFFIXES.get(task, '')}.pt"


# -- Legacy VRAM estimates (fallback heuristic only) --------------------------------

# VRAM usage estimates per image at 1024px (GB).
# FLOPs-scaled from prior YOLOv8 measurements using Ultralytics published
# detect FLOPs (m 68.2/78.9, l 86.4/165.2, x 193.9/257.8). seg/pose values are
# conservative placeholders (mask / pose head penalty over detect FLOPs).
# Local probe cache overrides these when present (see load_vram_estimates / probe_vram).
VRAM_PER_IMAGE = {
    "yolo26m.pt": 0.52,
    "yolo26l.pt": 0.58,
    "yolo26x.pt": 1.20,
    "yolo26m-seg.pt": 0.65,
    "yolo26l-seg.pt": 0.75,
    "yolo26x-seg.pt": 1.55,
    "yolo26m-pose.pt": 0.60,
    "yolo26l-pose.pt": 0.70,
    "yolo26x-pose.pt": 1.45,
}

ESTIMATES_PATH = WEIGHTS_DIR / "vram_estimates.json"


def load_vram_estimates():
    if not ESTIMATES_PATH.exists():
        return {}
    try:
        with open(ESTIMATES_PATH, encoding="utf-8") as f:
            data = json.load(f)
        out = {}
        for name, meta in (data.get("estimates") or {}).items():
            val = meta.get("gb_per_image_1024") if isinstance(meta, dict) else None
            if isinstance(val, (int, float)) and val > 0:
                out[str(name)] = float(val)
        return out
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as e:
        print(f"[WARNING] Ignoring corrupt VRAM cache {ESTIMATES_PATH}: {e}")
        return {}


def save_vram_estimates(estimates: dict, device_name: str) -> None:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    merged = load_vram_estimates()
    merged.update(estimates)
    try:
        import ultralytics
        ultra_ver = getattr(ultralytics, "__version__", "unknown")
    except Exception:
        ultra_ver = "unknown"
    payload = {
        "version": 1,
        "ultralytics": ultra_ver,
        "device_name": device_name,
        "updated_at": local_iso(),
        "estimates": {
            k: {"gb_per_image_1024": round(v, 4)} for k, v in sorted(merged.items())
        },
    }
    with open(ESTIMATES_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def get_vram_per_image(model: str) -> float:
    cached = load_vram_estimates().get(model)
    if cached is not None:
        return cached
    return float(VRAM_PER_IMAGE.get(model, 1.10))


def _write_probe_dataset(root: Path, imgsz: int = 640, task: str = "detect", kpt_count: int = 17) -> Path:
    """Synthetic single-image probe dataset; task-aware so -seg/-pose weights train against valid labels."""
    img_dir = root / "images"
    lbl_dir = root / "labels"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    img = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    cv2.imwrite(str(img_dir / "probe.jpg"), img)
    if task == "pose":
        k = max(1, int(kpt_count))
        keypoints = " ".join(f"{0.30 + 0.35 * ((i * 7) % 5) / 4:.2f} {0.35 + 0.30 * ((i * 5) % 4) / 3:.2f} 2"
                             for i in range(k))
        (lbl_dir / "probe.txt").write_text(f"0 0.5 0.5 0.4 0.4 {keypoints}\n", encoding="utf-8")
        yaml_text = (f"train: images\nval: images\nnc: 1\nnames: ['obj']\n"
                     f"kpt_shape: [{k}, 3]\nflip_idx: {list(range(k))}\n")
    else:
        # detect and segment both train fine on an empty label file (segmenter yields zero masks)
        (lbl_dir / "probe.txt").write_text("", encoding="utf-8")
        yaml_text = "train: images\nval: images\nnc: 1\nnames: ['obj']\n"
    yaml_path = root / "data.yaml"
    yaml_path.write_text(yaml_text, encoding="utf-8")
    return yaml_path


def probe_model_vram(model_name: str, device: str = "0") -> float:
    """Return measured GB/image @1024 from a tiny synthetic train step @640."""
    weights = resolve_pretrained_weights(model_name)
    model = YOLO(str(weights))
    mname = Path(model_name).name.lower()
    probe_task = "pose" if "-pose" in mname else ("segment" if "-seg" in mname else "detect")
    used_batch = 1
    with tempfile.TemporaryDirectory(prefix="yolo_vram_probe_") as tmp:
        tmp_path = Path(tmp)
        data_yaml = _write_probe_dataset(tmp_path, imgsz=640, task=probe_task)
        last_err = None
        for batch in (2, 1):
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                model.train(
                    data=str(data_yaml),
                    epochs=1,
                    imgsz=640,
                    batch=batch,
                    device=device,
                    project=str(tmp_path / "runs"),
                    name="probe",
                    exist_ok=True,
                    verbose=False,
                    plots=False,
                    save=False,
                    workers=0,
                    patience=0,
                )
                used_batch = batch
                last_err = None
                break
            except torch.cuda.OutOfMemoryError as e:
                last_err = e
                torch.cuda.empty_cache()
        if last_err is not None:
            raise last_err
        peak_bytes = float(torch.cuda.max_memory_allocated())
    gb_per_img_640 = (peak_bytes / (1024 ** 3)) / float(used_batch)
    return gb_per_img_640 * ((1024 / 640) ** 2)


def ensure_vram_estimates(models: list, device: str, force: bool = False) -> dict:
    if not cuda_available_for(device):
        print("[VRAM Probe] CUDA unavailable - using built-in FLOPs-scaled estimates")
        return {m: get_vram_per_image(m) for m in models}
    cached = load_vram_estimates()
    need = [m for m in models if force or m not in cached]
    if not need:
        return {m: get_vram_per_image(m) for m in models}
    print(f"[VRAM Probe] Measuring: {', '.join(need)}")
    measured = {}
    try:
        device_name = torch.cuda.get_device_name(0)
    except Exception:
        device_name = "cuda"
    for m in need:
        try:
            measured[m] = probe_model_vram(m, device=device)
            print(f"  {m}: {measured[m]:.3f} GB/img @1024")
        except Exception as e:
            print(f"[WARNING] Probe failed for {m}: {e} - using built-in fallback")
    if measured:
        save_vram_estimates(measured, device_name)
    return {m: get_vram_per_image(m) for m in models}


# -- Static Estimate Helpers (starting heuristics / CPU and --no-memory-probe caps) -

def snap_to_standard(size: int) -> int:
    """Snap a raw pixel size to the nearest standard YOLO imgsz."""
    return min(STANDARD_IMGSZ, key=lambda s: abs(s - size))


def native_capped_imgsz(native_size):
    """Largest standard imgsz that does NOT upscale beyond native dataset resolution."""
    if not native_size:
        return None
    cap = snap_to_standard(int(native_size))
    if cap > native_size:
        lower = [s for s in STANDARD_IMGSZ if s <= native_size]
        cap = max(lower) if lower else 320
    return cap


def imgsz_candidates(native_size):
    """Auto imgsz ladder: native-capped value first, then lower standards."""
    cap = native_capped_imgsz(native_size) or 640
    out = [cap] + [s for s in reversed(STANDARD_IMGSZ) if s < cap]
    return out[:4]


def calc_batch(vram_gb, model: str, imgsz: int, multi_scale: float = 0.0, vram_frac: float = 0.68) -> int:
    """
    Conservative static estimate of the largest ladder batch that fits.

    Sizing against the PEAK multi_scale frame (imgsz * (1 + multi_scale)) so
    peak-size augmentation bursts do not OOM. Returns 0 when even batch 2 is
    not estimated safe. Static estimates are starting heuristics only; dynamic
    probe measurements are authoritative for safety.
    """
    if not vram_gb:
        return 0
    effective_imgsz = imgsz * (1 + multi_scale) if multi_scale > 0 else imgsz
    scale = (effective_imgsz / 1024) ** 2
    vram_per_img = get_vram_per_image(model) * scale
    batch = int((vram_gb * vram_frac) / vram_per_img)
    for b in reversed(BATCH_LADDER):
        if batch >= b:
            return b
    return 0


def calc_workers(cpu_cores: int, ram_gb: float) -> int:
    """Calculate optimal dataloader workers (capped at 8 for Windows stability)."""
    max_by_cpu = max(1, cpu_cores - 2)
    max_by_ram = int(ram_gb / 2) if ram_gb else 4
    workers = min(max_by_cpu, max_by_ram, 8)
    return max(1, workers)


def auto_workers_limit():
    """Auto-config worker ceiling without CPU RSS proof (plan: 4 or lower)."""
    return 4


def model_candidates(image_count: int, vram_gb, for_cpu: bool, task: str = "detect") -> list:
    """
    Quality-first model chain for the dataset task: the largest task model is
    the first attempt for every dataset size; smaller models are appended only
    as probe-failure fallbacks. Fit is decided by dynamic probes, this order
    is fallback only. CPU runs cap the chain at 'l' (practicality cap).
    """
    order = model_order_for_task(task)
    if for_cpu:
        chain = ["l", "m", "n"]
    else:
        chain = ["x", "l", "m", "n"]
    suffix = MODEL_TASK_SUFFIXES.get(task, "")
    return [f"yolo26{size}{suffix}.pt" for size in chain]


# -- Dynamic Memory Calibration ----------------------------------------------------

class CalibrateFail(Exception):
    """No safe configuration exists (or an explicit override is unsafe without --force-unsafe)."""


class ProbeOOMError(Exception):
    """Probe run failed with an out-of-memory style error."""


class ProbeUnverifiedError(Exception):
    """Probe run failed for a non-OOM reason; candidate safety is unverified."""


# looks_like_oom is defined in common/hardware.py and re-exported above.


def probe_fraction(image_count: int) -> float:
    """Representative probe subset size (16-128 train images); all images for small datasets."""
    if image_count <= 128:
        return 1.0
    return 64.0 / float(image_count)


@contextlib.contextmanager
def silence_ultralytics_training_banners():
    """
    Hide Ultralytics INFO lines like "Starting training for 1 epochs..." during
    hardware memory probes so they are not mistaken for the real training run.
    """
    try:
        from ultralytics.utils import LOGGER as ul_logger  # type: ignore
    except Exception:
        yield
        return
    prev = ul_logger.level
    ul_logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        try:
            ul_logger.setLevel(prev)
        except Exception:
            pass


def probe_train_once(model_name, device, data_yaml, imgsz, batch, multi_scale, mosaic, mixup,
                     copy_paste, fraction, cuda_index, workers=0, degrees=0.0, flipud=0.0,
                     cache=False, sample_gpu_util=False):
    """
    One 1-epoch real-dataset probe run on a representative subset (never a
    synthetic blank image probe). Returns (peak_gpu_allocated_gb,
    peak_gpu_reserved_gb, peak_cpu_rss_gb[, avg_gpu_util]).

    A SAFE peak comparison is a memory estimate only — not a convergence guarantee.
    """
    weights = resolve_pretrained_weights(model_name)
    model = YOLO(str(weights))

    peak_rss = [0.0]
    util_samples = []
    stop_event = threading.Event()

    def rss_loop():
        while not stop_event.wait(2.0):
            rss = process_rss_bytes()
            if rss is not None:
                peak_rss[0] = max(peak_rss[0], rss / 2**30)
            if sample_gpu_util and cuda_index is not None:
                util = sample_gpu_utilization(cuda_index)
                if util is not None:
                    util_samples.append(util)

    rss_thread = threading.Thread(target=rss_loop, daemon=True) if PSUTIL_AVAILABLE else None
    if rss_thread:
        rss_thread.start()

    try:
        if cuda_index is not None:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(cuda_index)
        # Plain-English status: this is a hardware/config fit test, not training.
        print(
            f"[Hardware test] Checking fit: {Path(str(model_name)).name} @ {imgsz}px "
            f"batch {batch} workers {workers} (not the real training run — probe weights discarded)"
        )
        with tempfile.TemporaryDirectory(prefix="yolo_mem_probe_") as tmp:
            with silence_ultralytics_training_banners():
                model.train(
                    data=str(data_yaml),
                    epochs=1,
                    fraction=fraction,
                    imgsz=int(imgsz),
                    batch=int(batch),
                    multi_scale=float(multi_scale),
                    mosaic=float(mosaic),
                    mixup=float(mixup),
                    copy_paste=float(copy_paste),
                    degrees=float(degrees),
                    flipud=float(flipud),
                    close_mosaic=0,   # keep mosaic ACTIVE during 1-epoch probes
                    workers=int(workers),
                    val=False,
                    save=False,
                    plots=False,
                    verbose=False,
                    cache=cache,
                    device=device,
                    project=str(Path(tmp) / "runs"),
                    name="memory_probe",
                    exist_ok=True,
                    save_dir=str(Path(tmp) / "runs" / "memory_probe"),
                )
    except Exception as e:
        if cuda_index is not None:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        raise (ProbeOOMError(str(e)) if looks_like_oom(e) else ProbeUnverifiedError(str(e))) from e
    finally:
        stop_event.set()
        if rss_thread is not None:
                rss_thread.join(timeout=10)

    alloc_gb = None
    reserved_gb = None
    if cuda_index is not None:
        alloc_gb = torch.cuda.max_memory_allocated(cuda_index) / 2**30
        reserved_gb = torch.cuda.max_memory_reserved(cuda_index) / 2**30
        torch.cuda.empty_cache()

    if sample_gpu_util:
        avg_util = (sum(util_samples) / len(util_samples)) if util_samples else None
        return alloc_gb, reserved_gb, peak_rss[0], avg_util
    return alloc_gb, reserved_gb, peak_rss[0]


def sample_gpu_utilization(cuda_index) -> float | None:
    """Optional GPU util % via pynvml; None when unavailable."""
    try:
        import pynvml  # type: ignore[import-untyped]
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(int(cuda_index))
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return float(util.gpu)
    except Exception:
        return None


def static_candidate_config(args, hw, image_count, native_size, use_gpu):
    """
    Static starting config (dry-run display, CPU paths, and --no-memory-probe
    fallback). Static estimates are heuristics; only probe measurements prove
    safety.
    """
    profile = MEMORY_PROFILES[args.memory_profile]
    notes = []
    model_list = [args.model] if args.model else model_candidates(
        image_count, hw.get("vram_gb"), for_cpu=not use_gpu, task=getattr(args, "dataset_task", "detect"))
    if args.imgsz is not None:
        imgsz_list = [int(args.imgsz)]
        if native_size and int(args.imgsz) > int(native_size):
            notes.append(f"explicit --imgsz {int(args.imgsz)} exceeds native dataset resolution {int(native_size)}px (override honored)")
    else:
        imgsz_list = imgsz_candidates(native_size)
    workers = int(args.workers) if args.workers is not None else min(
        calc_workers(hw.get("cpu_cores") or 4, hw.get("ram_gb")), auto_workers_limit())

    for model in model_list:
        for imgsz in imgsz_list:
            if use_gpu:
                est0 = calc_batch(hw["vram_gb"], model, imgsz, 0.0, profile["gpu"])
                est_ms = calc_batch(hw["vram_gb"], model, imgsz, 0.5, profile["gpu"])
                explicit_unsafe = False
                if args.batch is not None:
                    batch = int(args.batch)
                    multi_scale = 0.5 if est_ms >= batch else 0.0
                    if est0 < batch:
                        explicit_unsafe = True
                        notes.append(f"explicit batch {batch} exceeds conservative static ceiling {est0} for {model}@{imgsz}px")
                else:
                    multi_scale = 0.5 if est_ms >= MIN_HEALTHY_BATCH else 0.0
                    batch = min(calc_batch(hw["vram_gb"], model, imgsz, multi_scale, profile["gpu"]),
                                HEALTHY_BATCH_CANDIDATES[0])
                    # Prefer a healthy batch when the static estimate allows it.
                    if batch >= MIN_HEALTHY_BATCH:
                        batch = max(b for b in HEALTHY_BATCH_CANDIDATES if b <= batch)
                    elif batch < MIN_AUTO_BATCH:
                        continue
                    else:
                        notes.append(
                            f"static estimate only fits batch {batch} (< {MIN_HEALTHY_BATCH}); "
                            "BN quality may be degraded — dynamic probes will try harder"
                        )
            else:
                multi_scale = 0.0
                explicit_unsafe = False
                batch = int(args.batch) if args.batch is not None else 4
                if args.batch is not None and args.batch > 16:
                    notes.append(f"CPU batch {args.batch} is unverified and likely slow")
            return {
                "model": model, "imgsz": imgsz, "batch": batch, "workers": workers,
                "multi_scale": multi_scale, "explicit_unsafe": explicit_unsafe, "notes": notes,
            }

    if args.model or args.imgsz is not None or args.batch is not None:
        return {
            "model": model_list[0], "imgsz": imgsz_list[0], "batch": int(args.batch or 1),
            "workers": workers, "multi_scale": 0.0, "explicit_unsafe": True,
            "notes": notes + ["explicit config does not fit within conservative static caps"],
        }
    return None


def _short(s: str) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= 300 else s[:300] + " ..."


def build_chosen(model, imgsz, batch, workers, multi_scale, mosaic, mixup=0.0, copy_paste=0.0,
                 degrees=0.0, flipud=0.0, cache=False):
    """
    Canonical chosen-config dict used for cache keys and training kwargs.
    Geometry (degrees/flipud) defaults to 0 (opt-in). close_mosaic/cos_lr/nbs are
    shared training constants so probes, training and the cache see the same truth.
    """
    return {
        "model": model, "imgsz": int(imgsz), "batch": int(batch), "workers": int(workers),
        "multi_scale": float(multi_scale), "mosaic": float(mosaic),
        "mixup": float(mixup), "copy_paste": float(copy_paste),
        "degrees": float(degrees), "flipud": float(flipud),
        "close_mosaic": 60, "cos_lr": True, "nbs": 64,
        "cache": cache,
    }


def calibrate_training_config(args, logger, cache, hw_fp, hw, device, data_yaml, train_path, ds_fp):
    """
    Select the training config via cached measured-safe selection, static-only
    mode, or dynamic 1-epoch real-dataset probe calibration.

    Returns (chosen_config, provenance) where provenance is one of
    'cache', 'static', 'measured', 'force-unsafe'. Raises CalibrateFail.
    """
    use_gpu = cuda_available_for(device)
    profile_name = args.memory_profile
    profile = MEMORY_PROFILES[profile_name]
    image_count = ds_fp.get("image_count") or count_images(train_path)
    native_size = ds_fp.get("median_img_size")
    cuda_index = cuda_index_for(device) if use_gpu else None
    fraction = probe_fraction(image_count)
    sel_key = selection_signature(args, ds_fp)

    cached = lookup_cached_selection(cache, hw_fp, ds_fp, sel_key, logger)
    if cached:
        sel, entry = cached
        chosen = dict(sel["config"])
        logger.log("[Calibrate] Cached MEASURED-SAFE selection matches this hardware+device+dataset+profile; probes skipped.")
        logger.log(f"[Calibrate] Cached config: {sel.get('config_key')}")
        logger.log(
            "[Calibrate] Recorded peaks: GPU alloc "
            f"{entry.get('peak_gpu_allocated_gb')} GB | GPU reserved {entry.get('peak_gpu_reserved_gb')} GB | "
            f"CPU RSS {entry.get('peak_cpu_rss_gb')} GB (measured {entry.get('updated_at')})"
        )
        print("[Calibration] Reusing measured-safe config from weights/memory_profiles.json (delete the file or pass --no-memory-probe to recalibrate).")
        return chosen, "cache"

    if args.no_memory_probe:
        cfg = static_candidate_config(args, hw, image_count, native_size, use_gpu)
        if cfg is None:
            raise CalibrateFail(
                "No config fits the conservative static caps and probes are disabled (--no-memory-probe). "
                "Drop --no-memory-probe for dynamic probe calibration."
            )
        if cfg["explicit_unsafe"] and not args.force_unsafe:
            raise CalibrateFail(
                "Static estimate marks the explicit override unsafe: " + "; ".join(cfg["notes"]) +
                ". Let dynamic probes calibrate instead, or pass --force-unsafe."
            )
        degrees, flipud = resolve_geometry_aug(args)
        task = getattr(args, "dataset_task", ds_fp.get("task", "detect"))
        ladder0 = aug_ladder_for_task(task)[0]
        mix0 = resolve_mixup(args, task, ladder0[2])
        chosen = build_chosen(
            cfg["model"], cfg["imgsz"], cfg["batch"], cfg["workers"], cfg["multi_scale"], 1.0,
            mixup=mix0, copy_paste=ladder0[1], degrees=degrees, flipud=flipud,
        )
        for note in cfg["notes"]:
            logger.log(f"[Calibrate static] {note}")
        logger.log("[Calibrate] Probes skipped (--no-memory-probe): static conservative config; NOT measured-safe.")
        print(f"[Calibration] Static-only config (--no-memory-probe): {chosen['model']} @ {chosen['imgsz']}px "
              f"batch {chosen['batch']} workers {chosen['workers']}")
        record_calibration(cache, logger, hw_fp, ds_fp, sel_key, chosen, (None, None, None),
                           safe=not cfg["explicit_unsafe"], measured=False, make_selection=not cfg["explicit_unsafe"])
        return chosen, "static"

    # ---- dynamic probe calibration (quality-v2: batch-floor-first) ------------
    gpu_cap = hw["vram_gb"] * profile["gpu"] if (use_gpu and hw.get("vram_gb")) else None
    cpu_cap = hw["ram_gb"] * profile["cpu"] if hw.get("ram_gb") else None
    cap_str = (f"GPU cap {gpu_cap:.2f} GB" if gpu_cap else "GPU cap n/a") + \
              (f", CPU cap {cpu_cap:.2f} GB" if cpu_cap else ", CPU cap n/a")
    logger.log(f"[Calibrate] Dynamic probes on device '{device}', profile '{profile_name}' ({cap_str}); "
               f"probe fraction {fraction:.3f} of {image_count} train images, 1 epoch, save off.")
    logger.log(
        f"[Calibrate] Selection policy quality-v2: largest model that fits batch>={MIN_HEALTHY_BATCH}, "
        "task-aware aug; lighten aug before dropping below the healthy batch floor. "
        "SAFE = memory estimate only, not a convergence guarantee."
    )
    print(
        "[Hardware test] Finding a safe training config for this machine "
        "(short memory probes on a small data subset — not the real training run; probe weights are discarded)..."
    )

    task = getattr(args, "dataset_task", ds_fp.get("task", "detect"))
    ladder = aug_ladder_for_task(task)
    degrees, flipud = resolve_geometry_aug(args)
    models = [args.model] if args.model else model_candidates(
        image_count, hw.get("vram_gb"), for_cpu=not use_gpu, task=task)
    if args.imgsz is not None:
        imgszs = [int(args.imgsz)]
        if native_size and int(args.imgsz) > int(native_size):
            logger.log(f"[Calibrate] Note: explicit --imgsz {int(args.imgsz)} exceeds native dataset resolution {int(native_size)}px (override honored).")
    else:
        imgszs = imgsz_candidates(native_size)
        label_stats = ds_fp.get("label_stats") or {}
        imgszs = prefer_imgsz_for_objects(imgszs, label_stats, native_size)

    base_workers = int(args.workers) if args.workers is not None else min(
        calc_workers(hw.get("cpu_cores") or 4, hw.get("ram_gb")), auto_workers_limit())

    probe_memo = {}
    last_probe_elapsed = [None]  # mutable cell: sec for final verification probe

    def evaluate_peaks(peaks, label):
        alloc, res, rss = peaks[0], peaks[1], peaks[2]
        safe = True
        reasons = []
        if gpu_cap is not None and res is not None and res > gpu_cap:
            safe = False
            reasons.append(f"GPU peak reserved {res:.2f} GB > {profile_name} cap {gpu_cap:.2f} GB")
        if cpu_cap is not None and rss is not None and rss > cpu_cap:
            safe = False
            reasons.append(f"CPU peak RSS {rss:.2f} GB > {profile_name} cap {cpu_cap:.2f} GB")
        peak_str = (f"GPU alloc {alloc:.2f}/reserved {res:.2f} GB, " if res is not None else "") + \
                   (f"CPU RSS {rss:.2f} GB" if rss else "CPU RSS n/a")
        logger.log(f"[Calibrate] {label}: {'SAFE' if safe else 'UNSAFE'} ({'; '.join(reasons) if reasons else peak_str})")
        return safe

    def run_probe(chosen_cfg, label, sample_gpu_util=False, timed=False):
        key = config_signature(**chosen_cfg)
        if key in probe_memo and not sample_gpu_util and not timed:
            safe = probe_memo[key][1]
            logger.log(f"[Calibrate] {label}: reusing earlier probe result for {key} ({'SAFE' if safe else 'UNSAFE'})")
            return probe_memo[key]
        t0 = time.perf_counter()
        try:
            peaks = probe_train_once(
                chosen_cfg["model"], device, data_yaml, chosen_cfg["imgsz"], chosen_cfg["batch"],
                chosen_cfg["multi_scale"], chosen_cfg["mosaic"], chosen_cfg["mixup"], chosen_cfg["copy_paste"],
                fraction, cuda_index, workers=chosen_cfg["workers"],
                degrees=chosen_cfg.get("degrees", 0.0), flipud=chosen_cfg.get("flipud", 0.0),
                cache=chosen_cfg.get("cache", False), sample_gpu_util=sample_gpu_util,
            )
        except ProbeOOMError as e:
            logger.log(f"[Calibrate] {label}: OOM ({_short(str(e))})")
            probe_memo[key] = ((None, None, None), False, None)
            return probe_memo[key]
        except ProbeUnverifiedError as e:
            raise CalibrateFail(
                f"Probe for {chosen_cfg['model']} @ {chosen_cfg['imgsz']}px batch {chosen_cfg['batch']} failed for a "
                f"non-memory reason, so its safety cannot be verified: {_short(str(e))}"
            ) from e
        elapsed = time.perf_counter() - t0
        if timed:
            last_probe_elapsed[0] = elapsed
        gpu_util = peaks[3] if sample_gpu_util and len(peaks) > 3 else None
        mem_peaks = peaks[:3]
        safe = evaluate_peaks(mem_peaks, label)
        record_calibration(cache, logger, hw_fp, ds_fp, sel_key, chosen_cfg, mem_peaks, safe, measured=True, make_selection=False)
        probe_memo[key] = (mem_peaks, safe, gpu_util)
        return probe_memo[key]

    def make_cfg(model, imgsz, batch, workers, mosaic, copy_paste, mixup, multi_scale, cache=False):
        mix = resolve_mixup(args, task, mixup)
        return build_chosen(
            model, imgsz, batch, workers, multi_scale, mosaic,
            mixup=mix, copy_paste=copy_paste, degrees=degrees, flipud=flipud, cache=cache,
        )

    def finalize_with_workers(model, imgsz, aug_cfg, degraded=False):
        """Final verification with real workers; optional throughput bumps."""
        nonlocal_cfg = None
        final_safe = False
        gpu_util = None
        batch = aug_cfg["batch"]
        if args.workers is not None:
            cfg = make_cfg(model, imgsz, batch, base_workers,
                           aug_cfg["mosaic"], aug_cfg["copy_paste"], aug_cfg["mixup"], aug_cfg["multi_scale"])
            peaks, safe, gpu_util = run_probe(cfg, f"final verify: workers {base_workers} (explicit)",
                                              sample_gpu_util=True, timed=True)
            if safe:
                nonlocal_cfg, final_safe = cfg, True
            elif args.force_unsafe:
                logger.log("[Calibrate] --force-unsafe: accepting explicit workers despite UNSAFE final verification probe (this config is NOT recorded as measured-safe).")
                nonlocal_cfg = cfg
            else:
                raise CalibrateFail(
                    f"Explicit --workers {base_workers} is unsafe on the final verification probe. "
                    "Lower --workers or pass --force-unsafe."
                )
        else:
            for w in sorted({base_workers, 2, 0}, reverse=True):
                if w > base_workers:
                    continue
                cfg = make_cfg(model, imgsz, batch, w,
                               aug_cfg["mosaic"], aug_cfg["copy_paste"], aug_cfg["mixup"], aug_cfg["multi_scale"])
                peaks, safe, gpu_util = run_probe(cfg, f"final verify: workers {w}",
                                                  sample_gpu_util=True, timed=True)
                if safe:
                    nonlocal_cfg, final_safe = cfg, True
                    break
        if nonlocal_cfg is None:
            return None, False

        # Throughput: if GPU util is low, try more workers / disk cache (probe-gated).
        if final_safe and args.workers is None and gpu_util is not None and gpu_util < GPU_UTIL_DATALOADER_BOUND:
            logger.log(f"[Calibrate] GPU util ~{gpu_util:.0f}% < {GPU_UTIL_DATALOADER_BOUND:.0f}% — treating as dataloader-bound.")
            more_workers = min(auto_workers_limit(), max(nonlocal_cfg["workers"] * 2, nonlocal_cfg["workers"] + 2, base_workers))
            if more_workers > nonlocal_cfg["workers"]:
                cand = dict(nonlocal_cfg)
                cand["workers"] = more_workers
                _, w_safe, _ = run_probe(cand, f"throughput: workers {more_workers}")
                if w_safe:
                    logger.log(f"[Calibrate] throughput: raising workers {nonlocal_cfg['workers']} -> {more_workers}")
                    nonlocal_cfg = cand
            # disk cache (ram only if dataset estimate fits comfortably)
            if not nonlocal_cfg.get("cache"):
                disk_cand = dict(nonlocal_cfg)
                disk_cand["cache"] = "disk"
                _, c_safe, _ = run_probe(disk_cand, "throughput: cache=disk")
                if c_safe:
                    logger.log("[Calibrate] throughput: enabling cache=disk (probe-safe)")
                    nonlocal_cfg = disk_cand
                else:
                    # estimate train images bytes roughly; only try ram if << available RAM
                    try:
                        avail = (hw.get("ram_gb") or 0) * 0.35
                        est_gb = (image_count * (nonlocal_cfg["imgsz"] ** 2) * 3) / (2 ** 30)
                        if est_gb > 0 and est_gb < avail:
                            ram_cand = dict(nonlocal_cfg)
                            ram_cand["cache"] = "ram"
                            _, r_safe, _ = run_probe(ram_cand, "throughput: cache=ram")
                            if r_safe:
                                logger.log(f"[Calibrate] throughput: enabling cache=ram (est {est_gb:.1f} GB < {avail:.1f} GB budget)")
                                nonlocal_cfg = ram_cand
                    except Exception:
                        pass

        if degraded and nonlocal_cfg["batch"] < MIN_HEALTHY_BATCH:
            warn = (
                f"[WARNING] No config fit batch>={MIN_HEALTHY_BATCH}; using batch={nonlocal_cfg['batch']}. "
                "BatchNorm running stats will be noisier — expect lower quality than a healthy-batch run."
            )
            logger.log(warn)
            print(warn)

        # Stash probe timing for ETA (scale probe fraction → full epoch estimate)
        if last_probe_elapsed[0] is not None and fraction > 0:
            nonlocal_cfg["_probe_sec"] = last_probe_elapsed[0]
            nonlocal_cfg["_probe_fraction"] = fraction
            nonlocal_cfg["_est_sec_per_epoch"] = last_probe_elapsed[0] / max(fraction, 1e-6)

        warn_object_size_vs_imgsz(logger, ds_fp.get("label_stats") or {}, nonlocal_cfg["imgsz"], native_size)
        return nonlocal_cfg, final_safe

    tried = []
    for model in models:
        for imgsz in imgszs:
            tried.append(f"{model}@{imgsz}px")
            if args.batch is not None:
                healthy = [int(args.batch)]
                degraded = []
            else:
                healthy = list(HEALTHY_BATCH_CANDIDATES)
                degraded = list(DEGRADED_BATCH_CANDIDATES)

            # Phase A: batch >= floor — lighten aug before leaving this phase.
            chosen_aug = None
            for b in healthy:
                for rung, (mosaic, copy_paste, mixup, multi_scale) in enumerate(ladder):
                    cfg = make_cfg(model, imgsz, b, 0, mosaic, copy_paste, mixup, multi_scale)
                    _, safe, _ = run_probe(
                        cfg,
                        f"healthy batch {b} aug rung {rung} (mosaic {mosaic}, copy_paste {copy_paste}, "
                        f"mixup {mixup}, multi_scale {multi_scale})",
                    )
                    if safe:
                        chosen_aug = cfg
                        break
                if chosen_aug is not None:
                    break

            degraded_pick = False
            if chosen_aug is None and degraded:
                logger.log(
                    f"[Calibrate] {model}@{imgsz}px: no probe-safe config at batch>={MIN_HEALTHY_BATCH}; "
                    f"trying degraded batches {degraded}."
                )
                for b in degraded:
                    for rung, (mosaic, copy_paste, mixup, multi_scale) in enumerate(ladder):
                        cfg = make_cfg(model, imgsz, b, 0, mosaic, copy_paste, mixup, multi_scale)
                        _, safe, _ = run_probe(
                            cfg,
                            f"degraded batch {b} aug rung {rung} (mosaic {mosaic}, copy_paste {copy_paste}, "
                            f"mixup {mixup}, multi_scale {multi_scale})",
                        )
                        if safe:
                            chosen_aug = cfg
                            degraded_pick = True
                            break
                    if chosen_aug is not None:
                        break

            if chosen_aug is None:
                if args.batch is not None and args.force_unsafe:
                    mosaic, copy_paste, mixup, multi_scale = ladder[-1]
                    chosen_aug = make_cfg(model, imgsz, int(args.batch), 0, mosaic, copy_paste, mixup, multi_scale)
                    logger.log(f"[Calibrate] --force-unsafe: accepting explicit batch {args.batch} at {model}@{imgsz}px.")
                else:
                    logger.log(f"[Calibrate] {model}@{imgsz}px: no probe-safe batch/aug; trying next candidate.")
                    continue

            chosen_cfg, final_probe_safe = finalize_with_workers(
                model, imgsz, chosen_aug, degraded=degraded_pick or chosen_aug["batch"] < MIN_HEALTHY_BATCH
            )
            if chosen_cfg is None:
                logger.log(f"[Calibrate] {model}@{imgsz}px: dataloader workers unsafe at every level; trying next candidate.")
                continue

            key = config_signature(**{k: v for k, v in chosen_cfg.items() if not str(k).startswith("_")})
            peaks = probe_memo.get(key, ((None, None, None), final_probe_safe, None))[0]
            record_calibration(
                cache, logger, hw_fp, ds_fp, sel_key,
                {k: v for k, v in chosen_cfg.items() if not str(k).startswith("_")},
                peaks, safe=final_probe_safe, measured=True, make_selection=final_probe_safe,
            )
            if not final_probe_safe:
                logger.log(f"[Calibrate] SELECTED --force-unsafe config (final verification probe UNSAFE): {key}")
                print(f"[Calibration] Force-unsafe config: {chosen_cfg['model']} @ {chosen_cfg['imgsz']}px batch {chosen_cfg['batch']} workers {chosen_cfg['workers']} multi_scale {chosen_cfg['multi_scale']} mosaic {chosen_cfg['mosaic']}")
                return chosen_cfg, "force-unsafe"
            logger.log(f"[Calibrate] SELECTED measured-safe config: {key}")
            print(f"[Calibration] Safe config: {chosen_cfg['model']} @ {chosen_cfg['imgsz']}px batch {chosen_cfg['batch']} workers {chosen_cfg['workers']} multi_scale {chosen_cfg['multi_scale']} mosaic {chosen_cfg['mosaic']}")
            return chosen_cfg, "measured"

    if args.force_unsafe:
        model0 = args.model or models[0]
        imgsz0 = int(args.imgsz) if args.imgsz is not None else imgszs[0]
        batch0 = int(args.batch) if args.batch is not None else MIN_AUTO_BATCH
        workers0 = int(args.workers) if args.workers is not None else 0
        mosaic, copy_paste, mixup, multi_scale = ladder[-1]
        chosen = make_cfg(model0, imgsz0, batch0, workers0, mosaic, copy_paste, mixup, 0.0)
        logger.log("[Calibrate] --force-unsafe: NO probe-safe config found; proceeding with an explicitly UNSAFE config anyway.")
        record_calibration(cache, logger, hw_fp, ds_fp, sel_key, chosen, (None, None, None), safe=False, measured=False, make_selection=True)
        return chosen, "force-unsafe"

    raise CalibrateFail(
        "No probe-safe configuration found. Tried: " + ", ".join(tried) + ".\n"
        "Suggestions: try --memory-profile balanced or aggressive first, then lower --imgsz "
        "(e.g. 416 or 320) or a smaller model (--model yolo26m.pt), more free RAM, "
        "or --force-unsafe to train anyway (not recommended)."
    )


# -- Argument Parsing ----------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train a YOLO26 object detector")

    parser.add_argument(
        "--input", type=str, default=None,
        help="Dataset DIRECTORY to auto-detect (YOLO with/without data.yaml, COCO JSON, VOC XML). Required for fresh training.",
    )
    parser.add_argument(
        "--task", choices=["auto", "detect", "segment", "pose"], default="auto",
        help="Override the auto-derived annotation task (default: auto).",
    )
    parser.add_argument(
        "--converted-dir", type=str, default=None,
        help="Override output location for COCO/VOC -> YOLO conversion (default: <dataset>_yolo sibling of the source).",
    )
    parser.add_argument(
        "--yes", action="store_true",
        help="Accept auto-detection writes into the dataset folder (synthesized data.yaml, added kpt_shape, label-structure repair) without prompting.",
    )
    parser.add_argument(
        "--fix-label-structure", action="store_true",
        help="Repair a segment dataset that mixes a few 5-field bbox-only label lines into polygons in place (backup + manifest under the dataset folder); refused unless segment dominates every split at >=95% of label lines. Applies the repair without a [y/N] prompt.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from last checkpoint. Use --name to specify which run.",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Force a new auto-suffixed run even if the requested run name already exists.",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Override auto-selected model (e.g. yolo26x.pt)",
    )
    parser.add_argument(
        "--epochs", type=int, default=600,
        help="Number of training epochs (default: 600)",
    )
    parser.add_argument(
        "--imgsz", type=int, default=None,
        help="Override auto-calculated image size in pixels",
    )
    parser.add_argument(
        "--batch", type=int, default=None,
        help="Override auto-calculated batch size (safety-probed unless --force-unsafe)",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Override auto-calculated worker count (safety-probed unless --force-unsafe)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to train on: 0 for GPU, cpu for CPU (default: 0; CPU fallback when CUDA unavailable)",
    )
    parser.add_argument(
        "--name", type=str, default=None,
        help="Run name to resume, or name for a new run (default: detector_v1)",
    )
    parser.add_argument(
        "--patience", type=int, default=100,
        help="Early stopping patience in epochs (default: 100)",
    )
    parser.add_argument(
        "--degrees", type=float, default=0.0,
        help="Random rotation degrees (+/-). Default 0 (opt-in; was previously hardcoded 180).",
    )
    parser.add_argument(
        "--flipud", type=float, default=0.0,
        help="Vertical flip probability. Default 0.0 (opt-in; was previously hardcoded 0.5).",
    )
    parser.add_argument(
        "--mixup", type=float, default=None,
        help="Mixup probability. Default: 0.0 for detect/pose; segment ladder may use 0.15 at the top rung.",
    )
    parser.add_argument(
        "--time-budget", type=float, default=None,
        help="Max training wall-clock hours (passed as Ultralytics time=). Early stop / patience may end sooner.",
    )
    parser.add_argument(
        "--skip-pilot", action="store_true",
        help="Skip the short pre-flight learnability pilot and start the long run after calibration.",
    )
    parser.add_argument(
        "--memory-profile", choices=["safe", "balanced", "aggressive"], default="safe",
        help="Dynamic memory safety profile (default: safe)",
    )
    parser.add_argument(
        "--no-memory-probe", action="store_true",
        help="Skip dynamic probe calibration; conservative static caps only.",
    )
    parser.add_argument(
        "--force-unsafe", action="store_true",
        help="Proceed even when dynamic calibration marks the selected config unsafe.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Resolve run name, validate dataset, fingerprint, write log, print config decision - then exit before probes/training.",
    )
    parser.add_argument(
        "--no-guard", action="store_true",
        help="Disable runtime memory guard enforcement (monitoring only).",
    )
    parser.add_argument(
        "--probe-vram", action="store_true",
        help="Measure/refresh legacy per-model GB/image estimates (yolo26m/l/x or --model); exit if --input omitted",
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    return parser.parse_args()


# -- Run-Name Resolution (collision-aware, crash-safe) ----------------------------

def interactive_stdin() -> bool:
    try:
        return sys.stdin is not None and sys.stdin.isatty()
    except Exception:
        return False


def prompt_choice(logfn, question: str, options, default_hint=None) -> str:
    """Terminal prompt whose question and answer are also recorded via logfn."""
    logfn(f"[PROMPT] {question} Options: {' / '.join(options)}")
    while True:
        try:
            answer = input(f"  {question} [{'/'.join(options)}]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            logfn("[PROMPT] No answer received (stdin closed); treating as quit.")
            return options[-1]
        if not answer and default_hint:
            answer = default_hint.strip().lower()
        for label in options:
            if answer == label.lower() or answer == label[0].lower():
                logfn(f"[PROMPT] answer: {label}")
                return label
        print(f"  Please answer one of: {', '.join(options)}")


def ultralytics_increment_name(name: str) -> str:
    """Mirror Ultralytics increment_path: Corrosion -> Corrosion2, v3 -> v32 (appends 2, 3, ...)."""
    n = 2
    while n < 9999:
        candidate = f"{name}{n}"
        if not (RUNS_DIR / candidate).exists():
            return candidate
        n += 1
    raise RuntimeError(f"Too many runs starting with '{name}' under {RUNS_DIR}")


def resolve_fresh_attempt(args):
    """
    Decide the final run name/mode for a fresh invocation.

    Returns (run_name, run_dir, mode, messages). mode: 'fresh' or 'resume'
    (user chose to resume an existing colliding run). --resume on the command
    line is dispatched earlier and never reaches this function.

    - No collision: requested name (default detector_v1).
    - --fresh: auto-suffix, never resume.
    - Colliding OMITTED --name: silent auto-suffix.
    - Colliding explicit --name: interactive Resume/New/Quit prompt; a
      non-interactive terminal fails with a clear error instead.
    """
    requested = args.name if args.name else "detector_v1"
    explicit = args.name is not None
    run_dir = RUNS_DIR / requested
    messages = []
    logfn = messages.append

    def say(msg):
        messages.append(msg)
        print(f"  {msg}")

    if not run_dir.exists():
        return requested, run_dir, "fresh", messages

    if args.fresh:
        new_name = ultralytics_increment_name(requested)
        say(f"[Run] '{requested}' already exists; --fresh: auto-suffixed to '{new_name}'.")
        return new_name, RUNS_DIR / new_name, "fresh", messages

    if not explicit:
        new_name = ultralytics_increment_name(requested)
        say(f"[Run] Default run name '{requested}' already exists; auto-suffixed to '{new_name}' (pass --name to control it).")
        return new_name, RUNS_DIR / new_name, "fresh", messages

    checkpoint = run_dir / "weights" / "last.pt"
    say(f"[Run] Run '{requested}' already exists at: {run_dir}")

    if not interactive_stdin():
        say("[Run] stdin is not interactive: refusing to overwrite or silently resume a colliding run name.")
        say(f"[Run] Options: --resume --name {requested} (continue it) | --fresh --name {requested} (new suffixed run) | a different --name.")
        sys.exit(1)

    has_ckpt = checkpoint.is_file()
    options = ["Resume", "New", "Quit"] if has_ckpt else ["New", "Quit"]
    question = (f"Run '{requested}' already exists at {run_dir}."
                + (" Its weights/last.pt checkpoint exists." if has_ckpt
                   else " It has NO weights/last.pt checkpoint, so it cannot be resumed."))
    answer = prompt_choice(logfn, question, options)

    if answer == "Resume":
        say(f"[Run] Chose RESUME for existing run '{requested}'.")
        return requested, run_dir, "resume", messages
    if answer == "New":
        new_name = ultralytics_increment_name(requested)
        say(f"[Run] Chose NEW run: auto-suffixed to '{new_name}'.")
        return new_name, RUNS_DIR / new_name, "fresh", messages
    say("[Run] Chose QUIT.")
    sys.exit(0)


# -- Resume Checkpoint Lookup -----------------------------------------------------

def find_last_checkpoint(name: str = None) -> Path:
    """
    Find last.pt to resume from.
    - If --name given: looks in train_detector/runs/detect/<name>/weights/last.pt
    - If no --name:    finds the most recently modified run folder automatically
    """
    runs_dir = RUNS_DIR

    if not runs_dir.exists():
        print(f"[ERROR] No runs directory found at: {runs_dir}")
        sys.exit(1)

    if name:
        last_pt = runs_dir / name / "weights" / "last.pt"
        if not last_pt.exists():
            print(f"[ERROR] No checkpoint found for run '{name}'")
            print(f"        Expected: {last_pt}")
            print(f"        (A run folder without weights/last.pt cannot be resumed; use --fresh for a new run.)")
            sys.exit(1)
        return last_pt
    else:
        run_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
        if not run_dirs:
            print(f"[ERROR] No runs found in: {runs_dir}")
            sys.exit(1)

        run_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)

        for run_dir in run_dirs:
            last_pt = run_dir / "weights" / "last.pt"
            if last_pt.exists():
                return last_pt

        print(f"[ERROR] No last.pt checkpoint found in any run under: {runs_dir}")
        print(f"        Crashed runs without weights/last.pt cannot be resumed; use --fresh for a new run.")
        sys.exit(1)


# -- Dataset Validation -----------------------------------------------------------

def resolve_yaml_split_path(yaml_path: Path, split_val: str) -> Path:
    """Resolve a train/val/test path from data.yaml relative to the yaml file."""
    split_path = Path(str(split_val).replace("\\", "/"))
    if split_path.is_absolute():
        return split_path.resolve()

    resolved = (yaml_path.parent / split_path).resolve()
    if resolved.is_dir():
        return resolved

    # Legacy datasets: data.yaml at dataset root but paths like ../train/images
    # (copied from older toolkit scripts). Strip leading .. and resolve again.
    stripped = Path(*[part for part in split_path.parts if part != ".."])
    if stripped != split_path:
        fallback = (yaml_path.parent / stripped).resolve()
        if fallback.is_dir():
            return fallback

    return resolved


def validate_dataset(yaml_path: Path) -> None:
    """Validate data.yaml exists, has required fields, and paths contain images."""

    if not yaml_path.exists():
        print(f"[ERROR] data.yaml not found: {yaml_path}")
        sys.exit(1)

    if yaml_path.name != "data.yaml":
        print(f"[WARNING] Expected data.yaml, got: {yaml_path.name}")

    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    required_keys = ["train", "val", "nc", "names"]
    all_errors: list[str] = []
    for key in required_keys:
        if key not in data:
            all_errors.append(f"data.yaml is missing required key: '{key}'")

    if all_errors:
        print("[ERROR] Validation failed:")
        for err in all_errors:
            print(f"  - {err}")
        sys.exit(1)

    nc = data["nc"]
    names = data["names"]
    if not isinstance(names, (list, dict)):
        all_errors.append(f"names must be a list or dict, got {type(names).__name__}")
    else:
        name_count = len(names)
        if nc != name_count:
            all_errors.append(f"nc ({nc}) does not match len(names) ({name_count})")

    for split_key in ("train", "val"):
        split_val = data.get(split_key)
        if not split_val:
            all_errors.append(f"data.yaml '{split_key}' is empty or missing")
            continue
        split_path = resolve_yaml_split_path(yaml_path, str(split_val))
        if not split_path.is_dir():
            all_errors.append(f"{split_key} directory not found: {split_path}")
            continue
        img_count = count_images(split_path)
        if img_count < 1:
            all_errors.append(f"{split_key} directory has no images: {split_path}")

    if all_errors:
        print("[ERROR] Validation failed:")
        for err in all_errors:
            print(f"  - {err}")
        sys.exit(1)

    train_path = resolve_yaml_split_path(yaml_path, str(data["train"]))
    val_path = resolve_yaml_split_path(yaml_path, str(data["val"]))

    print(f"\n[Dataset]")
    print(f"  yaml     : {yaml_path}")
    print(f"  classes  : {data['nc']}")
    print(f"  names    : {data['names']}")
    print(f"  train    : {train_path}  ({count_images(train_path)} imgs)")
    print(f"  val      : {val_path}  ({count_images(val_path)} imgs)")
    if "test" in data:
        print(f"  test     : {data['test']}")


# -- Dataset Format Auto-Detection --------------------------------------------------
# --input points at a dataset DIRECTORY. Formats: YOLO (data.yaml present or
# synthesized after confirmation), COCO JSON, VOC XML. The annotation task
# (detect/segment/pose) is derived from the data unless --task overrides it.
# COCO/VOC convert once into a reusable sibling <folder>_yolo copy.

CONVERSION_DIR_SUFFIX = "_yolo"
CONVERSION_MANIFEST_NAME = ".conversion_manifest.json"


def split_role(name: str):
    """Map a dataset subdirectory name to a split role: train*/val*/valid*/validation*/test*."""
    n = name.lower()
    if n.startswith("train"):
        return "train"
    if n.startswith("val"):
        return "val"          # covers val, valid*, validation*
    if n.startswith("test"):
        return "test"
    return None


def classify_label_tokens(vals):
    """Classify the value tokens (after the class id) of a YOLO label line -> (task, kpts)."""
    nf = len(vals) + 1
    seg_ok = nf >= 7 and (nf - 1) % 2 == 0 and (nf - 1) // 2 >= 3
    pose_ok = nf >= 8 and (nf - 5) % 3 == 0 and (nf - 5) // 3 >= 1
    if nf == 5:
        return "detect", None
    if pose_ok and seg_ok:
        # A field count can parse as both segment (1+2N) and pose (1+4+3K), e.g. 11 tokens.
        # Normalized polygon coordinates must lie in [0,1]; pose visibility values are 0/1/2,
        # so an out-of-[0,1] value proves pose. Otherwise treat as segment.
        if any(v < 0.0 or v > 1.0 for v in vals):
            return "pose", (nf - 5) // 3
        return "segment", None
    if seg_ok:
        return "segment", None
    if pose_ok:
        return "pose", (nf - 5) // 3
    return None, None


def classify_label_file(path: Path):
    """Uniform structure of one YOLO label file -> (task, kpts, max_class_id) or None (empty).

    Raises ValueError when the file mixes line structures or contains bad lines.
    """
    task = None
    kpts = None
    max_class = -1
    for lineno, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split()
        try:
            cls = float(parts[0])
            vals = [float(t) for t in parts[1:]]
        except (ValueError, IndexError):
            raise ValueError(f"line {lineno}: not a numeric YOLO label line")
        if cls != int(cls) or cls < 0:
            raise ValueError(f"line {lineno}: invalid class id '{parts[0]}'")
        t, k = classify_label_tokens(vals)
        if t is None:
            raise ValueError(f"line {lineno}: unrecognized field count {len(vals) + 1}")
        if task is None:
            task, kpts = t, k
        elif t != task or k != kpts:
            raise ValueError(f"line {lineno}: structure '{t}' differs from earlier lines '{task}'")
        max_class = max(max_class, int(cls))
    if task is None:
        return None
    return task, kpts, max_class


def scan_yolo_labels(label_dirs):
    """Scan YOLO label .txt files across split label dirs.

    Returns (tasks: dict task->files, ksets: dict k->files, bad: [(file, reason)], max_class_id, n_label_files).
    """
    tasks = {}
    ksets = {}
    bad = []
    max_class = -1
    n_files = 0
    for d in label_dirs:
        if d is None or not d.is_dir():
            continue
        for lp in sorted(d.glob("*.txt")):
            try:
                res = classify_label_file(lp)
            except (ValueError, OSError) as e:
                bad.append((str(lp), str(e)))
                continue
            if res is None:
                continue
            n_files += 1
            t, k, mc = res
            tasks.setdefault(t, []).append(str(lp))
            if k is not None:
                ksets.setdefault(k, []).append(str(lp))
            max_class = max(max_class, mc)
    return tasks, ksets, bad, max_class, n_files


def _sample_list(files, cap=10):
    s = "; ".join(files[:cap])
    if len(files) > cap:
        s += f" ... (+{len(files) - cap} more)"
    return s


def _fail_label_scan(tasks, ksets, bad):
    print("[ERROR] YOLO label structure problems in this dataset:")
    if bad:
        print(f"  - {len(bad)} malformed label file(s) (non-numeric line / unrecognized field count):")
        for f, why in bad[:10]:
            print(f"      {f}: {why}")
        if len(bad) > 10:
            print(f"      ... (+{len(bad) - 10} more)")
    if len(tasks) > 1:
        print("  - Mixed label line structures in one dataset:")
        for t, files in tasks.items():
            print(f"      {t}: {_sample_list(files)}")
    if len(ksets) > 1:
        print("  - Pose keypoint counts differ across files:")
        for k, files in ksets.items():
            print(f"      kpt count {k}: {_sample_list(files)}")
    sys.exit(1)


def _label_dirs_for_yaml(yaml_path: Path, data: dict):
    dirs = []
    for key in ("train", "val", "test"):
        v = data.get(key)
        if not v:
            continue
        try:
            sp = resolve_yaml_split_path(yaml_path, str(v))
        except Exception:
            continue
        ld = sp.parent / "labels"
        dirs.append(ld if ld.is_dir() else sp)
    return dirs


def _label_dirs_with_roles(yaml_path: Path, data: dict):
    """Same split->label-dir resolution as _label_dirs_for_yaml, but each dir is tagged with its role."""
    pairs = []
    for key in ("train", "val", "test"):
        v = data.get(key)
        if not v:
            continue
        try:
            sp = resolve_yaml_split_path(yaml_path, str(v))
        except Exception:
            continue
        ld = sp.parent / "labels"
        pairs.append((key, ld if ld.is_dir() else sp))
    return pairs


def derive_task_from_yolo_labels(label_dirs):
    """Derive (task, kpt_count, max_class_id, had_labels) from uniform YOLO labels; hard-fail on mixed."""
    tasks, ksets, bad, max_class, n_files = scan_yolo_labels(label_dirs)
    if bad or len(tasks) > 1 or len(ksets) > 1:
        _fail_label_scan(tasks, ksets, bad)
    if not tasks:
        print("[Dataset Format] no YOLO label content found in this dataset; defaulting to task=detect.")
        return "detect", None, max_class, False
    task = next(iter(tasks))
    k = next(iter(ksets)) if ksets else None
    return task, k, max_class, True


def _confirm_dataset_write(logger, args, target: Path, change_lines, reason: str) -> bool:
    """Decision 7: never write into a user dataset folder without asking; --yes runs non-interactively."""
    print(f"[Dataset Write] {reason}.")
    print(f"  File: {target}")
    for ln in change_lines:
        print(f"    {ln}")
    if getattr(args, "yes", False):
        print("[Dataset Write] --yes given: applying without a prompt.")
        return True
    if not interactive_stdin():
        print("[Dataset Write] stdin is not interactive; refusing to write into the dataset folder.")
        return False
    answer = prompt_choice(logger.log, f"Apply this change to {target}?", ["y", "N"])
    return answer == "y"


def _declined_dataset_write(target: Path):
    print("[Dataset Write] Declined; nothing was written to the dataset folder.")
    print(f"  Options: add the lines above to {target} yourself, or re-run with --yes to apply them automatically.")
    sys.exit(1)


def _resolve_passthrough_yaml(yaml_files, args, logger):
    """data.yaml already present: pass through, derive task from label structure, offer pose kpt_shape auto-fix."""
    data_yaml = next((p for p in yaml_files if p.name.lower() == "data.yaml"), yaml_files[0])
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    task, k_derived, max_class, had = derive_task_with_repair(
        _label_dirs_with_roles(data_yaml, data), data_yaml.parent, args, logger)
    kpt = data.get("kpt_shape")
    if task == "pose":
        if kpt is None:
            k = k_derived or 0
            addition = [f"kpt_shape: [{k}, 3]", f"flip_idx: {list(range(k))}"]
            if not _confirm_dataset_write(logger, args, data_yaml, addition,
                                          reason="pose labels detected but data.yaml has no kpt_shape"):
                _declined_dataset_write(data_yaml)
            with open(data_yaml, "a", encoding="utf-8") as f:
                f.write("\n".join(addition) + "\n")
            print(f"[Dataset Write] updated {data_yaml}")
        else:
            kk = kpt[0] if isinstance(kpt, (list, tuple)) and kpt else None
            if kk is not None and k_derived is not None and int(kk) != int(k_derived):
                print(f"[ERROR] data.yaml kpt_shape ({kk} keypoints) does not match the label lines ({k_derived} keypoints).")
                sys.exit(1)
    elif kpt is not None:
        print(f"[WARNING] data.yaml has kpt_shape but the labels are '{task}'; kpt_shape ignored.")
    args.dataset_had_labels = had
    return data_yaml, task


def _try_synthesize_yaml(input_dir: Path, args, logger):
    """YOLO pre-split folder without data.yaml: synthesize data.yaml after confirmation."""
    split_dirs = {}
    for child in sorted(input_dir.iterdir()):
        if child.is_dir():
            role = split_role(child.name)
            if role and role not in split_dirs:
                split_dirs[role] = child

    images_by_split = {}
    for role, d in split_dirs.items():
        img_dir = d / "images" if (d / "images").is_dir() else d
        if image_files(img_dir):
            images_by_split[role] = img_dir
    if "train" not in images_by_split or "val" not in images_by_split:
        return None

    role_pairs = []
    for role in ("train", "val", "test"):
        if role not in split_dirs:
            continue
        ld = split_dirs[role] / "labels"
        role_pairs.append((role, ld if ld.is_dir() else images_by_split.get(role, split_dirs[role])))
    task, k, max_class, had = derive_task_with_repair(role_pairs, input_dir, args, logger)

    names = None
    for fname in ("classes.txt", "labels.txt"):
        cf = input_dir / fname
        if cf.is_file():
            names = [ln.strip() for ln in cf.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
            break
    if names is None:
        if max_class < 0:
            print(f"[ERROR] No data.yaml, no labels, and no classes.txt in {input_dir} — cannot infer class names.")
            sys.exit(1)
        names = [f"class_{i}" for i in range(max_class + 1)]
    elif max_class >= len(names):
        print(f"[ERROR] Label class id {max_class} is out of range for the {len(names)} names in classes.txt/labels.txt.")
        sys.exit(1)

    out = {"train": f"{split_dirs['train'].name}/images", "val": f"{split_dirs['val'].name}/images"}
    if "test" in images_by_split:
        out["test"] = f"{split_dirs['test'].name}/images"
    out["nc"] = len(names)
    out["names"] = names
    if task == "pose":
        k = k or 0
        out["kpt_shape"] = [k, 3]
        out["flip_idx"] = list(range(k))

    content = yaml.dump(out, default_flow_style=False, sort_keys=False)
    target = input_dir / "data.yaml"
    if not _confirm_dataset_write(logger, args, target, content.rstrip().splitlines(),
                                  reason=f"YOLO dataset has no data.yaml; synthesizing one (task={task})"):
        _declined_dataset_write(target)
    target.write_text(content, encoding="utf-8")
    print(f"[Dataset Write] wrote {target}")
    args.dataset_had_labels = had
    return target, task


def _looks_like_flat_folder(input_dir: Path) -> bool:
    if (input_dir / "images").is_dir() and image_files(input_dir / "images"):
        return True
    return any(f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS for f in input_dir.iterdir())


def apply_task_override(derived, args, had_labels):
    """Plan taxonomy: --task override wins over derived task; impossible overrides hard-error."""
    override = getattr(args, "task", "auto")
    if override == "auto":
        return derived
    if derived == override:
        return derived
    if not had_labels:
        print(f"[WARNING] --task {override} requested on a dataset with no labels; ensure the labels match this task.")
        return override
    print(f"[ERROR] Dataset label structure is '{derived}' but --task {override} was requested.")
    print("        YOLO labels are used as-is (no relabeling): drop --task or provide labels of the requested structure.")
    sys.exit(1)


# -- Mixed-Label Structure Repair ---------------------------------------------------
# A segment dataset may contain a tiny minority of 5-field bbox-only lines that Roboflow
# left as rectangles. Ultralytics silently drops EVERY polygon when box/segment counts
# differ (ultralytics/data/dataset.py mixed-dataset branch), so the toolkit hard-rejects
# mixed structures. This opt-in repair normalises only the bbox minority into exact 4-vertex
# polygons, in place, with backups and an audit manifest. Strict scanning stays intact;
# repair is additive and only entered when a live minority exists. Repair the source dataset
# before tiling (tile_yolo_dataset propagates mixes rather than rejecting them).

LABEL_REPAIR_MANIFEST_NAME = ".label_repair_manifest.json"
LABEL_REPAIR_BACKUP_DIRNAME = ".label_repair_backup"
LABEL_REPAIR_FORMAT = "label_repair"
LABEL_REPAIR_VERSION = 1
LABEL_REPAIR_MIN_RATIO = 0.95


class _LabelRepairAbort(Exception):
    """Raised when a repair write must be aborted and already-written files rolled back."""


def _fmt_coord(v: float) -> str:
    """Round-trip-exact number text matching the export style (whole numbers as 1/0)."""
    s = repr(v)
    if "e" in s or "E" in s:
        s = f"{v:.17f}".rstrip("0")
        if s.endswith("."):
            s = s[:-1]
    if s.endswith(".0"):
        s = s[:-2]
    return s


def _to_fraction(value) -> Fraction:
    """Parse a coordinate token into an exact decimal Fraction (binary float noise avoided)."""
    if isinstance(value, Fraction):
        return value
    if isinstance(value, str):
        return Fraction(value)
    return Fraction(repr(float(value)))


def bbox_line_to_polygon(cls, cx, cy, w, h):
    """Expand one bbox line into a 4-vertex ring, clamped to [0,1]. Returns a dict.

    Arithmetic is exact decimal (Fraction) so coordinates survive round-trip: cx - w/2 done in
    binary floats would produce values like 0.30976562500000004 that corrupt the export precision.
    vertices order x1,y1 -> x2,y1 -> x2,y2 -> x1,y2 so the 9-token line re-reads as segment.
    zero_area True when clamping collapses the box (x1==x2 or y1==y2) -> the line should drop.
    Returned coords are floats (nearest double to each exact coordinate).
    """
    cx, cy, w, h = _to_fraction(cx), _to_fraction(cy), _to_fraction(w), _to_fraction(h)
    half_w = w / 2
    half_h = h / 2
    x1 = cx - half_w
    y1 = cy - half_h
    x2 = cx + half_w
    y2 = cy + half_h
    zero = Fraction(0)
    one = Fraction(1)
    clamped = False

    def clamp(val: Fraction) -> Fraction:
        nonlocal clamped
        if val < zero:
            clamped = True
            return zero
        if val > one:
            clamped = True
            return one
        return val

    x1, y1, x2, y2 = clamp(x1), clamp(y1), clamp(x2), clamp(y2)
    zero_area = (x1 == x2) or (y1 == y2)
    coords = [x1, y1, x2, y1, x2, y2, x1, y2]
    return {"coords": [float(c) for c in coords], "clamped": clamped, "zero_area": zero_area}


def analyze_label_lines(label_dirs_with_roles, backup_root):
    """Per-line label census (independent of classify_label_file's intra-file raise).

    label_dirs_with_roles: [(role, Path)]; backup_root: dataset folder artifacts live in.
    Returns counts by role and merged, dominant kind/ratio, split-majority verdicts, the
    minority line list, unparseable lines, and the backup root. Empty files count 0 lines.
    """
    backup_root = Path(backup_root)
    by_role = {}
    role_total = {}
    merged = {}
    all_lines = []          # (relpath, abspath, lineno, kind, token_count, out_of_range)
    unparseable = []        # (relpath, abspath, lineno, reason)
    label_dirs = []

    for role, d in label_dirs_with_roles:
        if d is None:
            continue
        d = Path(d)
        label_dirs.append(d)
        counts = {}
        total = 0
        if not d.is_dir():
            by_role[role] = counts
            role_total[role] = 0
            continue
        for lp in sorted(d.glob("*.txt")):
            try:
                text = lp.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                unparseable.append((str(lp), str(lp), 0, f"unreadable: {exc}"))
                continue
            try:
                relpath = str(lp.relative_to(backup_root))
            except ValueError:
                relpath = str(lp)
            for lineno, raw in enumerate(text.splitlines(), 1):
                stripped = raw.strip()
                if not stripped:
                    continue
                parts = stripped.split()
                try:
                    cls = float(parts[0])
                    vals = [float(tok) for tok in parts[1:]]
                except (ValueError, IndexError):
                    unparseable.append((relpath, str(lp), lineno, "non-numeric YOLO label line"))
                    continue
                if cls != int(cls) or cls < 0:
                    unparseable.append((relpath, str(lp), lineno, f"invalid class id '{parts[0]}'"))
                    continue
                kind, _kpts = classify_label_tokens(vals)
                if kind is None:
                    unparseable.append((relpath, str(lp), lineno, f"unrecognized field count {len(vals) + 1}"))
                    continue
                counts[kind] = counts.get(kind, 0) + 1
                merged[kind] = merged.get(kind, 0) + 1
                total += 1
                out_of_range = any(v < 0.0 or v > 1.0 for v in vals)
                all_lines.append((relpath, str(lp), lineno, kind, len(vals) + 1, out_of_range))
        by_role[role] = counts
        role_total[role] = total

    total_lines = sum(merged.values())
    dominant = max(merged, key=lambda k: merged[k]) if merged else None
    dominant_ratio = (merged[dominant] / total_lines) if total_lines else 0.0

    split_verdicts = {}
    split_majority_ok = True
    for role, counts in by_role.items():
        total = role_total[role]
        if total == 0:
            split_verdicts[role] = (None, 0, 0)
            continue
        role_dom = max(counts, key=lambda k: counts[k])
        seg = counts.get("segment", 0)
        split_verdicts[role] = (role_dom, seg, total)
        if role_dom != "segment" or seg <= total / 2:
            split_majority_ok = False

    minority = []
    if dominant is not None:
        for relpath, abspath, lineno, kind, token, oob in all_lines:
            if kind != dominant:
                minority.append((relpath, abspath, lineno, kind, token, oob))

    return {
        "by_role": by_role,
        "role_total": role_total,
        "line_counts": merged,
        "total_lines": total_lines,
        "dominant": dominant,
        "dominant_ratio": dominant_ratio,
        "split_majority_ok": split_majority_ok,
        "split_verdicts": split_verdicts,
        "minority": minority,
        "unparseable": unparseable,
        "backup_root": backup_root,
        "label_dirs": label_dirs,
    }


def repairability(analysis):
    """Both guard granularities: global >=95% segment dominance AND segment majority in
    every split that has label lines, bbox-only in-range minority, no unparseable lines."""
    total = analysis["total_lines"]
    if total == 0:
        return False, "dataset has no label lines to repair"
    dominant = analysis["dominant"]
    if dominant != "segment":
        return False, f"dominant label kind is '{dominant}', not segment (only segment datasets are repairable)"
    ratio = analysis["dominant_ratio"]
    if ratio < LABEL_REPAIR_MIN_RATIO:
        return False, f"segment is only {ratio * 100:.2f}% of label lines (need >= {LABEL_REPAIR_MIN_RATIO * 100:.0f}%)"
    if not analysis["split_majority_ok"]:
        offenders = [
            r for r, (_d, _seg, tot) in analysis["split_verdicts"].items()
            if tot > 0 and analysis["by_role"][r].get("segment", 0) <= tot / 2
        ]
        return False, "segment is not the strict majority in split(s): " + ", ".join(sorted(offenders))
    for relpath, _abs, lineno, kind, token, oob in analysis["minority"]:
        if kind != "detect":
            return False, f"minority line {relpath}:{lineno} is '{kind}' (only 5-field bbox lines can be repaired)"
        if token != 5:
            return False, f"minority line {relpath}:{lineno} has {token} fields (expected a 5-field bbox line)"
        if oob:
            return False, f"minority line {relpath}:{lineno} has coordinates outside [0,1] (refusing to clamp polygons/rectangles silently)"
    if analysis["unparseable"]:
        relpath, _abs, lineno, reason = analysis["unparseable"][0]
        return False, f"unparseable line {relpath}:{lineno}: {reason}"
    return True, "repairable"


def _detect_line_ending(text: str) -> str:
    if "\r\n" in text:
        return "\r\n"
    if "\r" in text:
        return "\r"
    return "\n"


def _label_repair_artifact_gate(backup_root: Path):
    """Mirror the .conversion_manifest.json refusal: refuse foreign same-named artifacts,
    write nothing. Returns the existing tool manifest dict, or None."""
    manifest_path = backup_root / LABEL_REPAIR_MANIFEST_NAME
    backup_dir = backup_root / LABEL_REPAIR_BACKUP_DIRNAME
    manifest = None
    if manifest_path.is_dir():
        print(f"[Label Repair] {manifest_path} is a directory, not this tool's manifest file; refusing to write.")
        sys.exit(1)
    if manifest_path.is_file():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            loaded = None
        if not isinstance(loaded, dict):
            print(f"[Label Repair] {manifest_path} exists but is not a repair created by this tool "
                  f"(no usable {LABEL_REPAIR_MANIFEST_NAME}); it will not be modified. Move/remove it or "
                  "use a different dataset.")
            sys.exit(1)
        if loaded.get("format") != LABEL_REPAIR_FORMAT or loaded.get("version") != LABEL_REPAIR_VERSION:
            print(f"[Label Repair] {manifest_path} is not a {LABEL_REPAIR_FORMAT} v{LABEL_REPAIR_VERSION} manifest "
                  "created by this tool; refusing to write.")
            sys.exit(1)
        manifest = loaded
    if backup_dir.exists() and not backup_dir.is_dir():
        print(f"[Label Repair] {backup_dir} exists but is not a directory created by this tool; refusing to write.")
        sys.exit(1)
    if backup_dir.is_dir() and manifest is None:
        print(f"[Label Repair] {backup_dir} exists without a valid {LABEL_REPAIR_MANIFEST_NAME}; it is not managed "
              "by this tool, so nothing will be written. Move/remove it first.")
        sys.exit(1)
    return manifest


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _restore_label_backups(restore_map, backup_root, logger):
    for relpath, orig_bytes in restore_map.items():
        target = backup_root / relpath
        try:
            target.write_bytes(orig_bytes)
            print(f"[Label Repair] restored {relpath} from backup.")
        except OSError as exc:
            print(f"[Label Repair] WARNING: could not restore {relpath}: {exc}")


def _print_label_repair_diagnosis(tasks, ksets, bad, analysis, reason):
    print("[ERROR] YOLO label structure problems in this dataset:")
    if bad:
        print(f"  - {len(bad)} malformed label file(s) (non-numeric line / unrecognized field count):")
        for f, why in bad[:10]:
            print(f"      {f}: {why}")
        if len(bad) > 10:
            print(f"      ... (+{len(bad) - 10} more)")
    if len(tasks) > 1:
        print("  - Mixed label line structures in one dataset:")
        for t, files in tasks.items():
            print(f"      {t}: {_sample_list(files)}")
    if len(ksets) > 1:
        print("  - Pose keypoint counts differ across files:")
        for k, files in ksets.items():
            print(f"      kpt count {k}: {_sample_list(files)}")
    dominant = analysis["dominant"]
    ratio = analysis["dominant_ratio"]
    total = analysis["total_lines"]
    print(f"  - Dominant kind: {dominant}; dominant ratio: {ratio * 100:.2f}%; total label lines: {total}")
    print("  - Per-split label line counts:")
    for role in ("train", "val", "test"):
        if role not in analysis["by_role"]:
            continue
        tot = analysis["role_total"][role]
        kinds = analysis["by_role"][role]
        if tot == 0:
            print(f"      {role}: 0 label lines")
            continue
        verdict = "segment majority" if kinds.get("segment", 0) > tot / 2 else "segment NOT majority"
        kind_txt = ", ".join(f"{k}={c}" for k, c in sorted(kinds.items()))
        print(f"      {role}: {kind_txt}  ({verdict})")
    minority = analysis["minority"]
    if minority:
        print(f"  - Minority (non-dominant) label lines: {len(minority)}")
        for relpath, _abs, lineno, kind, token, oob in minority[:10]:
            flag = " [coords out of [0,1]]" if oob else ""
            print(f"      {relpath}:{lineno}  kind={kind} fields={token}{flag}")
        if len(minority) > 10:
            print(f"      ... (+{len(minority) - 10} more)")
    if analysis["unparseable"]:
        print(f"  - Unparseable lines: {len(analysis['unparseable'])}")
        for relpath, _abs, lineno, why in analysis["unparseable"][:10]:
            print(f"      {relpath}:{lineno}  {why}")
        if len(analysis["unparseable"]) > 10:
            print(f"      ... (+{len(analysis['unparseable']) - 10} more)")
    print(f"  - Repair refused: {reason}")
    print("  - To attempt an in-place repair (bbox lines -> polygons, with backup + manifest), run:")
    print(f"      python train_detector/train_detector.py --input \"{analysis['backup_root']}\" --fix-label-structure --dry-run")


def derive_task_with_repair(label_dirs_with_roles, backup_root, args, logger):
    """Strict-first derive; only when the strict scan would fail do we analyse repairability.

    Clean datasets take the exact same path (and prints) as derive_task_from_yolo_labels and
    touch nothing. On a live minority we run the guards, then consent + repair, then re-derive.
    """
    dirs = [d for _role, d in label_dirs_with_roles if d is not None]
    tasks, ksets, bad, max_class, _n = scan_yolo_labels(dirs)
    if not (bad or len(tasks) > 1 or len(ksets) > 1):
        if not tasks:
            print("[Dataset Format] no YOLO label content found in this dataset; defaulting to task=detect.")
            return "detect", None, max_class, False
        task = next(iter(tasks))
        k = next(iter(ksets)) if ksets else None
        return task, k, max_class, True

    analysis = analyze_label_lines(label_dirs_with_roles, backup_root)
    ok, reason = repairability(analysis)
    if not ok:
        _print_label_repair_diagnosis(tasks, ksets, bad, analysis, reason)
        sys.exit(1)
    repair_label_structures(analysis, args, logger)
    tasks2, ksets2, bad2, max_class2, _n2 = scan_yolo_labels(dirs)
    if bad2 or len(tasks2) > 1 or len(ksets2) > 1:
        # repair_label_structures should have restored and exited; keep a defensive net.
        _fail_label_scan(tasks2, ksets2, bad2)
    task2 = next(iter(tasks2)) if tasks2 else "detect"
    k2 = next(iter(ksets2)) if ksets2 else None
    return task2, k2, max_class2, True


def _build_repair_file_plan(analysis):
    """Group minority lines by file and compute byte-minimal rewritten content + audit records."""
    backup_root = analysis["backup_root"]
    minority_by_file = {}
    for relpath, abspath, lineno, _kind, _token, _oob in analysis["minority"]:
        minority_by_file.setdefault((relpath, abspath), set()).add(lineno)

    plans = []
    for (relpath, abspath), target_lines in sorted(minority_by_file.items()):
        path = Path(abspath)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise _LabelRepairAbort(f"could not read {relpath}: {exc}")
        nl = _detect_line_ending(text)
        had_trailing = text.endswith(("\n", "\r"))
        lines = text.splitlines()
        entries = []
        new_lines = []
        converted = clamped = dropped = 0
        for idx, raw in enumerate(lines, 1):
            if idx in target_lines:
                parts = raw.split()
                cls = int(float(parts[0]))
                # Pass the verbatim token strings so polygon math stays exact-decimal.
                res = bbox_line_to_polygon(cls, *parts[1:])
                if res["zero_area"]:
                    entries.append({
                        "relpath": relpath, "lineno": idx, "original": raw, "repaired": "",
                        "clamped": bool(res["clamped"]), "dropped": True,
                    })
                    dropped += 1
                    continue
                new_line = f"{cls} " + " ".join(_fmt_coord(v) for v in res["coords"])
                new_lines.append(new_line)
                converted += 1
                if res["clamped"]:
                    clamped += 1
                entries.append({
                    "relpath": relpath, "lineno": idx, "original": raw, "repaired": new_line,
                    "clamped": bool(res["clamped"]), "dropped": False,
                })
            else:
                new_lines.append(raw)  # byte-identical original content for unchanged lines
        new_text = nl.join(new_lines)
        if had_trailing and new_lines:
            new_text += nl
        plans.append({
            "relpath": relpath, "path": path, "new_text": new_text,
            "entries": entries, "converted": converted, "clamped": clamped, "dropped": dropped,
        })
    return plans


def _print_repair_change_summary(analysis, plans):
    dominant = analysis["dominant"]
    ratio = analysis["dominant_ratio"]
    total = analysis["total_lines"]
    files_changed = len(plans)
    lines_converted = sum(p["converted"] for p in plans)
    clamped = sum(p["clamped"] for p in plans)
    dropped = sum(p["dropped"] for p in plans)
    print("[Label Repair] segment-dominant dataset with a bbox-only minority:")
    print(f"  dominant: {dominant}  ratio: {ratio * 100:.2f}%  total label lines: {total}")
    print("  per-split label line counts:")
    for role in ("train", "val", "test"):
        if role not in analysis["by_role"]:
            continue
        tot = analysis["role_total"][role]
        kinds = analysis["by_role"][role]
        if tot == 0:
            print(f"    {role}: 0 label lines")
            continue
        verdict = "segment majority" if kinds.get("segment", 0) > tot / 2 else "segment NOT majority"
        kind_txt = ", ".join(f"{k}={c}" for k, c in sorted(kinds.items()))
        print(f"    {role}: {kind_txt}  ({verdict})")
    print(f"  files to change: {files_changed}  lines to convert: {lines_converted}  clamped: {clamped}  dropped: {dropped}")
    print("  change preview:")
    for plan in plans:
        for e in plan["entries"]:
            arrow = "(dropped)" if e["dropped"] else e["repaired"]
            print(f"    {e['relpath']}:{e['lineno']}")
            print(f"      - {e['original']}")
            print(f"      + {arrow}")
    return files_changed, lines_converted, clamped, dropped


def repair_label_structures(analysis, args, logger):
    """Consent-gated, byte-minimal in-place repair with backup, manifest, cache cleanup and
    a strict re-scan that fully restores every written .bak on any failure."""
    backup_root = analysis["backup_root"]
    manifest = _label_repair_artifact_gate(backup_root)
    backup_dir = backup_root / LABEL_REPAIR_BACKUP_DIRNAME

    # Pristine backup identities from any prior tool run (first original wins).
    pristine_sha = {}
    existing_entries = []
    if manifest and isinstance(manifest.get("entries"), list):
        for e in manifest["entries"]:
            if isinstance(e, dict) and "relpath" in e:
                existing_entries.append(e)
                if e.get("original_sha256"):
                    pristine_sha[e["relpath"]] = e["original_sha256"]

    plans = _build_repair_file_plan(analysis)
    if not plans:
        # Idempotency: a live minority was expected but nothing maps to a writable file.
        return False
    files_changed, lines_converted, clamped, dropped = _print_repair_change_summary(analysis, plans)

    command = (f"python train_detector/train_detector.py --input \"{backup_root}\" "
               "--fix-label-structure --dry-run")

    if getattr(args, "fix_label_structure", False) or getattr(args, "yes", False):
        why = "fix-label-structure" if getattr(args, "fix_label_structure", False) else "yes"
        logger.log(f"[Label Repair] --{why} given: applying repair without a prompt.")
    elif interactive_stdin():
        answer = prompt_choice(logger.log, "Apply this label-structure repair to the dataset?", ["y", "N"])
        if answer != "y":
            print("[Label Repair] Declined; nothing was written to the dataset folder.")
            print(f"  Options: apply manually, or re-run with --fix-label-structure:")
            print(f"    {command}")
            sys.exit(1)
    else:
        print("[Label Repair] stdin is not interactive; refusing to write into the dataset folder.")
        print(f"  Re-run with --fix-label-structure (or --yes) to apply:")
        print(f"    {command}")
        sys.exit(1)

    restore_map = {}
    try:
        for plan in plans:
            relpath = plan["relpath"]
            path = plan["path"]
            try:
                rel_for_backup = path.relative_to(backup_root)
            except ValueError:
                raise _LabelRepairAbort(f"{relpath} is outside the dataset folder {backup_root}; cannot back it up")
            bak_path = backup_dir / (str(rel_for_backup) + ".bak")
            orig_bytes = path.read_bytes()
            if bak_path.exists():
                expected = pristine_sha.get(relpath)
                if expected is None:
                    raise _LabelRepairAbort(
                        f"backup {bak_path} exists with no manifest entry to verify it against; refusing to overwrite")
                if _sha256_bytes(bak_path.read_bytes()) != expected:
                    raise _LabelRepairAbort(
                        f"existing backup {bak_path} no longer matches its manifest original_sha256; refusing to use it as a rollback target")
            else:
                bak_path.parent.mkdir(parents=True, exist_ok=True)
                bak_path.write_bytes(orig_bytes)
                pristine_sha[relpath] = _sha256_bytes(orig_bytes)
            restore_map[relpath] = orig_bytes
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write(plan["new_text"])
    except _LabelRepairAbort as exc:
        _restore_label_backups(restore_map, backup_root, logger)
        print(f"[Label Repair] Aborted before completing: {exc}")
        print("  No partial repair was left in place.")
        sys.exit(1)
    except OSError as exc:
        _restore_label_backups(restore_map, backup_root, logger)
        print(f"[Label Repair] Write failed: {exc}")
        print("  All already-written files were restored from their backups.")
        sys.exit(1)

    # Strict re-scan: never leave a half-repaired dataset.
    tasks2, ksets2, bad2, _mc2, _n2 = scan_yolo_labels(analysis["label_dirs"])
    if bad2 or len(tasks2) > 1 or len(ksets2) > 1:
        _restore_label_backups(restore_map, backup_root, logger)
        print("[Label Repair] post-repair strict scan still failed; restored every repaired file.")
        _fail_label_scan(tasks2, ksets2, bad2)

    # Manifest audit (merge with prior entries, dedupe by relpath+lineno).
    merged_entries = list(existing_entries)
    seen = {(e.get("relpath"), e.get("lineno")) for e in merged_entries if isinstance(e, dict)}
    for plan in plans:
        relpath = plan["relpath"]
        sha = pristine_sha.get(relpath, _sha256_bytes(restore_map.get(relpath, b"")))
        for e in plan["entries"]:
            key = (e["relpath"], e["lineno"])
            if key in seen:
                continue
            seen.add(key)
            merged_entries.append({
                "relpath": e["relpath"],
                "split": Path(e["relpath"]).parts[0] if Path(e["relpath"]).parts else "",
                "lineno": e["lineno"],
                "original": e["original"],
                "repaired": e["repaired"],
                "clamped": e["clamped"],
                "dropped": e["dropped"],
                "original_sha256": sha,
                "created_at": local_iso(),
            })

    split_counts = {role: {**counts, "total_lines": analysis["role_total"][role]}
                    for role, counts in analysis["by_role"].items()}
    manifest_out = {
        "format": LABEL_REPAIR_FORMAT,
        "version": LABEL_REPAIR_VERSION,
        "dominant_task": "segment",
        "created_at": local_iso(),
        "dominant_ratio": analysis["dominant_ratio"],
        "split_counts": split_counts,
        "files_changed": files_changed,
        "lines_converted": lines_converted,
        "clamped": clamped,
        "dropped": dropped,
        "entries": merged_entries,
    }
    (backup_root / LABEL_REPAIR_MANIFEST_NAME).write_text(
        json.dumps(manifest_out, indent=2), encoding="utf-8")

    # Delete Ultralytics label caches adjacent to repaired labels dirs (Ultralytics keys the
    # cache on summed label+image byte size; deleting makes the effect deterministic).
    for plan in plans:
        labels_dir = plan["path"].parent
        cache_path = labels_dir.with_suffix(".cache")
        if cache_path.is_file():
            try:
                cache_path.unlink()
                print(f"[Label Repair] removed stale cache {cache_path}")
            except OSError as exc:
                print(f"[Label Repair] WARNING: could not remove {cache_path}: {exc}")

    print(f"[Label Repair] applied: files_changed={files_changed} lines_converted={lines_converted} "
          f"clamped={clamped} dropped={dropped} backups_dir={backup_dir}")
    logger.log(f"[Label Repair] applied files_changed={files_changed} lines_converted={lines_converted} "
               f"clamped={clamped} dropped={dropped} manifest={backup_root / LABEL_REPAIR_MANIFEST_NAME}")
    return True


# -- COCO Conversion Helpers -------------------------------------------------------

def looks_like_coco(input_dir: Path) -> bool:
    """COCO per-split (`<split>/_annotations.coco.json`) or canonical (`annotations/*.json` + `images/`)."""
    for child in input_dir.iterdir():
        if child.is_dir() and (child / "_annotations.coco.json").is_file():
            return True
    ann_dir = input_dir / "annotations"
    if ann_dir.is_dir() and any(p.is_file() and p.suffix.lower() == ".json" for p in ann_dir.iterdir()):
        if (input_dir / "images").is_dir():
            return True
    return False


def _norm_polygon(poly):
    if not poly:
        return []
    if isinstance(poly[0], (list, tuple)):
        return [(float(p[0]), float(p[1])) for p in poly if isinstance(p, (list, tuple)) and len(p) >= 2]
    pts = []
    for i in range(0, len(poly) - 1, 2):
        pts.append((float(poly[i]), float(poly[i + 1])))
    return pts


def _poly_area(pts):
    s = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def _largest_polygon(segs):
    best, best_area = None, -1.0
    for poly in segs:
        pts = _norm_polygon(poly)
        if len(pts) < 3:
            continue
        a = _poly_area(pts)
        if a > best_area:
            best_area, best = a, pts
    return best


def _bbox_rect_pts(bbox):
    x, y, w, h = [float(v) for v in bbox[:4]]
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


def _polygon_is_bbox_like(poly_pts, bbox):
    if not bbox or len(bbox) != 4 or len(poly_pts) != 4:
        return False
    for (px, py), (rx, ry) in zip(poly_pts, _bbox_rect_pts(bbox)):
        if abs(px - rx) > 1e-6 or abs(py - ry) > 1e-6:
            return False
    return True


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _read_coco_json(jp: Path, tag: str) -> dict:
    try:
        data = json.loads(jp.read_text(encoding="utf-8-sig"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"[ERROR] Could not read COCO annotation file {jp}: {e}")
        sys.exit(1)
    if not isinstance(data, dict) or not isinstance(data.get("images"), list):
        print(f"[ERROR] {tag}: {jp} is not a valid COCO JSON (missing 'images' list).")
        sys.exit(1)
    return data


def _load_coco_split_jsons(input_dir: Path):
    """Map split role -> [(json_path, coco_dict)]; hard-error on role collisions or unreadable jsons."""
    splits = {}
    for child in sorted(input_dir.iterdir()):
        if child.is_dir() and (child / "_annotations.coco.json").is_file():
            role = split_role(child.name)
            jp = child / "_annotations.coco.json"
            if role is None:
                print(f"[COCO] warning: folder '{child.name}' holds a COCO json but is not a train/val/test split name; ignored.")
                continue
            if role in splits:
                print(f"[ERROR] Two COCO folders map to the same split '{role}': {splits[role][0][0].parent} and {jp.parent}.")
                sys.exit(1)
            splits[role] = [(jp, _read_coco_json(jp, child.name))]
    ann_dir = input_dir / "annotations"
    if ann_dir.is_dir():
        for jp in sorted(ann_dir.iterdir()):
            if not (jp.is_file() and jp.suffix.lower() == ".json"):
                continue
            low = jp.name.lower()
            if "train" in low:
                role = "train"
            elif "val" in low:
                role = "val"
            elif "test" in low:
                role = "test"
            else:
                print(f"[COCO] warning: cannot infer a split from {jp.name}; ignored.")
                continue
            if role in splits:
                print(f"[ERROR] Split '{role}' provided twice (per-split json and {jp}); keep one layout per dataset.")
                sys.exit(1)
            splits.setdefault(role, []).append((jp, _read_coco_json(jp, jp.name)))
    return splits


def _coco_categories(splits):
    """Category map identical across split jsons: sorted-by-id -> class index, names, per-id metadata."""
    base_role, base = None, {}
    for role in sorted(splits):
        ids = {}
        for jp, data in splits[role]:
            for c in data.get("categories", []):
                try:
                    cid = int(c["id"])
                except (KeyError, ValueError, TypeError):
                    print(f"[ERROR] Category without a valid id in {jp}.")
                    sys.exit(1)
                ids[cid] = (c.get("name") or f"class_{cid}", int(c.get("num_keypoints", 0) or 0))
        if base_role is None:
            base_role, base = role, ids
            continue
        if set(ids) != set(base):
            print(f"[ERROR] COCO category sets differ between splits '{base_role}' and '{role}'; every split json must declare the same categories.")
            sys.exit(1)
        for cid in sorted(base):
            if base[cid][0] != ids[cid][0]:
                print(f"[ERROR] Category id {cid} is named {base[cid][0]!r} in '{base_role}' but {ids[cid][0]!r} in '{role}'.")
                sys.exit(1)
    ordered = sorted(base.items(), key=lambda kv: kv[0])
    cat_map = {cid: idx for idx, (cid, _) in enumerate(ordered)}
    names = [name for _, (name, _) in ordered]
    return cat_map, names, base


def _derive_coco_task(splits):
    """Task from COCO content: keypoint annotations > genuine polygons > plain bboxes."""
    kpt_counts = set()
    has_polygon = False
    for entries in splits.values():
        for _, data in entries:
            nk = {}
            for c in data.get("categories", []):
                try:
                    nk[int(c["id"])] = int(c.get("num_keypoints", 0) or 0)
                except (KeyError, ValueError, TypeError):
                    pass
            for ann in data.get("annotations", []):
                if ann.get("iscrowd"):
                    continue
                cid = int(ann.get("category_id", -1))
                if nk.get(cid, 0) > 0 and ann.get("keypoints"):
                    kpt_counts.add(nk[cid])
                bbox = ann.get("bbox")
                segs = ann.get("segmentation")
                if isinstance(segs, list):
                    for poly in segs:
                        pts = _norm_polygon(poly)
                        if len(pts) >= 3 and not _polygon_is_bbox_like(pts, bbox):
                            has_polygon = True
    if kpt_counts:
        if len(kpt_counts) > 1:
            print(f"[ERROR] COCO keypoint categories declare different keypoint counts {sorted(kpt_counts)}; pose conversion needs one uniform keypoint shape.")
            sys.exit(1)
        return "pose", next(iter(kpt_counts))
    if has_polygon:
        return "segment", None
    return "detect", None


def _coco_source_fingerprint(input_dir: Path, out_dir: Path):
    out_abs = out_dir.resolve()
    files = []
    for p in sorted(input_dir.rglob("*")):
        if not p.is_file():
            continue
        rp = p.resolve()
        if rp == out_abs or out_abs in rp.parents:
            continue
        st = p.stat()
        files.append({"path": str(p.relative_to(input_dir)).replace("\\", "/"),
                      "size": st.st_size, "mtime": int(st.st_mtime)})
    return {"files": files}


def _coco_ann_to_yolo(ann, im, cat_map, task, kpt_count, stats):
    try:
        cid = int(ann["category_id"])
    except (KeyError, ValueError, TypeError):
        stats["dropped"] += 1
        return None
    if cid not in cat_map:
        stats["unknown_category"] += 1
        return None
    cls = cat_map[cid]
    try:
        W = float(im.get("width") or 0.0)
        H = float(im.get("height") or 0.0)
    except (TypeError, ValueError):
        W = H = 0.0
    if W <= 0 or H <= 0:
        print(f"[ERROR] COCO image id {im.get('id')} has no valid width/height; cannot normalize labels.")
        sys.exit(1)

    bbox = None
    raw_bbox = ann.get("bbox")
    if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4:
        x, y, w, h = [float(v) for v in raw_bbox]
        if w > 0 and h > 0:
            bbox = (x, y, w, h)

    if bbox is None:
        poly = _largest_polygon(ann.get("segmentation") or [])
        if poly:
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            w, h = max(xs) - min(xs), max(ys) - min(ys)
            if w > 0 and h > 0:
                bbox = (min(xs), min(ys), w, h)

    if task == "detect":
        if bbox is None:
            stats["dropped"] += 1
            return None
        x, y, w, h = bbox
        return (f"{cls} {_clamp01((x + w / 2.0) / W):.6f} {_clamp01((y + h / 2.0) / H):.6f} "
                f"{_clamp01(w / W):.6f} {_clamp01(h / H):.6f}")

    if task == "segment":
        poly = _largest_polygon(ann.get("segmentation") or [])
        if poly:
            if sum(1 for s in (ann.get("segmentation") or []) if len(_norm_polygon(s)) >= 3) > 1:
                stats["multi_polygon"] += 1
        elif bbox is not None:
            poly = _bbox_rect_pts(bbox)
        else:
            stats["dropped"] += 1
            return None
        vals = []
        for px, py in poly:
            vals.append(f"{_clamp01(px / W):.6f}")
            vals.append(f"{_clamp01(py / H):.6f}")
        return f"{cls} " + " ".join(vals)

    # pose
    if bbox is None:
        stats["dropped"] += 1
        return None
    x, y, w, h = bbox
    parts = [str(cls), f"{_clamp01((x + w / 2.0) / W):.6f}", f"{_clamp01((y + h / 2.0) / H):.6f}",
             f"{_clamp01(w / W):.6f}", f"{_clamp01(h / H):.6f}"]
    kps = [float(v) if isinstance(v, (int, float)) else 0.0 for v in (ann.get("keypoints") or [])]
    kps += [0.0, 0.0, 0.0] * max(0, kpt_count - len(kps) // 3)
    for i in range(kpt_count):
        kx, ky, kv = kps[3 * i], kps[3 * i + 1], kps[3 * i + 2]
        try:
            kv = int(kv)
        except (TypeError, ValueError):
            kv = 0
        if kv not in (1, 2):
            kv = 0
        if kv == 0:
            kx, ky = 0.0, 0.0
        elif kv == 1:
            kx, ky = 0.0, 0.0
        else:
            kx, ky = _clamp01(kx / W), _clamp01(ky / H)
        parts.append(f"{kx:.6f}")
        parts.append(f"{ky:.6f}")
        parts.append(str(kv))
    return " ".join(parts)


def convert_coco_dataset(input_dir: Path, args):
    """Convert COCO JSON datasets into a *_yolo YOLO copy. Returns (data.yaml, final task, action).

    Reuse rules: existing conversion with same fingerprint and task is reused; changed source rebuilds
    (wipe allowed only because our manifest proves the folder is ours); an existing folder without our
    manifest is never modified.
    """
    splits = _load_coco_split_jsons(input_dir)
    if "train" not in splits or "val" not in splits:
        found = ", ".join(sorted(splits)) if splits else "none"
        print(f"[ERROR] COCO dataset needs train and val annotation JSONs; found splits: {found}")
        sys.exit(1)

    cat_map, names, _cat_meta = _coco_categories(splits)
    derived, kpt_count = _derive_coco_task(splits)

    override = getattr(args, "task", "auto")
    task = derived
    if override != "auto" and override != derived:
        if override == "pose":
            print("[ERROR] --task pose requested but this COCO dataset has no keypoint annotations (no category declares num_keypoints > 0).")
            print("        Use --task detect or --task segment, or export this dataset with keypoints.")
            sys.exit(1)
        task = override  # detect-on-polygons: geometry reduced to bboxes; segment-on-bboxes: rectangular polygons

    if args.converted_dir:
        out_dir = normalize_path(args.converted_dir)
    else:
        out_dir = input_dir.parent / f"{input_dir.name}{CONVERSION_DIR_SUFFIX}"

    _validate_converted_dir(input_dir, out_dir)
    fp = _coco_source_fingerprint(input_dir, out_dir)
    fp_id = fingerprint_hash(fp)
    manifest_path = out_dir / CONVERSION_MANIFEST_NAME
    gate = _conversion_reuse_gate(out_dir, fp_id, "coco", task)
    if gate == "reuse":
        return out_dir / "data.yaml", task, "reused-conversion"

    stats = {"missing_images": 0, "iscrowd_skipped": 0, "dropped": 0, "multi_polygon": 0, "unknown_category": 0}
    image_counts = {}
    print(f"[COCO] converting {input_dir} -> {out_dir} (task={task})")
    try:
        for role in ("train", "val", "test"):
            if role not in splits:
                continue
            images_dir = out_dir / role / "images"
            labels_dir = out_dir / role / "labels"
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)
            seen_ids = set()
            anns_by_image = {}
            images = []
            json_dirs = set()
            for jp, data in splits[role]:
                json_dirs.add(jp.parent)
                for im in data.get("images", []):
                    try:
                        iid = int(im["id"])
                    except (KeyError, ValueError, TypeError):
                        continue
                    if iid in seen_ids:
                        continue
                    seen_ids.add(iid)
                    images.append((jp, im))
                for ann in data.get("annotations", []):
                    try:
                        anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)
                    except (KeyError, ValueError, TypeError):
                        continue
            referenced = {str(im.get("file_name") or "").replace("\\", "/") for _, im in images}
            referenced_names = {Path(r).name for r in referenced if r}
            copied = 0
            for jp, im in images:
                file_name = str(im.get("file_name") or "").replace("\\", "/")
                if not file_name:
                    stats["missing_images"] += 1
                    continue
                src = None
                for base in (jp.parent, jp.parent / "images", input_dir / "images", input_dir):
                    cand = base / file_name
                    if cand.is_file():
                        src = cand
                        break
                if src is None:
                    stats["missing_images"] += 1
                    print(f"[COCO] warning: image '{file_name}' from {jp.name} not found next to the json or in images/; skipped.")
                    continue
                dst = images_dir / Path(file_name).name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                copied += 1
                lines = []
                for ann in anns_by_image.get(int(im["id"]), []):
                    if ann.get("iscrowd"):
                        stats["iscrowd_skipped"] += 1
                        continue
                    line = _coco_ann_to_yolo(ann, im, cat_map, task, kpt_count, stats)
                    if line:
                        lines.append(line)
                (labels_dir / f"{Path(file_name).stem}.txt").write_text(
                    "".join(ln + "\n" for ln in lines), encoding="utf-8")
            image_counts[role] = copied
            extra = []
            for jd in json_dirs | {input_dir / "images"}:
                probe_dirs = [jd, jd / "images"] if jd != input_dir / "images" else [jd]
                for probe in probe_dirs:
                    if probe.is_dir():
                        for f in image_files(probe):
                            rel = str(f.relative_to(jd)).replace("\\", "/")
                            if f.name not in referenced_names and rel not in referenced:
                                extra.append(f)
            if extra:
                print(f"[COCO] warning: {len(extra)} image(s) on disk in split '{role}' are not referenced by the COCO json and were not copied: {_sample_list([str(f) for f in extra])}")

        yaml_out = {"train": "train/images", "val": "val/images"}
        if "test" in splits:
            yaml_out["test"] = "test/images"
        yaml_out["nc"] = len(names)
        yaml_out["names"] = names
        if task == "pose":
            yaml_out["kpt_shape"] = [kpt_count, 3]
            yaml_out["flip_idx"] = list(range(kpt_count))
        (out_dir / "data.yaml").write_text(yaml.dump(yaml_out, default_flow_style=False, sort_keys=False), encoding="utf-8")

        manifest_data = {
            "version": 1,
            "format": "coco",
            "task": task,
            "source_dir": str(input_dir),
            "fingerprint_id": fp_id,
            "n_source_files": len(fp["files"]),
            "image_counts": image_counts,
            "created": local_iso(),
        }
        manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
    except OSError as e:
        print(f"[ERROR] Could not write the converted dataset to {out_dir}: {e}")
        print("        This location may be read-only or unavailable: pass --converted-dir <writable location> and re-run.")
        sys.exit(1)

    if stats["iscrowd_skipped"]:
        print(f"[COCO] skipped {stats['iscrowd_skipped']} iscrowd annotation(s) (crowd regions are not valid YOLO labels).")
    if stats["dropped"]:
        print(f"[COCO] warning: {stats['dropped']} annotation(s) dropped (zero-area or unusable geometry).")
    if stats["multi_polygon"]:
        print(f"[COCO] {stats['multi_polygon']} annotation(s) had multiple polygons: kept the largest by area (YOLO labels take one polygon per object).")
    if stats["unknown_category"]:
        print(f"[COCO] warning: {stats['unknown_category']} annotation(s) referenced a category missing from the jsons; skipped.")
    counts = ", ".join(f"{k}:{v}" for k, v in image_counts.items())
    print(f"[COCO] converted {sum(image_counts.values())} images ({counts}); labels in {out_dir}/<split>/labels/")
    return out_dir / "data.yaml", task, "converted"


# -- Conversion Manifest Gate ------------------------------------------------------

def _validate_converted_dir(input_dir: Path, out_dir: Path) -> None:
    """Hard-error when --converted-dir overlaps the input dataset: a conversion rebuild
    (wipe) must never be able to delete the user's raw source images/annotations."""
    in_abs = input_dir.resolve()
    out_abs = out_dir.resolve()
    if out_abs == in_abs:
        print(f"[ERROR] --converted-dir points at the input dataset itself: {out_abs}")
        print("        Use a separate output location (or omit --converted-dir for the default sibling copy).")
        sys.exit(1)
    if out_abs in in_abs.parents:
        print(f"[ERROR] --converted-dir ({out_abs}) is inside the input dataset ({in_abs});")
        print("        conversion output would live inside its own source.")
        sys.exit(1)
    if in_abs in out_abs.parents:
        print(f"[ERROR] --converted-dir ({out_abs}) contains the input dataset ({in_abs});")
        print("        a conversion rebuild could wipe the source dataset.")
        sys.exit(1)


def _conversion_reuse_gate(out_dir: Path, fp_id: str, fmt: str, task: str) -> str:
    """Manifest-driven reuse rules (decision 3). Returns 'reuse' or 'convert' (after wiping our own old output).

    An existing non-empty output folder without our manifest is treated as unmanaged: never modified,
    never wiped - hard error with guidance.
    """
    manifest_path = out_dir / CONVERSION_MANIFEST_NAME
    if out_dir.is_dir() and any(out_dir.iterdir()):
        manifest = None
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                manifest = None
        if manifest is None or not isinstance(manifest, dict):
            print(f"[ERROR] {out_dir} already exists but is not a conversion created by this tool (no usable {CONVERSION_MANIFEST_NAME}).")
            print("        It will not be modified or deleted: pass --converted-dir <fresh location>, or move/remove the existing folder.")
            sys.exit(1)
        reusable = (isinstance(manifest, dict) and manifest.get("format") == fmt
                    and manifest.get("task") == task and manifest.get("fingerprint_id") == fp_id
                    and (out_dir / "data.yaml").is_file())
        if reusable:
            print(f"[{fmt.upper()}] reusing previous conversion at {out_dir} (source fingerprint unchanged).")
            print(f"[{fmt.upper()}] previous manifest: task={manifest.get('task')}, image_counts={manifest.get('image_counts')}")
            return "reuse"
        print(f"[{fmt.upper()}] {out_dir} holds an earlier conversion but the source or requested task changed — rebuilding from scratch.")
        shutil.rmtree(out_dir)
    return "convert"


# -- VOC Conversion Helpers --------------------------------------------------------

_VOC_HELPER_MODULE = None


def _load_voc_helpers():
    """Import voc_to_yolo.py helpers by file path (plan decision 4); its __main__ guard keeps import side-effect free."""
    global _VOC_HELPER_MODULE
    if _VOC_HELPER_MODULE is None:
        voc_path = Path(__file__).resolve().parent.parent / "voc_to_yolo" / "voc_to_yolo.py"
        if not voc_path.is_file():
            print(f"[ERROR] VOC conversion needs the helpers in {voc_path}, but that file is missing.")
            sys.exit(1)
        import importlib.util
        spec = importlib.util.spec_from_file_location("voc_to_yolo_converters", voc_path)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            print(f"[ERROR] Could not load VOC helpers from {voc_path}: {e}")
            sys.exit(1)
        _VOC_HELPER_MODULE = module
    return _VOC_HELPER_MODULE


def looks_like_voc(input_dir: Path) -> bool:
    if next(input_dir.rglob("*.xml"), None) is not None:
        return True
    return any(p.is_file() and p.suffix.lower() == ".xml" for p in input_dir.rglob("*.XML"))


def _voc_pairs_in_split(split_dir: Path):
    """Pair <stem>.xml with same-stem image (flat, images/, or JPEGImages/ neighbours). Dedupes by stem."""
    pairs = []
    seen_stems = set()
    xml_files = sorted(split_dir.rglob("*.xml"))
    xml_stems = {p.stem for p in xml_files}
    unpaired_xmls = []
    for xp in xml_files:
        stem = xp.stem
        if stem in seen_stems:
            continue
        img = None
        for base in (xp.parent, xp.parent / "images", xp.parent / "JPEGImages",
                     xp.parent.parent / "images", xp.parent.parent / "JPEGImages"):
            for ext in sorted(IMAGE_EXTENSIONS):
                cand = base / f"{stem}{ext}"
                if cand.is_file():
                    img = cand
                    break
            if img is not None:
                break
        if img is None:
            unpaired_xmls.append(xp)
            continue
        seen_stems.add(stem)
        pairs.append((img, xp))
    paired_paths = {img for img, _ in pairs}
    unpaired_images = [f for f in sorted(split_dir.rglob("*"))
                       if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS and f not in paired_paths]
    return pairs, unpaired_images, unpaired_xmls


def convert_voc_dataset(input_dir: Path, args):
    """Pascal VOC -> YOLO conversion into a *_yolo copy. VOC is detect-only (no polygons/keypoints in XML)."""
    override = getattr(args, "task", "auto")
    if override in ("segment", "pose"):
        print(f"[ERROR] --task {override} is not supported for VOC datasets: VOC XML annotations are bounding boxes only.")
        print("        VOC conversion is always task=detect. Use a COCO or YOLO dataset for segmentation/pose training.")
        sys.exit(1)

    splits = {}
    warnings = {}
    for child in sorted(input_dir.iterdir()):
        if not child.is_dir():
            continue
        role = split_role(child.name)
        if role is None:
            if next(child.rglob("*.xml"), None) is not None:
                print(f"[VOC] warning: folder '{child.name}' holds VOC XMLs but is not a train/val/test split name; ignored.")
            continue
        if role in splits:
            print(f"[ERROR] Two folders map to the same split '{role}': {sorted(splits)[0]} and {child}.")
            sys.exit(1)
        splits[role] = child

    if "train" not in splits or "val" not in splits:
        print(f"[ERROR] This VOC dataset is not pre-split into train/val folders (found: {', '.join(sorted(splits)) or 'none'}).")
        print("        Auto-split it first, then pass the output folder to --input:")
        print("          python voc_to_yolo/voc_to_yolo.py --input <this folder> --output <split dataset>")
        sys.exit(1)

    helpers = _load_voc_helpers()
    role_pairs = {}
    for role, split_dir in splits.items():
        pairs, unpaired_images, unpaired_xmls = _voc_pairs_in_split(split_dir)
        role_pairs[role] = pairs
        warnings[role] = (unpaired_images, unpaired_xmls)

    for role in ("train", "val"):
        if not role_pairs.get(role):
            print(f"[ERROR] VOC split '{role}' has no image+XML pairs; nothing to convert.")
            sys.exit(1)

    all_pairs = [p for role in ("train", "val", "test") for p in role_pairs.get(role, [])]
    names = helpers.discover_classes(all_pairs)  # alphabetically sorted -> class ids (same rule as voc_to_yolo.py)
    if not names:
        print("[ERROR] No annotated objects found in any VOC XML; nothing to train on.")
        sys.exit(1)
    cls_map = {name: i for i, name in enumerate(names)}

    if args.converted_dir:
        out_dir = normalize_path(args.converted_dir)
    else:
        out_dir = input_dir.parent / f"{input_dir.name}{CONVERSION_DIR_SUFFIX}"

    _validate_converted_dir(input_dir, out_dir)
    fp = _coco_source_fingerprint(input_dir, out_dir)
    fp_id = fingerprint_hash(fp)
    action = _conversion_reuse_gate(out_dir, fp_id, "voc", "detect")
    if action == "reuse":
        return out_dir / "data.yaml", "detect", "reused-conversion"

    stats = {"parse_failed": 0, "unknown_name": 0}
    image_counts = {}
    print(f"[VOC] converting {input_dir} -> {out_dir} (task=detect, classes discovered alphabetically)")
    try:
        for role in ("train", "val", "test"):
            if role not in role_pairs:
                continue
            images_dir = out_dir / role / "images"
            labels_dir = out_dir / role / "labels"
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)
            copied = 0
            for img_p, xml_p in role_pairs[role]:
                dst = images_dir / img_p.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img_p, dst)
                copied += 1
                lines = []
                try:
                    img_w, img_h, objects = helpers.parse_voc_xml(xml_p)
                except (ValueError, Exception) as e:
                    stats["parse_failed"] += 1
                    print(f"[VOC] warning: {xml_p}: {e}; empty label file written for {img_p.name}.")
                else:
                    for ob in objects:
                        name = ob["name"]
                        if name not in cls_map:
                            stats["unknown_name"] += 1
                            continue
                        xc, yc, bw, bh = helpers.voc_bbox_to_yolo(ob["xmin"], ob["ymin"], ob["xmax"], ob["ymax"], img_w, img_h)
                        lines.append(f"{cls_map[name]} {_clamp01(xc):.6f} {_clamp01(yc):.6f} {_clamp01(bw):.6f} {_clamp01(bh):.6f}")
                (labels_dir / f"{img_p.stem}.txt").write_text("".join(ln + "\n" for ln in lines), encoding="utf-8")
            image_counts[role] = copied
            unpaired_images, unpaired_xmls = warnings.get(role, ([], []))
            if unpaired_images:
                print(f"[VOC] warning: {len(unpaired_images)} image(s) in split '{role}' are unpaired (no XML file) and were not copied: {_sample_list([str(f) for f in unpaired_images])}")
            if unpaired_xmls:
                print(f"[VOC] warning: {len(unpaired_xmls)} XML file(s) in split '{role}' have no matching image and were ignored: {_sample_list([str(f) for f in unpaired_xmls])}")

        yaml_out = {"train": "train/images", "val": "val/images"}
        if "test" in role_pairs:
            yaml_out["test"] = "test/images"
        yaml_out["nc"] = len(names)
        yaml_out["names"] = names
        (out_dir / "data.yaml").write_text(yaml.dump(yaml_out, default_flow_style=False, sort_keys=False), encoding="utf-8")

        manifest_data = {
            "version": 1,
            "format": "voc",
            "task": "detect",
            "source_dir": str(input_dir),
            "fingerprint_id": fp_id,
            "n_source_files": len(fp["files"]),
            "image_counts": image_counts,
            "created": local_iso(),
        }
        (out_dir / CONVERSION_MANIFEST_NAME).write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
    except OSError as e:
        print(f"[ERROR] Could not write the converted dataset to {out_dir}: {e}")
        print("        This location may be read-only or unavailable: pass --converted-dir <writable location> and re-run.")
        sys.exit(1)

    if stats["parse_failed"]:
        print(f"[VOC] {stats['parse_failed']} XML file(s) could not be parsed; their images got empty label files.")
    if stats["unknown_name"]:
        print(f"[VOC] warning: {stats['unknown_name']} object(s) named outside the discovered class set; skipped.")
    counts = ", ".join(f"{k}:{v}" for k, v in image_counts.items())
    print(f"[VOC] converted {sum(image_counts.values())} images ({counts}); classes: {names}")
    return out_dir / "data.yaml", "detect", "converted"


def resolve_dataset_input(args, logger):
    """Single entry point: --input dataset directory -> (data.yaml path, task)."""
    input_dir = normalize_path(args.input)

    if input_dir.is_file():
        print(f"[ERROR] --input must be a dataset FOLDER, not a file: {input_dir}")
        if input_dir.suffix.lower() in (".yaml", ".yml"):
            print("        Pass the dataset directory containing this yaml; its data.yaml is auto-detected.")
        print("        (To continue an existing run use: --resume --name <run> — the stored data path is reused.)")
        sys.exit(1)
    if not input_dir.is_dir():
        print(f"[ERROR] Dataset directory not found: {input_dir}")
        sys.exit(1)

    had_labels = False
    yaml_path = None
    fmt = action = None
    task_resolved = False

    yaml_files = sorted(p for p in input_dir.iterdir()
                        if p.is_file() and p.suffix.lower() in (".yaml", ".yml"))
    if yaml_files:
        # yaml + COCO json both present -> yaml wins (pass-through)
        yaml_path, task = _resolve_passthrough_yaml(yaml_files, args, logger)
        fmt, action = "yolo", "pass-through"
        had_labels = getattr(args, "dataset_had_labels", False)
    else:
        if looks_like_coco(input_dir):
            yaml_path, task, action = convert_coco_dataset(input_dir, args)
            fmt = "coco"
            task_resolved = True
        elif looks_like_voc(input_dir):
            yaml_path, task, action = convert_voc_dataset(input_dir, args)
            fmt = "voc"
            task_resolved = True
        else:
            synthesized = _try_synthesize_yaml(input_dir, args, logger)
            if synthesized:
                yaml_path, task = synthesized
                fmt, action = "yolo", "synthesized"
                had_labels = getattr(args, "dataset_had_labels", False)
            elif _looks_like_flat_folder(input_dir):
                print("[ERROR] Dataset folder looks flat (images and labels without a train/val split).")
                print("        Split it first:")
                print("          python flat_yolo_split/flat_yolo_split.py --input <this folder> --output <split dataset>")
                print("        Then pass the output dataset folder to --input.")
                sys.exit(1)
            else:
                found = sorted(e.name for e in input_dir.iterdir())
                print("[ERROR] Could not detect a supported dataset format in:")
                print(f"        {input_dir}")
                print(f"        Contents found: {', '.join(found[:20]) if found else '(empty)'}")
                print("        Supported formats: YOLO folder with data.yaml; YOLO train/val images+labels folders")
                print("        (data.yaml synthesized); COCO (_annotations.coco.json per split, or annotations/ + images/);")
                print("        Pascal VOC XML.")
                sys.exit(1)

    if not task_resolved:
        task = apply_task_override(task, args, had_labels)

    print(f"\n[Dataset Format]")
    print(f"  format: {fmt}")
    print(f"  task: {task}")
    print(f"  action: {action}")
    print(f"  yaml: {str(yaml_path).replace(chr(92), '/')}")
    return yaml_path, task


# -- Log Reporting Helpers ---------------------------------------------------------

def log_hardware(logger, hw, device, hw_fp):
    logger.log("[Hardware]")
    logger.log(f"  resolved device : {device}")
    logger.log(f"  GPU             : {hw['gpu_name']}")
    logger.log(f"  VRAM            : {hw['vram_gb']:.1f} GB" if hw['vram_gb'] else "  VRAM            : N/A (CPU)")
    logger.log(f"  CPU cores       : {hw['cpu_cores']}")
    logger.log(f"  RAM             : {hw['ram_gb']:.1f} GB" if hw['ram_gb'] else "  RAM             : Unknown")
    logger.log(f"  fingerprint id  : {fingerprint_hash(hw_fp)}")
    logger.log(f"  fingerprint     : {json.dumps(hw_fp, sort_keys=True, default=str)}")


def log_dataset(logger, ds_fp):
    logger.log("[Dataset fingerprint]")
    for key, val in ds_fp.items():
        logger.log(f"  {key}: {val}")
    logger.log(f"  fingerprint id  : {fingerprint_hash(ds_fp)}")


def log_checkpoint_status(logger, run_dir, tag=""):
    weights_dir = run_dir / "weights"
    logger.log(f"[Checkpoints]{(' ' + tag) if tag else ''} in {run_dir / 'weights'}")
    if not weights_dir.is_dir():
        logger.log("  no weights/ directory exists yet")
        return
    for fname in ("last.pt", "best.pt"):
        weight = weights_dir / fname
        if weight.is_file():
            stat = weight.stat()
            modified = local_iso(datetime.fromtimestamp(stat.st_mtime).astimezone())
            logger.log(f"  {fname}: {stat.st_size / 1e6:.1f} MB, modified {modified}")
        else:
            logger.log(f"  {fname}: MISSING")


def log_results_csv(logger, run_dir):
    csv_path = run_dir / "results.csv"
    if not csv_path.is_file():
        logger.log("[Results] no results.csv in run directory")
        return
    try:
        lines = csv_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    except OSError as e:
        logger.log(f"[Results] could not read results.csv: {e}")
        return
    if len(lines) < 2:
        logger.log("[Results] results.csv empty (no completed epochs)")
        return
    header = [h.strip() for h in lines[0].split(",")]
    last = [v.strip() for v in lines[-1].split(",")]
    logger.log(f"[Results] {len(lines) - 1} epoch row(s); final row: {', '.join(lines[-1].split(','))}")
    for name, value in zip(header, last):
        if "map" in name.lower() or "precision" in name.lower() or "recall" in name.lower():
            logger.log(f"[Results] {name} = {value}")


def log_resume_checkpoint_info(logger, last_pt: Path):
    """Log checkpoint metadata; returns the checkpoint dict (or None) for further checks."""
    try:
        ckpt = torch.load(str(last_pt), map_location="cpu", weights_only=False)
    except Exception as e:
        logger.log(f"[Resume] could not read checkpoint metadata: {e}")
        return None
    if not isinstance(ckpt, dict):
        logger.log("[Resume] checkpoint is not a Ultralytics dict; skipping metadata report")
        return None
    logger.log(f"[Resume] checkpoint epoch  : {ckpt.get('epoch')}")
    logger.log(f"[Resume] best fitness      : {ckpt.get('best_fitness')}")
    train_args = ckpt.get("train_args")
    if isinstance(train_args, dict):
        for key in ("model", "imgsz", "batch", "workers", "device", "epochs", "patience", "multi_scale", "mosaic"):
            logger.log(f"[Resume] ckpt {key:<12}: {train_args.get(key)}")
    return ckpt


# -- Config Display ------------------------------------------------------------------

def chosen_augmentation(chosen: dict) -> dict:
    """Full Ultralytics augmentation kwargs built from a calibrated selection."""
    aug = {
        "degrees": chosen.get("degrees", 0.0),
        "flipud": chosen.get("flipud", 0.0),
        "copy_paste": chosen["copy_paste"],
        "mixup": chosen["mixup"],
        "mosaic": chosen["mosaic"],
        "multi_scale": chosen["multi_scale"],
        "close_mosaic": chosen.get("close_mosaic", 60),
        "cos_lr": chosen.get("cos_lr", True),
        "nbs": chosen.get("nbs", 64),
    }
    if chosen.get("cache"):
        aug["cache"] = chosen["cache"]
    return aug


def explain_augmentation(task: str, aug: dict) -> str:
    """One-line why each aug knob is on/off."""
    bits = [
        f"degrees={aug['degrees']} ({'opt-in' if aug['degrees'] else 'off — no random rotation'})",
        f"flipud={aug['flipud']} ({'opt-in' if aug['flipud'] else 'off — no vertical flip'})",
    ]
    if task == "segment":
        bits.append(f"copy_paste={aug['copy_paste']} (segment: pastes polygons when >0)")
    else:
        bits.append(f"copy_paste={aug['copy_paste']} (off for {task}: CopyPaste is a no-op without polygons)")
    bits.append(f"mixup={aug['mixup']} ({'on' if aug['mixup'] else 'off'})")
    bits.append(f"mosaic={aug['mosaic']} ({'on' if aug['mosaic'] else 'off'})")
    bits.append(f"multi_scale={aug['multi_scale']} ({'on' if aug['multi_scale'] else 'off'})")
    return "; ".join(bits)


def format_duration(seconds) -> str:
    if seconds is None:
        return "unknown"
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def print_auto_config(logger, chosen, provenance, args, hw, ds_fp, device):
    aug = chosen_augmentation(chosen)
    task = ds_fp.get("task") or getattr(args, "dataset_task", "detect")

    logger.log("[Hardware Detected]")
    logger.log(f"  GPU        : {hw['gpu_name']}")
    logger.log(f"  VRAM       : {hw['vram_gb']:.1f} GB" if hw['vram_gb'] else "  VRAM       : N/A (CPU)")
    logger.log(f"  CPU cores  : {hw['cpu_cores']}")
    logger.log(f"  RAM        : {hw['ram_gb']:.1f} GB" if hw['ram_gb'] else "  RAM        : Unknown")
    alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    logger.log(f"  CUDA alloc : PYTORCH_CUDA_ALLOC_CONF={alloc_conf or '(unset)'}")

    logger.log(f"\n[Dataset]")
    logger.log(f"  train imgs : {ds_fp.get('image_count')}")
    logger.log(f"  task       : {task}")
    native = ds_fp.get("median_img_size")
    logger.log(f"  native res : {native}px" if native else "  native res : unknown")
    ls = ds_fp.get("label_stats") or {}
    if ls:
        logger.log(
            f"  label stats: median box {ls.get('median_box_w')}x{ls.get('median_box_h')} "
            f"(frac), p25 {ls.get('p25_box_w')}x{ls.get('p25_box_h')}, "
            f"boxes/img {ls.get('boxes_per_image')}, empty {ls.get('empty_label_fraction')}"
        )
    if ds_fp.get("val_label_stats"):
        warn_train_val_label_stats(logger, ls, ds_fp["val_label_stats"])

    logger.log(
        f"\n[Selection policy] quality-v2: largest model that fits a healthy batch "
        f"(>={MIN_HEALTHY_BATCH}), task-aware aug, object-resolving imgsz"
    )
    logger.log(
        "  lighten multi_scale/mosaic before dropping below the healthy batch floor; "
        "downgrades/step-ups only on measured probe/adaptive signals"
    )

    logger.log(f"\n[Selected Config] (source: {provenance})")
    logger.log(f"  model      : {chosen['model']}")
    logger.log(f"  imgsz      : {chosen['imgsz']}px")
    logger.log(f"  batch      : {chosen['batch']}")
    if chosen["batch"] < MIN_HEALTHY_BATCH:
        logger.log(f"  WARNING    : batch < {MIN_HEALTHY_BATCH} — BatchNorm quality may be degraded")
    if chosen["batch"] < 64:
        effective = chosen["batch"] * ((64 + chosen["batch"] - 1) // chosen["batch"])
        logger.log(
            f"  effective batch : {effective}  "
            "(nbs=64 gradient accumulation; does not fix small-batch BN noise)"
        )
    logger.log(f"  workers    : {chosen['workers']}")
    if chosen.get("cache"):
        logger.log(f"  cache      : {chosen['cache']}")
    if args.batch is not None or args.workers is not None or provenance == "force-unsafe":
        logger.log("  NOTE: config was pinned/forced by user flags; dynamic safety may have been overridden (--force-unsafe).")

    est = chosen.get("_est_sec_per_epoch")
    epochs = int(getattr(args, "epochs", 600) or 600)
    if est:
        logger.log(f"\n[Schedule ETA] ~{format_duration(est)}/epoch (from final probe, scaled by fraction)")
        logger.log(f"  full schedule ({epochs} epochs): ~{format_duration(est * epochs)}")
        logger.log("  note: patience / --time-budget can end sooner — ETA is a schedule ceiling, not a promise")
        if getattr(args, "time_budget", None):
            logger.log(f"  time budget : {args.time_budget} hours (Ultralytics time=)")

    logger.log(f"\n[Augmentation Config]")
    logger.log(f"  degrees      : {aug['degrees']}")
    logger.log(f"  flipud       : {aug['flipud']}")
    logger.log(f"  copy_paste   : {aug['copy_paste']}")
    logger.log(f"  mixup        : {aug['mixup']}")
    logger.log(f"  mosaic       : {aug['mosaic']}")
    ms_status = "(enabled)" if aug["multi_scale"] > 0 else "(disabled)"
    logger.log(f"  multi_scale  : {aug['multi_scale']}  {ms_status}")
    logger.log(f"  close_mosaic : {aug['close_mosaic']}")
    logger.log(f"  cos_lr       : {aug['cos_lr']}")
    logger.log(f"  why          : {explain_augmentation(task, aug)}")


# -- Resume calibration ----------------------------------------------------------
def native_cap_for_resume(native_size):
    return native_capped_imgsz(native_size) if native_size else None


def ckpt_config_unsafe(ckpt_cfg: dict, native_size, caps: dict) -> list:
    reasons = []
    imgsz = ckpt_cfg.get("imgsz")
    if native_size and isinstance(imgsz, int) and imgsz > int(native_size):
        reasons.append(f"imgsz {imgsz} > native {native_size}")
    ms = ckpt_cfg.get("multi_scale", 0.0)
    try:
        if float(ms) > 0.0:
            reasons.append(f"multi_scale {ms} > 0")
    except Exception:
        pass
    # caps check: batch/imgsz beyond caps? check gpu fraction via profile? use simple caps
    # profile caps handled by caller; here just note if imgsz exceeds standard cap
    return reasons


def calibrate_resume_config(args, logger, cache, hw, hw_fp, ds_fp, ckpt_cfg, run_dir,
                            yaml_path=None, device="0"):
    """
    Return (chosen_cfg dict, provenance) for a resume run.

    Quality-first resume parity: the checkpoint's own multi_scale is KEPT when a
    probe on the current hardware says the checkpoint augmentation combo is safe;
    it is reset to 0.0 only when the probe says unsafe or cannot verify it
    (no dataset yaml, CPU device, --no-memory-probe). --force-unsafe skips the
    probe and keeps the checkpoint values unvalidated (explicit opt-in).
    """
    native_size = ds_fp.get("median_img_size")
    # Resolve architecture model for probes/supervisor; actual training still resumes from last.pt.
    model_name = _infer_resume_architecture_model(run_dir, ckpt_cfg, task=ds_fp.get("task", "detect"))

    imgsz = ckpt_cfg.get("imgsz", 640)
    # cap imgsz to native
    if native_size and isinstance(imgsz, int) and imgsz > int(native_size):
        imgsz = int(native_size)
        # snap down to standard
        cands = [s for s in STANDARD_IMGSZ if s <= int(native_size)]
        if cands:
            imgsz = max(c for c in cands if c <= imgsz) if any(c <= imgsz for c in cands) else cands[0]
    batch = ckpt_cfg.get("batch", 4)
    workers = ckpt_cfg.get("workers", 2)
    # workers cap
    workers = min(int(workers), 4)
    try:
        multi_scale = float(ckpt_cfg.get("multi_scale", 0.0) or 0.0)
    except Exception:
        multi_scale = 0.0
    mosaic = ckpt_cfg.get("mosaic", 1.0)
    try:
        mosaic = float(mosaic)
    except Exception:
        mosaic = 1.0
    mixup = ckpt_cfg.get("mixup", 0.0)
    copy_paste = ckpt_cfg.get("copy_paste", 0.0)
    try:
        mixup = float(mixup)
        copy_paste = float(copy_paste)
    except Exception:
        mixup, copy_paste = 0.0, 0.0
    try:
        degrees = float(ckpt_cfg.get("degrees", 0.0) or 0.0)
    except Exception:
        degrees = 0.0
    try:
        flipud = float(ckpt_cfg.get("flipud", 0.0) or 0.0)
    except Exception:
        flipud = 0.0

    # unsafe reasons from original ckpt config (pre-probe flags; multi_scale is
    # probe-verified below instead of being zeroed unconditionally)
    unsafe_reasons = ckpt_config_unsafe(ckpt_cfg, native_size, {})
    # also if imgsz capped
    if ckpt_cfg.get("imgsz") != imgsz:
        unsafe_reasons.append(f"imgsz capped {ckpt_cfg.get('imgsz')} -> {imgsz}")
    if unsafe_reasons:
        logger.log("[RESUME CALIBRATE] original checkpoint config flagged for verification: " + "; ".join(unsafe_reasons))

    # Resume keeps the checkpoint's own augmentation (including degrees/flipud).
    chosen = build_chosen(model_name, imgsz, batch, workers, multi_scale, mosaic,
                          mixup=mixup, copy_paste=copy_paste, degrees=degrees, flipud=flipud,
                          cache=ckpt_cfg.get("cache", False))

    # cache lookup for resume selections
    cache_key = "resume_selections"
    pid = fingerprint_hash(hw_fp)
    ds_key = f"{fingerprint_hash(ds_fp)}|intent:{SELECTION_INTENT}"
    profile = cache.get("hardware_profiles", {}).get(pid) if cache.get("hardware_profiles") else None
    rsel = None
    if profile:
        rsel = profile.get(cache_key, {}).get(ds_key)
    if rsel:
        entry = rsel.get("config") if isinstance(rsel, dict) else None
        if isinstance(entry, dict) and rsel.get("verified") is not True:
            logger.log("[Resume Calibrate] cached resume selection was stored unverified (--force-unsafe); re-probing to revalidate.")
        elif isinstance(entry, dict):
            cached = dict(entry)
            cached_model = str(cached.get("model") or "")
            stale_checkpoint_model = (
                not cached_model
                or Path(cached_model).name.lower() in ("last.pt", "best.pt")
                or "weights" in cached_model.lower().replace("/", "\\")
            )
            if stale_checkpoint_model:
                logger.log(
                    f"[Resume Calibrate] ignoring stale cached resume selection model '{cached_model or '<empty>'}'; "
                    f"using inferred architecture '{model_name}'."
                )
                cached["model"] = model_name
                # keep inferred safe downscales
                cached["imgsz"] = min(int(cached.get("imgsz", chosen["imgsz"])), chosen["imgsz"])
                cached["multi_scale"] = 0.0
                cached["workers"] = min(int(cached.get("workers", chosen["workers"])), 4)
                try:
                    pid2, profile2 = get_hw_profile(cache, hw_fp)
                    profile2.setdefault("resume_selections", {})
                    profile2["resume_selections"][ds_key] = {
                        "config": dict(cached),
                        "verified": True,  # conservative inferred caps + multi_scale 0.0
                        "updated_at": local_iso(),
                    }
                    cache["updated_at"] = local_iso()
                    save_memory_cache(cache, logger)
                except Exception as e:
                    logger.log(f"[Resume Calibrate] cache rewrite failed: {e}")
            else:
                logger.log("[Resume Calibrate] using cached resume selection.")
            return cached, "cache"

    # --- probe-verify the checkpoint's augmentation combo (multi_scale) on the
    # current hardware; keep it when measured safe, reset only when unsafe or
    # unverifiable (supervisor's pre-attempt probe guards the rest).
    probe_verified = False
    if multi_scale > 0.0:
        if args.force_unsafe:
            logger.log(f"[Resume Calibrate] --force-unsafe: keeping checkpoint multi_scale {multi_scale} without probe validation (explicit opt-in).")
        elif getattr(args, "no_memory_probe", False):
            chosen["multi_scale"] = 0.0
            logger.log(f"[Resume Calibrate] --no-memory-probe: cannot verify checkpoint multi_scale {multi_scale}; reset to 0.0 (probes explicitly disabled).")
        elif yaml_path is None or not Path(yaml_path).exists():
            chosen["multi_scale"] = 0.0
            logger.log(f"[Resume Calibrate] no dataset yaml available to probe; cannot verify checkpoint multi_scale {multi_scale}; reset to 0.0.")
        elif not cuda_available_for(device):
            chosen["multi_scale"] = 0.0
            logger.log(f"[Resume Calibrate] device '{device}' is not CUDA: cannot probe checkpoint multi_scale {multi_scale}; reset to 0.0 (CPU resume stays conservative).")
        else:
            try:
                image_count = 100
                try:
                    data = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
                    tp = resolve_yaml_split_path(Path(yaml_path), str(data["train"]))
                    image_count = count_images(tp)
                except Exception:
                    pass
                fraction = probe_fraction(image_count)
                logger.log(
                    f"[Resume Calibrate] probing checkpoint augmentation: model={model_name} "
                    f"imgsz={chosen['imgsz']} batch={chosen['batch']} workers={chosen['workers']} "
                    f"multi_scale={multi_scale} mosaic={mosaic} mixup={mixup} copy_paste={copy_paste} "
                    f"fraction={fraction:.3f}"
                )
                peaks = probe_train_once(
                    model_name, device, Path(yaml_path), chosen["imgsz"], chosen["batch"],
                    multi_scale, mosaic, mixup, copy_paste, fraction, cuda_index_for(device),
                    workers=chosen["workers"],
                )
                profile = GUARD_PROFILE_CAPS.get(args.memory_profile, GUARD_PROFILE_CAPS["safe"])
                gpu_cap = (hw.get("vram_gb") or 0.0) * profile["gpu"]
                cpu_cap = (hw.get("ram_gb") or 0.0) * profile["cpu"]
                alloc, res, rss = peaks
                reasons = []
                if res is not None and res > gpu_cap:
                    reasons.append(f"GPU peak reserved {res:.2f} GB > {args.memory_profile} cap {gpu_cap:.2f} GB")
                if rss is not None and rss > cpu_cap:
                    reasons.append(f"CPU peak RSS {rss:.2f} GB > {args.memory_profile} cap {cpu_cap:.2f} GB")
                if reasons:
                    logger.log("[Resume Calibrate] checkpoint augmentation probe UNSAFE: " + "; ".join(reasons) + f"; resetting multi_scale {multi_scale} -> 0.0.")
                else:
                    probe_verified = True
                    logger.log(f"[Resume Calibrate] checkpoint augmentation probe SAFE (GPU reserved {res} GB, CPU RSS {rss} GB); keeping multi_scale {multi_scale}.")
            except Exception as e:
                logger.log(f"[Resume Calibrate] checkpoint augmentation probe failed ({_short(str(e))}); resetting multi_scale {multi_scale} -> 0.0.")
            if not probe_verified:
                chosen["multi_scale"] = 0.0
    elif args.force_unsafe:
        logger.log("[Resume Calibrate] --force-unsafe: calibrated-down resume config kept without probe validation.")

    # record cache
    try:
        pid2, profile2 = get_hw_profile(cache, hw_fp)
        profile2.setdefault("resume_selections", {})
        profile2["resume_selections"][ds_key] = {
            "config": dict(chosen),
            # verified: safe to reuse without re-probing. True when the config was
            # probe-measured safe, or its multi_scale is 0.0 (kept checkpoint value,
            # probe-proven unsafe fallback, or conservative unprobed cap fallback).
            # False only when --force-unsafe kept an unvalidated multi_scale > 0.
            "verified": bool(chosen.get("multi_scale", 0.0) == 0.0 or probe_verified),
            "updated_at": local_iso(),
        }
        cache["updated_at"] = local_iso()
        save_memory_cache(cache, logger)
    except Exception as e:
        logger.log(f"[Resume Calibrate] cache save failed: {e}")

    provenance = "resume-calibrated"
    logger.log(f"[Resume Calibrate] chosen: model={chosen['model']} imgsz={chosen['imgsz']} batch={chosen['batch']} workers={chosen['workers']} multi_scale={chosen['multi_scale']} mosaic={chosen['mosaic']}")
    return chosen, provenance


def aug_step_up_candidate(cfg: dict, task: str = "detect"):
    """
    Next-heavier task aug ladder rung for cfg's augmentation combo, or None when
    already heaviest or off-ladder (e.g. a checkpoint's custom aug values).
    Never changes the model.
    """
    ladder = aug_ladder_for_task(task)
    cur = (cfg["mosaic"], cfg["copy_paste"], cfg["mixup"], cfg["multi_scale"])
    cur_r = (round(cur[0], 2), round(cur[1], 2), round(cur[2], 2), round(cur[3], 2))
    for i, t in enumerate(ladder):
        if (round(t[0], 2), round(t[1], 2), round(t[2], 2), round(t[3], 2)) == cur_r:
            if i == 0:
                return None
            t = ladder[i - 1]
            new = dict(cfg)
            new["mosaic"], new["copy_paste"], new["mixup"], new["multi_scale"] = t
            return new
    return None


def aug_step_down_candidate(cfg: dict, task: str = "detect"):
    """Next-lighter task aug ladder rung, or None if already lightest / off-ladder."""
    ladder = aug_ladder_for_task(task)
    cur = (cfg["mosaic"], cfg["copy_paste"], cfg["mixup"], cfg["multi_scale"])
    cur_r = (round(cur[0], 2), round(cur[1], 2), round(cur[2], 2), round(cur[3], 2))
    for i, t in enumerate(ladder):
        if (round(t[0], 2), round(t[1], 2), round(t[2], 2), round(t[3], 2)) == cur_r:
            if i + 1 >= len(ladder):
                return None
            t = ladder[i + 1]
            new = dict(cfg)
            new["mosaic"], new["copy_paste"], new["mixup"], new["multi_scale"] = t
            return new
    return None


# -- Adaptive supervisor ---------------------------------------------------------
def run_attempt(args, logger, run_dir, last_pt, chosen, target_epochs, device, guard,
                yaml_path, ckpt_cfg, mode, attempt_no, state):
    """Run a single training attempt. Returns AttemptTracker."""
    run_dir = Path(run_dir)
    weights_dir = run_dir / "weights"
    # snapshot backups before attempt
    snapshot_backups(run_dir, logger)

    # Determine resume vs fresh
    is_resume = (mode == "resume") or (last_pt is not None and Path(last_pt).is_file() and attempt_no > 1) or (mode == "fresh" and attempt_no > 1)
    # For fresh first attempt, no checkpoint; subsequent attempts resume from last.pt
    if mode == "fresh" and attempt_no == 1:
        resume_pt = None
    else:
        resume_pt = last_pt if (last_pt and Path(last_pt).is_file()) else (weights_dir / "last.pt")
        if not (resume_pt and Path(resume_pt).is_file()):
            resume_pt = None
    # Terminal Ultralytics checkpoint (epoch == -1) cannot be resumed. Do NOT claim the
    # target schedule completed here; the supervisor classifies the real stop epoch.
    if resume_pt is not None and ckpt_completed_epoch(resume_pt) == -1:
        logger.log(f"[Attempt {attempt_no}] checkpoint {resume_pt} is finalized by Ultralytics; nothing to resume.")
        tracker = AttemptTracker(
            logger=logger, guard=guard, run_dir=run_dir, target_epochs=target_epochs,
            intended_cfg=dict(chosen), state=state,
            on_emergency=None, on_adaptive=None, on_complete=None, on_crash=None,
        )
        tracker.status = "attempt-terminal"
        return tracker

    # Build forced_args for the guarded trainer
    # Determine optimizer to force for consistency across attempts
    opt_name = "MuSGD"
    if resume_pt is not None and Path(resume_pt).is_file():
        try:
            _c = torch.load(str(resume_pt), map_location="cpu", weights_only=False)
            _opt = _c.get("optimizer", {})
            groups = _opt.get("param_groups", [])
            if any(g.get("param_group") == "muon" or g.get("use_muon") for g in groups):
                opt_name = "MuSGD"
            else:
                opt_name = "AdamW"
        except Exception:
            opt_name = "MuSGD"
    # Optimizer / lr pins match Ultralytics optimizer=auto for large iteration
    # counts (MuSGD, lr0=0.01, momentum=0.9, warmup_bias_lr=0.0). Kept for
    # reproducibility across adaptive resume attempts — see ultralytics
    # engine/trainer.py build_optimizer auto path.
    forced_args = {
        "imgsz": int(chosen["imgsz"]),
        "batch": int(chosen["batch"]),
        "workers": int(chosen["workers"]),
        "multi_scale": float(chosen["multi_scale"]),
        "mosaic": float(chosen["mosaic"]),
        "mixup": float(chosen["mixup"]),
        "copy_paste": float(chosen["copy_paste"]),
        "degrees": float(chosen.get("degrees", 0.0)),
        "flipud": float(chosen.get("flipud", 0.0)),
        "data": str(yaml_path) if yaml_path is not None else (ckpt_cfg.get("data") if isinstance(ckpt_cfg, dict) else None),
        "device": device,
        "epochs": int(target_epochs),
        "patience": args.patience,
        "save_dir": str(run_dir),
        "optimizer": opt_name,
        "lr0": 0.01,
        "momentum": 0.9,
        "warmup_bias_lr": 0.0,
    }
    if chosen.get("cache"):
        forced_args["cache"] = chosen["cache"]
    if getattr(args, "time_budget", None):
        forced_args["time"] = float(args.time_budget)
    # remove None data
    if forced_args.get("data") is None:
        forced_args.pop("data", None)

    tracker = AttemptTracker(
        logger=logger, guard=guard, run_dir=run_dir, target_epochs=target_epochs,
        intended_cfg=dict(chosen), state=state,
        on_emergency=None, on_adaptive=None, on_complete=None, on_crash=None,
        patience=int(getattr(args, "patience", 100) or 100),
    )

    # Every attempt targets the full schedule. Mid-run stops (adaptive pressure,
    # emergency, crash) are handled by the tracker + supervisor downgrade/resume
    # cycle — not by a per-attempt epoch cap. The old provisional attempt_epochs
    # logic was dead on resumes anyway: guarded forced_args pins Ultralytics'
    # args.epochs to target_epochs in check_resume.

    # build trainer via partial (task-correct: -seg/-pose weights need SegmentationTrainer/PoseTrainer;
    # the detect trainer on seg/pose models crashes its own val NMS with tuple predictions)
    attempt_task = resolve_attempt_task(args, ckpt_cfg, chosen)
    trainer_cls = functools.partial(guarded_trainer_for_task(attempt_task), forced_args=forced_args, tracker=tracker, logger=logger)

    # callbacks
    cb_map = tracker.callbacks()
    # wrap guard into callbacks via model.add_callback style? We pass via trainer _callbacks
    # DetectionTrainer expects _callbacks dict of event->list of callables
    callbacks_for_trainer = {}
    for ev, fn in cb_map.items():
        callbacks_for_trainer[ev] = [fn]
    # also keep monitor epoch end
    # guard sampling thread started by caller

    status = "running"
    model = None
    try:
        if resume_pt is not None:
            model = YOLO(str(resume_pt))
        else:
            model = YOLO(str(resolve_pretrained_weights(chosen["model"])))
        # attach guard epoch callback
        model.add_callback("on_train_epoch_end", guard.on_epoch_end)
        guard.start()
        tracker_state = {"resume": resume_pt is not None}
        # build args for trainer
        train_kwargs = {}
        if resume_pt is not None:
            train_kwargs["resume"] = True
            train_kwargs["model"] = str(resume_pt)
        else:
            train_kwargs["data"] = str(yaml_path) if yaml_path is not None else forced_args.get("data")
            train_kwargs["model"] = str(resolve_pretrained_weights(chosen["model"]))
            train_kwargs["epochs"] = int(target_epochs)
            train_kwargs["imgsz"] = int(chosen["imgsz"])
            train_kwargs["batch"] = int(chosen["batch"])
            train_kwargs["workers"] = int(chosen["workers"])
            train_kwargs["device"] = device
            train_kwargs["project"] = str(RUNS_DIR)
            train_kwargs["name"] = run_dir.name
            train_kwargs["exist_ok"] = True
            train_kwargs["save_dir"] = str(run_dir)
            train_kwargs["patience"] = args.patience
            train_kwargs["save"] = True
            train_kwargs["plots"] = True
            train_kwargs["verbose"] = True
            train_kwargs.update(chosen_augmentation(chosen))
            if getattr(args, "time_budget", None):
                train_kwargs["time"] = float(args.time_budget)

        if resume_pt is not None:
            # Resume extends from the checkpoint epoch to the full target schedule;
            # guarded forced_args pin args.epochs to the same value after check_resume.
            train_kwargs["epochs"] = int(target_epochs)
            if getattr(args, "time_budget", None):
                train_kwargs["time"] = float(args.time_budget)

        # Inject trainer class via model.train(trainer=...)
        # callbacks attach via model.add_callback; trainer class passed separately below
        for ev, fns in callbacks_for_trainer.items():
            for fn in fns:
                model.add_callback(ev, fn)

        logger.log(f"[Attempt {attempt_no}] starting: task={attempt_task} trainer={guarded_trainer_for_task(attempt_task).__name__} resume={resume_pt is not None} epochs={train_kwargs.get('epochs')} batch={chosen['batch']} imgsz={chosen['imgsz']}")
        try:
            model.train(trainer=trainer_cls, **train_kwargs)
        except KeyboardInterrupt:
            tracker.status = "interrupted"
            logger.log(f"[Attempt {attempt_no}] interrupted by user.")
        except Exception as e:
            tracker.crash_exc = e
            if looks_like_oom(e):
                tracker.status = "crashed"
                logger.log(f"[Attempt {attempt_no}] OOM crash: {_short(str(e))}")
            else:
                tracker.status = "crashed"
                logger.log(f"[Attempt {attempt_no}] crashed: {_short(str(e))}")
                logger.log(traceback.format_exc())
        finally:
            try:
                guard.stop()
            except Exception:
                pass

        if tracker.status == "running":
            tracker.on_train_end(None)
        # if emergency triggered, status already set by tracker? on_train_end sets status; but emergency flag set. Re-evaluate:
        if tracker.emergency_triggered:
            tracker.status = "partial-emergency"
        if tracker.adaptive_triggered and not tracker.emergency_triggered:
            tracker.status = "stopped-adaptive"

    except Exception as e:
        tracker.status = "crashed"
        tracker.crash_exc = e
        logger.log(f"[Attempt {attempt_no}] unexpected: {e}")
        logger.log(traceback.format_exc())
        try:
            guard.stop()
        except Exception:
            pass

    # post-attempt: sync batch from trainer args? best-effort already in tracker
    # update state completed_epoch
    ce = tracker.completed_epoch
    if ce is None:
        # derive from checkpoint
        rp = run_dir / "weights" / "last.pt"
        if rp.is_file():
            ce = ckpt_completed_epoch(rp)
    if isinstance(ce, int) and ce >= 0:
        state["completed_epoch"] = ce
    state["last_attempt_partial"] = (tracker.status == "partial-emergency")
    # record history
    state.setdefault("history", []).append({
        "attempt": attempt_no,
        "status": tracker.status,
        "completed_epoch": ce,
        "config": dict(chosen),
    })
    write_adaptive_state(run_dir, state)

    # if partial-emergency, restore backups (attempt failed) -- but keep state for trim
    if tracker.status == "partial-emergency":
        logger.log("[Adaptive] partial epoch detected; restoring backups to pre-attempt state.")
        restore_backups(run_dir, logger)
        # The backup restore rolled this attempt's epochs back on disk. Rewind the
        # recorded evidence to what the disk actually represents, otherwise the
        # pending-hidden staleness gate (pending_resumable_checkpoint) sees an inflated
        # completed_epoch and rejects the hidden snapshot that matches the durable
        # checkpoint, terminalizing an otherwise resumable run.
        durable = durable_progress_epoch(run_dir, state)
        if durable is not None and isinstance(ce, int) and ce >= 0 and ce > durable:
            logger.log(f"[Adaptive] rewinding partial-emergency evidence {ce} -> {durable} "
                       "(backup restore rolled the attempt's epochs back).")
            ce = durable
            state["completed_epoch"] = ce
            hist = state.get("history") or []
            if hist and isinstance(hist[-1], dict) and hist[-1].get("attempt") == attempt_no:
                hist[-1]["completed_epoch"] = ce
            write_adaptive_state(run_dir, state)
        # trim csv to the durable completed epoch
        if isinstance(ce, int) and ce >= 0:
            trim_results_csv(run_dir, ce, logger)

    return tracker


def run_adaptive_supervisor(args, logger, run_dir, run_name, last_pt, chosen, target_epochs,
                            device, hw, guard, state, yaml_path, ckpt_cfg, mode):
    run_dir = Path(run_dir)
    state = state or {}
    initial_ckpt_epoch = ckpt_completed_epoch(last_pt) if (last_pt and Path(last_pt).is_file()) else 0
    state.setdefault("completed_epoch", initial_ckpt_epoch if initial_ckpt_epoch >= 0 else 0)
    # Evidence is the source of truth at init: legacy runs stored target_epochs in
    # completed_epoch when an adaptive/patience stop finalized the checkpoint early
    # (the epoch == -1 bug), and stale foreign state files can carry the same lie.
    derived = derive_actual_epoch(
        run_dir, state,
        ckpt_epoch=initial_ckpt_epoch if initial_ckpt_epoch >= 0 else None,
        target_epochs=target_epochs,
    )
    stored = state.get("completed_epoch")
    if derived != stored:
        state["completed_epoch"] = derived
        if isinstance(stored, int):
            logger.log(f"[Supervisor] corrected completed_epoch {stored} -> {derived} from run evidence "
                       "(tracker/history/results.csv).")
    state.setdefault("target_epochs", target_epochs)
    state.setdefault("provisional", True)
    state.setdefault("stable_epochs", 0)
    state.setdefault("downgrade_count", 0)
    state.setdefault("history", [])
    task = getattr(args, "dataset_task", None) or (ckpt_cfg.get("task") if isinstance(ckpt_cfg, dict) else None) or "detect"

    def downgrade_config(cfg):
        """
        Quality-v2 runtime downgrade (never changes model):
          multi_scale -> 0, then batch down to healthy floor, then imgsz, then workers,
          then lighter mosaic/mixup rung, and only last batch below the floor.
        """
        # 1. multi_scale -> 0
        if float(cfg.get("multi_scale", 0.0) or 0.0) > 0.0:
            new = dict(cfg)
            new["multi_scale"] = 0.0
            return new, "multi_scale"
        # 2. batch down while staying >= MIN_HEALTHY_BATCH
        b = int(cfg["batch"])
        healthy_lower = [x for x in BATCH_LADDER_DOWN if MIN_HEALTHY_BATCH <= x < b]
        if healthy_lower:
            new = dict(cfg)
            new["batch"] = max(healthy_lower)
            logger.log(f"[Adaptive] BATCH CHANGE (downgrade toward floor): {b} -> {new['batch']}")
            return new, "batch"
        # 3. imgsz down
        cands = [s for s in STANDARD_IMGSZ if s < cfg["imgsz"]]
        if cands:
            new = dict(cfg)
            new["imgsz"] = max(cands)
            return new, "imgsz"
        # 4. workers down
        if cfg["workers"] > 0:
            new = dict(cfg)
            new["workers"] = max(0, cfg["workers"] // 2)
            if new["workers"] == cfg["workers"]:
                new["workers"] = 0
            return new, "workers"
        # 5. lighter aug rung (mosaic/mixup/copy_paste) before unhealthy batch
        lighter = aug_step_down_candidate(cfg, task=task)
        if lighter is not None:
            return lighter, "aug"
        # 6. batch below the healthy floor
        below = [x for x in BATCH_LADDER_DOWN if x < b]
        if below:
            new = dict(cfg)
            new["batch"] = max(below)
            warn = (
                f"[WARNING] BATCH CHANGE below healthy floor {MIN_HEALTHY_BATCH}: "
                f"{b} -> {new['batch']} (BatchNorm quality degraded)"
            )
            logger.log(warn)
            print(warn)
            return new, "batch-below-floor"
        return None, None

    probe_cache = {}

    def probe_config_safe(cfg):
        # quick probe validation; reuse probe_train_once if possible
        probe_model = cfg.get("model")
        try:
            pm = str(probe_model or "")
            if Path(pm).name.lower() in ("last.pt", "best.pt") or "weights" in pm.lower().replace("/", "\\"):
                probe_model = _infer_resume_architecture_model(run_dir, ckpt_cfg, task=task)
        except Exception:
            pass
        if getattr(args, "no_memory_probe", False):
            logger.log(f"[Probe] --no-memory-probe set; skipping probe for {probe_model} (cfg={cfg}).")
            return True
        dev = device
        use_gpu = cuda_available_for(dev)
        if not use_gpu:
            logger.log(f"[Probe] non-GPU device '{dev}'; skipping probe for {probe_model}.")
            return True
        sig_cfg = {k: v for k, v in cfg.items() if not str(k).startswith("_")}
        sig = config_signature(**sig_cfg)
        if sig in probe_cache:
            verdict = probe_cache[sig]
            logger.log(f"[Probe] reusing measured verdict for {probe_model} @ {cfg.get('imgsz')}px batch {cfg.get('batch')} (no new probe run).")
            return verdict
        cuda_index = cuda_index_for(dev)
        image_count = 100
        try:
            if yaml_path and Path(yaml_path).exists():
                data = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
                tp = resolve_yaml_split_path(yaml_path, str(data["train"]))
                image_count = count_images(tp)
        except Exception:
            pass
        fraction = probe_fraction(image_count)
        logger.log(
            f"[Probe] validating model={probe_model} imgsz={cfg['imgsz']} batch={cfg['batch']} "
            f"workers={cfg['workers']} multi_scale={cfg['multi_scale']} mosaic={cfg['mosaic']} fraction={fraction}"
        )
        try:
            peaks = probe_train_once(
                probe_model, dev, yaml_path, cfg["imgsz"], cfg["batch"], cfg["multi_scale"],
                cfg["mosaic"], cfg["mixup"], cfg["copy_paste"], fraction, cuda_index, workers=cfg["workers"],
                degrees=cfg.get("degrees", 0.0), flipud=cfg.get("flipud", 0.0),
                cache=cfg.get("cache", False),
            )
            # evaluate against profile caps — SAFE is a memory estimate only
            profile = GUARD_PROFILE_CAPS.get(args.memory_profile, GUARD_PROFILE_CAPS["safe"])
            vram = hw.get("vram_gb") or 0.0
            ram = hw.get("ram_gb") or 0.0
            gpu_cap = vram * profile["gpu"]
            cpu_cap = ram * profile["cpu"]
            alloc, res, rss = peaks[0], peaks[1], peaks[2]
            if res is not None and res > gpu_cap:
                logger.log(f"[Probe] UNSAFE {probe_model}: reserved {res:.2f} GB > cap {gpu_cap:.2f} GB")
                probe_cache[sig] = False
                return False
            if rss is not None and rss > cpu_cap:
                logger.log(f"[Probe] UNSAFE {probe_model}: CPU RSS {rss:.2f} GB > cap {cpu_cap:.2f} GB")
                probe_cache[sig] = False
                return False
            logger.log(
                f"[Probe] SAFE {probe_model}: alloc={alloc} reserved={res} rss={rss} "
                f"(caps gpu={gpu_cap:.2f} cpu={cpu_cap:.2f}; memory estimate only, not convergence)"
            )
            probe_cache[sig] = True
            return True
        except Exception as e:
            logger.log(f"[Probe] FAILED for {probe_model}: {type(e).__name__}: {_short(str(e))}")
            probe_cache[sig] = False
            return False

    # Augmentation/batch step-ups are only meaningful where a real probe can
    # measure them; on CPU or --no-memory-probe there is no measured verdict.
    step_up_probing_ok = (not getattr(args, "no_memory_probe", False)) and cuda_available_for(device)

    current_ckpt = last_pt
    initial_hidden = pending_resumable_checkpoint(run_dir, state)
    if initial_hidden is not None and not valid_resume_checkpoint(current_ckpt):
        current_ckpt = initial_hidden
        mode = "resume"
        logger.log("[Supervisor] weights/last.pt is not resumable (finalized); continuing from hidden .adaptive/resumable_last.pt.")
    attempt_no = 0
    max_attempts = 1 + MAX_ADAPTIVE_CONFIG_DOWNGRADES
    current_cfg = dict(chosen)
    final_status = "no-safe-config"

    def finish_completed_run():
        """Target schedule really reached: finalize state and drop the hidden snapshot."""
        state["completed_epoch"] = target_epochs
        state["provisional"] = False
        state["resume_pending"] = False
        state.pop("last_checkpoint_finalized", None)
        hidden_path = adaptive_dir(run_dir) / RESUMABLE_LAST_NAME
        try:
            hidden_path.unlink(missing_ok=True)
        except OSError as e:
            # Windows busy-file safety: keep the file but make resume ignore it.
            state["hidden_checkpoint_invalid"] = True
            logger.log(f"[Adaptive] hidden snapshot cleanup failed ({e}); marked invalid, resume will ignore it.")

    def finish_terminal_run(ckpt_epoch, actual_epoch, status_name):
        """Terminal stop below target: record the real epoch, never target_epochs."""
        state["completed_epoch"] = max(0, int(actual_epoch))
        if ckpt_epoch == -1:
            state["last_checkpoint_finalized"] = True
        state["resume_pending"] = False
        logger.log(f"[Supervisor] {status_name} after {max(0, int(actual_epoch))}/{target_epochs} epochs "
                   f"(target schedule not reached).")

    while attempt_no < max_attempts:
        attempt_no += 1
        # ensure safe config before attempt (probe validate)
        if not probe_config_safe(current_cfg) and not getattr(args, "force_unsafe", False):
            # downgrade until safe
            steps = 0
            while not probe_config_safe(current_cfg) and steps < MAX_ADAPTIVE_CONFIG_DOWNGRADES:
                res = downgrade_config(current_cfg)
                if res[0] is None:
                    break
                current_cfg, reason = res
                logger.log(f"[Adaptive] pre-attempt downgrade ({reason}): {current_cfg}")
                steps += 1
            if not probe_config_safe(current_cfg) and not getattr(args, "force_unsafe", False):
                hidden_pre = pending_resumable_checkpoint(run_dir, state)
                if hidden_pre is not None:
                    # A valid hidden snapshot is a real continuation point: do NOT
                    # terminalize the run; manual --resume will use the hidden ckpt
                    # with the last downgraded safe config.
                    he = ckpt_completed_epoch(hidden_pre)
                    state["completed_epoch"] = max(0, derive_actual_epoch(
                        run_dir, state,
                        ckpt_epoch=he if isinstance(he, int) and he >= 0 else None,
                        target_epochs=target_epochs))
                    final_status = "no-safe-config"
                    write_adaptive_state(run_dir, state)
                    logger.log("[Supervisor] no safe config for current hardware; run stays resumable "
                               f"from {ADAPTIVE_DIRNAME}/{RESUMABLE_LAST_NAME}.")
                    break
                logger.log("[Adaptive] no safe config found after downgrades.")
                weights_last = run_dir / "weights" / "last.pt"
                wl_epoch = ckpt_completed_epoch(weights_last) if weights_last.is_file() else None
                actual_pre = derive_actual_epoch(
                    run_dir, state,
                    ckpt_epoch=wl_epoch if isinstance(wl_epoch, int) and wl_epoch >= 0 else None,
                    target_epochs=target_epochs)
                finish_terminal_run(wl_epoch, actual_pre, "no-safe-config")
                final_status = "no-safe-config"
                write_adaptive_state(run_dir, state)
                break

        # provisional handling
        state["provisional"] = state.get("stable_epochs", 0) < 1
        tracker = run_attempt(args, logger, run_dir, current_ckpt, current_cfg, target_epochs,
                              device, guard, yaml_path, ckpt_cfg, mode, attempt_no, state)
        st = tracker.status
        logger.log(f"[Supervisor] attempt {attempt_no} status={st}")

        # Resolve the checkpoint the NEXT attempt resumes from. A clean adaptive stop
        # keeps a hidden unstripped snapshot; the public weights/last.pt was finalized
        # (epoch = -1) by final_eval(), so it must not be the continuation point.
        hidden_ckpt = pending_resumable_checkpoint(run_dir, state)
        if st == "stopped-adaptive" and hidden_ckpt is not None:
            current_ckpt = hidden_ckpt
        else:
            current_ckpt = run_dir / "weights" / "last.pt"
        ckpt_epoch = ckpt_completed_epoch(current_ckpt) if current_ckpt.is_file() else None
        # actual_epoch is derived from tracker/history/results.csv evidence;
        # epoch == -1 NEVER maps to "completed the target schedule".
        actual_epoch = derive_actual_epoch(run_dir, state, tracker=tracker, ckpt_epoch=ckpt_epoch,
                                           target_epochs=target_epochs)

        if st == "completed-target" or (actual_epoch >= target_epochs
                                        and st not in ("partial-emergency", "crashed", "interrupted", "attempt-terminal")):
            final_status = "completed-target"
            state["stable_epochs"] = state.get("stable_epochs", 0) + 1
            finish_completed_run()
            write_adaptive_state(run_dir, state)
            logger.log(f"[Supervisor] target schedule reached ({actual_epoch}/{target_epochs} epochs) after attempt {attempt_no}.")
            break

        if st == "completed-attempt" and ckpt_epoch == -1:
            # Patience/early stop: Ultralytics finalized last.pt; this is NOT target completion.
            final_status = "early-stopped"
            finish_terminal_run(ckpt_epoch, actual_epoch, "early-stopped")
            write_adaptive_state(run_dir, state)
            break

        if st in ("completed-attempt",):
            # attempt completed cleanly with a still-resumable checkpoint
            state["completed_epoch"] = max(0, actual_epoch)
            if state.get("provisional", True):
                # mark proven if no adaptive/emergency/oom
                if not tracker.adaptive_triggered and not tracker.emergency_triggered and not tracker.oom_auto_reduced:
                    state["provisional"] = False
                    state["stable_epochs"] = 1
                    logger.log("[Adaptive] config proven after a clean attempt; step-up eligibility enabled.")
                else:
                    state["provisional"] = True
                    if tracker.oom_auto_reduced:
                        probe_cache.clear()  # attempt halved batch mid-run; measurements stale
            state["stable_epochs"] = state.get("stable_epochs", 0) + (1 if not tracker.adaptive_triggered and not tracker.emergency_triggered else 0)
            write_adaptive_state(run_dir, state)
            # step-up check
            if (state.get("stable_epochs", 0) >= STABLE_EPOCHS_BEFORE_STEP_UP
                    and guard.peak_reserved() < (hw.get("vram_gb", 0) * STABLE_STEP_UP_GPU_FRACTION)
                    and tracker.oom_auto_reduced is False):
                if not step_up_probing_ok:
                    logger.log("[Adaptive] step-up skipped: probing unavailable (CPU device or --no-memory-probe); step-ups require measured probe verdicts.")
                else:
                    # Quality-v2 step-up = reverse of downgrade: batch toward floor,
                    # workers, imgsz, batch above floor, then heavier aug / multi_scale.
                    # Never changes model. Every candidate must pass probe_config_safe.
                    stepped = False
                    b = int(current_cfg["batch"])
                    # batch up toward / past the healthy floor
                    next_batches = [x for x in BATCH_LADDER_UP if x > b]
                    if next_batches:
                        new_batch = next_batches[0]
                        cand = dict(current_cfg)
                        cand["batch"] = new_batch
                        if probe_config_safe(cand):
                            logger.log(f"[Adaptive] BATCH CHANGE (step-up): {b} -> {new_batch}")
                            state.setdefault("history", []).append({
                                "event": "batch-step-up", "from": b, "to": new_batch,
                                "epoch": state.get("completed_epoch"),
                            })
                            current_cfg = cand
                            stepped = True
                        else:
                            logger.log(f"[Adaptive] batch step-up to {new_batch} probe-rejected.")
                    if not stepped and current_cfg.get("workers", 0) < 8:
                        cand = dict(current_cfg)
                        cand["workers"] = min(8, max(2, current_cfg["workers"] * 2 or 2))
                        if cand["workers"] != current_cfg["workers"] and probe_config_safe(cand):
                            logger.log(f"[Adaptive] step-up approved (workers): {current_cfg['workers']} -> {cand['workers']}")
                            current_cfg = cand
                            stepped = True
                    if not stepped:
                        higher = [s for s in STANDARD_IMGSZ if s > current_cfg["imgsz"]]
                        if higher:
                            cand = dict(current_cfg)
                            cand["imgsz"] = min(higher)
                            if probe_config_safe(cand):
                                logger.log(f"[Adaptive] step-up approved (imgsz): {current_cfg['imgsz']} -> {cand['imgsz']}")
                                current_cfg = cand
                                stepped = True
                    if not stepped:
                        cand = aug_step_up_candidate(current_cfg, task=task)
                        if cand is not None and probe_config_safe(cand):
                            cur_a = (current_cfg['mosaic'], current_cfg['copy_paste'], current_cfg['mixup'], current_cfg['multi_scale'])
                            new_a = (cand['mosaic'], cand['copy_paste'], cand['mixup'], cand['multi_scale'])
                            logger.log(f"[Adaptive] step-up approved (augmentation ladder): {cur_a} -> {new_a}")
                            current_cfg = cand
                            stepped = True
                    if not stepped:
                        logger.log("[Adaptive] step-up not approved (unsafe or not eligible).")
            # continue loop (next attempt resumes and runs on)
            mode = "resume"
            continue

        if st == "stopped-adaptive":
            if hidden_ckpt is not None:
                # Clean epoch-boundary stop with a pending hidden snapshot: downgrade,
                # then resume the SAME run from the hidden checkpoint.
                logger.log("[Adaptive] adaptive pressure; downgrading config and resuming from hidden .adaptive/resumable_last.pt.")
                probe_cache.clear()  # runtime pressure invalidates earlier measured verdicts
                res = downgrade_config(current_cfg)
                if res[0] is None:
                    # No safe downgrade remains. Manual --resume must not silently retry
                    # the config that already caused memory pressure.
                    final_status = "stopped-adaptive"
                    finish_terminal_run(ckpt_epoch, actual_epoch, "stopped-adaptive")
                    write_adaptive_state(run_dir, state)
                    logger.log("[Supervisor] no safe downgrade remains; the run is finished.")
                    break
                current_cfg, reason = res
                state["downgrade_count"] = state.get("downgrade_count", 0) + 1
                state["provisional"] = True
                state.setdefault("history", []).append({
                    "event": "adaptive-downgrade", "reason": reason,
                    "config": {k: current_cfg.get(k) for k in (
                        "model", "imgsz", "batch", "workers", "multi_scale", "mosaic", "mixup", "copy_paste",
                    )},
                    "completed_epoch": state.get("completed_epoch"),
                })
                if reason and "batch" in str(reason):
                    logger.log(f"[Adaptive] BATCH CHANGE recorded in state history ({reason}).")
                # resume_pending stays True: the hidden checkpoint is the continuation
                # point for the next (downgraded) attempt.
                write_adaptive_state(run_dir, state)
                mode = "resume"
                continue
            # No hidden snapshot (legacy run or failed snapshot save): classify the
            # evidence honestly, never as a fake completed-target.
            final_status = "stopped-adaptive"
            finish_terminal_run(ckpt_epoch, actual_epoch, "stopped-adaptive")
            write_adaptive_state(run_dir, state)
            logger.log("[Supervisor] no hidden resumable checkpoint; cannot continue automatically.")
            break

        if st == "attempt-terminal":
            # run_attempt was handed a finalized checkpoint; try a pending hidden one more time.
            if hidden_ckpt is not None:
                current_ckpt = hidden_ckpt
                mode = "resume"
                logger.log("[Supervisor] finalized last.pt; continuing from hidden .adaptive/resumable_last.pt instead.")
                continue
            _, hist_status = adaptive_history_last_entry(state)
            final_status = hist_status if hist_status in TERMINAL_EARLY_STATUSES else "early-stopped"
            finish_terminal_run(ckpt_epoch, actual_epoch, final_status)
            write_adaptive_state(run_dir, state)
            break

        if st == "partial-emergency":
            # restore already done in run_attempt; downgrade config
            logger.log("[Adaptive] emergency partial epoch; downgrading config.")
            probe_cache.clear()  # emergency means an earlier verdict just failed in practice
            res = downgrade_config(current_cfg)
            if res[0] is None:
                final_status = "partial-emergency"
                finish_terminal_run(ckpt_epoch, actual_epoch, "partial-emergency")
                write_adaptive_state(run_dir, state)
                break
            current_cfg, reason = res
            state["downgrade_count"] = state.get("downgrade_count", 0) + 1
            state["provisional"] = True
            write_adaptive_state(run_dir, state)
            mode = "resume"
            continue

        if st == "crashed":
            # OOM or other crash; downgrade
            logger.log("[Adaptive] crash; downgrading config.")
            probe_cache.clear()  # crash invalidates earlier measured verdicts
            res = downgrade_config(current_cfg)
            if res[0] is None:
                final_status = "crashed"
                finish_terminal_run(ckpt_epoch, actual_epoch, "crashed")
                write_adaptive_state(run_dir, state)
                break
            current_cfg, reason = res
            state["downgrade_count"] = state.get("downgrade_count", 0) + 1
            state["provisional"] = True
            write_adaptive_state(run_dir, state)
            mode = "resume"
            continue

        if st in ("interrupted",):
            final_status = "interrupted"
            finish_terminal_run(ckpt_epoch, actual_epoch, "interrupted")
            write_adaptive_state(run_dir, state)
            break

        # unknown
        final_status = st
        break
    else:
        # Attempt budget exhausted without any terminal status. With a pending hidden
        # snapshot the run is still resumable: keep resume_pending, record the real
        # epoch, and tell the user --resume will continue from the hidden checkpoint.
        # (finish_terminal_run is deliberately NOT called — it clears the pending flag.)
        hidden_fin = pending_resumable_checkpoint(run_dir, state)
        if state.get("resume_pending") and hidden_fin is not None:
            weights_last = run_dir / "weights" / "last.pt"
            wl_epoch = ckpt_completed_epoch(weights_last) if weights_last.is_file() else None
            state["completed_epoch"] = max(0, derive_actual_epoch(
                run_dir, state,
                ckpt_epoch=wl_epoch if isinstance(wl_epoch, int) and wl_epoch >= 0 else None,
                target_epochs=target_epochs))
            final_status = "stopped-adaptive"
            write_adaptive_state(run_dir, state)
            logger.log("[Supervisor] attempt budget exhausted; --resume will continue from "
                       f"{ADAPTIVE_DIRNAME}/{RESUMABLE_LAST_NAME}.")

    if final_status == "no-safe-config":
        if state.get("resume_pending"):
            logger.log("[Adaptive] no safe config available for a fresh attempt; the pending hidden "
                       f"snapshot stays available — {ADAPTIVE_DIRNAME}/{RESUMABLE_LAST_NAME} remains resumable.")
        else:
            logger.log("[Adaptive] no safe config available; aborting.")
    state["final_status"] = final_status
    write_adaptive_state(run_dir, state)
    return final_status


# -- Resume Logic --------------------------------------------------------------

def resolve_resume_device(args, run_args):
    """Determine the resume device from checkpoint args (Ultralytics resume keeps ckpt config)."""
    dev = run_args.get("device") if isinstance(run_args, dict) else None
    if dev is None:
        resolved = "0"
    elif isinstance(dev, (list, tuple)):
        resolved = str(dev[0]) if dev else "0"
    else:
        resolved = str(dev)
    if resolved != "cpu" and not cuda_available_for(resolved):
        print(f"[Resume] CUDA device '{resolved}' is not currently available; Ultralytics will fall back to CPU.")
    return resolved


def resume_training(args):
    last_pt = find_last_checkpoint(args.name)
    run_dir = last_pt.parent.parent
    run_name = run_dir.name
    args_yaml = run_dir / "args.yaml"

    logger = RunLogger(run_dir, resume=True)
    logger.start_tee()
    try:
        logger.log("=" * 72)
        logger.log("TRAIN DETECTOR RESUME LOG")
        logger.log(f"local start time : {local_iso()}")
        logger.log(f"command line     : {subprocess.list2cmdline(sys.argv)}")
        logger.log(f"run name         : {run_name}")
        logger.log(f"run dir          : {run_dir}")
        logger.log(f"checkpoint       : {last_pt}")
        logger.log("=" * 72)

        run_args = None
        if args_yaml.exists():
            try:
                with open(args_yaml, encoding="utf-8") as f:
                    run_args = yaml.safe_load(f)
            except Exception as e:
                logger.log(f"[Resume] WARNING: could not read args.yaml: {e}")

        device = resolve_resume_device(args, run_args)
        args.device = device
        hw = detect_hardware()
        hw_fp = hardware_device_fingerprint(device)
        log_hardware(logger, hw, device, hw_fp)
        log_checkpoint_status(logger, run_dir, "(before resume)")
        ckpt_meta = log_resume_checkpoint_info(logger, last_pt)

        ckpt_epoch = ckpt_meta.get("epoch") if isinstance(ckpt_meta, dict) else None
        state = load_adaptive_state(run_dir)
        target_hint = int(args.epochs) if args.epochs else 600
        if isinstance(run_args, dict) and run_args.get("epochs"):
            try:
                target_hint = int(run_args["epochs"])
            except (TypeError, ValueError):
                pass

        if not (isinstance(ckpt_epoch, (int, float)) and ckpt_epoch >= 0):
            # weights/last.pt is finalized (epoch -1) or unreadable: look for a pending
            # hidden snapshot before declaring anything finished.
            alt_ckpt, resume_info = choose_resume_checkpoint(run_dir, last_pt, target_epochs=target_hint)
            if alt_ckpt is not None:
                logger.log(f"[Resume] weights/last.pt is finalized (epoch marker -1); using hidden adaptive snapshot "
                           f"{alt_ckpt} (epoch {resume_info.get('epoch')}) for the pending adaptive continuation.")
                last_pt = alt_ckpt
                ckpt_meta = log_resume_checkpoint_info(logger, last_pt)
                ckpt_epoch = ckpt_meta.get("epoch") if isinstance(ckpt_meta, dict) else None
            else:
                train_args = ckpt_meta.get("train_args") if isinstance(ckpt_meta, dict) else None
                total = train_args.get("epochs") if isinstance(train_args, dict) else None
                total = int(total) if isinstance(total, (int, float)) and total else target_hint
                # choose_resume_checkpoint already derived the honest stopped-after count;
                # consume it instead of re-deriving (a second call site would drift).
                actual = resume_info.get("stopped_after")
                if not isinstance(actual, int):
                    actual = derive_actual_epoch(run_dir, state, tracker=None,
                                                 ckpt_epoch=ckpt_epoch if isinstance(ckpt_epoch, int) else None,
                                                 target_epochs=total)
                best_pt = run_dir / "weights" / "best.pt"
                # Claim "finished the full schedule" only when the evidence says so.
                completed = actual >= total or state.get("final_status") == "completed-target"
                if completed:
                    total_txt = f"its full {total}-epoch schedule"
                    msg = (f"[Resume] Run '{run_name}' already finished {total_txt}; "
                           "last.pt is the final checkpoint (epoch marker -1), so there is nothing to resume.")
                    resume_state = "already-finished"
                else:
                    msg = (f"[Resume] Run '{run_name}' was stopped after {actual} epochs (target {total}). "
                           "last.pt is finalized (epoch marker -1) and not resumable; no pending hidden adaptive "
                           "checkpoint exists for this run. "
                           f"Start a new run from {best_pt} with lower workers/batch if continuation is desired.")
                    resume_state = "finalized-not-resumable"
                logger.log(msg)
                log_results_csv(logger, run_dir)
                resume_log_path = str(logger.log_path)
                logger.close()
                print(f"\n[Resume] {msg}")
                print(f"  Run log: {resume_log_path}")
                return best_pt, run_name, resume_state

        if getattr(args, "dry_run", False):
            logger.log("[Resume Dry Run] inspecting checkpoint and calibration plan; no training.")
            print(f"\n[Resume Dry Run] run={run_name} checkpoint={last_pt}")
            print(f"  ckpt epoch: {ckpt_epoch}")
            if isinstance(ckpt_meta, dict):
                ta = ckpt_meta.get("train_args")
                if isinstance(ta, dict):
                    for k in ("model", "imgsz", "batch", "workers", "multi_scale", "mosaic", "mixup", "copy_paste", "epochs"):
                        print(f"  ckpt {k}: {ta.get(k)}")
            logger.log("[Resume Dry Run] done.")
            best_pt = run_dir / "weights" / "best.pt"
            logger.close()
            return best_pt, run_name, "dry-run"

        if state.get("last_attempt_partial"):
            logger.log("[Adaptive] prior attempt partial detected; restoring backups.")
            restore_backups(run_dir, logger)
            ce = state.get("completed_epoch")
            if isinstance(ce, int) and ce >= 0:
                trim_results_csv(run_dir, ce, logger)

        data_str = None
        if args.input:
            resolved_yaml, resume_task = resolve_dataset_input(args, logger)
            args.dataset_task = resume_task
            data_str = str(resolved_yaml)
        elif isinstance(run_args, dict) and run_args.get("data"):
            data_str = str(run_args.get("data"))
        if data_str and args_yaml.exists() and run_args is not None:
            try:
                run_args["data"] = data_str
                with open(args_yaml, "w", encoding="utf-8") as f:
                    for k, v in run_args.items():
                        if k == "data":
                            f.write(f"data: '{data_str}'\n")
                        else:
                            yaml.dump({k: v}, f, default_flow_style=True, allow_unicode=True)
                logger.log(f"[Resume] data path patched in args.yaml: {data_str}")
            except Exception as e:
                logger.log(f"[Resume] WARNING: failed to patch args.yaml data path: {e}")

        yaml_path = None
        if data_str:
            try:
                yaml_path = normalize_path(data_str)
            except Exception:
                yaml_path = None
        train_path = None
        ds_fp = {}
        ckpt_cfg = {}
        if isinstance(ckpt_meta, dict):
            ta = ckpt_meta.get("train_args")
            if isinstance(ta, dict):
                ckpt_cfg = dict(ta)
        if yaml_path is not None and yaml_path.exists():
            try:
                validate_dataset(yaml_path)
                data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
                train_path = resolve_yaml_split_path(yaml_path, str(data["train"]))
                resume_task = getattr(args, "dataset_task", None)
                ckpt_task = ckpt_cfg.get("task")
                if resume_task not in ("detect", "segment", "pose") and isinstance(ckpt_task, str):
                    resume_task = ckpt_task
                if resume_task not in ("detect", "segment", "pose"):
                    try:
                        label_dirs = [train_path.parent / "labels"] if train_path is not None else []
                        resume_task = derive_task_from_yolo_labels(label_dirs)[0]
                    except Exception:
                        resume_task = "detect"
                args.dataset_task = resume_task
                if isinstance(ckpt_task, str) and ckpt_task in ("detect", "segment", "pose") and resume_task != ckpt_task:
                    print(f"[ERROR] Resume with --input resolves task={resume_task} but checkpoint '{run_name}' trained task={ckpt_task}.")
                    print("        Task/architecture mismatch: start a new run with a different --name instead of --resume.")
                    sys.exit(1)
                ds_fp = dataset_fingerprint(yaml_path, train_path, resume_task)
                log_dataset(logger, ds_fp)
            except Exception as e:
                logger.log(f"[Resume] dataset fingerprint failed: {e}")

        cache = load_memory_cache(logger)
        chosen, prov = calibrate_resume_config(args, logger, cache, hw, hw_fp, ds_fp, ckpt_cfg, run_dir,
                                               yaml_path=yaml_path, device=device)
        logger.log(f"[Resume] calibrated config provenance: {prov}")

        target_epochs = args.epochs if args.epochs else 600
        if ckpt_cfg.get("epochs") and (args.epochs is None or args.epochs == 600):
            target_epochs = ckpt_cfg.get("epochs")

        guard_enabled = not getattr(args, "no_guard", False)
        guard = make_guard(logger, hw, device, args, guard_enabled=guard_enabled)

        if not state:
            state = {"run_name": run_name, "target_epochs": target_epochs, "history": []}
        state["target_epochs"] = target_epochs
        write_adaptive_state(run_dir, state)

        best_pt = run_dir / "weights" / "best.pt"

        status = run_adaptive_supervisor(
            args=args, logger=logger, run_dir=run_dir, run_name=run_name,
            last_pt=last_pt, chosen=chosen, target_epochs=target_epochs,
            device=device, hw=hw, guard=guard, state=state, yaml_path=yaml_path,
            ckpt_cfg=ckpt_cfg, mode="resume",
        )

        log_checkpoint_status(logger, run_dir, f"(after resume, status: {status})")
        log_results_csv(logger, run_dir)
        logger.log(f"[Resume] final status: {status}; best.pt: {best_pt if best_pt.is_file() else 'not produced'}")

        if status in ("completed-target", "completed", "early-stopped"):
            print(f"\n{'='*60}")
            print(f"  Training complete")
            print(f"{'='*60}")
            print(f"  best.pt  : {best_pt}")
            print(f"  last.pt  : {last_pt}")
            print(f"{'='*60}\n")
        else:
            report_run_outcome(run_dir, state, status, target_epochs, run_name, logger)

        return best_pt, run_name, status
    except KeyboardInterrupt:
        logger.log("[Resume] Interrupted by user (Ctrl+C).")
        print("\n[Resume] Interrupted by user. See run log: " + str(logger.log_path))
        sys.exit(130)
    except SystemExit:
        raise
    except Exception:
        logger.log("[Resume] UNEXPECTED CRASH:")
        logger.log(traceback.format_exc())
        print("\n[Resume] Unexpected error; full traceback logged to " + str(logger.log_path))
        sys.exit(1)
    finally:
        logger.close()


# -- Fresh Training ------------------------------------------------------------
def resolve_train_device(args, logger=None):
    device = args.device if args.device else "0"
    if device != "cpu" and not cuda_available_for(device):
        msg = f"[Device] Requested CUDA device '{device}' is not available; falling back to CPU."
        print(msg)
        if logger:
            logger.log(msg)
        device = "cpu"
    args.device = device
    return device


def _write_subset_list(image_paths, out_file: Path):
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(str(p) for p in image_paths) + ("\n" if image_paths else ""), encoding="utf-8")


def build_pilot_dataset(yaml_path: Path, run_dir: Path, logger, train_n=PILOT_TRAIN_IMAGES, val_n=PILOT_VAL_IMAGES):
    """
    Build a temporary pilot data.yaml under <run>/.pilot/ pointing at image-list
    subsets. Does not touch the real run's weights/.
    """
    pilot_dir = Path(run_dir) / PILOT_DIRNAME
    if pilot_dir.exists():
        shutil.rmtree(pilot_dir, ignore_errors=True)
    pilot_dir.mkdir(parents=True, exist_ok=True)
    data = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8")) or {}
    train_images = resolve_yaml_split_path(yaml_path, str(data["train"]))
    val_key = data.get("val") or data["train"]
    val_images = resolve_yaml_split_path(yaml_path, str(val_key))
    train_files = image_files(train_images)
    val_files = image_files(val_images) if val_images else []
    if len(train_files) > train_n:
        step = len(train_files) / float(train_n)
        train_files = [train_files[int(i * step)] for i in range(train_n)]
    if val_files and len(val_files) > val_n:
        step = len(val_files) / float(val_n)
        val_files = [val_files[int(i * step)] for i in range(val_n)]
    elif not val_files:
        val_files = train_files[: max(1, min(val_n, len(train_files)))]
    train_list = pilot_dir / "train.txt"
    val_list = pilot_dir / "val.txt"
    _write_subset_list(train_files, train_list)
    _write_subset_list(val_files, val_list)
    pilot_yaml = {
        "path": str(pilot_dir.resolve()),
        "train": str(train_list.resolve()),
        "val": str(val_list.resolve()),
        "nc": data.get("nc"),
        "names": data.get("names"),
    }
    if data.get("kpt_shape") is not None:
        pilot_yaml["kpt_shape"] = data["kpt_shape"]
    if data.get("flip_idx") is not None:
        pilot_yaml["flip_idx"] = data["flip_idx"]
    out_yaml = pilot_dir / "data.yaml"
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(pilot_yaml, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    logger.log(f"[Pilot] subset dataset: {len(train_files)} train / {len(val_files)} val -> {out_yaml}")
    return out_yaml, pilot_dir


def pilot_trend_ok(history: list) -> bool:
    """Train loss decreasing AND check-set score rising from the first epoch."""
    if len(history) < 2:
        return True  # too short to fail
    losses = [h["loss"] for h in history if h.get("loss") is not None]
    scores = [h["score"] for h in history if h.get("score") is not None]
    if len(losses) < 2 or len(scores) < 2:
        return True
    loss_down = losses[-1] < losses[0] - 1e-6
    score_up = scores[-1] > scores[0] + 1e-4
    return bool(loss_down and score_up)


def run_preflight_pilot(args, logger, run_dir, chosen, yaml_path, device, hw, guard, ds_fp):
    """
    Short learnability check using the chosen model/aug on a subset.
    Uses MemoryGuard; artifacts stay under <run>/.pilot/ (never weights/).
    Returns (ok: bool, sample_grid_path or None, diagnostics: str).
    """
    if getattr(args, "skip_pilot", False):
        logger.log("[Pilot] skipped (--skip-pilot).")
        return True, None, "skipped"
    pilot_yaml, pilot_dir = build_pilot_dataset(yaml_path, run_dir, logger)
    pilot_project = pilot_dir / "runs"
    pilot_name = "pilot"
    epochs = min(PILOT_EPOCHS, max(3, int(getattr(args, "epochs", 5) or 5)))
    if int(getattr(args, "epochs", 600) or 600) < 3:
        epochs = max(1, int(args.epochs))
    logger.log(f"[Pilot] starting {epochs}-epoch learnability check with {chosen['model']} "
               f"@ {chosen['imgsz']}px batch {chosen['batch']}")
    print(f"[Pilot] Running a short learnability check ({epochs} epochs on a ~{PILOT_TRAIN_IMAGES}-image subset)...")
    history = []
    sample_grid = None
    try:
        weights = resolve_pretrained_weights(chosen["model"])
        model = YOLO(str(weights))
        # Lightweight metric capture via callback
        state = {"t0": None}

        def on_fit_end(trainer):
            ep = getattr(trainer, "epoch", 0)
            loss = _metric_from_trainer(trainer, "train/box_loss", "train/loss", "loss")
            score = _metric_from_trainer(
                trainer, "metrics/mAP50(B)", "metrics/mAP50(M)", "metrics/mAP50", "mAP50",
            )
            history.append({"epoch": int(ep) + 1, "loss": loss, "score": score})
            logger.log(f"[Pilot] epoch {int(ep)+1}: loss={loss} score={score}")

        model.add_callback("on_fit_epoch_end", on_fit_end)
        if guard is not None:
            try:
                guard.start()
            except Exception:
                pass
        aug = chosen_augmentation(chosen)
        model.train(
            data=str(pilot_yaml),
            epochs=epochs,
            imgsz=int(chosen["imgsz"]),
            batch=int(chosen["batch"]),
            workers=int(chosen["workers"]),
            device=device,
            project=str(pilot_project),
            name=pilot_name,
            exist_ok=True,
            save=True,
            plots=True,
            verbose=False,
            patience=epochs,
            **aug,
        )
    except Exception as e:
        logger.log(f"[Pilot] failed: {_short(str(e))}")
        print(f"[Pilot] WARNING: pilot failed ({e}); continuing to full run (use diagnostics if training looks stuck).")
        return True, None, f"pilot-error: {e}"
    finally:
        if guard is not None:
            try:
                guard.stop()
            except Exception:
                pass

    # Find sample batch grid if Ultralytics wrote one
    for candidate in (
        pilot_project / pilot_name / "train_batch0.jpg",
        pilot_project / pilot_name / "train_batch1.jpg",
        pilot_project / pilot_name / "val_batch0_labels.jpg",
    ):
        if candidate.is_file():
            sample_grid = candidate
            break
    if sample_grid:
        logger.log(f"[Pilot] sample grid: {sample_grid}")
        print(f"[Pilot] Augmented sample grid: {sample_grid}")

    # Remove pilot weights so they cannot be mistaken for the real run
    for wdir in (pilot_project / pilot_name / "weights",):
        if wdir.is_dir():
            shutil.rmtree(wdir, ignore_errors=True)

    ok = pilot_trend_ok(history)
    if ok:
        logger.log("[Pilot] PASS: train loss trending down and check-set score rising.")
        print("[Pilot] PASS — learning signal looks healthy; starting the full run.")
        return True, sample_grid, "pass"
    ls = ds_fp.get("label_stats") or {}
    diag = (
        "Pilot FAIL: train loss and check-set score both look flat.\n"
        f"  label stats: median box {ls.get('median_box_w')}x{ls.get('median_box_h')}, "
        f"boxes/img {ls.get('boxes_per_image')}, empty {ls.get('empty_label_fraction')}\n"
        f"  sample grid: {sample_grid or '(none)'}\n"
        "  likely causes: wrong/empty labels, objects too small for imgsz, "
        "class mismatch, or augmentation destroying boxes.\n"
        "  re-run with --skip-pilot only if you have verified the labels manually."
    )
    logger.log("[Pilot] " + diag.replace("\n", "\n[Pilot] "))
    print("\n[Pilot] " + diag)
    return False, sample_grid, diag


def train(args):
    if not args.input:
        print("[ERROR] --input is required for fresh training.")
        print("        Use --resume to continue an existing run.")
        sys.exit(1)

    input_dir = normalize_path(args.input)
    if input_dir.is_file():
        print(f"[ERROR] --input must be a dataset FOLDER, not a file: {input_dir}")
        if input_dir.suffix.lower() in (".yaml", ".yml"):
            print("        Pass the dataset directory containing this yaml; its data.yaml is auto-detected.")
        print("        (To continue an existing run use: --resume --name <run> — the stored data path is reused.)")
        sys.exit(1)
    if not input_dir.is_dir():
        print(f"[ERROR] Dataset directory not found: {input_dir}")
        sys.exit(1)

    requested = args.name if args.name else "detector_v1"

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_name, run_dir, mode, run_messages = resolve_fresh_attempt(args)

    run_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(run_dir, resume=False)
    logger.start_tee()
    try:
        for message in run_messages:
            logger.log(message)

        if mode == "resume":
            logger.log(f"[Run] Colliding --name resolved by user choice: delegating to the RESUME flow for '{run_name}'.")
            logger.close()
            args.name = run_name
            best_pt, _, _ = resume_training(args)
            return {"mode": "resume", "best": best_pt, "run_name": run_name, "status": "delegated"}

        logger.log_run_startup(args, requested, run_name, run_dir, "fresh")

        device = resolve_train_device(args, logger)
        hw = detect_hardware()
        hw_fp = hardware_device_fingerprint(device)
        log_hardware(logger, hw, device, hw_fp)

        yaml_path, dataset_task = resolve_dataset_input(args, logger)
        args.dataset_task = dataset_task
        validate_dataset(yaml_path)
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        train_path = resolve_yaml_split_path(yaml_path, str(data["train"]))
        ds_fp = dataset_fingerprint(yaml_path, train_path, dataset_task)
        log_dataset(logger, ds_fp)

        cache = load_memory_cache(logger)

        if args.dry_run:
            return run_dry_summary(logger, args, cache, hw, hw_fp, ds_fp, device, run_name, run_dir)

        log_checkpoint_status(logger, run_dir, "(before training)")

        try:
            chosen, provenance = calibrate_training_config(
                args, logger, cache, hw_fp, hw, device, yaml_path, train_path, ds_fp
            )
        except CalibrateFail as e:
            logger.log(f"[Calibrate] FAILED: {e}")
            print(f"\n[ERROR] Memory calibration failed:\n  {e}")
            print(f"  Run diagnostics log: {logger.log_path}")
            return {"mode": "fresh", "best": None, "run_name": run_name, "status": "calibration-failed"}
        except KeyboardInterrupt:
            logger.log("[Run] Interrupted by user (Ctrl+C) during calibration; training never started, nothing was saved.")
            print("\n[Run] Interrupted during calibration probes. No training was started.")
            return {"mode": "fresh", "best": None, "run_name": run_name, "status": "interrupted"}

        print_auto_config(logger, chosen, provenance, args, hw, ds_fp, device)
        print(f"[Config ({provenance})] model={chosen['model']} imgsz={chosen['imgsz']}px batch={chosen['batch']} "
              f"workers={chosen['workers']} multi_scale={chosen['multi_scale']} mosaic={chosen['mosaic']} "
              f"(full detail logged to {logger.log_path.name})")

        print(f"\n[Training Config]")
        print(f"  epochs     : {args.epochs}")
        print(f"  image size : {chosen['imgsz']}px")
        print(f"  patience   : {args.patience} epochs early stopping")
        if getattr(args, "time_budget", None):
            print(f"  time budget: {args.time_budget} hours")
        print(f"  device     : {device}")
        print(f"  output     : {run_dir}")
        print()

        monitor = make_guard(logger, hw, device, args, guard_enabled=not getattr(args, "no_guard", False))
        model = None
        # fresh first attempt goes through supervisor with mode=fresh, no checkpoint
        guard_enabled = not getattr(args, "no_guard", False)
        guard = monitor

        pilot_ok, sample_grid, pilot_diag = run_preflight_pilot(
            args, logger, run_dir, chosen, yaml_path, device, hw, guard, ds_fp,
        )
        if not pilot_ok:
            logger.log("[Run] Aborted before long training: pilot learnability check failed.")
            return {
                "mode": "fresh", "best": None, "run_name": run_name, "status": "pilot-failed",
                "diagnostics": pilot_diag, "sample_grid": str(sample_grid) if sample_grid else None,
            }

        state = {"run_name": run_name, "target_epochs": args.epochs, "completed_epoch": 0,
                 "provisional": True, "stable_epochs": 0, "downgrade_count": 0, "history": []}
        write_adaptive_state(run_dir, state)

        chosen_cfg = {k: v for k, v in chosen.items() if not str(k).startswith("_")}
        status = run_adaptive_supervisor(
            args=args, logger=logger, run_dir=run_dir, run_name=run_name,
            last_pt=None, chosen=chosen_cfg, target_epochs=args.epochs,
            device=device, hw=hw, guard=guard, state=state, yaml_path=yaml_path,
            ckpt_cfg={}, mode="fresh",
        )

        log_checkpoint_status(logger, run_dir, f"(after training, status: {status})")
        log_results_csv(logger, run_dir)

        best_pt = run_dir / "weights" / "best.pt"
        last_pt = run_dir / "weights" / "last.pt"

        if status in ("completed-target", "completed", "early-stopped"):
            logger.log(f"[Run] Training completed successfully. best.pt: {best_pt}")
            print(f"\n{'='*60}")
            print(f"  Training complete")
            print(f"{'='*60}")
            print(f"  best.pt  : {best_pt}")
            print(f"  last.pt  : {last_pt}")
            print(f"{'='*60}")
            print(f"\n  Next step: export to TensorRT for Manifold 3:")
            print(f"    python export_tensorrt.py --weights {best_pt}")
            print()
        else:
            logger.log(f"[Run] Training {status}; partial checkpoint state kept in {run_dir / 'weights'}.")
            report_run_outcome(run_dir, state, status, args.epochs, run_name, logger)

        return {"mode": "fresh", "best": best_pt, "run_name": run_name, "status": status, "yaml_path": str(yaml_path)}
    except KeyboardInterrupt:
        logger.log("[Run] Interrupted by user (Ctrl+C) before training finished or started.")
        print("\n[Run] Interrupted by user. See run log: " + str(logger.log_path))
        sys.exit(130)
    except SystemExit:
        raise
    except Exception:
        logger.log("[Run] UNEXPECTED CRASH:")
        logger.log(traceback.format_exc())
        print("\n[Run] Unexpected error; full traceback logged to " + str(logger.log_path))
        sys.exit(1)
    finally:
        logger.close()


# -- Dry Run ---------------------------------------------------------------------

def run_dry_summary(logger, args, cache, hw, hw_fp, ds_fp, device, run_name, run_dir):
    logger.log("[Dry Run] --dry-run exercised run-name handling, dataset validation, fingerprinting and log writing; "
               "NO probe runs and NO training were performed.")

    sel_key = selection_signature(args, ds_fp)
    cached = lookup_cached_selection(cache, hw_fp, ds_fp, sel_key, logger)
    if cached:
        sel, entry = cached
        logger.log(f"[Dry Run] Cache lookup HIT for selection key {sel_key}")
        logger.log(f"[Dry Run] Measured-safe cached config: {sel.get('config_key')}")
        logger.log(f"[Dry Run] Recorded peaks: GPU alloc {entry.get('peak_gpu_allocated_gb')} GB | "
                   f"GPU reserved {entry.get('peak_gpu_reserved_gb')} GB | CPU RSS {entry.get('peak_cpu_rss_gb')} GB")
    else:
        logger.log(f"[Dry Run] Cache lookup MISS for selection key {sel_key}; a real run would calibrate with probe runs.")

    task = getattr(args, "dataset_task", None) or ds_fp.get("task", "detect")
    ladder = aug_ladder_for_task(task)
    degrees, flipud = resolve_geometry_aug(args)
    logger.log(f"[Dry Run] Planned selection order: models largest-first × imgsz (object-size aware) × "
               f"healthy batches {HEALTHY_BATCH_CANDIDATES} × task aug ladder ({len(ladder)} rungs); "
               f"degraded batches {DEGRADED_BATCH_CANDIDATES} only if nothing fits batch>={MIN_HEALTHY_BATCH}.")
    logger.log(f"[Dry Run] Task-aware aug ladder ({task}): {ladder}")

    cfg = static_candidate_config(args, hw, ds_fp.get("image_count") or 0, ds_fp.get("median_img_size"), cuda_available_for(device))
    if cfg:
        for note in cfg["notes"]:
            logger.log(f"[Dry Run] static note: {note}")
        mosaic, copy_paste, mixup, multi_scale = ladder[0]
        mixup = resolve_mixup(args, task, mixup)
        chosen = build_chosen(
            cfg["model"], cfg["imgsz"], cfg["batch"], cfg["workers"],
            cfg["multi_scale"] if cfg["multi_scale"] else multi_scale, mosaic,
            mixup=mixup, copy_paste=copy_paste, degrees=degrees, flipud=flipud,
        )
        print_auto_config(logger, chosen, "dry-run static estimate (no probes)", args, hw, ds_fp, device)
        print(f"[Dry Run] Static config preview: {chosen['model']} @ {chosen['imgsz']}px batch {chosen['batch']} "
              f"workers={chosen['workers']} multi_scale={chosen['multi_scale']} mosaic={chosen['mosaic']} "
              f"(full detail logged to {logger.log_path.name})")
        if cfg["explicit_unsafe"]:
            logger.log("[Dry Run] static estimate flags the explicit override as UNSAFE (would require --force-unsafe or dynamic probes).")
    else:
        logger.log("[Dry Run] no candidate fits conservative static caps; a real run would probe lower (imgsz/model/batch ladder).")

    logger.log(f"[Dry Run] resolved run name : {run_name}")
    logger.log(f"[Dry Run] resolved run dir  : {run_dir}")
    logger.log(f"[Dry Run] run log written to: {logger.log_path}")

    print(f"\n[Dry Run] Resolved run name : {run_name}")
    print(f"[Dry Run] Run directory     : {run_dir}")
    print(f"[Dry Run] Run log           : {logger.log_path}")
    print(f"[Dry Run] Dataset validated, fingerprints computed, log written. No probes, no training.")
    return {"mode": "dry", "best": None, "run_name": run_name, "status": "dry-run"}


# -- Post-Training Validation --------------------------------------------------

def task_metric_lines(model_task, metrics) -> list:
    """Extra per-task validation metrics to display/log when the metrics object exposes them.
    The mAP@0.50 box gate stays the pass/fail truth; seg/pose lines are informational."""
    t = str(model_task or "").strip().lower()
    out = []
    if t == "segment":
        seg = getattr(metrics, "seg", None)
        if seg is not None:
            for attr, label in (("map50", "mAP@0.50 (seg)"), ("map", "mAP@0.50:0.95 (seg)")):
                val = getattr(seg, attr, None)
                if isinstance(val, (int, float)):
                    out.append((label, float(val)))
    elif t == "pose":
        pose = getattr(metrics, "pose", None)
        if pose is not None:
            for attr, label in (("mp", "Pose mP"), ("mr", "Pose mR"),
                                ("map50", "mAP@0.50 (pose)"), ("map", "mAP@0.50:0.95 (pose)")):
                val = getattr(pose, attr, None)
                if isinstance(val, (int, float)):
                    out.append((label, float(val)))
    return out


def validate(yaml_path: Path, args, run_name: str):
    run_dir = RUNS_DIR / run_name
    best_pt = run_dir / "weights" / "best.pt"
    run_args_yaml = run_dir / "args.yaml"

    print(f"\n[Validation] Running on best.pt...")
    print(f"  Validating: {best_pt}")

    if not best_pt.exists():
        print(f"[ERROR] best.pt not found at: {best_pt}")
        sys.exit(1)

    val_imgsz = 1024
    if run_args_yaml.exists():
        with open(run_args_yaml, encoding="utf-8") as f:
            run_args = yaml.safe_load(f)
        val_imgsz = run_args.get("imgsz", 1024)

    device = args.device if args.device else "0"
    model   = YOLO(str(best_pt))
    metrics = model.val(
        data=str(yaml_path),
        imgsz=val_imgsz,
        device=device,
        project=str(run_dir.parent),
        name=f"{run_name}-val",
    )

    map50    = metrics.box.map50
    map50_95 = metrics.box.map
    task_lines = task_metric_lines(getattr(model, "task", None), metrics)

    try:
        log_files = sorted((run_dir / "Logs").glob("run-*.log"))
        if log_files:
            with open(log_files[-1], "a", encoding="utf-8") as f:
                f.write(f"\n[Validation] best.pt at imgsz={val_imgsz} (saved under {run_dir.parent / (run_name + '-val')})\n")
                f.write(f"  mAP@0.50   = {map50:.4f}\n")
                f.write(f"  mAP@0.50-95 = {map50_95:.4f}\n")
                for label, value in task_lines:
                    f.write(f"  {label} = {value:.4f}\n")
    except OSError:
        pass

    print(f"\n[Metrics]")
    print(f"  mAP@0.50       : {map50:.4f}  (target: > 0.80)")
    print(f"  mAP@0.50:0.95  : {map50_95:.4f}")
    for label, value in task_lines:
        print(f"  {label:<15}: {value:.4f}")

    if map50 >= 0.80:
        print(f"\n  PASS -- model meets target accuracy (mAP@0.50 >= 0.80)")
        print(f"  Ready to export to TensorRT for Manifold 3")
    else:
        print(f"\n  WARNING -- model below target accuracy (mAP@0.50 < 0.80)")
        print(f"  Consider: more data, more epochs, or lower confidence threshold")


# -- Entry Point ---------------------------------------------------------------

def main():
    args = parse_args()

    if args.probe_vram:
        models = [args.model] if args.model else ["yolo26m.pt", "yolo26l.pt", "yolo26x.pt"]
        ensure_vram_estimates(models, device=args.device or "0", force=True)
        if not args.input and not args.resume:
            sys.exit(0)

    if args.resume:
        resume_training(args)
        return

    outcome = train(args)
    if (
        outcome
        and outcome.get("mode") == "fresh"
        and outcome.get("status") in ("completed", "completed-target")
        and not args.dry_run
    ):
        validate(Path(outcome["yaml_path"]), args, outcome["run_name"])


if __name__ == "__main__":
    main()
