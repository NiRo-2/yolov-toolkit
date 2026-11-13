r"""
YOLO26 Object Detector Training Script
Auto-detects hardware, dataset size and image resolution to calculate optimal training config.
Every fresh or resumed training invocation writes a per-run text log under <run_dir>/Logs/.
Memory/model/batch selection is dynamically calibrated with real-dataset probe runs for the
selected hardware, device, dataset and augmentation config; static estimates are fallback only.

Usage:
    # Fresh training run (auto-configures everything, runs dynamic memory calibration)
    python train_detector/train_detector.py --input /path/to/data.yaml --name my_detector_v1

    # Override any auto-calculated value (overrides are safety-checked unless --force-unsafe)
    python train_detector/train_detector.py --input /path/to/data.yaml --name my_detector_v1 --model yolo26x.pt --batch 8
    python train_detector/train_detector.py --input /path/to/data.yaml --name my_detector_v1 --imgsz 1024

    # Run-name collision control
    python train_detector/train_detector.py --input /path/to/data.yaml --name my_detector_v1 --fresh
    python train_detector/train_detector.py --input /path/to/data.yaml --name my_detector_v1 --resume

    # Resume a crashed/interrupted run
    python train_detector/train_detector.py --resume --name my_detector_v1

    # Resume without --name: auto-finds the most recent run
    python train_detector/train_detector.py --resume

    # Test run-name handling, dataset validation, fingerprinting and logging without training
    python train_detector/train_detector.py --input /path/to/data.yaml --name DryRunTest --device cpu --dry-run

    --input   path to data.yaml file (required for fresh training)
              works with Windows paths: c:\Users\Ni\Desktop\project\data.yaml
    --resume  resume from last checkpoint (fails clearly if no checkpoint exists)
    --fresh   force a new auto-suffixed run even if the run name already exists
    --name    run name to resume, or name for new run (default: detector_v1)
    --model   override auto-selected model (yolo26n/m/l/x)
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
    train_detector/weights/memory_profiles.json       <- measured dynamic memory cache (schema v2)
    train_detector/weights/vram_estimates.json        <- legacy fallback estimates only
    train_detector/runs/detect/<name>/Logs/run-<localtime><offset>.log     <- fresh run log
    train_detector/runs/detect/<name>/Logs/resume-<localtime><offset>.log  <- resume run log
    train_detector/runs/detect/<name>/weights/best.pt <- use this for TensorRT export
    train_detector/runs/detect/<name>/weights/last.pt
    train_detector/runs/detect/<name>/results.png
    train_detector/runs/detect/<name>/confusion_matrix.png
"""

import argparse
import functools
import hashlib
import json
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
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer

try:
    import psutil
    PSUTIL_AVAILABLE = True
except Exception:
    PSUTIL_AVAILABLE = False


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

# Memory safety profiles: maximum allowed measured peak as a fraction of total memory.
# safe @16GB VRAM -> ~10.9 GB peak reserved ceiling.
MEMORY_PROFILES = {
    "safe":       {"gpu": 0.68, "cpu": 0.38},
    "balanced":   {"gpu": 0.75, "cpu": 0.45},
    "aggressive": {"gpu": 0.85, "cpu": 0.55},
}

PREFERRED_AUTO_BATCH = 8   # auto batch target; nbs=64 gradient accumulation keeps effective batch at 64
MIN_AUTO_BATCH = 4         # auto batch floor; batch 2 is only usable with an explicit --force-unsafe run
BATCH_LADDER = [2, 4, 8, 16, 32, 64]
STANDARD_IMGSZ = [320, 416, 512, 640, 768, 1024, 1280, 1536]
MODEL_QUALITY_ORDER = ["yolo26n.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt"]

MEMORY_CACHE_PATH = WEIGHTS_DIR / "memory_profiles.json"


# -- Adaptive safe-resume / crash-prevention constants ----------------------------
# Quality-first: never exceed these fractions of total memory during a run.
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
ADAPTIVE_EPOCH_ATTEMPT_CAP = 1     # provisional config: 1 epoch per attempt
MAX_ADAPTIVE_CONFIG_DOWNGRADES = 12
AUG_LADDER = [
    # (mosaic, copy_paste, mixup, multi_scale)
    (1.0, 0.3, 0.15, 0.0),
    (1.0, 0.3, 0.0,  0.0),
    (1.0, 0.0, 0.0,  0.0),
    (0.5, 0.3, 0.15, 0.0),
    (0.5, 0.3, 0.0,  0.0),
    (0.5, 0.0, 0.0,  0.0),
    (0.0, 0.0, 0.0,  0.0),
]
BATCH_LADDER_DOWN = [8, 4, 2, 1]
IMGSZ_LADDER_DOWN = None  # computed from native cap at runtime
ADAPTIVE_DIRNAME = ".adaptive"
ADAPTIVE_STATE_FILE = "state.json"


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

def detect_hardware():
    """Detect GPU VRAM, CPU cores and RAM."""
    info = {
        "vram_gb":   None,
        "cpu_cores": os.cpu_count() or 4,
        "ram_gb":    None,
        "gpu_name":  "CPU",
    }

    try:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["vram_gb"] = props.total_memory / (1024 ** 3)
            info["gpu_name"] = props.name
    except Exception:
        pass

    if PSUTIL_AVAILABLE:
        try:
            info["ram_gb"] = psutil.virtual_memory().total / (1024 ** 3)
        except Exception:
            pass

    return info


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

def process_rss_bytes():
    """Recursive process RSS in bytes, or None when psutil is unavailable/fails."""
    if not PSUTIL_AVAILABLE:
        return None
    try:
        proc = psutil.Process()
        total = proc.memory_info().rss
        try:
            children = proc.children(recursive=True)
        except (psutil.AccessDenied, psutil.Error):
            children = []
        for child in children:
            try:
                total += child.memory_info().rss
            except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ProcessLookupError):
                continue
        return total
    except Exception:
        return None


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


# -- Guarded trainer subclass ----------------------------------------------------
class GuardedDetectionTrainer(DetectionTrainer):
    """
    DetectionTrainer subclass that:
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


def dataset_fingerprint(yaml_path: Path, train_path: Path) -> dict:
    """Content stats of the dataset used for calibration and cache validity."""
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

    return {
        "data_yaml": str(yaml_path).replace("\\", "/"),
        "image_count": len(files),
        "nc": int(data.get("nc") or 0),
        "median_img_size": int(statistics.median(sizes)) if sizes else None,
        "max_sampled_img_size": int(max(sizes)) if sizes else None,
        "avg_labels_per_sampled_image": round(sum(label_counts) / len(label_counts), 2) if label_counts else None,
    }


def fingerprint_hash(fp: dict) -> str:
    payload = json.dumps(fp, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


# -- Dynamic Memory Cache (memory_profiles.json, schema v2) ------------------------
# Legacy vram_estimates.json is a model-only fallback heuristic; it is NOT final
# safety truth. Only measured, fingerprint-matched entries in this cache can
# skip recalibration.

def load_memory_cache(logger=None) -> dict:
    if not MEMORY_CACHE_PATH.exists():
        return {"version": 2, "hardware_profiles": {}}
    try:
        with open(MEMORY_CACHE_PATH, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or data.get("version") != 2:
            if logger:
                logger.log(f"[Cache] {MEMORY_CACHE_PATH} is not schema version 2; starting fresh.")
            return {"version": 2, "hardware_profiles": {}}
        data.setdefault("hardware_profiles", {})
        return data
    except (OSError, json.JSONDecodeError) as e:
        if logger:
            logger.log(f"[Cache] Ignoring corrupt memory cache {MEMORY_CACHE_PATH}: {e}")
        return {"version": 2, "hardware_profiles": {}}


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


def config_signature(model, imgsz, batch, workers, multi_scale, mosaic=1.0, mixup=0.15, copy_paste=0.3) -> str:
    """Cache config key: model|imgsz|batch|workers|ms|mosaic|mixup|copy_paste."""
    return "|".join([
        str(model), str(imgsz), str(batch), str(workers),
        f"ms{sig_num(multi_scale)}",
        f"mosaic{sig_num(mosaic)}",
        f"mixup{sig_num(mixup)}",
        f"copy{sig_num(copy_paste)}",
    ])


def selection_signature(args, ds_fp: dict) -> str:
    """Which calibration request a cached selection answers."""
    explicit = {k: v for k, v in {
        "model": args.model, "imgsz": args.imgsz, "batch": args.batch, "workers": args.workers,
    }.items() if v is not None}
    sig = json.dumps(explicit, sort_keys=True, separators=(",", ":")) if explicit else "auto"
    return f"ds:{fingerprint_hash(ds_fp)}|profile:{args.memory_profile}|{sig}"


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
class AttemptTracker:
    def __init__(self, logger, guard, run_dir, target_epochs, intended_cfg, state,
                 on_emergency, on_adaptive, on_complete, on_crash):
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
        self.status = "running"
        self.emergency_triggered = False
        self.adaptive_triggered = False
        self.oom_auto_reduced = False
        self.start_epoch = None
        self.completed_epoch = None
        self.actual_cfg = None
        self.crash_exc = None
        self._stop_handled = False

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
        if isinstance(epoch, int):
            self.completed_epoch = epoch + 1
        if self.guard.consume_adaptive():
            self.adaptive_triggered = True
            try:
                trainer.stop = True
            except Exception:
                pass
            self.logger.log("[Attempt] adaptive stop requested at epoch end (clean).")

    def on_train_end(self, trainer):
        # classify
        try:
            final_epoch = getattr(trainer, "epoch", None)
            if isinstance(final_epoch, int):
                self.completed_epoch = final_epoch + 1
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


def ckpt_completed_epoch(last_pt: Path) -> int:
    try:
        ckpt = torch.load(str(last_pt), map_location="cpu", weights_only=False)
        e = ckpt.get("epoch", -1)
        if isinstance(e, (int, float)) and e >= 0:
            return int(e) + 1
        if e == -1:
            return -1  # finished
    except Exception:
        pass
    return 0


# -- Legacy VRAM estimates (fallback heuristic only) --------------------------------

# VRAM usage estimates per image at 1024px (GB).
# FLOPs-scaled from prior YOLOv8 measurements using Ultralytics published
# detect FLOPs (m 68.2/78.9, l 86.4/165.2, x 193.9/257.8). Local probe cache
# overrides these when present (see load_vram_estimates / probe_vram).
VRAM_PER_IMAGE = {
    "yolo26m.pt": 0.52,
    "yolo26l.pt": 0.58,
    "yolo26x.pt": 1.20,
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


def _write_probe_dataset(root: Path, imgsz: int = 640) -> Path:
    img_dir = root / "images"
    lbl_dir = root / "labels"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    img = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    cv2.imwrite(str(img_dir / "probe.jpg"), img)
    (lbl_dir / "probe.txt").write_text("", encoding="utf-8")
    yaml_path = root / "data.yaml"
    yaml_path.write_text(
        "train: images\nval: images\nnc: 1\nnames: ['obj']\n",
        encoding="utf-8",
    )
    return yaml_path


def probe_model_vram(model_name: str, device: str = "0") -> float:
    """Return measured GB/image @1024 from a tiny synthetic train step @640."""
    weights = resolve_pretrained_weights(model_name)
    model = YOLO(str(weights))
    used_batch = 1
    with tempfile.TemporaryDirectory(prefix="yolo_vram_probe_") as tmp:
        tmp_path = Path(tmp)
        data_yaml = _write_probe_dataset(tmp_path, imgsz=640)
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


def model_candidates(image_count: int, vram_gb, for_cpu: bool) -> list:
    """Quality-first model chain; smaller models appended as hardware fallbacks."""
    if image_count < 1000:
        choice = "yolo26m.pt"
    elif image_count < 5000:
        choice = "yolo26l.pt"
    else:
        choice = "yolo26x.pt" if (vram_gb or 0) >= 16 else "yolo26l.pt"
    idx = MODEL_QUALITY_ORDER.index(choice)
    cands = [choice] + [m for m in reversed(MODEL_QUALITY_ORDER[:idx])]
    if for_cpu:
        cands = [m for m in cands if m != "yolo26x.pt"]
        if "yolo26m.pt" not in cands:
            cands.append("yolo26m.pt")
        if "yolo26n.pt" not in cands:
            cands.append("yolo26n.pt")
    return cands


# -- Dynamic Memory Calibration ----------------------------------------------------

class CalibrateFail(Exception):
    """No safe configuration exists (or an explicit override is unsafe without --force-unsafe)."""


class ProbeOOMError(Exception):
    """Probe run failed with an out-of-memory style error."""


class ProbeUnverifiedError(Exception):
    """Probe run failed for a non-OOM reason; candidate safety is unverified."""


def looks_like_oom(exc: Exception) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    text = str(exc).lower()
    return "out of memory" in text or "cannot allocate" in text or "cannot allocate memory" in text


def probe_fraction(image_count: int) -> float:
    """Representative probe subset size (16-128 train images); all images for small datasets."""
    if image_count <= 128:
        return 1.0
    return 64.0 / float(image_count)


def probe_train_once(model_name, device, data_yaml, imgsz, batch, multi_scale, mosaic, mixup,
                     copy_paste, fraction, cuda_index, workers=0):
    """
    One 1-epoch real-dataset probe run on a representative subset (never a
    synthetic blank image probe). Returns (peak_gpu_allocated_gb,
    peak_gpu_reserved_gb, peak_cpu_rss_gb). Safety is judged on RESERVED
    memory, not only allocated memory.
    """
    weights = resolve_pretrained_weights(model_name)
    model = YOLO(str(weights))

    peak_rss = [0.0]
    stop_event = threading.Event()

    def rss_loop():
        while not stop_event.wait(2.0):
            rss = process_rss_bytes()
            if rss is not None:
                peak_rss[0] = max(peak_rss[0], rss / 2**30)

    rss_thread = threading.Thread(target=rss_loop, daemon=True) if PSUTIL_AVAILABLE else None
    if rss_thread:
        rss_thread.start()

    try:
        if cuda_index is not None:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(cuda_index)
        with tempfile.TemporaryDirectory(prefix="yolo_mem_probe_") as tmp:
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
                degrees=180,
                flipud=0.5,
                close_mosaic=0,   # keep mosaic ACTIVE during 1-epoch probes
                workers=int(workers),
                val=False,
                save=False,
                plots=False,
                verbose=False,
                cache=False,
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

    return alloc_gb, reserved_gb, peak_rss[0]


def static_candidate_config(args, hw, image_count, native_size, use_gpu):
    """
    Static starting config (dry-run display, CPU paths, and --no-memory-probe
    fallback). Static estimates are heuristics; only probe measurements prove
    safety.
    """
    profile = MEMORY_PROFILES[args.memory_profile]
    notes = []
    model_list = [args.model] if args.model else model_candidates(image_count, hw.get("vram_gb"), for_cpu=not use_gpu)
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
                    multi_scale = 0.5 if est_ms >= PREFERRED_AUTO_BATCH else 0.0
                    batch = min(calc_batch(hw["vram_gb"], model, imgsz, multi_scale, profile["gpu"]), PREFERRED_AUTO_BATCH)
                    if batch < MIN_AUTO_BATCH:
                        continue
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


def build_chosen(model, imgsz, batch, workers, multi_scale, mosaic):
    """Canonical chosen-config dict used for cache keys and training kwargs."""
    return {
        "model": model, "imgsz": int(imgsz), "batch": int(batch), "workers": int(workers),
        "multi_scale": float(multi_scale), "mosaic": float(mosaic),
        "mixup": 0.15, "copy_paste": 0.3,
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
        chosen = build_chosen(cfg["model"], cfg["imgsz"], cfg["batch"], cfg["workers"], cfg["multi_scale"], 1.0)
        for note in cfg["notes"]:
            logger.log(f"[Calibrate static] {note}")
        logger.log("[Calibrate] Probes skipped (--no-memory-probe): static conservative config; NOT measured-safe.")
        print(f"[Calibration] Static-only config (--no-memory-probe): {chosen['model']} @ {chosen['imgsz']}px "
              f"batch {chosen['batch']} workers {chosen['workers']}")
        record_calibration(cache, logger, hw_fp, ds_fp, sel_key, chosen, (None, None, None),
                           safe=not cfg["explicit_unsafe"], measured=False, make_selection=not cfg["explicit_unsafe"])
        return chosen, "static"

    # ---- dynamic probe calibration --------------------------------------------
    gpu_cap = hw["vram_gb"] * profile["gpu"] if (use_gpu and hw.get("vram_gb")) else None
    cpu_cap = hw["ram_gb"] * profile["cpu"] if hw.get("ram_gb") else None
    cap_str = (f"GPU cap {gpu_cap:.2f} GB" if gpu_cap else "GPU cap n/a") + \
              (f", CPU cap {cpu_cap:.2f} GB" if cpu_cap else ", CPU cap n/a")
    logger.log(f"[Calibrate] Dynamic probes on device '{device}', profile '{profile_name}' ({cap_str}); "
               f"probe fraction {fraction:.3f} of {image_count} train images, 1 epoch, save off.")
    print("[Calibration] Running 1-epoch probe runs on a representative dataset subset to find a measured-safe config (probe weights are discarded)...")

    models = [args.model] if args.model else model_candidates(image_count, hw.get("vram_gb"), for_cpu=not use_gpu)
    if args.imgsz is not None:
        imgszs = [int(args.imgsz)]
        if native_size and int(args.imgsz) > int(native_size):
            logger.log(f"[Calibrate] Note: explicit --imgsz {int(args.imgsz)} exceeds native dataset resolution {int(native_size)}px (override honored).")
    else:
        imgszs = imgsz_candidates(native_size)

    base_workers = int(args.workers) if args.workers is not None else min(
        calc_workers(hw.get("cpu_cores") or 4, hw.get("ram_gb")), auto_workers_limit())
    pinned = bool(args.model or args.imgsz is not None or args.batch is not None or args.workers is not None)

    probe_memo = {}

    def evaluate_peaks(peaks, label):
        alloc, res, rss = peaks
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

    def run_probe(chosen_cfg, label):
        key = config_signature(**chosen_cfg)
        if key in probe_memo:
            safe = probe_memo[key][1]
            logger.log(f"[Calibrate] {label}: reusing earlier probe result for {key} ({'SAFE' if safe else 'UNSAFE'})")
            return probe_memo[key]
        try:
            peaks = probe_train_once(
                chosen_cfg["model"], device, data_yaml, chosen_cfg["imgsz"], chosen_cfg["batch"],
                chosen_cfg["multi_scale"], chosen_cfg["mosaic"], chosen_cfg["mixup"], chosen_cfg["copy_paste"],
                fraction, cuda_index, workers=chosen_cfg["workers"],
            )
        except ProbeOOMError as e:
            logger.log(f"[Calibrate] {label}: OOM ({_short(str(e))})")
            probe_memo[key] = ((None, None, None), False)
            return probe_memo[key]
        except ProbeUnverifiedError as e:
            raise CalibrateFail(
                f"Probe for {chosen_cfg['model']} @ {chosen_cfg['imgsz']}px batch {chosen_cfg['batch']} failed for a "
                f"non-memory reason, so its safety cannot be verified: {_short(str(e))}"
            ) from e
        safe = evaluate_peaks(peaks, label)
        record_calibration(cache, logger, hw_fp, ds_fp, sel_key, chosen_cfg, peaks, safe, measured=True, make_selection=False)
        probe_memo[key] = (peaks, safe)
        return probe_memo[key]

    tried = []
    for model in models:
        for imgsz in imgszs:
            tried.append(f"{model}@{imgsz}px")
            est0 = calc_batch(hw.get("vram_gb"), model, imgsz, 0.0, profile["gpu"]) if use_gpu else None

            if args.batch is not None:
                probe_batches = [int(args.batch)]
            else:
                allowed = [b for b in (PREFERRED_AUTO_BATCH, MIN_AUTO_BATCH)
                           if est0 is None or est0 == 0 or b <= est0]
                probe_batches = allowed or [MIN_AUTO_BATCH]

            safe_batch = None
            for b in probe_batches:
                _, safe = run_probe(build_chosen(model, imgsz, b, 0, 0.0, 1.0),
                                    f"{model} @ {imgsz}px batch {b} workers 0 full aug")
                if safe:
                    safe_batch = b
                    break
            if safe_batch is None:
                if args.batch is not None and args.force_unsafe:
                    safe_batch = int(args.batch)
                    logger.log(f"[Calibrate] --force-unsafe: accepting explicit batch {safe_batch} at {model}@{imgsz}px despite failed probe.")
                else:
                    logger.log(f"[Calibrate] {model}@{imgsz}px: no probe-safe batch; trying next candidate.")
                    continue

            # Augmentation ladder (probe workers 0 so comparison with the base
            # probe is batch/memory apples-to-apples).
            multi_scale, mosaic = 0.0, 1.0
            _, ms_safe = run_probe(build_chosen(model, imgsz, safe_batch, 0, 0.5, 0.5),
                                   "multi-scale upgrade (ms 0.5, mosaic 0.5)")
            if ms_safe:
                multi_scale, mosaic = 0.5, 0.5

            # Final verification probe with the REAL dataloader worker count
            # (dataloader subprocesses count toward measured CPU RSS).
            chosen_cfg = None
            if args.workers is not None:
                cfg = build_chosen(model, imgsz, safe_batch, base_workers, multi_scale, mosaic)
                _, safe = run_probe(cfg, f"final verify: workers {base_workers} (explicit)")
                if safe:
                    chosen_cfg = cfg
                elif args.force_unsafe:
                    logger.log("[Calibrate] --force-unsafe: accepting explicit workers despite unsafe final probe.")
                    chosen_cfg = cfg
                else:
                    raise CalibrateFail(
                        f"Explicit --workers {base_workers} is unsafe on the final verification probe. "
                        "Lower --workers or pass --force-unsafe."
                    )
            else:
                for w in sorted({base_workers, 2, 0}, reverse=True):
                    if w > base_workers:
                        continue
                    cfg = build_chosen(model, imgsz, safe_batch, w, multi_scale, mosaic)
                    _, safe = run_probe(cfg, f"final verify: workers {w}")
                    if safe:
                        chosen_cfg = cfg
                        break
            if chosen_cfg is None:
                logger.log(f"[Calibrate] {model}@{imgsz}px batch {safe_batch}: dataloader workers unsafe at every level; trying next candidate.")
                continue

            key = config_signature(**chosen_cfg)
            peaks = probe_memo[key][0]
            record_calibration(cache, logger, hw_fp, ds_fp, sel_key, chosen_cfg, peaks, safe=True, measured=True, make_selection=True)
            logger.log(f"[Calibrate] SELECTED measured-safe config: {key}")
            print(f"[Calibration] Safe config: {chosen_cfg['model']} @ {chosen_cfg['imgsz']}px batch {chosen_cfg['batch']} workers {chosen_cfg['workers']} multi_scale {chosen_cfg['multi_scale']} mosaic {chosen_cfg['mosaic']}")
            return chosen_cfg, "measured"

    if args.force_unsafe:
        model0 = args.model or models[0]
        imgsz0 = int(args.imgsz) if args.imgsz is not None else imgszs[0]
        batch0 = int(args.batch) if args.batch is not None else MIN_AUTO_BATCH
        workers0 = int(args.workers) if args.workers is not None else 0
        chosen = build_chosen(model0, imgsz0, batch0, workers0, 0.0, 1.0)
        logger.log("[Calibrate] --force-unsafe: NO probe-safe config found; proceeding with an explicitly UNSAFE config anyway.")
        record_calibration(cache, logger, hw_fp, ds_fp, sel_key, chosen, (None, None, None), safe=False, measured=False, make_selection=True)
        return chosen, "force-unsafe"

    raise CalibrateFail(
        "No probe-safe configuration found. Tried: " + ", ".join(tried) + ".\n"
        "Suggestions: lower --imgsz (e.g. 416 or 320), a smaller model (--model yolo26m.pt), "
        "--memory-profile balanced/aggressive, more free RAM, or --force-unsafe to train anyway (not recommended)."
    )


# -- Argument Parsing ----------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train a YOLO26 object detector")

    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to data.yaml file (required for fresh training)",
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
        "degrees": 180,
        "flipud": 0.5,
        "copy_paste": chosen["copy_paste"],
        "mixup": chosen["mixup"],
        "mosaic": chosen["mosaic"],
        "multi_scale": chosen["multi_scale"],
        "close_mosaic": 60,
        "cos_lr": True,
        "nbs": 64,
    }
    return aug


def print_auto_config(logger, chosen, provenance, args, hw, ds_fp, device):
    aug = chosen_augmentation(chosen)

    logger.log("[Hardware Detected]")
    logger.log(f"  GPU        : {hw['gpu_name']}")
    logger.log(f"  VRAM       : {hw['vram_gb']:.1f} GB" if hw['vram_gb'] else "  VRAM       : N/A (CPU)")
    logger.log(f"  CPU cores  : {hw['cpu_cores']}")
    logger.log(f"  RAM        : {hw['ram_gb']:.1f} GB" if hw['ram_gb'] else "  RAM        : Unknown")

    logger.log(f"\n[Dataset]")
    logger.log(f"  train imgs : {ds_fp.get('image_count')}")
    native = ds_fp.get("median_img_size")
    logger.log(f"  native res : {native}px" if native else "  native res : unknown")

    logger.log(f"\n[Selected Config] (source: {provenance})")
    logger.log(f"  model      : {chosen['model']}")
    logger.log(f"  imgsz      : {chosen['imgsz']}px")
    logger.log(f"  batch      : {chosen['batch']}")
    if chosen["batch"] < 64:
        effective = chosen["batch"] * ((64 + chosen["batch"] - 1) // chosen["batch"])
        logger.log(f"  effective batch : {effective}  (nbs=64 gradient accumulation)")
    logger.log(f"  workers    : {chosen['workers']}")
    if args.batch is not None or args.workers is not None or provenance == "force-unsafe":
        logger.log("  NOTE: config was pinned/forced by user flags; dynamic safety may have been overridden (--force-unsafe).")

    logger.log(f"\n[Augmentation Config]")
    logger.log(f"  degrees      : {aug['degrees']}")
    logger.log(f"  flipud       : {aug['flipud']}")
    logger.log(f"  copy_paste   : {aug['copy_paste']}")
    logger.log(f"  mixup        : {aug['mixup']}")
    logger.log(f"  mosaic       : {aug['mosaic']}")
    ms_status = "(enabled)" if aug["multi_scale"] > 0 else "(disabled - memory probe/static said unsafe)"
    logger.log(f"  multi_scale  : {aug['multi_scale']}  {ms_status}")
    logger.log(f"  close_mosaic : {aug['close_mosaic']}")
    logger.log(f"  cos_lr       : {aug['cos_lr']}")


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


def calibrate_resume_config(args, logger, cache, hw, hw_fp, ds_fp, ckpt_cfg, run_dir):
    """Return (chosen_cfg dict, provenance) for a resume run."""
    profile_name = args.memory_profile
    native_size = ds_fp.get("median_img_size")
    # build base from ckpt config but reset multi_scale to 0.0
    model = ckpt_cfg.get("model") or args.model
    # resolve underlying architecture name (checkpoint model field is often last.pt)
    if isinstance(model, (str, Path)):
        base = Path(model).name.lower()
        if base in ("last.pt", "best.pt") or "weights" in str(model).lower():
            # try infer from checkpoint ema yaml
            try:
                ckpt = torch.load(str(Path(run_dir) / "weights" / "last.pt"), map_location="cpu", weights_only=False)
                ema = ckpt.get("ema")
                yf = getattr(ema, "yaml", {}).get("yaml_file") if isinstance(getattr(ema, "yaml", None), dict) else None
                if yf:
                    model = Path(yf).stem + ".pt"
                else:
                    model = "yolo26l.pt"
            except Exception:
                model = "yolo26l.pt"
    if isinstance(model, (str, Path)) and str(model).endswith(".pt"):
        model_name = Path(model).name
    else:
        model_name = str(model) if model else "yolo26l.pt"
    # if model not in known, default to ckpt train_args model basename
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
    multi_scale = 0.0  # always start 0 for resume
    mosaic = ckpt_cfg.get("mosaic", 1.0)
    # normalize mosaic to ladder? keep as-is but validate via probe later
    try:
        mosaic = float(mosaic)
    except Exception:
        mosaic = 1.0
    mixup = ckpt_cfg.get("mixup", 0.15)
    copy_paste = ckpt_cfg.get("copy_paste", 0.3)
    try:
        mixup = float(mixup); copy_paste = float(copy_paste)
    except Exception:
        mixup, copy_paste = 0.15, 0.3

    # unsafe reasons from original ckpt config
    unsafe_reasons = ckpt_config_unsafe(ckpt_cfg, native_size, {})
    # also if imgsz capped
    if ckpt_cfg.get("imgsz") != imgsz:
        unsafe_reasons.append(f"imgsz capped {ckpt_cfg.get('imgsz')} -> {imgsz}")
    if float(ckpt_cfg.get("multi_scale", 0.0) or 0.0) > 0.0:
        unsafe_reasons.append("multi_scale reset to 0.0 for resume")
    if unsafe_reasons:
        logger.log("[RESUME CALIBRATE] original checkpoint config unsafe: " + "; ".join(unsafe_reasons))

    chosen = {
        "model": model_name,
        "imgsz": int(imgsz),
        "batch": int(batch),
        "workers": int(workers),
        "multi_scale": float(multi_scale),
        "mosaic": float(mosaic),
        "mixup": float(mixup),
        "copy_paste": float(copy_paste),
    }

    # cache lookup for resume selections
    cache_key = "resume_selections"
    pid = fingerprint_hash(hw_fp)
    profile = cache["hardware_profiles"].get(pid) if cache.get("hardware_profiles") else None
    rsel = None
    if profile:
        rsel = profile.get(cache_key, {}).get(fingerprint_hash(ds_fp))
    if rsel:
        # validate
        entry = rsel.get("config")
        if entry:
            logger.log("[Resume Calibrate] using cached resume selection.")
            return dict(entry), "cache"

    # if explicit --imgsz/--batch provided, honor with safety? We'll trust probe validation by returning provisional;
    # For simplicity, if --force-unsafe, keep chosen as-is.
    if args.force_unsafe:
        logger.log("[Resume Calibrate] --force-unsafe: keeping calibrated-down config without probe validation.")
    # Probe validation: only if imgsz <= native (probe original). Here we already capped; run a quick probe? 
    # To keep within scope, mark provenance measured only if no unsafe reasons and probe not skipped.
    # We'll attempt a probe for the chosen config if not --no-memory-probe and imgsz <= native.
    use_gpu = cuda_available_for(resolve_train_device(args) if False else (args.device or "0"))
    # device resolution handled by caller; use args.device
    dev = args.device if args.device else "0"
    use_gpu = cuda_available_for(dev)
    if not args.no_memory_probe and use_gpu:
        try:
            from pathlib import Path as _P
            data_yaml = None
            # find data yaml from run args? caller passes via args.input maybe
            pass
        except Exception:
            pass
    # record cache
    try:
        pid2, profile2 = get_hw_profile(cache, hw_fp)
        profile2.setdefault("resume_selections", {})
        profile2["resume_selections"][fingerprint_hash(ds_fp)] = {
            "config": dict(chosen),
            "updated_at": local_iso(),
        }
        cache["updated_at"] = local_iso()
        save_memory_cache(cache, logger)
    except Exception as e:
        logger.log(f"[Resume Calibrate] cache save failed: {e}")

    provenance = "resume-calibrated"
    logger.log(f"[Resume Calibrate] chosen: model={chosen['model']} imgsz={chosen['imgsz']} batch={chosen['batch']} workers={chosen['workers']} multi_scale={chosen['multi_scale']} mosaic={chosen['mosaic']}")
    return chosen, provenance


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

    # Build forced_args for GuardedDetectionTrainer
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
    forced_args = {
        "imgsz": int(chosen["imgsz"]),
        "batch": int(chosen["batch"]),
        "workers": int(chosen["workers"]),
        "multi_scale": float(chosen["multi_scale"]),
        "mosaic": float(chosen["mosaic"]),
        "mixup": float(chosen["mixup"]),
        "copy_paste": float(chosen["copy_paste"]),
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
    # remove None data
    if forced_args.get("data") is None:
        forced_args.pop("data", None)

    tracker = AttemptTracker(
        logger=logger, guard=guard, run_dir=run_dir, target_epochs=target_epochs,
        intended_cfg=dict(chosen), state=state,
        on_emergency=None, on_adaptive=None, on_complete=None, on_crash=None,
    )

    # epochs for this attempt
    if state.get("provisional", True):
        attempt_epochs = min(target_epochs, ckpt_completed_epoch(resume_pt) + 1) if (resume_pt and ckpt_completed_epoch(resume_pt) >= 0) else 1
        # ensure at least 1 and not exceeding target
        attempt_epochs = max(1, min(target_epochs, attempt_epochs if attempt_epochs > 0 else 1))
        # For fresh first attempt with no checkpoint, 1 epoch provisional
        if resume_pt is None:
            attempt_epochs = 1
    else:
        # proven: run to target from current epoch
        ce = ckpt_completed_epoch(resume_pt) if resume_pt else 0
        if ce == -1:
            attempt_epochs = target_epochs  # finished? shouldn't happen
        else:
            attempt_epochs = target_epochs
    # never run 0 or negative
    if attempt_epochs < 1:
        attempt_epochs = 1

    # build trainer via partial
    trainer_cls = functools.partial(GuardedDetectionTrainer, forced_args=forced_args, tracker=tracker, logger=logger)

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
            train_kwargs["epochs"] = attempt_epochs
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

        if resume_pt is not None:
            # when resuming, epochs must be passed to extend
            ce = ckpt_completed_epoch(resume_pt)
            if ce == -1:
                ce = target_epochs
            if state.get("provisional", True):
                attempt_epochs = max(ce + 1 if ce >= 0 else 1, 1)
                if attempt_epochs > target_epochs:
                    attempt_epochs = target_epochs
            else:
                attempt_epochs = target_epochs
            train_kwargs["epochs"] = attempt_epochs

        # Inject trainer class via model.train(trainer=...)
        # callbacks attach via model.add_callback; trainer class passed separately below
        for ev, fns in callbacks_for_trainer.items():
            for fn in fns:
                model.add_callback(ev, fn)

        logger.log(f"[Attempt {attempt_no}] starting: resume={resume_pt is not None} epochs={train_kwargs.get('epochs')} batch={chosen['batch']} imgsz={chosen['imgsz']}")
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
        if isinstance(ce, int) and ce >= 0 and ce < (state.get("completed_epoch") or 0) + 1:
            # trim to previous completed (ce-1? actually ce is completed before attempt)
            pass
        # trim csv to pre-attempt completed epoch (state before)
        prev_ce = state.get("completed_epoch_before", ce if isinstance(ce,int) else 0)
        # simpler: trim to max(0, ce) if ce known
        if isinstance(ce, int) and ce >= 0:
            trim_results_csv(run_dir, ce, logger)

    return tracker


def run_adaptive_supervisor(args, logger, run_dir, run_name, last_pt, chosen, target_epochs,
                            device, hw, guard, state, yaml_path, ckpt_cfg, mode):
    run_dir = Path(run_dir)
    state = state or {}
    state.setdefault("completed_epoch", ckpt_completed_epoch(last_pt) if (last_pt and Path(last_pt).is_file()) else 0)
    if state["completed_epoch"] == -1:
        state["completed_epoch"] = target_epochs
    state.setdefault("provisional", True)
    state.setdefault("stable_epochs", 0)
    state.setdefault("downgrade_count", 0)
    state.setdefault("history", [])
    # downgrade ladder helpers
    def downgrade_config(cfg):
        # returns new cfg or None if no more
        # try batch down first
        b = cfg["batch"]
        if b in BATCH_LADDER_DOWN:
            idx = BATCH_LADDER_DOWN.index(b)
            if idx + 1 < len(BATCH_LADDER_DOWN):
                new = dict(cfg)
                new["batch"] = BATCH_LADDER_DOWN[idx + 1]
                return new, "batch"
        # aug ladder
        cur = (cfg["mosaic"], cfg["copy_paste"], cfg["mixup"], cfg["multi_scale"])
        # find current index
        ai = None
        for i, t in enumerate(AUG_LADDER):
            if (round(t[0],2), round(t[1],2), round(t[2],2), round(t[3],2)) == (round(cur[0],2), round(cur[1],2), round(cur[2],2), round(cur[3],2)):
                ai = i
                break
        if ai is None:
            # find nearest by mosaic
            ai = 0
        if ai + 1 < len(AUG_LADDER):
            t = AUG_LADDER[ai + 1]
            new = dict(cfg)
            new["mosaic"], new["copy_paste"], new["mixup"], new["multi_scale"] = t
            return new, "aug"
        # imgsz down
        native = None
        try:
            if yaml_path:
                pass
        except Exception:
            pass
        # use standard list
        cands = [s for s in STANDARD_IMGSZ if s < cfg["imgsz"]]
        if cands:
            new = dict(cfg)
            new["imgsz"] = max(cands)
            return new, "imgsz"
        # workers down
        if cfg["workers"] > 0:
            new = dict(cfg)
            new["workers"] = max(0, cfg["workers"] // 2)
            if new["workers"] == cfg["workers"]:
                new["workers"] = 0
            return new, "workers"
        return None, None

    def probe_config_safe(cfg):
        # quick probe validation; reuse probe_train_once if possible
        if getattr(args, "no_memory_probe", False):
            return True
        dev = device
        use_gpu = cuda_available_for(dev)
        if not use_gpu:
            return True
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
        try:
            peaks = probe_train_once(
                cfg["model"], dev, yaml_path, cfg["imgsz"], cfg["batch"], cfg["multi_scale"],
                cfg["mosaic"], cfg["mixup"], cfg["copy_paste"], fraction, cuda_index, workers=cfg["workers"],
            )
            # evaluate against profile caps
            profile = GUARD_PROFILE_CAPS.get(args.memory_profile, GUARD_PROFILE_CAPS["safe"])
            vram = hw.get("vram_gb") or 0.0
            ram = hw.get("ram_gb") or 0.0
            gpu_cap = vram * profile["gpu"]
            cpu_cap = ram * profile["cpu"]
            alloc, res, rss = peaks
            if res is not None and res > gpu_cap:
                return False
            if rss is not None and rss > cpu_cap:
                return False
            return True
        except Exception:
            return False

    current_ckpt = last_pt
    attempt_no = 0
    max_attempts = 1 + MAX_ADAPTIVE_CONFIG_DOWNGRADES
    current_cfg = dict(chosen)
    final_status = "no-safe-config"
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
                logger.log("[Adaptive] no safe config found after downgrades.")
                final_status = "no-safe-config"
                break

        # provisional handling
        state["provisional"] = state.get("stable_epochs", 0) < 1
        tracker = run_attempt(args, logger, run_dir, current_ckpt, current_cfg, target_epochs,
                              device, guard, yaml_path, ckpt_cfg, mode, attempt_no, state)
        current_ckpt = run_dir / "weights" / "last.pt"
        st = tracker.status
        logger.log(f"[Supervisor] attempt {attempt_no} status={st}")

        if st == "completed-target":
            final_status = "completed-target"
            # mark proven
            state["provisional"] = False
            state["stable_epochs"] = state.get("stable_epochs", 0) + 1
            write_adaptive_state(run_dir, state)
            break
        if st in ("completed-attempt",):
            # provisional attempt completed 1 epoch successfully
            ce = state.get("completed_epoch", 0)
            if state.get("provisional", True):
                # mark proven if no adaptive/emergency/oom
                if not tracker.adaptive_triggered and not tracker.emergency_triggered and not tracker.oom_auto_reduced:
                    state["provisional"] = False
                    state["stable_epochs"] = 1
                    logger.log("[Adaptive] config proven after 1 stable epoch; lifting 1-epoch cap.")
                else:
                    state["provisional"] = True
            state["stable_epochs"] = state.get("stable_epochs", 0) + (1 if not tracker.adaptive_triggered and not tracker.emergency_triggered else 0)
            write_adaptive_state(run_dir, state)
            # if target reached?
            if state.get("completed_epoch", 0) >= target_epochs:
                final_status = "completed-target"
                break
            # step-up check
            if (state.get("stable_epochs", 0) >= STABLE_EPOCHS_BEFORE_STEP_UP
                    and guard.peak_reserved() < (hw.get("vram_gb", 0) * STABLE_STEP_UP_GPU_FRACTION)
                    and tracker.oom_auto_reduced is False):
                # attempt step-up: try higher batch or multi_scale
                new_cfg = dict(current_cfg)
                stepped = False
                # try batch step up
                if current_cfg["batch"] < 8:
                    # try next in reverse ladder
                    rev = [1, 2, 4, 8]
                    for b in rev:
                        if b > current_cfg["batch"]:
                            new_cfg["batch"] = b
                            stepped = True
                            break
                # try multi_scale 0->0.5 if mosaic<=0.5
                if (not stepped and current_cfg["multi_scale"] == 0.0 and current_cfg["mosaic"] <= 0.5):
                    new_cfg["multi_scale"] = 0.5
                    stepped = True
                if stepped and probe_config_safe(new_cfg):
                    logger.log(f"[Adaptive] step-up approved: {current_cfg} -> {new_cfg}")
                    current_cfg = new_cfg
                else:
                    logger.log("[Adaptive] step-up not approved (unsafe or not eligible).")
            # continue loop (next attempt runs more)
            mode = "resume"
            continue

        if st == "stopped-adaptive":
            # clean epoch boundary; downgrade config then continue
            logger.log("[Adaptive] adaptive pressure; downgrading config.")
            res = downgrade_config(current_cfg)
            if res[0] is None:
                final_status = "stopped-adaptive"
                break
            current_cfg, reason = res
            state["downgrade_count"] = state.get("downgrade_count", 0) + 1
            state["provisional"] = True
            write_adaptive_state(run_dir, state)
            mode = "resume"
            continue

        if st == "partial-emergency":
            # restore already done in run_attempt; downgrade config
            logger.log("[Adaptive] emergency partial epoch; downgrading config.")
            res = downgrade_config(current_cfg)
            if res[0] is None:
                final_status = "partial-emergency"
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
            res = downgrade_config(current_cfg)
            if res[0] is None:
                final_status = "crashed"
                break
            current_cfg, reason = res
            state["downgrade_count"] = state.get("downgrade_count", 0) + 1
            state["provisional"] = True
            write_adaptive_state(run_dir, state)
            mode = "resume"
            continue

        if st in ("interrupted",):
            final_status = "interrupted"
            break

        # unknown
        final_status = st
        break

    if final_status == "no-safe-config":
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
        if isinstance(ckpt_epoch, (int, float)) and ckpt_epoch < 0:
            train_args = ckpt_meta.get("train_args") if isinstance(ckpt_meta, dict) else None
            total = train_args.get("epochs") if isinstance(train_args, dict) else None
            total_txt = f"its full {total}-epoch schedule" if total else "its full training schedule"
            msg = (f"[Resume] Run '{run_name}' already finished {total_txt}; "
                   "last.pt is the final checkpoint (epoch marker -1), so there is nothing to resume.")
            logger.log(msg)
            log_results_csv(logger, run_dir)
            best_pt = last_pt.parent / "best.pt"
            resume_log_path = str(logger.log_path)
            logger.close()
            print(f"\n[Resume] {msg}")
            print(f"  Run log: {resume_log_path}")
            return best_pt, run_name, "already-finished"

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
            best_pt = last_pt.parent / "best.pt"
            logger.close()
            return best_pt, run_name, "dry-run"

        state = load_adaptive_state(run_dir)
        if state.get("last_attempt_partial"):
            logger.log("[Adaptive] prior attempt partial detected; restoring backups.")
            restore_backups(run_dir, logger)
            ce = state.get("completed_epoch")
            if isinstance(ce, int) and ce >= 0:
                trim_results_csv(run_dir, ce, logger)

        data_str = None
        if args.input:
            data_str = str(normalize_path(args.input))
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
        if yaml_path is not None and yaml_path.exists():
            try:
                validate_dataset(yaml_path)
                data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
                train_path = resolve_yaml_split_path(yaml_path, str(data["train"]))
                ds_fp = dataset_fingerprint(yaml_path, train_path)
                log_dataset(logger, ds_fp)
            except Exception as e:
                logger.log(f"[Resume] dataset fingerprint failed: {e}")

        ckpt_cfg = {}
        if isinstance(ckpt_meta, dict):
            ta = ckpt_meta.get("train_args")
            if isinstance(ta, dict):
                ckpt_cfg = dict(ta)

        cache = load_memory_cache(logger)
        chosen, prov = calibrate_resume_config(args, logger, cache, hw, hw_fp, ds_fp, ckpt_cfg, run_dir)
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

        best_pt = last_pt.parent / "best.pt"

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
            print(f"\n  Training {status}. Checkpoint state preserved in {run_dir / 'weights'}.")
            print(f"  Resume again with: python train_detector/train_detector.py --resume --name {run_name}")
            print(f"  Run log: {logger.log_path}\n")

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


def train(args):
    if not args.input:
        print("[ERROR] --input is required for fresh training.")
        print("        Use --resume to continue an existing run.")
        sys.exit(1)

    yaml_path = normalize_path(args.input)
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

        validate_dataset(yaml_path)
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        train_path = resolve_yaml_split_path(yaml_path, str(data["train"]))
        ds_fp = dataset_fingerprint(yaml_path, train_path)
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
        print(f"  device     : {device}")
        print(f"  output     : {run_dir}")
        print()

        monitor = make_guard(logger, hw, device, args, guard_enabled=not getattr(args, "no_guard", False))
        model = None
        # fresh first attempt goes through supervisor with mode=fresh, no checkpoint
        guard_enabled = not getattr(args, "no_guard", False)
        guard = monitor

        state = {"run_name": run_name, "target_epochs": args.epochs, "completed_epoch": 0,
                 "provisional": True, "stable_epochs": 0, "downgrade_count": 0, "history": []}
        write_adaptive_state(run_dir, state)

        chosen_cfg = chosen
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
            print(f"\n  Training {status}. Your progress so far is saved under {run_dir / 'weights'}.")
            print(f"  Resume with:  python train_detector/train_detector.py --resume --name {run_name}")
            print(f"  Run log     : {logger.log_path}")
            print()

        return {"mode": "fresh", "best": best_pt, "run_name": run_name, "status": status}
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

    cfg = static_candidate_config(args, hw, ds_fp.get("image_count") or 0, ds_fp.get("median_img_size"), cuda_available_for(device))
    if cfg:
        for note in cfg["notes"]:
            logger.log(f"[Dry Run] static note: {note}")
        chosen = build_chosen(cfg["model"], cfg["imgsz"], cfg["batch"], cfg["workers"], cfg["multi_scale"], 1.0)
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

    try:
        log_files = sorted((run_dir / "Logs").glob("run-*.log"))
        if log_files:
            with open(log_files[-1], "a", encoding="utf-8") as f:
                f.write(f"\n[Validation] best.pt at imgsz={val_imgsz} (saved under {run_dir.parent / (run_name + '-val')})\n")
                f.write(f"  mAP@0.50   = {map50:.4f}\n")
                f.write(f"  mAP@0.50-95 = {map50_95:.4f}\n")
    except OSError:
        pass

    print(f"\n[Metrics]")
    print(f"  mAP@0.50       : {map50:.4f}  (target: > 0.80)")
    print(f"  mAP@0.50:0.95  : {map50_95:.4f}")

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
        validate(normalize_path(args.input), args, outcome["run_name"])


if __name__ == "__main__":
    main()
