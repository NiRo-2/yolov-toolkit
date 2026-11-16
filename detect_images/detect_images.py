r"""
YOLO Image Detection Script

Runs a trained YOLO model over a directory of images,
draws bounding boxes with confidence scores, and saves results.

Usage:
    python detect_images/detect_images.py --images /path/to/images --model /path/to/best.pt
    python detect_images/detect_images.py --images /path/to/images --model /path/to/best.pt --export-json

    --images      path to directory containing images (required)
                  supports Windows paths: c:\Users\Ni\Desktop\images
    --model       path to trained .pt model file (required)
    --conf        minimum confidence threshold, 0.0-1.0 (default: 0.25)
    --export-json export detections as JSON in detections/ (default: True)
    --recursive    scan subdirectories of --images for input images (default: True)
    --no-recursive only scan the top-level of --images
    --workers     worker threads for post-inference I/O (JSON + ExifTool + save)
                  integer or 'auto' (default: auto = min(8, os.cpu_count()))
    --batch       inference items per model() call. In tiled mode these are
                  tiles accumulated across source images; in non-tiled mode
                  they are whole images. Integer or 'auto'
                  (default: auto = hardware-derived from VRAM/RAM profile)
    --decode-workers  threads for cv2.imread + tile prep (default: auto)
    --device     inference device: auto | cpu | 0 (default: auto)
    --half       fp16 precision: auto | on | off (default: auto = on for CUDA, off for CPU)
    --memory-profile  safe | balanced | aggressive (default: balanced)
    --vram-frac  advanced: override the GPU memory fraction used by the profile
    --exiftool-stay-open  run exiftool as persistent -stay_open daemons
    --tiles       tile large images before detect, merging results with NMS
                  (default: True)
    --no-tiles    disable tiling; run whole-image inference (legacy behavior)
    --tile-imgsz  tile size in pixels when --tiles is on (default: 1024)
    --tile-overlap tile overlap fraction (default: 0.2)
    --tile-iou    NMS IoU threshold for merging tiled detections (default: 0.5)

By default, images are tiled before inference (tiling is skipped automatically
when an image already fits within one tile). Tiles from different source
images are batched into one model() call; detections from all tiles of one
image are merged with class-wise NMS. Pass --no-tiles to restore
whole-image inference.

The detection task (detect/segment/pose) comes from the loaded model
(YOLO.model.task), not a CLI flag. Segment models add polygon contours per
detection; on tiled runs, mask fragments of one object spanning several tiles
are chained transitively by box IoU (union-find) and composited into a single
whole polygon. Pose models add keypoints; across a merged tile group each
keypoint index takes the highest-visibility candidate within its member tile
extent. JSON exports gain polygons_px/polygons_norm (segment) and
keypoints_px/keypoints_norm (pose); all existing box fields stay unchanged for
every task.

When recursive scanning finds images in subfolders, outputs are still written
flat under detect_images/detections/<input_folder_name>/. Files in subfolders are renamed by
prepending the relative subpath joined with underscores
(e.g. sub/a/foo.jpg -> sub_a_foo.jpg) to avoid collisions.
JSON sidecars follow the same flattened naming convention: `image.file_name`
stores the flattened output image name, while `image.source_path` keeps the
original input path.

The pipeline is hardware-aware at startup: GPU VRAM, CPU cores and host RAM
are measured once and used to size the decode pool, the inference batch, the
post-processing workers, the precision and the device. A VRAM probe sizes the
first batch; an OOM halving ladder recovers mid-run, and a host RAM byte budget
keeps decoded arrays from queuing without limit. Pass --workers 1 --batch 1
--half off to reproduce the legacy sequential, single-image-at-a-time pipeline
byte-for-byte.

Output:
    detect_images/detections/<input_folder_name>/    <- annotated images + JSON
    detect_images/detections/<input_folder_name>/labels.txt  <- class_id -> class_name mapping
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, Tuple
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO  # type: ignore[union-attr]
from PIL import Image, ExifTags

try:
    import torch  # type: ignore[import-untyped]
except Exception:
    torch = None  # type: ignore[assignment]

# -- Precision kwargs -----------------------------------------------------------
# Ultralytics deprecated the ``half`` kwarg on YOLO model() calls in favor of
# ``quantize`` (integer bit width) on newer builds. Probe once at import time
# and route every inference call through ``_precision_kwargs`` so the correct
# kwarg is used without changing the CLI surface.
try:
    from ultralytics.cfg import DEFAULT_CFG_DICT as _UCFG  # noqa: E402
    _SUPPORTS_QUANTIZE = "quantize" in _UCFG
except Exception:
    _SUPPORTS_QUANTIZE = False


def _precision_kwargs(half_on: bool) -> dict:
    """Return the ultralytics precision kwarg dict for the requested half flag.

    ``False`` -> ``{}`` (FP32, unchanged).
    ``True`` on a build that supports ``quantize`` -> ``{"quantize": 16}`` (FP16).
    ``True`` on an older build -> ``{"half": True}`` (legacy kwarg).
    """
    if not half_on:
        return {}
    if _SUPPORTS_QUANTIZE:
        return {"quantize": 16}
    return {"half": True}

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from common.hardware import (  # noqa: E402
    BATCH_LADDER,
    MEMORY_PROFILES,
    detect_hardware,
    looks_like_oom,
    process_rss_bytes,
)
SCRIPT_DIR = Path(__file__).resolve().parent
_SIDECAR_DIR = _REPO_ROOT / "ortho_tag_sidecar"
if str(_SIDECAR_DIR) not in sys.path:
    sys.path.insert(0, str(_SIDECAR_DIR))
from ortho_tag_sidecar import (
    merge_pillow_gps_exif_into_metadata,
    verify_sidecar_json_file,
)

# tile_yolo_dataset is a repo-root package; _REPO_ROOT is already on sys.path
# above, so ``tile_yolo_dataset.tile_geometry`` resolves without polluting
# sys.path with the package directory itself (which would shadow the package
# for other modules and break cross-module imports).
from tile_yolo_dataset.tile_geometry import iter_tile_windows, nms_xyxy  # type: ignore


# -- Config --------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
DEFAULT_EXIFTOOL_DIR = _REPO_ROOT / "exiftool"
DEFAULT_EXIFTOOL_CANDIDATES = (
    DEFAULT_EXIFTOOL_DIR / "exiftool.exe",
    DEFAULT_EXIFTOOL_DIR / "exiftool",
)
DEFAULT_EXIFTOOL_PERL = DEFAULT_EXIFTOOL_DIR / "exiftool_files" / "perl.exe"
DEFAULT_EXIFTOOL_PL = DEFAULT_EXIFTOOL_DIR / "exiftool_files" / "exiftool.pl"
EXIFTOOL_DOWNLOAD_URL = "https://exiftool.org/"

# Box and label style
BOX_COLOR       = (0, 200, 0)      # green
BOX_THICKNESS   = 2
FONT            = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE      = 0.6
FONT_THICKNESS  = 2
LABEL_BG_COLOR  = (0, 200, 0)      # green background
LABEL_TX_COLOR  = (0, 0, 0)        # black text
LABEL_PADDING   = 4

# Segment mask overlay and pose keypoint style
MASK_ALPHA      = 0.35             # semi-transparent fill over the mask polygon
KEYPOINT_COLOR  = (0, 0, 255)      # red keypoints
KEYPOINT_RADIUS = 4


# -- Path Normalization --------------------------------------------------------

def normalize_path(raw: str) -> Path:
    """Handle Windows and Unix paths on any OS."""
    cleaned = raw.strip().strip('"').strip("'")
    cleaned = cleaned.replace("\\", "/")
    return Path(cleaned).resolve()


def flat_output_name(img_path: Path, images_root: Path) -> str:
    """Return a flat output basename for an image found under images_root.

    Top-level files keep their basename. Files in subfolders are prefixed with
    the relative parent path joined by underscores so flat outputs don't
    collide (e.g. sub/a/foo.jpg -> sub_a_foo.jpg).
    """
    try:
        rel = img_path.relative_to(images_root)
    except ValueError:
        return img_path.name
    parent_parts = rel.parent.parts
    if not parent_parts or parent_parts == (".",):
        return img_path.name
    return "_".join(parent_parts) + "_" + img_path.name


# -- Argument Parsing ----------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLO detection on a folder of images")

    parser.add_argument(
        "--images", type=str, required=True,
        help="Path to directory containing input images"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to trained .pt model file"
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Minimum confidence threshold 0.0-1.0 (default: 0.25)"
    )
    parser.add_argument(
        "--save-all", action="store_true", default=False,
        help="Save all images including those with no detections"
    )
    parser.add_argument(
        "--export-json", action="store_true", default=True,
        help="Export detections as JSON in detections/ (default: True)"
    )
    parser.add_argument(
        "--export-annotated-images", dest="export_annotated_images",
        action="store_true",
        help="Save annotated output images (default: True)"
    )
    parser.add_argument(
        "--no-export-annotated-images", dest="export_annotated_images",
        action="store_false",
        help="Disable annotated image export and write JSON-only outputs for detections"
    )
    parser.set_defaults(export_annotated_images=True)
    parser.add_argument(
        "--exiftool", type=str, default=None,
        help="Optional path to exiftool executable for full metadata copy"
    )
    parser.add_argument(
        "--allow-missing-exiftool", action="store_true", default=False,
        help="Allow run without exiftool (metadata preservation will be limited)"
    )
    parser.add_argument(
        "--verify-b3dm", action="store_true", default=False,
        help="After each JSON export, verify Ortho-Tag georeference keys; exit 1 if any sidecar fails"
    )
    parser.add_argument(
        "--recursive", dest="recursive", action="store_true",
        help="Scan subdirectories of --images for input images (default: True)"
    )
    parser.add_argument(
        "--no-recursive", dest="recursive", action="store_false",
        help="Only scan the top-level of --images"
    )
    parser.set_defaults(recursive=True)
    parser.add_argument(
        "--workers", type=str, default="auto",
        help="Worker threads for post-inference I/O (JSON + ExifTool + save). "
             "Integer or 'auto' (default: auto = min(8, os.cpu_count()))."
    )
    parser.add_argument(
        "--batch", type=str, default="auto",
        help="GPU inference batch size. Integer or 'auto' "
             "(default: auto = 8 if CUDA is available, else 1)."
    )
    parser.add_argument(
        "--tiles", dest="tiles", action="store_true",
        help="Tile large images before detect (default: True)"
    )
    parser.add_argument(
        "--no-tiles", dest="tiles", action="store_false",
        help="Disable tiling; whole-image inference"
    )
    parser.set_defaults(tiles=True)
    parser.add_argument(
        "--tile-imgsz", type=int, default=1024,
        help="Tile size in pixels when --tiles is on (default: 1024)"
    )
    parser.add_argument(
        "--tile-overlap", type=float, default=0.2,
        help="Tile overlap fraction (default: 0.2)"
    )
    parser.add_argument(
        "--tile-iou", type=float, default=0.5,
        help="NMS IoU for merging tiled detections (default: 0.5)"
    )
    parser.add_argument(
        "--decode-workers", type=str, default="auto",
        help="Threads for cv2.imread + tile prep. Integer or 'auto' "
             "(default: auto = clamp(cpu_count // 6, 1, 6))."
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Inference device: auto | cpu | 0 (default: auto; picks GPU 0 when CUDA is available)."
    )
    parser.add_argument(
        "--half", type=str, default="auto",
        help="FP16 precision: auto | on | off (default: auto = on for CUDA, off for CPU)."
    )
    parser.add_argument(
        "--memory-profile", type=str, default="balanced",
        choices=["safe", "balanced", "aggressive"],
        help="Memory ceiling profile: safe | balanced | aggressive (default: balanced)."
    )
    parser.add_argument(
        "--vram-frac", type=float, default=None,
        help="Advanced: override the GPU memory fraction used by the memory profile."
    )
    parser.add_argument(
        "--exiftool-stay-open", action="store_true", default=False,
        help="Run exiftool as persistent -stay_open daemons (min(4, workers)) instead of "
             "spawning one process per image."
    )

    return parser.parse_args()


def _cuda_available() -> bool:
    return bool(torch is not None and torch.cuda.is_available())


def resolve_device(value) -> Tuple[str, str]:
    """Resolve --device argument. Returns (device_label, source_label)."""
    if isinstance(value, str) and value.lower() == "auto":
        return ("cuda:0" if _cuda_available() else "cpu"), "auto"
    dev = str(value).strip().lower()
    if dev in ("", "cpu"):
        return "cpu", "manual"
    import re as _re
    m = _re.search(r"\d+", dev)
    if not _cuda_available():
        return "cpu", "manual (CUDA unavailable)"
    return f"cuda:{m.group()}" if m else "cuda:0", "manual"


def resolve_half(value, cuda: bool) -> Tuple[bool, str]:
    """Resolve --half argument. Returns (half_on, source_label)."""
    if isinstance(value, str) and value.lower() == "auto":
        return (True if cuda else False), "auto"
    v = str(value).strip().lower()
    if v in ("on", "true", "1", "yes"):
        return True, "manual"
    if v in ("off", "false", "0", "no"):
        return False, "manual"
    print(f"[ERROR] Invalid --half value: {value!r} (must be 'auto', 'on' or 'off')")
    sys.exit(1)


def resolve_profile(name: str) -> dict:
    """Return the memory profile dict for ``name`` (safe/balanced/aggressive)."""
    return MEMORY_PROFILES.get(str(name).strip().lower(), MEMORY_PROFILES["balanced"])


def resolve_workers(value, cpu_cores: int, decode_workers: int, half_on: bool,
                    device_is_cpu: bool) -> Tuple[int, str]:
    """Resolve --workers argument with hardware awareness.

    Post-processing threads are capped by available cores minus the decode
    pool, and further capped by host RAM when a byte budget is in play.
    """
    cpu_cores = max(1, cpu_cores)
    decode_workers = max(1, decode_workers)
    if isinstance(value, str) and value.lower() == "auto":
        # Leave headroom for the decode pool and the main thread.
        n = max(1, min(8, cpu_cores - decode_workers + 1))
        return n, "auto"
    try:
        n = int(value)
    except (TypeError, ValueError):
        print(f"[ERROR] Invalid --workers value: {value!r} (must be an integer or 'auto')")
        sys.exit(1)
    return max(1, n), "manual"


def resolve_decode_workers(value, cpu_cores: int) -> Tuple[int, str]:
    """Resolve --decode-workers argument."""
    if isinstance(value, str) and value.lower() == "auto":
        n = max(1, min(6, max(1, cpu_cores) // 6))
        return n, "auto"
    try:
        n = int(value)
    except (TypeError, ValueError):
        print(f"[ERROR] Invalid --decode-workers value: {value!r} (must be an integer or 'auto')")
        sys.exit(1)
    return max(1, n), "manual"


def resolve_batch(value, budget_bytes: int, per_item_bytes: int,
                  cuda: bool, half_on: bool) -> Tuple[int, str]:
    """Resolve --batch argument.

    ``auto`` plans a ladder-clamped batch from the VRAM budget and the measured
    per-item inference cost; explicit integers pass through (floor 1).
    """
    if isinstance(value, str) and value.lower() == "auto":
        if budget_bytes <= 0 or per_item_bytes <= 0:
            return (8 if cuda else 1), "auto"
        n = max(1, budget_bytes // max(1, per_item_bytes))
        n = max(1, min(n, 64))
        ladder = BATCH_LADDER
        best = ladder[0]
        for v in ladder:
            if v <= n:
                best = v
            else:
                break
        return best, "auto"
    try:
        n = int(value)
    except (TypeError, ValueError):
        print(f"[ERROR] Invalid --batch value: {value!r} (must be an integer or 'auto')")
        sys.exit(1)
    return max(1, n), "manual"


# -- VRAM probe, batch planning and OOM recovery ---------------------------------

def _empty_cuda():
    """Best-effort CUDA cache cleanup; no-op when torch is unavailable."""
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def measure_inference_vram(engine, imgsz, half, device="cuda:0", runs=2):
    """Measure peak reserved VRAM for one inference item at ``imgsz``.

    Runs ``runs`` warmup forwards, resets peak stats and returns the
    max_memory_reserved() delta scaled by a 1.15x fragmentation factor. Returns
    0 when CUDA is unavailable or the probe fails.
    """
    if torch is None or not torch.cuda.is_available():
        return 0
    import numpy as _np
    dummy = _np.zeros((imgsz, imgsz, 3), dtype=_np.uint8)
    try:
        for _ in range(runs):
            engine([dummy], imgsz=imgsz, verbose=False, device=device, **_precision_kwargs(half))
        torch.cuda.reset_peak_memory_stats()
        engine([dummy], imgsz=imgsz, verbose=False, device=device, **_precision_kwargs(half))
        peak = torch.cuda.max_memory_reserved()
        return int(peak * 1.15)
    except Exception:
        return 0


def plan_batch(budget_bytes, per_item_bytes, ladder=None):
    """Clamp a batch count to ``ladder`` (default BATCH_LADDER), floor 1."""
    if ladder is None:
        ladder = BATCH_LADDER
    if budget_bytes <= 0 or per_item_bytes <= 0:
        return 1
    n = max(1, budget_bytes // max(1, per_item_bytes))
    n = max(1, min(n, ladder[-1]))
    best = ladder[0]
    for v in ladder:
        if v <= n:
            best = v
        else:
            break
    return best


def run_batch(engine, items, imgsz, conf, half, task, batch, device="cuda:0"):
    """Run inference over ``items`` with an OOM halving ladder.

    Returns a flat list of (item, result) pairs. On CUDA OOM the batch is
    halved (up to 4 times); any other exception isolates the failing item and
    continues. A batch-1 OOM is a hard failure.
    """
    results = []
    if not items:
        return results
    remaining = list(items)
    working_batch = max(1, batch)
    halvings = 0
    while remaining:
        chunk = remaining[:working_batch]
        try:
            outs = engine(chunk, conf=conf, imgsz=imgsz, verbose=False, device=device, **_precision_kwargs(half))
            for it, r in zip(chunk, outs):
                results.append((it, r))
            remaining = remaining[working_batch:]
        except Exception as e:
            if looks_like_oom(e) and working_batch > 1 and halvings < 4:
                _empty_cuda()
                working_batch = max(1, working_batch // 2)
                halvings += 1
                continue
            if looks_like_oom(e):
                # batch 1 still OOMs -> hard fail.
                raise RuntimeError(
                    f"Inference OOM at batch 1 (imgsz={imgsz}, half={half}, device={device}). "
                    f"Lower --imgsz/--tile-imgsz, use --half off, or reduce --vram-frac."
                ) from e
            # Non-OOM failure: isolate the failing item and continue.
            bad = chunk[0]
            results.append((bad, None))
            remaining = remaining[1:]
            working_batch = max(1, min(working_batch, len(remaining) if remaining else 1))
    return results


def free_vram_info():
    """Return (free_bytes, total_bytes) for CUDA device 0, or (None, None)."""
    if torch is None or not torch.cuda.is_available():
        return None, None
    try:
        free, total = torch.cuda.mem_get_info(0)
        return int(free), int(total)
    except Exception:
        return None, None


class ByteBudget:
    """Host-RAM semaphore acquired before decode, released after post-processing.

    Sized from the memory profile: ``min(total_ram * cpu_frac, available_ram * 0.5)``.
    Per-image estimates are ``2 * w * h * 3`` (original held alive by tile views
    plus the annotated copy) plus a JSON allowance.
    """

    def __init__(self, budget_bytes: int):
        self._budget = max(0, int(budget_bytes))
        self._cond = threading.Condition()
        self._used = 0

    @property
    def budget(self) -> int:
        return self._budget

    def acquire(self, image_bytes: int) -> bool:
        if self._budget <= 0:
            return True
        with self._cond:
            while self._used + image_bytes > self._budget:
                self._cond.wait()
            self._used += image_bytes
            return True

    def release(self, image_bytes: int) -> None:
        if self._budget <= 0:
            return
        with self._cond:
            self._used = max(0, self._used - image_bytes)
            self._cond.notify_all()


def estimate_image_bytes(w: int, h: int, json_bytes: int = 4096) -> int:
    """Per-image host-RAM estimate: original + annotated copy + JSON allowance."""
    return 2 * max(0, int(w)) * max(0, int(h)) * 3 + json_bytes


class StayOpenExifToolPool:
    """Persistent ``exiftool -stay_open True -@ -`` process pool (opt-in).

    Each request is written to stdin terminated by ``-execute\\n`` and read
    until the ``{ready}`` marker. A watchdog respawns a hung process once,
    then falls back to a per-image spawn for that call.
    """

    def __init__(self, cmd, n_processes: int, timeout: float = 30.0):
        self._cmd = list(cmd)
        self._n = max(1, int(n_processes))
        self._timeout = float(timeout)
        self._procs = []
        self._lock = threading.Lock()
        self._watchdogs = []
        self._spawn_all()

    def _spawn_all(self):
        for _ in range(self._n):
            self._procs.append(self._spawn_one())

    def _spawn_one(self):
        try:
            proc = subprocess.Popen(
                self._cmd + ["-stay_open", "True", "-@", "-"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True, bufsize=1,
            )
        except Exception:
            return None
        watchdog = threading.Thread(target=self._watchdog, args=(proc,), daemon=True)
        watchdog.start()
        self._watchdogs.append(watchdog)
        return proc

    def _watchdog(self, proc):
        # No-op placeholder; liveness is checked per request via the {ready} read.
        return

    def _read_ready(self, proc, timeout):
        """Read stdout until ``{ready}`` or timeout.

        Returns ``(ok, lines)`` where ``lines`` is everything emitted *before*
        the ready marker (i.e. the actual command output), ready marker excluded.
        """
        lines = []
        try:
            while True:
                line = proc.stdout.readline()
                if not line:
                    return False, lines
                if line.strip() == "{ready}":
                    return True, lines
                lines.append(line)
        except Exception:
            return False, lines

    def run(self, args_list):
        """Run an exiftool command (list of args) and return (returncode, stdout, stderr)."""
        with self._lock:
            for attempt in range(2):
                proc = self._procs[attempt % len(self._procs)] if self._procs else None
                if proc is None or proc.poll() is not None:
                    proc = self._spawn_one()
                    if proc is None:
                        return self._fallback(args_list)
                    self._procs[attempt % len(self._procs)] = proc
                try:
                    payload = "\n".join(args_list) + "\n-execute\n"
                    proc.stdin.write(payload)
                    proc.stdin.flush()
                    ok, out_lines = self._read_ready(proc, self._timeout)
                    if not ok:
                        try:
                            proc.kill()
                        except Exception:
                            pass
                        continue
                    out = "".join(out_lines)
                    return 0, out, ""
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    continue
            return self._fallback(args_list)

    def _collect_output(self, proc):
        # Compatibility shim. Real output is captured by _read_ready, which
        # returns the lines emitted before {ready}; this hook is retained so
        # external callers that drain stdout directly still have an API.
        return []

    def _fallback(self, args_list):
        try:
            proc = subprocess.run(self._cmd + args_list, capture_output=True, text=True, check=False)
            return proc.returncode, proc.stdout, proc.stderr
        except Exception as e:
            return 1, "", str(e)

    def shutdown(self):
        with self._lock:
            for proc in self._procs:
                try:
                    if proc.poll() is None:
                        proc.stdin.write("-stay_open\nFalse\n-execute\n")
                        proc.stdin.flush()
                        proc.terminate()
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
            self._procs = []
            self._watchdogs = []


# -- Drawing -------------------------------------------------------------------

def _to_numpy(t):
    """Return a numpy array from a torch tensor or array-like."""
    if hasattr(t, "cpu"):
        return t.cpu().numpy()
    return np.asarray(t)


def _result_polygons(result) -> list:
    """Per-box polygon contours (source-image pixel coords) from result.masks.

    Masks are resized to the result's native shape, binarized, and reduced to
    the largest external contour via cv2.findContours (one polygon per box).
    """
    masks = getattr(result, "masks", None)
    if masks is None or getattr(masks, "data", None) is None:
        return []
    orig_shape = getattr(result, "orig_shape", None)
    polygons = []
    for mask in masks.data:
        arr = _to_numpy(mask)
        if orig_shape is not None and arr.shape[:2] != tuple(orig_shape):
            arr = cv2.resize(arr, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_LINEAR)
        binary = (arr > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            biggest = max(contours, key=cv2.contourArea)
            polygons.append([(float(px), float(py)) for px, py in biggest.reshape(-1, 2)])
        else:
            polygons.append([])
    return polygons


def _result_keypoints(result) -> list:
    """Per-box keypoint lists [[x, y, visibility], ...] in native image coords."""
    keypoints_obj = getattr(result, "keypoints", None)
    if keypoints_obj is None or getattr(keypoints_obj, "data", None) is None:
        return []
    out = []
    for kp in keypoints_obj.data:
        out.append([[float(x), float(y), float(vis)] for x, y, vis in _to_numpy(kp)])
    return out


def results_to_dets(result, x_offset: float = 0.0, y_offset: float = 0.0, task: str = "detect") -> list:
    """Adapt one ultralytics Results object into a flat det-dict list for its task.

    Base dict (every task): {"cls": int, "conf": float, "x1": float, "y1": float,
    "x2": float, "y2": float}. Coordinates are offset into source-image space
    (used when the Results came from a tile crop rather than the full image).
    Segment dets additionally carry "polygon": [(x, y), ...]; pose dets carry
    "keypoints": [[x, y, vis], ...]. Detect dets are byte-identical to the
    historical 5-field behavior.
    """
    polygons = _result_polygons(result) if task == "segment" else []
    keypoints = _result_keypoints(result) if task == "pose" else []
    dets = []
    for i, box in enumerate(result.boxes):
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        det = {
            "cls": int(box.cls[0]),
            "conf": float(box.conf[0]),
            "x1": x1 + x_offset,
            "y1": y1 + y_offset,
            "x2": x2 + x_offset,
            "y2": y2 + y_offset,
        }
        if task == "segment":
            det["polygon"] = [(px + x_offset, py + y_offset) for px, py in polygons[i]] if i < len(polygons) else []
        elif task == "pose":
            det["keypoints"] = [
                [kx + x_offset, ky + y_offset, vis] for kx, ky, vis in keypoints[i]
            ] if i < len(keypoints) else []
        dets.append(det)
    return dets


def _boxes_iou(first: dict, second: dict) -> float:
    inter_w = max(0.0, min(first["x2"], second["x2"]) - max(first["x1"], second["x1"]))
    inter_h = max(0.0, min(first["y2"], second["y2"]) - max(first["y1"], second["y1"]))
    intersection = inter_w * inter_h
    first_area = max(0.0, first["x2"] - first["x1"]) * max(0.0, first["y2"] - first["y1"])
    second_area = max(0.0, second["x2"] - second["x1"]) * max(0.0, second["y2"] - second["y1"])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def group_by_iou(dets: list, iou_thresh: float) -> list:
    """Group detections per class by box IoU, chained transitively via union-find.

    An object spanning three or more tiles merges even when its outer tiles'
    boxes don't overlap directly, as long as each neighbor pair crosses the
    threshold. Groups never mix classes.
    """
    parent = list(range(len(dets)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    by_class = defaultdict(list)
    for i, det in enumerate(dets):
        by_class[det["cls"]].append(i)
    for indices in by_class.values():
        for a_pos in range(len(indices)):
            for b_pos in range(a_pos + 1, len(indices)):
                ia, ib = indices[a_pos], indices[b_pos]
                if _boxes_iou(dets[ia], dets[ib]) >= iou_thresh:
                    ra, rb = find(ia), find(ib)
                    if ra != rb:
                        parent[rb] = ra

    grouped = defaultdict(list)
    for i, det in enumerate(dets):
        grouped[find(i)].append(det)
    return list(grouped.values())


def _merge_mask_group(group: list) -> dict:
    """Composite a group of tile mask fragments into one whole-polygon detection.

    Member polygons are OR-filled onto a canvas covering only the group's union
    bbox; cv2.findContours on the composite yields the merged polygon (no slice
    seam). Box = union bbox, conf = max.
    """
    ux1 = min(d["x1"] for d in group)
    uy1 = min(d["y1"] for d in group)
    ux2 = max(d["x2"] for d in group)
    uy2 = max(d["y2"] for d in group)
    canvas_w = max(1, int(round(ux2 - ux1)))
    canvas_h = max(1, int(round(uy2 - uy1)))
    canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    drew_any = False
    for det in group:
        points = np.array(
            [[int(round(px - ux1)), int(round(py - uy1))] for px, py in det.get("polygon", [])],
            dtype=np.int32,
        )
        if len(points) >= 3:
            cv2.fillPoly(canvas, [points], 1)
            drew_any = True
    polygon: list = []
    if drew_any:
        contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            biggest = max(contours, key=cv2.contourArea)
            polygon = [(float(px + ux1), float(py + uy1)) for px, py in biggest.reshape(-1, 2)]
    if not polygon:
        fallback = max(group, key=lambda d: len(d.get("polygon", [])))
        polygon = list(fallback.get("polygon", []))
    return {
        "cls": group[0]["cls"],
        "conf": max(d["conf"] for d in group),
        "x1": ux1, "y1": uy1, "x2": ux2, "y2": uy2,
        "polygon": polygon,
    }


def _merge_keypoint_group(group: list) -> dict:
    """Merge a tile group's keypoints by per-index visibility voting.

    A candidate counts only if its coordinates fall inside the member tile's
    extent; the surviving highest-visibility candidate wins. Indices with no
    valid candidate are kept as [0, 0, 0]. Box = union bbox, conf = max.
    """
    ux1 = min(d["x1"] for d in group)
    uy1 = min(d["y1"] for d in group)
    ux2 = max(d["x2"] for d in group)
    uy2 = max(d["y2"] for d in group)
    k_count = max((len(d.get("keypoints", [])) for d in group), default=0)
    merged_keypoints = []
    for k in range(k_count):
        best = None
        for det in group:
            kps = det.get("keypoints", [])
            if k >= len(kps):
                continue
            x, y, vis = kps[k]
            tile = det.get("tile")
            if tile is not None:
                tx1, ty1, tx2, ty2 = tile
                if x < tx1 - 1e-6 or x > tx2 + 1e-6 or y < ty1 - 1e-6 or y > ty2 + 1e-6:
                    continue
            if best is None or vis > best[2]:
                best = [x, y, vis]
        merged_keypoints.append(best if best is not None else [0.0, 0.0, 0.0])
    return {
        "cls": group[0]["cls"],
        "conf": max(d["conf"] for d in group),
        "x1": ux1, "y1": uy1, "x2": ux2, "y2": uy2,
        "keypoints": merged_keypoints,
    }


def merge_grouped_dets(dets: list, iou_thresh: float, task: str) -> list:
    """Merge a tiled image's segment/pose detections (union-find grouping + compositing).

    Single-member groups pass through unchanged (no canvas compositing).
    """
    merged = []
    for group in group_by_iou(dets, iou_thresh):
        if len(group) == 1:
            merged.append(group[0])
        elif task == "segment":
            merged.append(_merge_mask_group(group))
        else:
            merged.append(_merge_keypoint_group(group))
    return merged


def detect_image_tiled(model, image_bgr, conf, tile, overlap, iou, batch_size=1, task: str = "detect", half=False) -> list:
    """Run tiled detection over one image; returns a flat list of det dicts.

    When ``batch_size <= 1`` this runs the legacy per-tile path verbatim so
    ``--workers 1 --batch 1 --half off`` reproduces the previous release
    byte-for-byte. For ``batch_size > 1`` it delegates to :class:`TileBatcher`,
    which accumulates tiles across source images into one model() call.
    Crops all tile windows for this image and runs them through the model in
    chunks of at most ``batch_size`` tiles per call. ``imgsz=tile`` is passed
    explicitly so Ultralytics runs inference at the tile's native size instead
    of silently resizing crops to its default 640. When the image fits within
    one tile, ``iter_tile_windows`` returns a single whole-image window, so
    this is equivalent to whole-image inference. Detect dets from multiple
    tiles are merged with class-wise NMS; segment and pose dets are merged
    with transitive IoU chaining (`group_by_iou`) plus mask compositing or
    keypoint voting, and carry their per-tile extent in "tile".
    """
    if batch_size is None or int(batch_size) <= 1:
        h, w = image_bgr.shape[:2]
        windows = iter_tile_windows(w, h, tile, overlap)
        crops = [image_bgr[ty1:ty2, tx1:tx2] for (tx1, ty1, tx2, ty2) in windows]

        tile_results = []
        for crop in crops:
            tile_results.append(model([crop], conf=conf, imgsz=tile, verbose=False, **_precision_kwargs(half))[0])

        all_dets = []
        for (tx1, ty1, tx2, ty2), result in zip(windows, tile_results):
            dets = results_to_dets(result, x_offset=tx1, y_offset=ty1, task=task)
            if task != "detect":
                for det in dets:
                    det["tile"] = (tx1, ty1, tx2, ty2)
            all_dets.extend(dets)

        if len(windows) == 1:
            return all_dets
        if task == "detect":
            return nms_xyxy(all_dets, iou_thresh=iou)
        return merge_grouped_dets(all_dets, iou, task)

    batcher = TileBatcher(model, conf, tile, overlap, iou, batch_size, task)
    for (tx1, ty1, tx2, ty2) in iter_tile_windows(image_bgr.shape[1], image_bgr.shape[0], tile, overlap):
        batcher.submit(image_bgr[ty1:ty2, tx1:tx2], (tx1, ty1, tx2, ty2), job_id=0)
    batcher.flush()
    return batcher.results_for(0)


class TileBatcher:
    """Accumulate inference items (tiles across images) and run them in batches.

    Each submitted item carries a ``job_id``; results are routed back to the
    owning job and merged once the job's last tile lands. Per-image detections
    depend only on that image's own tiles, so global batching cannot change
    results: merge/NMS still runs once per image.
    """

    def __init__(self, model, conf, tile, overlap, iou, batch_size, task, device="cuda:0", half=False,
                 on_complete=None):
        self._model = model
        self._conf = conf
        self._tile = tile
        self._overlap = overlap
        self._iou = iou
        self._task = task
        self._device = device
        self._half = half
        self._batch = max(1, int(batch_size))
        self._pending = []            # (crop, window, job_id)
        self._remaining = {}          # job_id -> remaining tile count
        self._dets_by_job = defaultdict(list)  # job_id -> [det, ...]
        self._results = {}            # job_id -> merged det list
        self._completed_jobs = set()
        self._on_complete = on_complete

    def submit(self, crop, window, job_id):
        self._pending.append((crop, window, job_id))
        self._remaining[job_id] = self._remaining.get(job_id, 0) + 1

    def flush(self):
        if not self._pending:
            return
        # Batch across jobs up to self._batch items per model() call.
        while self._pending:
            chunk = self._pending[:self._batch]
            self._pending = self._pending[self._batch:]
            crops = [c for (c, _w, _j) in chunk]
            try:
                outs = self._model(crops, conf=self._conf, imgsz=self._tile,
                                   verbose=False, device=self._device,
                                   **_precision_kwargs(self._half))
            except Exception:
                outs = [None] * len(crops)
            for (crop, window, job_id), result in zip(chunk, outs):
                self._consume(job_id, window, result)

        # Finalise any jobs whose tiles all landed.
        for job_id in list(self._remaining.keys()):
            self._finalise(job_id)

    def _consume(self, job_id, window, result):
        if result is None:
            return
        tx1, ty1, tx2, ty2 = window
        dets = results_to_dets(result, x_offset=tx1, y_offset=ty1, task=self._task)
        if self._task != "detect":
            for det in dets:
                det["tile"] = (tx1, ty1, tx2, ty2)
        self._dets_by_job[job_id].extend(dets)
        self._remaining[job_id] -= 1
        if self._remaining[job_id] <= 0:
            self._finalise(job_id)

    def _finalise(self, job_id):
        if job_id in self._completed_jobs:
            return
        self._completed_jobs.add(job_id)
        dets = self._dets_by_job.pop(job_id, [])
        if len(dets) <= 1:
            self._results[job_id] = dets
        elif self._task == "detect":
            self._results[job_id] = nms_xyxy(dets, iou_thresh=self._iou)
        else:
            self._results[job_id] = merge_grouped_dets(dets, self._iou, self._task)
        if self._on_complete is not None:
            try:
                self._on_complete(job_id, self._results[job_id])
            except Exception:
                pass

    def results_for(self, job_id):
        return self._results.get(job_id, [])


def export_json(img_array, dets, class_names):
    """Export detections as JSON with pixel + YOLO normalized coordinates.

    dets: list of {"cls": int, "conf": float, "x1"/"y1"/"x2"/"y2": float}.
    """
    h, w = img_array.shape[:2]
    items = []
    for det in dets:
        x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])
        conf = det["conf"]
        cls_id = det["cls"]
        cls_name = class_names[cls_id]

        # YOLO normalized format (cx, cy, bw, bh) / (W, H)
        yolo_cx = (x1 + x2) / (2 * w)
        yolo_cy = (y1 + y2) / (2 * h)
        yolo_bw = (x2 - x1) / w
        yolo_bh = (y2 - y1) / h

        items.append({
            "class_id": cls_id,
            "class_name": cls_name,
            "confidence": round(conf, 4),
            "pixel": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "yolo": {"cx": round(yolo_cx, 6), "cy": round(yolo_cy, 6), "bw": round(yolo_bw, 6), "bh": round(yolo_bh, 6)},
            **({
                "polygons_px": [[int(round(px)), int(round(py))] for px, py in det["polygon"]],
                "polygons_norm": [[round(px / w, 6), round(py / h, 6)] for px, py in det["polygon"]],
            } if det.get("polygon") is not None else {}),
            **({
                "keypoints_px": [[round(kx, 2), round(ky, 2), round(vis, 4)] for kx, ky, vis in det["keypoints"]],
                "keypoints_norm": [[round(kx / w, 6), round(ky / h, 6), round(vis, 4)] for kx, ky, vis in det["keypoints"]],
            } if det.get("keypoints") is not None else {}),
        })
    return items


def extract_image_metadata(img_path: Path, exiftool_cmd, pool=None) -> Tuple[Dict, str]:
    """Extract full metadata via exiftool, with Pillow EXIF fallback.

    Returns (metadata_dict, source) where source is ``exiftool``, ``pillow``, or ``none``.
    When ``pool`` is a :class:`StayOpenExifToolPool` it is used instead of
    spawning a fresh process per image.
    """
    if exiftool_cmd:
        args_list = ["-j", "-a", "-u", "-G1", str(img_path)]
        if pool is not None:
            rc, out, _err = pool.run(args_list)
            if rc == 0 and out:
                try:
                    payload = json.loads(out)
                    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
                        return payload[0], "exiftool"
                except json.JSONDecodeError:
                    pass
        else:
            cmd = exiftool_cmd + args_list
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if proc.returncode == 0:
                try:
                    payload = json.loads(proc.stdout)
                    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
                        return payload[0], "exiftool"
                except json.JSONDecodeError:
                    pass

    # Fallback: limited EXIF from Pillow only.
    try:
        with Image.open(img_path) as im:
            exif = im.getexif()
            if not exif:
                return {}, "none"
            return {
                ExifTags.TAGS.get(tag_id, str(tag_id)): value
                for tag_id, value in exif.items()
            }, "pillow"
    except Exception:
        return {}, "none"


def _draw_mask_overlay(image, polygon, x1: int, y1: int, x2: int, y2: int) -> None:
    """Blend a semi-transparent colored mask for a polygon over its box region."""
    h, w = image.shape[:2]
    rx1, ry1 = max(0, min(x1, w - 1)), max(0, min(y1, h - 1))
    rx2, ry2 = max(0, min(x2, w - 1)), max(0, min(y2, h - 1))
    if rx2 <= rx1 or ry2 <= ry1:
        return
    region = image[ry1:ry2 + 1, rx1:rx2 + 1]
    mask = np.zeros(region.shape[:2], dtype=np.uint8)
    points = np.array([[int(px - rx1), int(py - ry1)] for px, py in polygon], dtype=np.int32)
    if len(points) < 3:
        return
    cv2.fillPoly(mask, [points], 1)
    colored = np.zeros_like(region)
    colored[mask == 1] = BOX_COLOR
    image[ry1:ry2 + 1, rx1:rx2 + 1] = cv2.addWeighted(region, 1.0, colored, MASK_ALPHA, 0)


def draw_detections(image, dets, class_names):
    """Draw detections: boxes + labels; semi-transparent mask fills for segment
    dets and keypoint circles for pose dets. Detect rendering is unchanged.

    dets: list of {"cls": int, "conf": float, "x1"/"y1"/"x2"/"y2": float,
    optionally "polygon": [(x, y), ...] or "keypoints": [[x, y, vis], ...]}.
    """
    for det in dets:
        x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])
        conf            = det["conf"]
        cls_id          = det["cls"]
        cls_name        = class_names[cls_id]

        # Segment mask overlay first, so the box and label stay on top
        polygon = det.get("polygon")
        if polygon:
            _draw_mask_overlay(image, polygon, x1, y1, x2, y2)

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)

        # Build label text
        label = f"{cls_name} {conf:.2f}"

        # Measure label size for background
        (lw, lh), _ = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)

        # Draw label background
        label_y = max(y1, lh + LABEL_PADDING * 2)
        cv2.rectangle(
            image,
            (x1, label_y - lh - LABEL_PADDING * 2),
            (x1 + lw + LABEL_PADDING * 2, label_y),
            LABEL_BG_COLOR,
            -1  # filled
        )

        # Draw label text
        cv2.putText(
            image, label,
            (x1 + LABEL_PADDING, label_y - LABEL_PADDING),
            FONT, FONT_SCALE, LABEL_TX_COLOR, FONT_THICKNESS
        )

        # Pose keypoints (drawn last so they stay visible over box/label)
        for kx, ky, vis in det.get("keypoints", []):
            if vis > 0:
                cv2.circle(image, (int(round(kx)), int(round(ky))), KEYPOINT_RADIUS, KEYPOINT_COLOR, -1)

    return image


def resolve_exiftool_command(exiftool_arg: Optional[str]):
    """Resolve exiftool invocation command and optional diagnostic reason."""
    if exiftool_arg:
        candidate = normalize_path(exiftool_arg)
        if not candidate.exists():
            return None, f"explicit --exiftool path does not exist: {candidate}"
        if "(-k)" in candidate.name.lower():
            return None, "explicit --exiftool points to exiftool(-k).exe, which is interactive and unsupported"
        return [str(candidate)], None

    for name in ("exiftool", "exiftool.exe"):
        resolved = shutil.which(name)
        if resolved and "(-k)" not in Path(resolved).name.lower():
            return [resolved], None

    for candidate in DEFAULT_EXIFTOOL_CANDIDATES:
        if candidate.exists() and "(-k)" not in candidate.name.lower():
            return [str(candidate)], None

    # Fallback to bundled Perl runtime when exiftool.exe is not present.
    if DEFAULT_EXIFTOOL_PERL.exists() and DEFAULT_EXIFTOOL_PL.exists():
        return [str(DEFAULT_EXIFTOOL_PERL), str(DEFAULT_EXIFTOOL_PL)], None

    k_variant = DEFAULT_EXIFTOOL_DIR / "exiftool(-k).exe"
    if k_variant.exists():
        return None, "found only exiftool(-k).exe in default directory; use exiftool.exe or bundled Perl runtime"
    return None, None


def require_exiftool_before_run(args, exiftool_cmd, exiftool_resolve_reason) -> None:
    """Exit before inference if annotated saves need exifTool but it is unavailable."""
    if args.exiftool and not exiftool_cmd:
        print("[ERROR] --exiftool path is invalid or unusable.")
        if exiftool_resolve_reason:
            print(f"        Reason: {exiftool_resolve_reason}")
        sys.exit(1)

    if not args.export_annotated_images or args.allow_missing_exiftool or exiftool_cmd:
        return

    print("[ERROR] exiftool is required when saving images to detections/.")
    if not DEFAULT_EXIFTOOL_DIR.exists():
        print(f"        Default directory missing: {DEFAULT_EXIFTOOL_DIR}")
        print("        Download exiftool and put it in this directory,")
        print(f"        URL: {EXIFTOOL_DOWNLOAD_URL}")
        print("        or pass --exiftool /path/to/exiftool(.exe).")
    else:
        print(f"        exiftool runtime not available in default directory: {DEFAULT_EXIFTOOL_DIR}")
        if exiftool_resolve_reason:
            print(f"        Reason: {exiftool_resolve_reason}")
        print("        Download exiftool and place exiftool.exe there,")
        print(f"        URL: {EXIFTOOL_DOWNLOAD_URL}")
        print("        or pass --exiftool /path/to/exiftool(.exe).")
    print("        To bypass (limited metadata copy), use --allow-missing-exiftool.")
    sys.exit(1)


def detection_output_dir(images_dir: Path) -> Path:
    """Default output folder under this script's directory, one subfolder per input dir."""
    return SCRIPT_DIR / "detections" / images_dir.name


def _image_size_lazy(p: Path):
    """Return (w, h) via a lazy Pillow open, or (0, 0) on failure."""
    try:
        from PIL import Image as _PILImage
        with _PILImage.open(p) as im:
            return im.size
    except Exception:
        return (0, 0)


def _run_parallel(image_paths, images_dir, total, model, model_task, args,
                  batch_size, workers, decode_workers, byte_budget,
                  exiftool_pool, post_process, safe_print, output_dir,
                  half_on, device):
    """Hardware-aware pipeline: decode pool -> bounded queue -> TileBatcher -> post pool.

    Each image is decoded exactly once: decode workers pull paths from a shared
    FIFO queue with ``get_nowait()`` until it is empty, instead of every worker
    iterating the full image list. In tiled mode tiles accumulate across
    decoded images and are flushed once ``working_batch`` tiles are pending,
    so a single ``model()`` call can mix tiles from several source images;
    per-image merge/NMS is unaffected because it is keyed by ``job_id``.
    """
    from concurrent.futures import ThreadPoolExecutor
    from queue import Queue, Empty

    path_queue = Queue()
    for p in image_paths:
        path_queue.put(p)

    in_queue = Queue(maxsize=max(1, decode_workers * 4))
    post_semaphore = threading.Semaphore(max(1, workers * 2))

    def decode_worker():
        while True:
            try:
                p = path_queue.get_nowait()
            except Empty:
                return
            im = cv2.imread(str(p))
            if im is None:
                safe_print(f"  SKIP (could not read): {p.name}")
                continue
            w, h = _image_size_lazy(p)
            img_bytes = estimate_image_bytes(w, h)
            if byte_budget is not None:
                byte_budget.acquire(img_bytes)
            try:
                in_queue.put((p, im, img_bytes), block=True)
            except Exception:
                if byte_budget is not None:
                    byte_budget.release(img_bytes)
                return

    decode_pool = ThreadPoolExecutor(max_workers=decode_workers,
                                     thread_name_prefix="decode")
    decode_futures = [decode_pool.submit(decode_worker) for _ in range(decode_workers)]

    working_batch = max(1, batch_size)
    batcher = None
    if args.tiles:
        batcher = TileBatcher(model, args.conf, args.tile_imgsz, args.tile_overlap,
                              args.tile_iou, working_batch, model_task,
                              device=device, half=half_on)

    def submit_post(idx, p, image, dets, flat_name, display_name):
        post_semaphore.acquire()
        try:
            post_process(idx, p, image, dets, flat_name, display_name)
        finally:
            post_semaphore.release()

    post_pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="post")
    job_counter = [0]
    job_lock = threading.Lock()
    meta_lock = threading.Lock()
    # job_id -> (idx, p, im, img_bytes, flat_name, display_name)
    job_meta = {}

    def next_job_id():
        with job_lock:
            job_counter[0] += 1
            return job_counter[0]

    def on_complete(job_id, dets):
        with meta_lock:
            meta = job_meta.pop(job_id, None)
        if meta is None:
            return
        idx, p, im, img_bytes, flat_name, display_name = meta
        try:
            post_pool.submit(submit_post, idx, p, im, dets, flat_name, display_name)
        finally:
            if byte_budget is not None:
                byte_budget.release(img_bytes)

    if batcher is not None:
        batcher._on_complete = on_complete

    def process_one(p, im, img_bytes):
        idx = job_counter[0] + 1
        try:
            display_name = str(p.relative_to(images_dir))
        except ValueError:
            display_name = p.name
        flat_name = flat_output_name(p, images_dir)
        job_id = next_job_id()
        if args.tiles:
            with meta_lock:
                job_meta[job_id] = (idx, p, im, img_bytes, flat_name, display_name)
            h, w = im.shape[:2]
            for (tx1, ty1, tx2, ty2) in iter_tile_windows(w, h, args.tile_imgsz, args.tile_overlap):
                batcher.submit(im[ty1:ty2, tx1:tx2], (tx1, ty1, tx2, ty2), job_id=job_id)
            # Flush once enough tiles are pending to fill a model() call; tiles
            # from several decoded images may share the call. Leftover tiles
            # are drained in the ``finally`` block.
            if len(batcher._pending) >= working_batch:
                batcher.flush()
        else:
            try:
                outs = model([im], conf=args.conf, verbose=False, device=device,
                             **_precision_kwargs(half_on))
                r = outs[0] if outs else None
            except Exception:
                r = None
            dets = results_to_dets(r, task=model_task) if r is not None else []
            post_pool.submit(submit_post, idx, p, im, dets, flat_name, display_name)
            if byte_budget is not None:
                byte_budget.release(img_bytes)

    try:
        while True:
            try:
                p, im, img_bytes = in_queue.get(timeout=0.5)
            except Empty:
                if all(f.done() for f in decode_futures):
                    break
                continue
            try:
                process_one(p, im, img_bytes)
            except Exception:
                if byte_budget is not None:
                    byte_budget.release(img_bytes)
    finally:
        decode_pool.shutdown(wait=True)
        if batcher is not None:
            batcher.flush()
        post_pool.shutdown(wait=True)


# -- Main ----------------------------------------------------------------------

def run(args):
    images_dir = normalize_path(args.images)
    model_path = normalize_path(args.model)
    output_dir = detection_output_dir(images_dir)

    # Validate inputs
    if not images_dir.exists():
        print(f"[ERROR] Images directory not found: {images_dir}")
        sys.exit(1)

    if not model_path.exists():
        print(f"[ERROR] Model file not found: {model_path}")
        sys.exit(1)

    if not (0.0 <= args.conf <= 1.0):
        print(f"[ERROR] Confidence must be between 0.0 and 1.0, got: {args.conf}")
        sys.exit(1)

    if args.tiles:
        if not (0.0 <= args.tile_overlap < 1.0):
            print(f"[ERROR] --tile-overlap must be in [0.0, 1.0), got: {args.tile_overlap}")
            sys.exit(1)
        if args.tile_imgsz <= 0:
            print(f"[ERROR] --tile-imgsz must be > 0, got: {args.tile_imgsz}")
            sys.exit(1)

    # Collect images.
    iterator = images_dir.rglob("*") if args.recursive else images_dir.iterdir()
    image_paths = []
    for p in iterator:
        if not p.is_file():
            continue
        if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        image_paths.append(p)
    image_paths.sort()

    if not image_paths:
        print(f"[ERROR] No images found in: {images_dir}")
        print(f"        Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}")
        if args.recursive:
            print("        (Searched recursively. Use --no-recursive to scan only the top-level.)")
        sys.exit(1)

    subfolder_count = 0
    if args.recursive:
        subfolders = {
            p.parent for p in image_paths
            if p.parent.resolve() != images_dir.resolve()
        }
        subfolder_count = len(subfolders)

    # Resolve hardware profile and concurrency.
    hw = detect_hardware()
    profile = resolve_profile(args.memory_profile)
    if args.vram_frac is not None:
        profile = dict(profile)
        profile["gpu"] = float(args.vram_frac)
    cuda = _cuda_available()
    device, device_src = resolve_device(args.device)
    device_is_cpu = device == "cpu"
    half_on, half_src = resolve_half(args.half, cuda and not device_is_cpu)
    cpu_cores = max(1, hw["cpu_cores"])
    decode_workers, decode_src = resolve_decode_workers(args.decode_workers, cpu_cores)

    # Host RAM byte budget: min(total * cpu_frac, available * 0.5).
    budget_bytes = 0
    if hw["ram_gb"]:
        try:
            import psutil as _psutil
            total_ram = _psutil.virtual_memory().total
            avail_ram = _psutil.virtual_memory().available
            budget_bytes = int(min(total_ram * profile["cpu"], avail_ram * 0.5))
        except Exception:
            budget_bytes = 0

    # VRAM budget: min(free_now * 0.9, total * profile.gpu).
    vram_budget_bytes = 0
    free_v, total_v = free_vram_info()
    if free_v is not None and total_v:
        vram_budget_bytes = int(min(free_v * 0.9, total_v * profile["gpu"]))

    # Per-image byte estimate: 2 * w * h * 3 (original + annotated copy) + JSON.
    # Resolved lazily per image; the budget caps how many decoded arrays are
    # alive at once.
    byte_budget = ByteBudget(budget_bytes) if budget_bytes > 0 else None

    # Load model before sizing the batch so the VRAM probe can run.
    model = YOLO(str(model_path))
    class_names = model.names
    model_task = getattr(model, "task", None) or "detect"

    # Thread oversubscription guard: cv2 decode threads must stay at 1 when
    # the decode pool has more than one worker.
    if decode_workers > 1:
        try:
            cv2.setNumThreads(1)
        except Exception:
            pass

    # Inference item size: tiles in tiled mode, whole images otherwise.
    item_imgsz = args.tile_imgsz if args.tiles else 640
    per_item_vram = 0
    if cuda and not device_is_cpu and vram_budget_bytes > 0:
        per_item_vram = measure_inference_vram(model, item_imgsz, half_on, device=device)
        if per_item_vram <= 0:
            per_item_vram = int(vram_budget_bytes // 8)  # conservative fallback

    workers, workers_src = resolve_workers(args.workers, cpu_cores, decode_workers,
                                           half_on, device_is_cpu)
    if device_is_cpu and torch is not None:
        try:
            torch.set_num_threads(max(1, cpu_cores - max(1, workers)))
        except Exception:
            pass
    batch_size, batch_src = resolve_batch(args.batch, vram_budget_bytes, per_item_vram,
                                          cuda and not device_is_cpu, half_on)

    exiftool_cmd, exiftool_resolve_reason = resolve_exiftool_command(args.exiftool)
    if args.exiftool_stay_open and exiftool_cmd:
        exiftool_pool = StayOpenExifToolPool(exiftool_cmd, min(4, workers))
    else:
        exiftool_pool = None
    require_exiftool_before_run(args, exiftool_cmd, exiftool_resolve_reason)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process images. The legacy single-image-at-a-time path (--workers 1
    # --batch 1 --half off --decode-workers 1) runs the original serial loop
    # verbatim so its output is byte-for-byte identical to the previous
    # release. It also skips the [Hardware] block so the printed output
    # matches the old [Config]-only layout.
    legacy = (batch_size <= 1 and workers <= 1 and decode_workers <= 1
              and str(args.decode_workers).strip().lower() in ("1", "1.0")
              and exiftool_pool is None)

    if not legacy:
        print(f"\n[Hardware]")
        print(f"  gpu        : {hw['gpu_name']}")
        print(f"  vram       : {('%.2f GB' % hw['vram_gb']) if hw['vram_gb'] else 'n/a'} total, "
              f"{('%.2f GB' % (free_v / 1024**3)) if free_v is not None else 'n/a'} free")
        print(f"  cpu cores  : {cpu_cores}")
        print(f"  ram        : {('%.2f GB' % hw['ram_gb']) if hw['ram_gb'] else 'n/a'} total")
        print(f"  profile    : {args.memory_profile}  (gpu={profile['gpu']}, cpu={profile['cpu']})")
        print(f"  device     : {device}  ({device_src})")
        print(f"  half       : {'on' if half_on else 'off'}  ({half_src})")
        print(f"  decode wkrs: {decode_workers}  ({decode_src})")
        print(f"  workers    : {workers}  ({workers_src})")
        print(f"  batch      : {batch_size}  ({batch_src})"
              + (f", per-item vram ~{per_item_vram/1024**2:.0f} MB" if per_item_vram else ""))

    print(f"\n[Config]")
    print(f"  images     : {images_dir}  ({len(image_paths)} found)")
    if args.recursive:
        print(f"  recursive  : True  ({subfolder_count} subfolder(s) walked)")
    else:
        print(f"  recursive  : False")
    print(f"  model      : {model_path}")
    print(f"  confidence : {args.conf}")
    print(f"  output     : {output_dir}")
    print(f"  save images: {args.export_annotated_images}")
    print(f"  workers    : {workers}  ({workers_src})")
    print(f"  batch      : {batch_size}  ({batch_src})")
    if args.tiles:
        print(f"  tiles      : True  (imgsz={args.tile_imgsz}, overlap={args.tile_overlap}, iou={args.tile_iou})")
    else:
        print(f"  tiles      : False  (whole-image inference)")
    print(f"  task       : {model_task}")
    print()

    total = len(image_paths)

    # Shared, thread-safe state mutated from worker threads.
    state = {
        "total_detections": 0,
        "exiftool_warned": False,
        "nonexiftool_json_metadata_warned": False,
        "b3dm_verify_failed": False,
    }
    state_lock = threading.Lock()
    print_lock = threading.Lock()

    def safe_print(*pargs, **pkwargs):
        with print_lock:
            print(*pargs, **pkwargs)

    # save_image_with_exif preserves EXIF metadata when writing annotated images.
    # Defined once (not per-image) so it is shared by all worker threads.
    def save_image_with_exif(src_path: Path, img_array, dest_path: Path):
        """Save img_array to dest_path preserving metadata from src_path.

        Uses Pillow for baseline metadata copy and exiftool (if available)
        to preserve full metadata blocks (XMP/MPF/vendor APP segments).
        """
        if not exiftool_cmd and not args.allow_missing_exiftool:
            return

        dest_ext = dest_path.suffix.lower()
        is_jpeg = dest_ext in {".jpg", ".jpeg"}

        # Read source metadata from both EXIF object and raw info payloads.
        with Image.open(src_path) as orig_img:
            info = dict(orig_img.info)
            exif_data = None

            # Prefer normalized EXIF table when available.
            try:
                exif = orig_img.getexif()
                if exif and len(exif) > 0:
                    exif_data = exif.tobytes()
            except Exception:
                exif_data = None

            # Fallback to raw EXIF bytes from the source container.
            if not exif_data:
                exif_data = info.get("exif")

            save_kwargs = {}
            if exif_data:
                save_kwargs["exif"] = exif_data

            icc_profile = info.get("icc_profile")
            if icc_profile:
                save_kwargs["icc_profile"] = icc_profile

            dpi = info.get("dpi")
            if dpi:
                save_kwargs["dpi"] = dpi

            # JFIF fields are JPEG-specific and may be rejected by other formats.
            if is_jpeg:
                for key in ("jfif", "jfif_version", "jfif_unit", "jfif_density"):
                    if key in info:
                        save_kwargs[key] = info[key]

            # Convert numpy array (BGR) to RGB Pillow Image
            rgb_img = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
            try:
                rgb_img.save(dest_path, **save_kwargs)
            except TypeError:
                # Fallback: keep widely supported metadata keys only.
                fallback_kwargs = {}
                for key in ("exif", "icc_profile", "dpi"):
                    if key in save_kwargs:
                        fallback_kwargs[key] = save_kwargs[key]
                rgb_img.save(dest_path, **fallback_kwargs)

        # Pillow cannot preserve all JPEG APP metadata blocks. If exiftool is
        # present, copy all writable metadata groups from source to output.
        if exiftool_cmd:
            copy_args = [
                "-overwrite_original",
                "-m",
                "-P",
                "-TagsFromFile",
                str(src_path),
                "-all:all",
                "-unsafe",
                str(dest_path),
            ]
            if exiftool_pool is not None:
                rc, _out, err = exiftool_pool.run(copy_args)
                if rc != 0:
                    safe_print(f"  [WARN] exiftool metadata copy failed for {dest_path.name}: {err.strip()}")
            else:
                cmd = exiftool_cmd + copy_args
                proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
                if proc.returncode != 0:
                    safe_print(f"  [WARN] exiftool metadata copy failed for {dest_path.name}: {proc.stderr.strip()}")
        elif is_jpeg:
            with state_lock:
                should_warn = not state["exiftool_warned"]
                state["exiftool_warned"] = True
            if should_warn:
                safe_print("  [WARN] exiftool not found; JPEG metadata copy is limited to Pillow-supported fields.")

    def post_process(idx, img_path, image, dets, flat_name, display_name):
        """Per-image post-inference work (JSON, ExifTool, image save). Runs in worker thread."""
        n_det = len(dets)
        with state_lock:
            state["total_detections"] += n_det

        annotated = draw_detections(image.copy(), dets, class_names)

        if args.export_json and n_det > 0:
            items = export_json(image, dets, class_names)
            metadata, meta_src = extract_image_metadata(img_path, exiftool_cmd, pool=exiftool_pool)
            metadata = merge_pillow_gps_exif_into_metadata(metadata, img_path)
            if meta_src != "exiftool":
                with state_lock:
                    should_warn = not state["nonexiftool_json_metadata_warned"]
                    state["nonexiftool_json_metadata_warned"] = True
                if should_warn:
                    safe_print(
                        "  [WARN] Sidecar JSON metadata was not produced by exiftool (Pillow fallback or empty EXIF). "
                        "Ortho-Tag B3DM expects exiftool -G1 keys (e.g. XMP-drone-dji:* for yaw/pitch/altitude). "
                        "GPS/FocalLength are merged from EXIF when possible; install exiftool or use --exiftool for full DJI pose."
                    )
            h, w = image.shape[:2]
            payload = {
                "image": {
                    "file_name": flat_name,
                    "source_path": str(img_path),
                    "width": w,
                    "height": h,
                },
                "metadata": metadata,
                "detections": items,
            }
            json_path = output_dir / (Path(flat_name).stem + ".json")
            with open(json_path, "w") as f:
                json.dump(payload, f, indent=2, default=str)
            safe_print(f"  [{idx}/{total}] {display_name}  ->  {json_path.name}")
            if args.verify_b3dm:
                ok_geo, miss_c, _miss_r = verify_sidecar_json_file(json_path)
                if not ok_geo:
                    with state_lock:
                        state["b3dm_verify_failed"] = True
                    safe_print(f"  [WARN] --verify-b3dm failed for {json_path.name}: {', '.join(miss_c)}")

        if n_det > 0:
            if not args.export_annotated_images:
                safe_print(f"  [{idx}/{total}] {display_name}  ->  {n_det} detection(s)  [json-only]")
                return
            out_path = output_dir / flat_name
            save_image_with_exif(img_path, annotated, out_path)
            safe_print(f"  [{idx}/{total}] {display_name}  ->  {n_det} detection(s)  [saved]")
        else:
            if args.save_all:
                out_path = output_dir / flat_name
                save_image_with_exif(img_path, annotated, out_path)
                safe_print(f"  [{idx}/{total}] {display_name}  ->  0 detections  [saved]")
            else:
                safe_print(f"  [{idx}/{total}] {display_name}  ->  0 detections  [skipped]")

# Process images. The legacy single-image-at-a-time path (--workers 1
    # --batch 1 --half off --decode-workers 1) runs the original serial loop
    # verbatim so its output is byte-for-byte identical to the previous
    # release. It also skips the [Hardware] block so the printed output
    # matches the old [Config]-only layout.
    legacy = (batch_size <= 1 and workers <= 1 and decode_workers <= 1
              and str(args.decode_workers).strip().lower() in ("1", "1.0")
              and exiftool_pool is None)

    if legacy:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for chunk_start in range(0, total, batch_size):
                chunk = image_paths[chunk_start:chunk_start + batch_size]
                valid = []
                for offset, p in enumerate(chunk):
                    idx = chunk_start + offset + 1
                    try:
                        display_name = str(p.relative_to(images_dir))
                    except ValueError:
                        display_name = p.name
                    im = cv2.imread(str(p))
                    if im is None:
                        safe_print(f"  [{idx}/{total}] SKIP (could not read): {display_name}")
                        continue
                    flat_name = flat_output_name(p, images_dir)
                    valid.append((idx, p, im, flat_name, display_name))

                if not valid:
                    continue

                if args.tiles:
                    for idx, p, im, flat_name, display_name in valid:
                        dets = detect_image_tiled(
                            model, im, args.conf,
                            args.tile_imgsz, args.tile_overlap, args.tile_iou,
                            batch_size, task=model_task, half=half_on,
                        )
                        pool.submit(post_process, idx, p, im, dets, flat_name, display_name)
                else:
                    images_for_model = [v[2] for v in valid]
                    results_list = list(model(images_for_model, conf=args.conf, verbose=False))

                    for (idx, p, im, flat_name, display_name), r in zip(valid, results_list):
                        dets = results_to_dets(r, task=model_task)
                        pool.submit(post_process, idx, p, im, dets, flat_name, display_name)
    else:
        _run_parallel(image_paths, images_dir, total, model, model_task, args,
                      batch_size, workers, decode_workers, byte_budget,
                      exiftool_pool, post_process, safe_print, output_dir,
                      half_on, device)

    if exiftool_pool is not None:
        exiftool_pool.shutdown()

    # Export labels.txt (class_id -> class_name mapping)
    if args.export_json:
        labels_path = output_dir / "labels.txt"
        with open(labels_path, "w") as f:
            for cid, cname in sorted(class_names.items(), key=lambda x: x[0]):
                f.write(f"{cid}\t{cname}\n")
        print(f"\n  {output_dir}/labels.txt  ({len(class_names)} class(es))")

    if args.verify_b3dm and state["b3dm_verify_failed"]:
        print("[ERROR] --verify-b3dm: one or more sidecars lack latitude/longitude in exiftool-style keys.")
        sys.exit(1)

    # Summary
    print(f"\n{'='*60}")
    print(f"  Done")
    print(f"{'='*60}")
    print(f"  images processed : {total}")
    print(f"  total detections : {state['total_detections']}")
    print(f"  output saved to  : {output_dir}")
    print(f"{'='*60}\n")


# -- Entry Point ---------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    run(args)