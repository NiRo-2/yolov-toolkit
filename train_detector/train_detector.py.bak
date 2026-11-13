r"""
YOLO26 Object Detector Training Script
Auto-detects hardware, dataset size and image resolution to calculate optimal training config.

Usage:
    # Fresh training run (auto-configures everything)
    python train_detector/train_detector.py --input /path/to/data.yaml --name my_detector_v1

    # Override any auto-calculated value
    python train_detector/train_detector.py --input /path/to/data.yaml --name my_detector_v1 --model yolo26x.pt --batch 8
    python train_detector/train_detector.py --input /path/to/data.yaml --name my_detector_v1 --imgsz 1024

    # Resume a crashed/interrupted run
    python train_detector/train_detector.py --resume --name my_detector_v1

    # Resume without --name: auto-finds the most recent run
    python train_detector/train_detector.py --resume

    --input   path to data.yaml file (required for fresh training)
              works with Windows paths: c:\Users\Ni\Desktop\project\data.yaml
    --resume  resume from last checkpoint
    --name    run name to resume, or name for new run (default: detector_v1)
    --model   override auto-selected model (yolo26m/l/x)
    --batch   override auto-calculated batch size
    --workers override auto-calculated worker count
    --imgsz   override auto-calculated image size in pixels
    --epochs  number of training epochs (default: 600)
    --patience early stopping patience in epochs (default: 100)
    --device  0 for GPU, cpu for CPU (default: 0)

Output:
    train_detector/weights/<model>.pt          <- Ultralytics pretrained backbones (cached)
    train_detector/runs/detect/<name>/weights/best.pt   <- use this for TensorRT export
    train_detector/runs/detect/<name>/weights/last.pt
    train_detector/runs/detect/<name>/results.png
    train_detector/runs/detect/<name>/confusion_matrix.png
"""

import argparse
import json
import os
import statistics
import sys
import tempfile
import yaml
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO


SCRIPT_DIR = Path(__file__).resolve().parent
RUNS_DIR = SCRIPT_DIR / "runs" / "detect"
WEIGHTS_DIR = SCRIPT_DIR / "weights"


def resolve_pretrained_weights(model: str) -> Path:
    """Resolve YOLO backbone name or path; cache under train_detector/weights/."""
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    raw = model.strip().strip('"').strip("'")
    path = Path(raw.replace("\\", "/"))

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


# -- Path Normalization --------------------------------------------------------

def normalize_path(raw: str) -> Path:
    """Handle Windows and Unix paths on any OS."""
    cleaned = raw.strip().strip('"').strip("'")
    cleaned = cleaned.replace("\\", "/")
    return Path(cleaned).resolve()


# -- Hardware Detection --------------------------------------------------------

def detect_hardware():
    """Detect GPU VRAM, CPU cores and RAM."""
    info = {
        "vram_gb":   None,
        "cpu_cores": os.cpu_count() or 4,
        "ram_gb":    None,
        "gpu_name":  "CPU",
    }

    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["vram_gb"] = props.total_memory / (1024 ** 3)
            info["gpu_name"] = props.name
    except Exception:
        pass

    try:
        import psutil
        info["ram_gb"] = psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        pass

    return info


# -- Dataset Detection ---------------------------------------------------------

def get_train_images_path(yaml_path: Path) -> Path:
    """Resolve the train images directory from data.yaml."""
    with open(yaml_path) as f:
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


def count_images(train_path: Path) -> int:
    """Count images in a directory."""
    if train_path is None:
        return 0
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    return sum(1 for f in train_path.rglob("*") if f.suffix.lower() in extensions)


def detect_image_size(train_path: Path) -> int:
    """
    Sample images spread across the dataset to detect native resolution.

    Returns the median largest dimension across sampled images.
    Median is used as a representative native resolution for auto-config.

    Samples up to 20 images evenly spread across the dataset for reliability
    without reading every file.
    """
    if train_path is None:
        return None

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    image_files = [f for f in train_path.rglob("*") if f.suffix.lower() in extensions]

    if not image_files:
        return None

    # Sample 10% of dataset evenly spread, capped at 100 images
    total = len(image_files)
    sample_count = min(max(1, int(total * 0.10)), 100)
    step = max(1, total // sample_count)
    sampled = image_files[::step][:sample_count]

    sizes = []

    # Try PIL first, fall back to OpenCV
    try:
        from PIL import Image as PILImage
        use_pil = True
    except ImportError:
        use_pil = False

    for img_path in sampled:
        try:
            if use_pil:
                with PILImage.open(img_path) as img:
                    sizes.append(max(img.size))
            else:
                import cv2
                img = cv2.imread(str(img_path))
                if img is not None:
                    h, w = img.shape[:2]
                    sizes.append(max(h, w))
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


# -- Auto Config ---------------------------------------------------------------

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


def load_vram_estimates() -> dict[str, float]:
    if not ESTIMATES_PATH.exists():
        return {}
    try:
        with open(ESTIMATES_PATH, encoding="utf-8") as f:
            data = json.load(f)
        out: dict[str, float] = {}
        for name, meta in (data.get("estimates") or {}).items():
            val = meta.get("gb_per_image_1024") if isinstance(meta, dict) else None
            if isinstance(val, (int, float)) and val > 0:
                out[str(name)] = float(val)
        return out
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as e:
        print(f"[WARNING] Ignoring corrupt VRAM cache {ESTIMATES_PATH}: {e}")
        return {}


def save_vram_estimates(estimates: dict[str, float], device_name: str) -> None:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    merged = load_vram_estimates()
    merged.update(estimates)
    try:
        import ultralytics
        ultra_ver = getattr(ultralytics, "__version__", "unknown")
    except Exception:
        ultra_ver = "unknown"
    from datetime import datetime, timezone
    payload = {
        "version": 1,
        "ultralytics": ultra_ver,
        "device_name": device_name,
        "updated_at": datetime.now(timezone.utc).isoformat(),
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
        last_err: Exception | None = None
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


def cuda_available_for(device: str) -> bool:
    if str(device).lower() == "cpu":
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def ensure_vram_estimates(models: list[str], device: str, force: bool = False) -> dict[str, float]:
    if not cuda_available_for(device):
        print("[VRAM Probe] CUDA unavailable — using built-in FLOPs-scaled estimates")
        return {m: get_vram_per_image(m) for m in models}
    cached = load_vram_estimates()
    need = [m for m in models if force or m not in cached]
    if not need:
        return {m: get_vram_per_image(m) for m in models}
    print(f"[VRAM Probe] Measuring: {', '.join(need)}")
    measured: dict[str, float] = {}
    try:
        device_name = torch.cuda.get_device_name(0)
    except Exception:
        device_name = "cuda"
    for m in need:
        try:
            measured[m] = probe_model_vram(m, device=device)
            print(f"  {m}: {measured[m]:.3f} GB/img @1024")
        except Exception as e:
            print(f"[WARNING] Probe failed for {m}: {e} — using built-in fallback")
    if measured:
        save_vram_estimates(measured, device_name)
    return {m: get_vram_per_image(m) for m in models}


MIN_BATCH = 8  # minimum viable batch size for stable training


def snap_to_standard(size: int) -> int:
    """Snap a raw pixel size to the nearest standard YOLO imgsz."""
    standards = [320, 416, 512, 640, 768, 1024, 1280, 1536]
    return min(standards, key=lambda s: abs(s - size))


def calc_max_batch_for_imgsz(vram_gb: float, model: str, imgsz: int) -> int:
    """Calculate what batch size we'd get at a given imgsz."""
    scale = (imgsz / 1024) ** 2
    vram_per_img = get_vram_per_image(model) * scale
    usable_vram = vram_gb * 0.85
    batch = int(usable_vram / vram_per_img)
    for b in [64, 32, 16, 8, 4]:
        if batch >= b:
            return b
    return 4


def select_model_and_imgsz(image_count: int, vram_gb: float) -> tuple:
    """
    Select best model and image size based on dataset size, VRAM and native image resolution.
    Priority: best quality within hardware limits.

    Key constraints:
    1. imgsz never exceeds native image resolution — upscaling adds no detail
    2. imgsz is only raised if batch stays >= 4 (acceptable when nbs=64 gradient
       accumulation is active; effective batch remains 64 regardless of per-step batch)

    Decision table (before native resolution cap):
    ┌─────────────┬──────────┬────────────┬────────┬──────────────────────────────────────────┐
    │ Train imgs  │ VRAM     │ Model      │ imgsz  │ Reason                                   │
    ├─────────────┼──────────┼────────────┼────────┼──────────────────────────────────────────┤
    │ < 1,000     │ >= 16GB  │ yolo26m    │ 1280   │ m is light, 1280 fits, avoids overfit    │
    │ 1,000-5,000 │ >= 16GB  │ yolo26l    │ 1024   │ l+1280 fits batch 8; 1024 kept for balance │
    │ > 5,000     │ >= 16GB  │ yolo26x    │ 1280†  │ x+1280 needs batch>=8 (~17.6GB+); else 1024│
    ├─────────────┼──────────┼────────────┼────────┼──────────────────────────────────────────┤
    │ < 1,000     │ >= 12GB  │ yolo26m    │ 1024   │ small dataset, safe imgsz                │
    │ 1,000-5,000 │ >= 12GB  │ yolo26l    │ 1024   │ VRAM limit, keep imgsz safe              │
    │ > 5,000     │ >= 12GB  │ yolo26l    │ 1024   │ l is safe ceiling at 12GB                │
    ├─────────────┼──────────┼────────────┼────────┼──────────────────────────────────────────┤
    │ < 1,000     │ >= 8GB   │ yolo26m    │ 640    │ low VRAM, keep it safe                   │
    │ 1,000-5,000 │ >= 8GB   │ yolo26m    │ 1024   │ m is safe at 8GB + 1024                  │
    │ > 5,000     │ >= 8GB   │ yolo26l    │ 1024   │ dataset justifies l at safe imgsz        │
    ├─────────────┼──────────┼────────────┼────────┼──────────────────────────────────────────┤
    │ any         │ < 8GB    │ yolo26m    │ 640    │ low VRAM, minimal safe config            │
    │ any         │ None/CPU │ yolo26m    │ 640    │ CPU fallback                             │
    └─────────────┴──────────┴────────────┴────────┴──────────────────────────────────────────┘

    † yolo26x@1280 only ships when it reaches MIN_BATCH (8) — that needs ~17.6GB+
      (calc_max_batch_for_imgsz(vram_gb, "yolo26x.pt", 1280) >= 8). At the 16GB
      threshold itself this yields batch 4, so imgsz falls back to 1024 (see the
      "Final safety check" below, which enforces the same floor generically).

    Any value can be overridden via --model, --imgsz, --batch, --workers flags.

    Returns: (model, imgsz)
    """
    if vram_gb is None:
        model, imgsz = "yolo26m.pt", 640
    elif vram_gb >= 16:
        if image_count < 1000:
            model, imgsz = "yolo26m.pt", 1280
        elif image_count < 5000:
            model, imgsz = "yolo26l.pt", 1024
        else:
            if calc_max_batch_for_imgsz(vram_gb, "yolo26x.pt", 1280) >= MIN_BATCH:
                model, imgsz = "yolo26x.pt", 1280
            else:
                model, imgsz = "yolo26x.pt", 1024
    elif vram_gb >= 12:
        if image_count < 1000:
            model, imgsz = "yolo26m.pt", 1024
        elif image_count < 5000:
            model, imgsz = "yolo26l.pt", 1024
        else:
            model, imgsz = "yolo26l.pt", 1024
    elif vram_gb >= 8:
        if image_count < 1000:
            model, imgsz = "yolo26m.pt", 640
        elif image_count < 5000:
            model, imgsz = "yolo26m.pt", 1024
        else:
            model, imgsz = "yolo26l.pt", 1024
    else:
        model, imgsz = "yolo26m.pt", 640

    # Final safety check: if chosen imgsz causes batch < MIN_BATCH, drop imgsz
    if vram_gb and calc_max_batch_for_imgsz(vram_gb, model, imgsz) < MIN_BATCH:
        imgsz = 1024
        if calc_max_batch_for_imgsz(vram_gb, model, imgsz) < MIN_BATCH:
            imgsz = 640

    return model, imgsz


def calc_batch(vram_gb: float, model: str, imgsz: int, multi_scale: float = 0.0) -> int:
    """Calculate max safe batch size for given VRAM and model."""
    if vram_gb is None:
        return 4

    # Size against peak multi_scale batches to avoid OOM (imgsz * (1 + multi_scale))
    effective_imgsz = imgsz * (1 + multi_scale) if multi_scale > 0 else imgsz
    scale = (effective_imgsz / 1024) ** 2
    vram_per_img = get_vram_per_image(model) * scale
    usable_vram = vram_gb * 0.85
    batch = int(usable_vram / vram_per_img)

    for b in [64, 32, 16, 8, 4]:
        if batch >= b:
            return b
    return 4


def calc_augmentation_config(vram_gb: float, model: str, imgsz: int) -> dict:
    """Quality-first augmentation: enable multi_scale whenever VRAM sustains batch >= 4.

    nbs=64 keeps gradient accumulation explicit so optimization stays stable regardless
    of the actual per-step batch size (effective batch remains 64).
    """
    # batch >= 4 is enough for multi_scale; nbs=64 gradient accumulation
    # compensates for smaller per-step batches (effective batch stays 64).
    multi_scale = 0.5 if calc_batch(vram_gb, model, imgsz, multi_scale=0.5) >= 4 else 0.0
    return {
        "degrees": 180,
        "flipud": 0.5,
        "copy_paste": 0.3,
        "mixup": 0.15,
        "multi_scale": multi_scale,
        "close_mosaic": 60,
        "cos_lr": True,
        "nbs": 64,
    }


def calc_workers(cpu_cores: int, ram_gb: float) -> int:
    """Calculate optimal dataloader workers."""
    max_by_cpu = max(1, cpu_cores - 2)
    max_by_ram = int(ram_gb / 2) if ram_gb else 4
    workers = min(max_by_cpu, max_by_ram, 8)  # cap at 8 for Windows stability
    return max(1, workers)


def auto_config(yaml_path: Path, args) -> dict:
    """Build full training config from hardware + dataset + native image resolution."""
    hw          = detect_hardware()
    train_path  = get_train_images_path(yaml_path)
    image_count = count_images(train_path)
    native_size = detect_image_size(train_path)

    # Select model + imgsz together
    auto_model, auto_imgsz = select_model_and_imgsz(image_count, hw["vram_gb"])
    model  = args.model if args.model else auto_model
    imgsz  = args.imgsz if args.imgsz else auto_imgsz
    if cuda_available_for(args.device):
        ensure_vram_estimates([model], device=args.device, force=False)
    augmentation = calc_augmentation_config(hw["vram_gb"], model, imgsz)
    batch   = args.batch   if args.batch   else calc_batch(
        hw["vram_gb"], model, imgsz, multi_scale=augmentation["multi_scale"]
    )
    workers = args.workers if args.workers else calc_workers(hw["cpu_cores"], hw["ram_gb"])

    return {
        "model":       model,
        "imgsz":       imgsz,
        "batch":       batch,
        "workers":     workers,
        "augmentation": augmentation,
        "image_count": image_count,
        "native_size": native_size,
        "hw":          hw,
    }


def print_auto_config(config: dict, args):
    """Print hardware detection results and final config."""
    hw = config["hw"]

    print(f"\n[Hardware Detected]")
    print(f"  GPU        : {hw['gpu_name']}")
    vram_str = f"{hw['vram_gb']:.1f} GB" if hw['vram_gb'] else "N/A (CPU)"
    print(f"  VRAM       : {vram_str}")
    print(f"  CPU cores  : {hw['cpu_cores']}")
    ram_str = f"{hw['ram_gb']:.1f} GB" if hw['ram_gb'] else "Unknown"
    print(f"  RAM        : {ram_str}")

    print(f"\n[Dataset]")
    print(f"  train imgs : {config['image_count']}")
    native = config['native_size']
    print(f"  native res : {native}px" if native else "  native res : unknown")

    print(f"\n[Auto Config]")
    model_src  = "override" if args.model   else "auto"
    imgsz_src  = "override" if args.imgsz   else "auto"
    batch_src  = "override" if args.batch   else "auto"
    worker_src = "override" if args.workers else "auto"
    print(f"  model      : {config['model']}  ({model_src})")
    print(f"  imgsz      : {config['imgsz']}px  ({imgsz_src})")
    print(f"  batch      : {config['batch']}  ({batch_src})")
    batch = config["batch"]
    nbs = config["augmentation"]["nbs"]
    if batch < 64:
        effective = batch * ((nbs + batch - 1) // batch)  # ceil(nbs / batch)
        print(f"  effective batch : {effective}  (gradient accumulation)")
    print(f"  workers    : {config['workers']}  ({worker_src})")

    aug = config["augmentation"]
    ms_status = "(enabled - VRAM ok)" if aug["multi_scale"] > 0 else "(disabled - insufficient VRAM)"
    print(f"\n[Augmentation Config]")
    print(f"  degrees      : {aug['degrees']}")
    print(f"  flipud       : {aug['flipud']}")
    print(f"  copy_paste   : {aug['copy_paste']}")
    print(f"  mixup        : {aug['mixup']}")
    print(f"  multi_scale  : {aug['multi_scale']}  {ms_status}")
    print(f"  close_mosaic : {aug['close_mosaic']}")
    print(f"  cos_lr       : {aug['cos_lr']}")


# -- Argument Parsing ----------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train a YOLO26 object detector")

    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to data.yaml file (required for fresh training)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from last checkpoint. Use --name to specify which run."
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Override auto-selected model (e.g. yolo26x.pt)"
    )
    parser.add_argument(
        "--epochs", type=int, default=600,
        help="Number of training epochs (default: 600)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=None,
        help="Override auto-calculated image size in pixels"
    )
    parser.add_argument(
        "--batch", type=int, default=None,
        help="Override auto-calculated batch size"
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Override auto-calculated worker count"
    )
    parser.add_argument(
        "--device", type=str, default="0",
        help="Device to train on: 0 for GPU, cpu for CPU (default: 0)"
    )
    parser.add_argument(
        "--name", type=str, default=None,
        help="Run name to resume, or name for a new run (default: detector_v1)"
    )
    parser.add_argument(
        "--patience", type=int, default=100,
        help="Early stopping patience in epochs (default: 100)"
    )
    parser.add_argument(
        "--probe-vram", action="store_true",
        help="Measure/refresh local VRAM estimates (yolo26m/l/x or --model); exit if --input omitted",
    )

    return parser.parse_args()


# -- Resume Logic --------------------------------------------------------------

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
        sys.exit(1)


def resume_training(args):
    last_pt   = find_last_checkpoint(args.name)
    args_yaml = last_pt.parent.parent / "args.yaml"

    print(f"\n[Resuming Training]")
    print(f"  checkpoint : {last_pt}")

    if args.input:
        data_str = args.input.strip().strip('"').strip("'")

        if args_yaml.exists():
            with open(args_yaml) as f:
                run_args = yaml.safe_load(f)
            run_args["data"] = data_str
            with open(args_yaml, "w") as f:
                for k, v in run_args.items():
                    if k == "data":
                        f.write(f"data: '{data_str}'\n")
                    else:
                        yaml.dump({k: v}, f, default_flow_style=True, allow_unicode=True)
            print(f"  data path  : {data_str}  (patched in args.yaml)")
        else:
            print(f"  [WARNING] args.yaml not found, cannot patch data path")
    else:
        print(f"  (all settings loaded from checkpoint automatically)")
    print()

    model = YOLO(str(last_pt))
    model.train(resume=True)

    best_pt = last_pt.parent / "best.pt"

    print(f"\n{'='*60}")
    print(f"  Training complete")
    print(f"{'='*60}")
    print(f"  best.pt  : {best_pt}")
    print(f"  last.pt  : {last_pt}")
    print(f"{'='*60}")
    print(f"\n  Next step: export to TensorRT for Manifold 3:")
    print(f"    python export_tensorrt.py --weights {best_pt}")
    print()

    return best_pt


# -- Dataset Validation --------------------------------------------------------

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

    with open(yaml_path) as f:
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


# -- Fresh Training ------------------------------------------------------------

def train(args):
    if not args.input:
        print("[ERROR] --input is required for fresh training.")
        print("        Use --resume to continue an existing run.")
        sys.exit(1)

    yaml_path  = normalize_path(args.input)
    output_dir = RUNS_DIR
    run_name   = args.name if args.name else "detector_v1"

    validate_dataset(yaml_path)

    config = auto_config(yaml_path, args)
    print_auto_config(config, args)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Training Config]")
    print(f"  epochs     : {args.epochs}")
    print(f"  image size : {config['imgsz']}px")
    print(f"  patience   : {args.patience} epochs early stopping")
    print(f"  device     : {args.device}")
    print(f"  output     : {output_dir / run_name}")
    print()

    model = YOLO(str(resolve_pretrained_weights(config["model"])))

    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=config["imgsz"],
        batch=config["batch"],
        workers=config["workers"],
        device=args.device,
        project=str(output_dir),
        name=run_name,
        patience=args.patience,
        save=True,
        plots=True,
        verbose=True,
        **config["augmentation"],
    )

    run_dir = output_dir / run_name
    best_pt = run_dir / "weights" / "best.pt"
    last_pt = run_dir / "weights" / "last.pt"

    print(f"\n{'='*60}")
    print(f"  Training complete")
    print(f"{'='*60}")
    print(f"  best.pt  : {best_pt}")
    print(f"  last.pt  : {last_pt}")
    print(f"  results  : {run_dir}")
    print(f"{'='*60}")
    print(f"\n  Next step: export to TensorRT for Manifold 3:")
    print(f"    python export_tensorrt.py --weights {best_pt}")
    print()

    return best_pt


# -- Post-Training Validation --------------------------------------------------

def validate(yaml_path: Path, args):
    run_name = args.name if args.name else "detector_v1"
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
        with open(run_args_yaml) as f:
            run_args = yaml.safe_load(f)
        val_imgsz = run_args.get("imgsz", 1024)

    model   = YOLO(str(best_pt))
    metrics = model.val(
        data=str(yaml_path),
        imgsz=val_imgsz,
        device=args.device,
    )

    map50    = metrics.box.map50
    map50_95 = metrics.box.map

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

if __name__ == "__main__":
    args = parse_args()

    if args.probe_vram:
        models = [args.model] if args.model else ["yolo26m.pt", "yolo26l.pt", "yolo26x.pt"]
        ensure_vram_estimates(models, device=args.device, force=True)
        if not args.input and not args.resume:
            sys.exit(0)

    if args.resume:
        resume_training(args)
    else:
        train(args)
        yaml_path = normalize_path(args.input)
        validate(yaml_path, args)
