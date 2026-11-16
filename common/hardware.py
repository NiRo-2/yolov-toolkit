r"""
Shared hardware helpers for the YOLO toolkit.

This module is intentionally small and self-contained: it detects GPU VRAM,
CPU cores and host RAM and exposes a handful of memory-safety primitives that
both ``train_detector.py`` and ``detect_images.py`` rely on.

``torch`` is imported *optionally* here so that ``detect_images.py`` keeps its
torch-free CPU path when CUDA is not installed. ``looks_like_oom`` returns
``False`` when ``torch is None``.
"""

import os

try:
    import psutil
    PSUTIL_AVAILABLE = True
except Exception:
    PSUTIL_AVAILABLE = False

try:
    import torch  # type: ignore[import-untyped]
except Exception:
    torch = None  # type: ignore[assignment]


# Memory safety profiles: maximum allowed measured peak as a fraction of total
# memory. Mirrors the profiles historically defined in train_detector.py.
MEMORY_PROFILES = {
    "safe":       {"gpu": 0.68, "cpu": 0.38},
    "balanced":   {"gpu": 0.75, "cpu": 0.45},
    "aggressive": {"gpu": 0.85, "cpu": 0.55},
}

# Batch-size ladder used by both training and inference sizing.
BATCH_LADDER = [2, 4, 8, 16, 32, 64]


def detect_hardware():
    """Detect GPU VRAM, CPU cores and RAM.

    Returns a dict with keys ``vram_gb`` (float or None), ``cpu_cores``,
    ``ram_gb`` (float or None) and ``gpu_name`` (str; "CPU" when no CUDA device).
    """
    info = {
        "vram_gb":   None,
        "cpu_cores": os.cpu_count() or 4,
        "ram_gb":    None,
        "gpu_name":  "CPU",
    }

    try:
        if torch is not None and torch.cuda.is_available():
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


def looks_like_oom(exc: Exception) -> bool:
    """True when ``exc`` looks like a CUDA out-of-memory error.

    Returns ``False`` when torch is unavailable (preserving the torch-free CPU
    path of detect_images.py).
    """
    if torch is None:
        return False
    try:
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:
        pass
    text = str(exc).lower()
    return "out of memory" in text or "cannot allocate" in text or "cannot allocate memory" in text


def resolve_profile(name: str) -> dict:
    """Return the memory profile dict for ``name`` (safe/balanced/aggressive).

    Unknown names fall back to ``balanced``.
    """
    key = str(name).strip().lower()
    return MEMORY_PROFILES.get(key, MEMORY_PROFILES["balanced"])


def clamp_to_ladder(n: int, ladder=None) -> int:
    """Clamp ``n`` to the nearest value in ``ladder`` (default BATCH_LADDER), floor 1."""
    if ladder is None:
        ladder = BATCH_LADDER
    n = max(1, int(n))
    best = ladder[0]
    for v in ladder:
        if v <= n:
            best = v
        else:
            break
    return best


def free_vram_bytes():
    """Return (free_bytes, total_bytes) for CUDA device 0, or (None, None)."""
    if torch is None or not torch.cuda.is_available():
        return None, None
    try:
        free, total = torch.cuda.mem_get_info(0)
        return int(free), int(total)
    except Exception:
        return None, None