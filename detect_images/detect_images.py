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
    --batch       GPU inference batch size; integer or 'auto'
                  (default: auto = 8 if CUDA is available, else 1)
    --tiles       tile large images before detect, merging results with NMS
                  (default: True)
    --no-tiles    disable tiling; run whole-image inference (legacy behavior)
    --tile-imgsz  tile size in pixels when --tiles is on (default: 1024)
    --tile-overlap tile overlap fraction (default: 0.2)
    --tile-iou    NMS IoU threshold for merging tiled detections (default: 0.5)

By default, images are tiled before inference (tiling is skipped automatically
when an image already fits within one tile). Each source image's tiles are
batched together in a single model call; tiles from different source images
are never batched together. Detections from all tiles of one image are merged
with class-wise NMS. Pass --no-tiles to restore whole-image inference.

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

By default the script batches inference across multiple images per GPU call
and runs post-inference I/O (sidecar JSON export, ExifTool metadata extract /
copy, Pillow image save) on a thread pool, overlapping ExifTool subprocesses
with the next batch's inference. Pass --workers 1 --batch 1 to revert to a
fully sequential, single-image-at-a-time pipeline. Memory cost grows with
batch size: roughly batch * imgsz^2 * 3 * 4 bytes on host plus a similar
amount of GPU RAM.

Output:
    detect_images/detections/<input_folder_name>/    <- annotated images + JSON
    detect_images/detections/<input_folder_name>/labels.txt  <- class_id → class_name mapping
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

_REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
_SIDECAR_DIR = _REPO_ROOT / "ortho_tag_sidecar"
if str(_SIDECAR_DIR) not in sys.path:
    sys.path.insert(0, str(_SIDECAR_DIR))
from ortho_tag_sidecar import (
    merge_pillow_gps_exif_into_metadata,
    verify_sidecar_json_file,
)

_TILE_DIR = _REPO_ROOT / "tile_yolo_dataset"
if str(_TILE_DIR) not in sys.path:
    sys.path.insert(0, str(_TILE_DIR))
from tile_geometry import iter_tile_windows, nms_xyxy  # type: ignore


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
        "--only-detections", action="store_true", default=True,
        help="Only save images that have at least one detection (default: True)"
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

    return parser.parse_args()


def resolve_workers(value) -> Tuple[int, str]:
    """Resolve --workers argument. Returns (n_workers, source_label)."""
    if isinstance(value, str) and value.lower() == "auto":
        return max(1, min(8, os.cpu_count() or 4)), "auto"
    try:
        n = int(value)
    except (TypeError, ValueError):
        print(f"[ERROR] Invalid --workers value: {value!r} (must be an integer or 'auto')")
        sys.exit(1)
    return max(1, n), "manual"


def resolve_batch(value) -> Tuple[int, str]:
    """Resolve --batch argument. Returns (batch_size, source_label)."""
    if isinstance(value, str) and value.lower() == "auto":
        cuda = bool(torch is not None and torch.cuda.is_available())
        return (8 if cuda else 1), ("auto, cuda" if cuda else "auto, cpu")
    try:
        n = int(value)
    except (TypeError, ValueError):
        print(f"[ERROR] Invalid --batch value: {value!r} (must be an integer or 'auto')")
        sys.exit(1)
    return max(1, n), "manual"


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


def detect_image_tiled(model, image_bgr, conf, tile, overlap, iou, batch_size=1, task: str = "detect") -> list:
    """Run tiled detection over one image; returns a flat list of det dicts.

    Crops all tile windows for this image and runs them through the model in
    chunks of at most `batch_size` tiles per call (tiles from different
    source images are never batched together). `imgsz=tile` is passed
    explicitly so Ultralytics runs inference at the tile's native size
    instead of silently resizing crops to its default 640. When the image
    fits within one tile, `iter_tile_windows` returns a single whole-image
    window, so this is equivalent to whole-image inference. Detect dets from
    multiple tiles are merged with class-wise NMS; segment and pose dets are
    merged with transitive IoU chaining (`group_by_iou`) plus mask compositing
    or keypoint voting, and carry their per-tile extent in "tile".
    """
    h, w = image_bgr.shape[:2]
    windows = iter_tile_windows(w, h, tile, overlap)
    crops = [image_bgr[ty1:ty2, tx1:tx2] for (tx1, ty1, tx2, ty2) in windows]

    tile_results = []
    for chunk_start in range(0, len(crops), max(1, batch_size)):
        chunk = crops[chunk_start:chunk_start + max(1, batch_size)]
        tile_results.extend(model(chunk, conf=conf, imgsz=tile, verbose=False))

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


def extract_image_metadata(img_path: Path, exiftool_cmd) -> Tuple[Dict, str]:
    """Extract full metadata via exiftool, with Pillow EXIF fallback.

    Returns (metadata_dict, source) where source is ``exiftool``, ``pillow``, or ``none``.
    """
    if exiftool_cmd:
        cmd = exiftool_cmd + ["-j", "-a", "-u", "-G1", str(img_path)]
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

    # Resolve concurrency and batching
    workers, workers_src = resolve_workers(args.workers)
    batch_size, batch_src = resolve_batch(args.batch)

    exiftool_cmd, exiftool_resolve_reason = resolve_exiftool_command(args.exiftool)
    require_exiftool_before_run(args, exiftool_cmd, exiftool_resolve_reason)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

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
    print()

    # Load model
    model = YOLO(str(model_path))
    class_names = model.names
    model_task = getattr(model, "task", None) or "detect"
    print(f"  task       : {model_task}")

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

    total = len(image_paths)

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
            cmd = exiftool_cmd + [
                "-overwrite_original",
                "-m",
                "-P",
                "-TagsFromFile",
                str(src_path),
                "-all:all",
                "-unsafe",
                str(dest_path),
            ]
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
            metadata, meta_src = extract_image_metadata(img_path, exiftool_cmd)
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

    # Process images in batches: main thread reads + runs inference, worker
    # pool fans out the per-image post-processing for that batch.
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
                # Per-source-image tiled detect: each image's own tiles are
                # run in chunks of `batch_size` inside detect_image_tiled,
                # but tiles from different source images are never batched
                # with each other.
                for idx, p, im, flat_name, display_name in valid:
                    dets = detect_image_tiled(
                        model, im, args.conf,
                        args.tile_imgsz, args.tile_overlap, args.tile_iou,
                        batch_size,
                        task=model_task,
                    )
                    pool.submit(post_process, idx, p, im, dets, flat_name, display_name)
            else:
                images_for_model = [v[2] for v in valid]
                results_list = list(model(images_for_model, conf=args.conf, verbose=False))

                for (idx, p, im, flat_name, display_name), r in zip(valid, results_list):
                    dets = results_to_dets(r, task=model_task)
                    pool.submit(post_process, idx, p, im, dets, flat_name, display_name)
    # ThreadPoolExecutor.__exit__ waits for all submitted tasks to complete.

    # Export labels.txt (class_id → class_name mapping)
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