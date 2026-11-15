"""Pure geometry helpers shared by YOLO dataset tiling and tiled inference."""

from __future__ import annotations

import math
import random
from collections import defaultdict


def iter_tile_windows(
    width: int, height: int, tile: int, overlap: float
) -> list[tuple[int, int, int, int]]:
    """Return overlapping tile windows, with final windows flush to image edges."""
    if width <= 0 or height <= 0 or tile <= 0:
        raise ValueError("width, height, and tile must be positive")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")
    if width <= tile and height <= tile:
        return [(0, 0, width, height)]

    stride = max(1, int(tile * (1 - overlap)))

    def starts(length: int) -> list[int]:
        if length <= tile:
            return [0]
        positions = list(range(0, length - tile + 1, stride))
        final = length - tile
        if positions[-1] != final:
            positions.append(final)
        return positions

    return [
        (x1, y1, min(x1 + tile, width), min(y1 + tile, height))
        for y1 in starts(height)
        for x1 in starts(width)
    ]


def yolo_line_to_xyxy(
    parts: list[float], img_w: int, img_h: int
) -> tuple[int, float, float, float, float]:
    """Convert a YOLO class/center/size record to pixel-space xyxy coordinates."""
    cls_id, cx, cy, box_w, box_h = parts
    cx *= img_w
    cy *= img_h
    box_w *= img_w
    box_h *= img_h
    return (
        int(cls_id),
        cx - box_w / 2,
        cy - box_h / 2,
        cx + box_w / 2,
        cy + box_h / 2,
    )


def xyxy_to_yolo_line(
    cls_id: int,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    tile_w: int,
    tile_h: int,
) -> str:
    """Serialize a pixel-space xyxy box as a normalized YOLO annotation line."""
    box_w = x2 - x1
    box_h = y2 - y1
    cx = (x1 + x2) / 2 / tile_w
    cy = (y1 + y2) / 2 / tile_h
    return f"{cls_id} {cx:.6f} {cy:.6f} {box_w / tile_w:.6f} {box_h / tile_h:.6f}"


def clip_box_to_tile(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    tx1: int,
    ty1: int,
    tx2: int,
    ty2: int,
) -> tuple[float, float, float, float] | None:
    """Clip an xyxy box to a tile and return coordinates relative to its origin."""
    clipped_x1 = max(x1, tx1)
    clipped_y1 = max(y1, ty1)
    clipped_x2 = min(x2, tx2)
    clipped_y2 = min(y2, ty2)
    if clipped_x1 >= clipped_x2 or clipped_y1 >= clipped_y2:
        return None
    return (
        clipped_x1 - tx1,
        clipped_y1 - ty1,
        clipped_x2 - tx1,
        clipped_y2 - ty1,
    )


def keep_clipped_box(
    orig_area: float,
    clipped: tuple[float, float, float, float],
    min_frac: float = 0.2,
) -> bool:
    """Return whether a clipped box retains the required original-area fraction."""
    if orig_area <= 0:
        return False
    x1, y1, x2, y2 = clipped
    return (x2 - x1) * (y2 - y1) / orig_area >= min_frac


def polygon_area(points: list[tuple[float, float]]) -> float:
    """Return the absolute area of a closed polygon using the shoelace formula."""
    n = len(points)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def clip_polygon_to_tile(
    points: list[tuple[float, float]],
    tx1: int,
    ty1: int,
    tx2: int,
    ty2: int,
) -> list[tuple[float, float]] | None:
    """Clip a pixel-space polygon to a tile (Sutherland–Hodgman).

    Returns the surviving vertices relative to the tile origin, or None when
    fewer than 3 vertices remain (a polygon line needs at least 3 points).
    """
    def clip_half(poly, inside, crossing):
        out: list[tuple[float, float]] = []
        prev = poly[-1]
        prev_in = inside(prev)
        for cur in poly:
            cur_in = inside(cur)
            if cur_in:
                if not prev_in:
                    out.append(crossing(prev, cur))
                out.append(cur)
            elif prev_in:
                out.append(crossing(prev, cur))
            prev, prev_in = cur, cur_in
        return out

    def cross_vertical(a, b, x):
        ax, ay = a
        bx, by = b
        if abs(bx - ax) < 1e-12:
            return (x, (ay + by) / 2)
        t = (x - ax) / (bx - ax)
        return (x, ay + t * (by - ay))

    def cross_horizontal(a, b, y):
        ax, ay = a
        bx, by = b
        if abs(by - ay) < 1e-12:
            return ((ax + bx) / 2, y)
        t = (y - ay) / (by - ay)
        return (ax + t * (bx - ax), y)

    poly: list[tuple[float, float]] = list(points)
    for inside, crossing in (
        (lambda p: p[0] >= tx1, lambda a, b: cross_vertical(a, b, tx1)),
        (lambda p: p[0] <= tx2, lambda a, b: cross_vertical(a, b, tx2)),
        (lambda p: p[1] >= ty1, lambda a, b: cross_horizontal(a, b, ty1)),
        (lambda p: p[1] <= ty2, lambda a, b: cross_horizontal(a, b, ty2)),
    ):
        if not poly:
            return None
        poly = clip_half(poly, inside, crossing)
    if len(poly) < 3:
        return None
    return [(x - tx1, y - ty1) for x, y in poly]


def polygon_to_yolo_line(
    cls_id: int,
    points: list[tuple[float, float]],
    tile_w: int,
    tile_h: int,
) -> str:
    """Serialize tile-relative polygon points as a normalized YOLO segmentation line."""
    parts = [str(cls_id)]
    for x, y in points:
        parts.append(f"{x / tile_w:.6f}")
        parts.append(f"{y / tile_h:.6f}")
    return " ".join(parts)


def keypoints_to_tile_line(
    keypoints: list[tuple[float, float, int]],
    tx1: int,
    ty1: int,
    tx2: int,
    ty2: int,
    img_w: int,
    img_h: int,
) -> str:
    """Serialize pixel keypoints as tile-normalized triplets using YOLO visibility rules.

    A visible keypoint (kv=2) inside the tile is clamped to the tile edge and stays
    kv=2; outside the tile it becomes (0, 0, 1) — labelled but invisible. kv=0/1
    entries pass through unchanged in source-image-normalized coordinates.
    """
    tile_w = tx2 - tx1
    tile_h = ty2 - ty1
    parts: list[str] = []
    for kx, ky, kv in keypoints:
        if kv == 2:
            if kx < tx1 or kx > tx2 or ky < ty1 or ky > ty2:
                parts.extend(("0.000000", "0.000000", "1"))
            else:
                nx = min(max(kx - tx1, 0.0), tile_w) / tile_w
                ny = min(max(ky - ty1, 0.0), tile_h) / tile_h
                parts.append(f"{nx:.6f}")
                parts.append(f"{ny:.6f}")
                parts.append("2")
        else:
            parts.append(f"{kx / img_w:.6f}")
            parts.append(f"{ky / img_h:.6f}")
            parts.append(str(kv))
    return " ".join(parts)


def select_empty_tiles(
    labelled_count: int,
    empty_indices: list[int],
    empty_frac: float = 0.10,
    seed: int = 42,
) -> list[int]:
    """Deterministically sample empty tiles without exceeding their output share."""
    if labelled_count <= 0 or empty_frac <= 0:
        return []
    if empty_frac >= 1:
        max_empty = len(empty_indices)
    else:
        max_empty = math.floor(labelled_count * empty_frac / (1 - empty_frac))
    chosen = list(empty_indices)
    random.Random(seed).shuffle(chosen)
    return chosen[:max_empty]


def nms_xyxy(dets: list[dict], iou_thresh: float = 0.5) -> list[dict]:
    """Apply greedy confidence-ordered non-maximum suppression per class."""
    by_class: dict[int, list[dict]] = defaultdict(list)
    for detection in dets:
        by_class[detection["cls"]].append(detection)

    kept: list[dict] = []
    for class_dets in by_class.values():
        candidates = sorted(class_dets, key=lambda det: det["conf"], reverse=True)
        while candidates:
            best = candidates.pop(0)
            kept.append(best)
            candidates = [
                candidate
                for candidate in candidates
                if _iou_xyxy(best, candidate) <= iou_thresh
            ]
    return kept


def _iou_xyxy(first: dict, second: dict) -> float:
    intersection_w = max(0.0, min(first["x2"], second["x2"]) - max(first["x1"], second["x1"]))
    intersection_h = max(0.0, min(first["y2"], second["y2"]) - max(first["y1"], second["y1"]))
    intersection = intersection_w * intersection_h
    first_area = max(0.0, first["x2"] - first["x1"]) * max(
        0.0, first["y2"] - first["y1"]
    )
    second_area = max(0.0, second["x2"] - second["x1"]) * max(
        0.0, second["y2"] - second["y1"]
    )
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0
