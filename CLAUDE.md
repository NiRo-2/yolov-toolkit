# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A YOLO Toolkit for building datasets, training detectors, and running inference with the latest Ultralytics YOLO (currently YOLO26). The pipeline flows: raw/VOC/flat inputs → dataset builders → `tile_yolo_dataset` (recommended) → training → tiled-by-default inference. Training takes a dataset **directory** and auto-detects the format (YOLO folder, COCO JSON, VOC XML) and the annotation task (detect/segment/pose).

PT → ONNX + X-AnyLabeling export lives in the separate [X-AnyLabel-toolkit](https://github.com/NiRo-2/X-AnyLabel-toolkit) repo (`scripts/yolov8_pt_to_xanylabeling_onnx/`).

## Scripts

| Script | Purpose |
|---|---|
| `vlm_yolo_prep/vlm_yolo_prep.py` | Auto-label raw photos using a local VLM (LM Studio) → YOLO dataset |
| `voc_to_yolo/voc_to_yolo.py` | Convert existing Pascal VOC XML annotations to YOLO format |
| `flat_yolo_split/flat_yolo_split.py` | Split flat folder of YOLO images + labels into train/val (+ optional test) and write `data.yaml` |
| `remap_yolo_labels/remap_yolo_labels.py` | Remap classes and merge one or more YOLO datasets into a new output dataset |
| `tile_yolo_dataset/tile_yolo_dataset.py` | Tile a YOLO dataset into overlapping windows for small-object training |
| `train_detector/train_detector.py` | Train YOLO26 detector/segmenter/pose model from a dataset directory with auto-configured hardware-aware hyperparameters |
| `detect_images/detect_images.py` | Run a trained model on an image folder with tiled inference on by default; draw boxes and export JSON detections |
| `ortho_tag_sidecar/ortho_tag_sidecar.py` | Pillow GPS → ExifTool `-G1`-style metadata helpers; CLI verifies one sidecar JSON for B3DM |
| `exiftool/_Run_exiftool.bat` | Windows helper to dump full image metadata to `./exiftool/outputs/` |
| `flat_yolo_split/_Run_flat_yolo_split_template.bat` | Template batch helper for flat YOLO folder → train/val split |
| `remap_yolo_labels/_Run_remap_yolo_labels_template.bat` | Template batch helper for personal multi-input remap/merge runs |

## Common Commands

**Install dependencies:**
```bash
pip install -r requirements.txt
```

`requirements.txt` requires `ultralytics>=8.4.0` for YOLO26 support. Update it with `pip install -U ultralytics`.

**Dataset from raw photos:**
```bash
python vlm_yolo_prep/vlm_yolo_prep.py --input C:/data/raw --output C:/data/dataset --objects screw bolt
```

**Dataset from VOC:**
```bash
python voc_to_yolo/voc_to_yolo.py --input C:/data/voc --output C:/data/dataset
```

**Dataset from flat YOLO folder:**
```bash
python flat_yolo_split/flat_yolo_split.py --input C:/data/labels --output C:/data/dataset
```

**Remap / merge datasets:**
```bash
python remap_yolo_labels/remap_yolo_labels.py --input C:/data/a --input C:/data/b --output C:/data/merged --map 0:bolt_a:Bolt --map 1:rusty_screw:Screw

# --input is repeatable/dynamic; pass as many as needed
python remap_yolo_labels/remap_yolo_labels.py --input C:/data/d1 --input C:/data/d2 --input C:/data/d3 --input C:/data/d4 --input C:/data/d5 --input C:/data/d6 --input C:/data/d7 --input C:/data/d8 --input C:/data/d9 --input C:/data/d10 --output C:/data/merged_many --map 0:bolt_a:Bolt --map 9:rusty_screw:Screw
```

`--map INDEX:OLD:NEW` indices follow input order (first `--input` is index 0).

**Tile a dataset for small-object training (recommended):**
```bash
python tile_yolo_dataset/tile_yolo_dataset.py --input C:/data/dataset --output C:/data/dataset_tiled
```

Default tiling uses `--imgsz 1024`, `--overlap 0.2`, and `--empty-frac 0.10`. Point `train_detector.py --input` at the tiled dataset folder; `--manifest` writes `tiles_manifest.json`.

**Train:**
```bash
# --input is a dataset DIRECTORY: YOLO (data.yaml present or synthesized), COCO JSON, or VOC XML.
# Format + task (detect/segment/pose) auto-detect; COCO/VOC convert once into a <folder>_yolo copy.
python train_detector/train_detector.py --input /path/to/dataset --name my_detector

# Override the auto-derived task, or relocate the conversion copy (source dir is never modified)
python train_detector/train_detector.py --input /path/to/dataset --name my_detector --task segment
python train_detector/train_detector.py --input /path/to/coco_ds --name my_detector --converted-dir C:/data/my_yolo --yes

python train_detector/train_detector.py --resume --name my_detector   # resume crashed run
```

**Probe VRAM estimates without training:**
```bash
python train_detector/train_detector.py --probe-vram
python train_detector/train_detector.py --probe-vram --model yolo26l.pt
```

**Detect:**
```bash
python detect_images/detect_images.py --images /path/to/images --model train_detector/runs/detect/my_detector/weights/best.pt --export-json
```

Detection tiles images by default (`--tile-imgsz 1024`, `--tile-overlap 0.2`, `--tile-iou 0.5`) and merges tile detections with class-wise NMS. Use `--no-tiles` for legacy whole-image inference.

**X-AnyLabeling (PT → ONNX + config)** — run from [X-AnyLabel-toolkit](https://github.com/NiRo-2/X-AnyLabel-toolkit):
```bash
python scripts/yolov8_pt_to_xanylabeling_onnx/yolov8_pt_to_xanylabeling_onnx.py /path/to/train_detector/runs/detect/my_detector/weights/best.pt
```

## Local outputs (gitignored)

| Script | Output location |
|---|---|
| `train_detector/train_detector.py` | `train_detector/runs/detect/<name>/`, `train_detector/weights/` |
| `detect_images/detect_images.py` | `detect_images/detections/<input_folder_name>/` |
| `exiftool/_Run_exiftool.bat` | `exiftool/outputs/` |

`.gitignore` also covers: `*_personal.bat`, legacy repo-root `runs/`, `exiftool/`, `*.pt`, `*.onnx`, `*.engine`, and external dataset `*.yaml` files. Dataset scripts write only to user `--output` paths.

## Architecture

All scripts share common patterns:
- `normalize_path()` — handles Windows/Unix path conversion
- YOLO dataset format: `train/val/[test]/images/` + `labels/` + `data.yaml`
- `data.yaml` structure: `train`, `val`, `test` (optional), `nc`, `names`
- Class IDs assigned in order (first listed = 0, second = 1, etc.)

**`train_detector/train_detector.py`** — auto-detects GPU VRAM, CPU cores, RAM, dataset size, and native image resolution. Selects optimal model (m/l/x), imgsz, batch size, and workers. Decision logic in `select_model_and_imgsz()` and `calc_batch()` with VRAM_PER_IMAGE estimates. `resolve_dataset_input()` accepts only a dataset directory: cascade is existing `data.yaml` passthrough → COCO JSON → VOC XML → YOLO label synthesis (needs `classes.txt` or unambiguous label files) → hard error. Task taxonomy detect/segment/pose derives from label line shape (detect `5` fields, segment `1+2N`, pose `1+4+3K`; ambiguous counts disambiguated by value ranges). COCO/VOC convert once into a `<folder>_yolo` sibling governed by `.conversion_manifest.json` (reuse/rebuild gate; foreign same-named dirs are never wiped). Per-task model chains `yolo26{n,m,l,x}[-seg|-pose].pt`. Attempt task precedence: resolved dataset task (CLI `--task` or derived from train labels) → stored run yaml/checkpoint train args → model-name suffix (`-pose` wins over `-seg`) → detect; `resolve_attempt_task()` + `GUARDED_TRAINERS` select the guarded trainer subclass (`GuardedDetectionTrainer`/`GuardedSegmentTrainer`/`GuardedPoseTrainer`) — Ultralytics requires the task-matched trainer for `-seg`/`-pose` weights (the detect trainer crashes its own val NMS on tuple predictions).

**`tile_yolo_dataset/tile_yolo_dataset.py`** — splits YOLO train/val[/test] images into overlapping square windows, clips and re-normalizes labels (dropping a clipped box if less than 20% of its original area survives), caps empty tiles at no more than 10% of each split's *total output tile count* (not 10% of the labelled-tile count — that works out to ~11% of labelled tiles), and writes a new YOLO dataset plus optional tile-provenance manifest. Streams per source image (writes labelled tiles immediately, re-reads a source image only for its selected empty tiles) so peak memory is one decoded image, not the whole split. Segment datasets: polygons are clipped per tile with Sutherland–Hodgman (`tile_geometry.py`), kept when ≥20% of the original shoelace polygon area survives, dropped below 3 vertices, and re-normalized tile-relative. Pose datasets require `kpt_shape: [K,3]` in `data.yaml`: bbox keeps the 20% rule; keypoints inside a tile clamp and stay kv=2, keypoints outside become `(0, 0, 1)` (labelled-invisible), kv 0/1 triplets pass through; `kpt_shape`/`flip_idx` are copied verbatim into the tiled `data.yaml`.

**`vlm_yolo_prep/vlm_yolo_prep.py`** — sends images to LM Studio's OpenAI-compatible API, parses JSON bbox responses, salvages partial JSON from truncated outputs, converts to YOLO format, splits dataset, writes data.yaml. Uses Qwen2.5-VL models.

**`voc_to_yolo/voc_to_yolo.py`** — discovers image/XML pairs (supports flat, VOC-standard, and pre-split layouts), parses Pascal VOC XML, converts to YOLO normalized coords, writes dataset.

**`flat_yolo_split/flat_yolo_split.py`** — reads `classes.txt` or `labels.txt` (not both), validates no anonymous/out-of-range class IDs, discovers image+label pairs in a flat folder, shuffles into train/val[/test], copies files and writes `data.yaml`.

**`remap_yolo_labels/remap_yolo_labels.py`** — accepts repeatable `--input` datasets, applies indexed remap rules (`--map index:old:new`), merges all sources into one new output dataset, preserves split structure, resolves filename collisions with source-suffixed renames, and updates merged `data.yaml`.

## Key Details

- `vlm_yolo_prep.py`: MAX_INFERENCE_SIZE (line ~431) controls VLM input resolution — match to LM Studio context setting (4000 for 32k, 2048 for 16k, 1280 for 8k)
- `train_detector/train_detector.py`: writes to `train_detector/runs/detect/<name>/`; caches pretrained backbones and CUDA VRAM probe results at `train_detector/weights/vram_estimates.json` via `resolve_pretrained_weights()` / `ensure_vram_estimates()`; `--probe-vram` refreshes measurements and exits without `--input`; falls back to FLOPs-scaled estimates when CUDA or probing is unavailable; default `--patience` 100; fresh-run augmentation: `degrees=180`, `flipud=0.5`, `copy_paste=0.3`, `mixup=0.15`, `multi_scale=0.5` (when VRAM supports batch >= 4 under peak multi-scale sizing), `close_mosaic=60`, `cos_lr=True`, `nbs=64` (not applied on `--resume`); imgsz only raised if batch stays >= 4 (nbs=64 keeps effective batch at 64); prints `effective batch` in `[Auto Config]` when batch < 64; capped to native image resolution (no upscaling); uses `statistics.median()` for native resolution detection
- `detect_images/detect_images.py`: JSON export includes both pixel coords (x1,y1,x2,y2) and YOLO normalized (cx,cy,bw,bh); labels.txt maps class_id to class_name
- `detect_images/detect_images.py`: the task (detect/segment/pose) comes from the loaded model (`YOLO(model).task`), not a CLI flag; detect outputs are unchanged. Segment dets add `polygons_px` / `polygons_norm` (largest `cv2.findContours` contour per box); pose dets add `keypoints_px` / `keypoints_norm` as `[x, y, visibility]`.
- `detect_images/detect_images.py`: on tiled runs, segment and pose dets merge per class via union-find grouping on box IoU ≥ `--tile-iou` with transitive chaining (objects spanning 3+ tiles merge): segment fragments are OR-composited on a canvas sized to the group's union bbox and re-contoured into one whole polygon (no slice seam); pose merges by per-index visibility voting limited to each member tile's extent, missing indices stay `[0,0,0]`. Drawing: semi-transparent segment mask fill (alpha 0.35) under the box/label, pose keypoints as filled red circles.
- `detect_images/detect_images.py`: annotated-image save path uses Pillow metadata transfer plus ExifTool (`--exiftool`, PATH, or repo-local `./exiftool/`) for full metadata groups; save-time fallback can be enabled with `--allow-missing-exiftool`
- `detect_images/detect_images.py`: subdirectory scanning is on by default (`--recursive`, disable with `--no-recursive`); outputs stay flat under `detect_images/detections/<input_folder_name>/` with relative-subpath underscore prefixing on collisions (`sub/a/foo.jpg` → `sub_a_foo.jpg`)
- `detect_images/detect_images.py`: parallel pipeline by default — `--batch` (default `auto = 8 if CUDA else 1`) batches GPU inference, and `--workers` (default `auto = min(8, os.cpu_count())`) runs post-inference I/O (JSON sidecar, ExifTool metadata extract / copy, Pillow image save) on a `ThreadPoolExecutor` while the main thread reads the next batch. Pass `--workers 1 --batch 1` to revert to fully sequential per-image processing. Inference itself stays single-threaded on one GPU; the speed-up comes from overlapping ExifTool subprocesses with the next batch's inference. Worker prints are serialized via a `threading.Lock` but may appear slightly out of order.
- `detect_images/detect_images.py`: tiled inference is default on (`--tiles`); use `--no-tiles` for whole-image inference. It uses `--tile-imgsz 1024`, `--tile-overlap 0.2`, batches each source image's tiles in chunks of `--batch` (default `8` on CUDA, `1` on CPU) at `imgsz=tile` — tiles from different source images are never batched together — then merges detections with class-wise NMS (`--tile-iou 0.5`).
- `ortho_tag_sidecar/ortho_tag_sidecar.py`: `merge_pillow_gps_exif_into_metadata()` fills `GPS:*` / basic `ExifIFD:*` when JSON lacks ExifTool-style keys; `--verify-b3dm` in `detect_images.py` uses the same checks as `python ortho_tag_sidecar/ortho_tag_sidecar.py <sidecar.json>`
- `remap_yolo_labels.py`: all input datasets are read-only; only `--output` is written. Final class IDs are aligned by final class names across all merged inputs.
- `flat_yolo_split.py`: requires `classes.txt` or `labels.txt` in `--input`; rejects both present; hard-fails on anonymous class names and invalid bbox lines before writing output
- All scripts create auto-versioned output dirs (`dataset` → `dataset_v2` → `dataset_v3`) when target exists and is non-empty
- Type checker false positive on `from ultralytics import YOLO` — already suppressed with `# type: ignore[exported]` — keep it suppressed with `# type: ignore[union-attr]`

## Out of scope

- X-AnyLabeling ONNX export configs for segment/pose models — lives in the separate [X-AnyLabel-toolkit](https://github.com/NiRo-2/X-AnyLabel-toolkit) repo; only detect-PT export is handled there today.
- COCO `iscrowd` mask decoding — `iscrowd` annotations are skipped (with a logged count) in every task, never turned into labels.
- VOC segmentation/pose conversion — Pascal VOC XML carries bbox-only annotations; VOC datasets always convert as `detect`, and `--task segment|pose` on a VOC directory is a hard error.
