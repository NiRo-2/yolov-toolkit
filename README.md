# YOLO Toolkit

A clean, practical toolkit for building datasets, training detectors, and running inference with the latest Ultralytics YOLO (currently YOLO26) — **automatically configures itself** based on your GPU, RAM, and dataset size — no manual tuning needed.

Built for real-world use: handles Windows paths, crash recovery, and scales from a laptop GPU to a workstation.

---

## Scripts

| Script | What it does |
|---|---|
| `vlm_yolo_prep/vlm_yolo_prep.py` | Build a labelled YOLO dataset from raw photos using a local VLM — no manual annotation needed |
| `voc_to_yolo/voc_to_yolo.py` | Convert an existing Pascal VOC annotated dataset to YOLO format |
| `flat_yolo_split/flat_yolo_split.py` | Split a flat folder of YOLO images + labels into train/val and write `data.yaml` |
| `remap_yolo_labels/remap_yolo_labels.py` | Copy an existing YOLO dataset to a new location and remap/merge class labels |
| `tile_yolo_dataset/tile_yolo_dataset.py` | Tile a YOLO dataset into overlapping windows for small-object training |
| `train_detector/train_detector.py` | Train a YOLO26 detector on any labelled dataset |
| `detect_images/detect_images.py` | Run a trained model on a folder of images; tiled inference is on by default |
| `ortho_tag_sidecar/ortho_tag_sidecar.py` | Verify detection JSON sidecars for Ortho-Tag / B3DM georeference keys (`Usage:` one `.json` path) |

**Related:** [X-AnyLabel-toolkit](https://github.com/NiRo-2/X-AnyLabel-toolkit) — PT → ONNX + X-AnyLabeling `config.yaml` export (`scripts/yolov8_pt_to_xanylabeling_onnx/`).

---

## Full Pipeline

```
[Raw photos]        [VOC dataset]        [Flat YOLO labels folder]
     |                   |                        |
     v                   v                        v
vlm_yolo_prep/     voc_to_yolo/          flat_yolo_split/
     |                   |                        |
     +-------------------+------------------------+
                     |
                     v
          Labelled YOLO dataset
          (train / val + data.yaml)
                     |
                     v
      tile_yolo_dataset/ (recommended
        for small-object training)
                     |
                     v
          train_detector/
                     |
                     v
              detect_images/
          (tiled inference by default)
```

For X-AnyLabeling ONNX export, use the companion repo [X-AnyLabel-toolkit](https://github.com/NiRo-2/X-AnyLabel-toolkit).

---

## Requirements

```bash
pip install -r requirements.txt
```

- Python 3.8+
- `ultralytics>=8.4.0` (currently YOLO26); update with `pip install -U ultralytics`
- PyTorch with CUDA (for GPU training) — install from [pytorch.org](https://pytorch.org/get-started/locally/)
- [LM Studio](https://lmstudio.ai/) with a vision model loaded (for `vlm_yolo_prep.py` only)
- Dataset in YOLO format (e.g. exported from [Roboflow](https://roboflow.com))

### Local outputs (gitignored)

Scripts that write into the repo use **script-local folders** (not the repo root). These paths are in `.gitignore`:

| Script | Default output location |
|---|---|
| `train_detector/train_detector.py` | `train_detector/runs/detect/<name>/`, `train_detector/weights/` (pretrained backbones) |
| `detect_images/detect_images.py` | `detect_images/detections/<input_folder_name>/` |
| `exiftool/_Run_exiftool.bat` | `exiftool/outputs/` (under gitignored `exiftool/`) |

Also ignored globally: `*_personal.bat`, `*.pt`, `*.onnx`, `*.engine`, and dataset `data.yaml` files written outside these folders.

Dataset-building scripts (`vlm_yolo_prep`, `voc_to_yolo`, `flat_yolo_split`, `remap_yolo_labels`) write only to paths you pass via `--output` (typically outside the repo).

If you still have an old repo-root `runs/` folder from before this layout, move it to `train_detector/runs/` (or delete after copying).

Pretrained Ultralytics backbones (`yolo26m.pt`, etc.) belong in `train_detector/weights/` — not the repo root. `train_detector.py` resolves and downloads them there automatically.

---

## Step 1a — Build a Dataset from Raw Photos (`vlm_yolo_prep.py`)

Sends each photo to a Vision-Language Model running locally in LM Studio, gets bounding box coordinates back, and writes a complete YOLO dataset with no manual labelling.

### Basic usage
```bash
python vlm_yolo_prep/vlm_yolo_prep.py \
    --input  C:/data/raw_photos \
    --output C:/data/dataset \
    --objects screw "hex bolt" "countersunk screw"
```

Classes are auto-assigned in the order you list them: `screw=0`, `hex bolt=1`, `countersunk screw=2`.

Annotated preview images are saved to `<output>/preview/` by default so you can visually verify detection quality before training.

### Windows batch file
```bat
python vlm_yolo_prep/vlm_yolo_prep.py ^
    --input  "C:\data\raw_photos" ^
    --output "C:\data\dataset"    ^
    --objects Screw               ^
    --model   qwen2.5-vl-7b-instruct ^
    --confidence 0.9              ^
    --downsample 2
```

### LM Studio setup
1. Download and open [LM Studio](https://lmstudio.ai/)
2. Download one of these models:
   - **Best quality:** `Qwen2.5-VL-72B-Instruct-GGUF` Q4_K_M — needs 16 GB VRAM + 128 GB RAM, set context to **32k**
   - **Faster:** `Qwen2.5-VL-7B-Instruct-GGUF` Q8_0 — needs 8 GB VRAM, set context to **16k**
3. Load the model and start the local server (default port 1234)

### All arguments

| Argument | Default | Description |
|---|---|---|
| `--input` | required | Folder of raw images |
| `--output` | required | Output folder (created if absent, auto-versioned if not empty) |
| `--objects` | required | Object names to detect. Spaces allowed: `"hex bolt"` `"red car"` |
| `--classes` | auto | Override class IDs: `screw:0 bolt:0 "hex bolt":1`. Auto-assigned by default |
| `--model` | `qwen2.5-vl-72b-instruct` | Model name as shown in LM Studio |
| `--url` | `http://localhost:1234/...` | LM Studio API endpoint |
| `--timeout` | `180` | API timeout in seconds (use 60 for the 7B model) |
| `--retries` | `2` | Retry attempts per image on API failure |
| `--confidence` | `0.0` | Min detection confidence — `0.0` keeps all, `0.9` keeps only high-confidence |
| `--downsample` | `1.0` | Divide image dimensions by this factor before sending to VLM |
| `--train` | `0.70` | Train split ratio |
| `--val` | `0.20` | Val split ratio (gets remainder when `--enable-test` is off) |
| `--seed` | `42` | Random seed for reproducible splits |
| `--enable-test` | off | Create a test split in addition to train and val |
| `--no-preview` | off | Disable annotated preview image export |

### Output structure

```
dataset/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
├── preview/          ← annotated images for visual QA (one per input photo)
└── data.yaml
```

Generated `data.yaml`:
```yaml
train: ../train/images
val: ../val/images
nc: 1
names: ['screw']
```

> **Note:** VLM-generated labels are a strong starting point but not perfect. Review the `preview/` folder and correct any bad detections in [Roboflow](https://roboflow.com) or [LabelImg](https://github.com/HumanSignal/labelImg) before final training.

---

## Step 1b — Convert an Existing VOC Dataset (`voc_to_yolo.py`)

If you already have a dataset annotated in Pascal VOC format (XML files), this script converts it to YOLO format in one command — no manual work needed.

### Basic usage
```bash
python voc_to_yolo/voc_to_yolo.py --input C:/data/voc --output C:/data/dataset
```

Classes are auto-discovered from the XML files and sorted alphabetically. Use `--classes` to control the order (and therefore the class IDs) explicitly.

### Supported VOC layouts

The script handles all common VOC folder structures automatically:

```
# Flat — images and XMLs in the same folder
input/
  image1.jpg  image1.xml
  image2.jpg  image2.xml

# Standard VOC — separate subfolders
input/
  JPEGImages/   image1.jpg
  Annotations/  image1.xml

# Pre-split — already divided into train/val/test
input/
  train/  image1.jpg  image1.xml
  val/    image2.jpg  image2.xml
```

### All arguments

| Argument | Default | Description |
|---|---|---|
| `--input` | required | Folder containing VOC images and XML annotation files |
| `--output` | required | Output folder for the YOLO dataset (created if absent, auto-versioned if not empty) |
| `--classes` | auto | Class names in order — controls ID assignment. Auto-discovered and sorted alphabetically if omitted |
| `--train` | `0.70` | Train split ratio |
| `--val` | `0.20` | Val split ratio (gets remainder when `--enable-test` is off) |
| `--seed` | `42` | Random seed for reproducible splits |
| `--enable-test` | off | Create a test split in addition to train and val |

### Examples

```bash
# Auto-discover classes, 70/30 split
python voc_to_yolo/voc_to_yolo.py --input C:/data/voc --output C:/data/dataset

# Control class order (first = ID 0, second = ID 1, ...)
python voc_to_yolo/voc_to_yolo.py --input C:/data/voc --output C:/data/dataset \
    --classes screw bolt "hex bolt"

# With test split
python voc_to_yolo/voc_to_yolo.py --input C:/data/voc --output C:/data/dataset --enable-test
```

---

## Step 1c — Split a Flat YOLO Folder (`flat_yolo_split.py`)

Use this when you already have YOLO-format labels in a **single flat folder** (images and `.txt` files together, not yet split into `train/` / `val/`). The script reads `classes.txt` or `labels.txt`, validates that there are no anonymous or out-of-range class IDs, then copies files into the standard training layout.

### Input layout

```
labels/
  classes.txt       # one class per line (line index = class ID)
  # OR labels.txt   # id<TAB>name per line (detect_images export style)
  image1.jpg
  image1.txt
  image2.jpg
  image2.txt
```

### Basic usage

```bash
python flat_yolo_split/flat_yolo_split.py --input C:/data/labels --output C:/data/dataset
```

### Windows batch file

Copy `flat_yolo_split/_Run_flat_yolo_split_template.bat` to `flat_yolo_split/_Run_flat_yolo_split_personal.bat` in the same folder, set `INPUT_DIR` and `OUTPUT_DIR`, then run it. Personal batch files are gitignored (`*_personal.bat`).

### All arguments

| Argument | Default | Description |
|---|---|---|
| `--input` | required | Flat folder with images, label `.txt` files, and `classes.txt` or `labels.txt` |
| `--output` | required | Output dataset root (auto-versioned if not empty) |
| `--train` | `0.70` | Train split ratio |
| `--val` | `0.20` | Val split ratio when `--enable-test` is on |
| `--seed` | `42` | Random seed for reproducible splits |
| `--enable-test` | off | Create a test split in addition to train and val |

### Validation (hard-fail before export)

- Rejects ambiguous input if both `classes.txt` and `labels.txt` exist
- Rejects anonymous class names (`unknown`, `anonymous`, etc.) and duplicate names
- Rejects bbox lines with invalid class IDs or coordinates outside `[0, 1]`
- Does not write `--output` until validation passes

### Example

```bash
python flat_yolo_split/flat_yolo_split.py ^
    --input  "d:\Nir\Projects\ScrewIdentifier\v3\labels" ^
    --output "d:\Nir\Projects\ScrewIdentifier\v3\yolo_dataset"
```

---

## Step 1d — Remap / Merge One or More YOLO Datasets (`remap_yolo_labels.py`)

Use this when you already have one or more YOLO datasets and want to:
- merge multiple classes into one
- merge only selected classes
- rename one or more classes
- merge multiple datasets into one output dataset

The script treats all inputs as read-only and writes only to a new output folder:
- first input is copied as the base tree
- labels are remapped by final class names
- additional inputs are merged by split (`train`/`val`/`test`)
- filename collisions are preserved by renaming incoming files with source suffixes
- final `data.yaml` is updated with merged `names` and `nc`

### Basic usage
```bash
python remap_yolo_labels/remap_yolo_labels.py --input C:/data/yolo --output C:/data/yolo_v2 --map 0:bolt_a:Bolt --map 0:bolt_b:Bolt
```

### Examples
```bash
# One dataset: merge selected classes
python remap_yolo_labels/remap_yolo_labels.py --input C:/data/yolo --output C:/data/yolo_merged \
    --map 0:bolt_a:Bolt --map 0:bolt_b:Bolt

# Merge two datasets and align names to shared IDs
python remap_yolo_labels/remap_yolo_labels.py \
    --input C:/data/yolo_a \
    --input C:/data/yolo_b \
    --output C:/data/yolo_merged \
    --map 0:bolt_a:Bolt \
    --map 0:bolt_b:Bolt \
    --map 1:rusty_screw:Screw

# Merge three datasets
python remap_yolo_labels/remap_yolo_labels.py \
    --input C:/data/a \
    --input C:/data/b \
    --input C:/data/c \
    --output C:/data/merged \
    --map 2:vague:Screw

# Merge many datasets (10 shown; repeat --input as needed)
python remap_yolo_labels/remap_yolo_labels.py \
    --input C:/data/d1 --input C:/data/d2 --input C:/data/d3 --input C:/data/d4 --input C:/data/d5 \
    --input C:/data/d6 --input C:/data/d7 --input C:/data/d8 --input C:/data/d9 --input C:/data/d10 \
    --output C:/data/merged_many \
    --map 0:bolt_a:Bolt --map 9:rusty_screw:Screw
```

`--input` is repeatable and dynamic (no script-level hard limit): pass as many `--input` flags as needed.
`--map INDEX:OLD:NEW` uses zero-based input order (`first --input = 0`, `second = 1`, ..., `tenth = 9`).

### Windows batch helper

Use `remap_yolo_labels/_Run_remap_yolo_labels_template.bat`:
- copy it to `remap_yolo_labels/_Run_remap_yolo_labels_personal.bat` (same folder)
- edit `INPUT_DATASET_1` (required), optional `INPUT_DATASET_2` through `INPUT_DATASET_10`, `OUTPUT_DATASET`, and `MAP_ARGS`
- run your personal copy

`*_personal.bat` is ignored by git in this repo.

---

## Recommended Training Prep — Tile a YOLO Dataset (`tile_yolo_dataset.py`)

For small objects in high-resolution images, tile the completed YOLO dataset before training. This creates a new dataset with overlapping image windows and clipped YOLO labels; the source dataset is not modified.

```bash
python tile_yolo_dataset/tile_yolo_dataset.py \
    --input C:/data/dataset \
    --output C:/data/dataset_tiled
```

Defaults use 1024-pixel square tiles with 20% overlap. A box that gets clipped to a tile edge is kept only if at least 20% of its original (pre-clip) area remains; smaller remainders are dropped. Empty tiles (no surviving boxes) are capped at no more than 10% of each split's **total output tile count** — not 10% of the labelled-tile count, which works out to roughly 11% of labelled tiles. Output keeps the normal YOLO `train` / `val` (and optional `test`) structure and writes a new `data.yaml`.

| Argument | Default | Description |
|---|---|---|
| `--input`, `-i` | required | YOLO dataset root or its `data.yaml` |
| `--output`, `-o` | required | Output tiled dataset directory (auto-versioned if non-empty) |
| `--imgsz` | `1024` | Square tile size in pixels |
| `--overlap` | `0.2` | Tile overlap fraction |
| `--empty-frac` | `0.10` | Maximum output fraction of empty tiles |
| `--seed` | `42` | Empty-tile sampling seed |
| `--manifest` | off | Write `tiles_manifest.json` with tile provenance |

Train with the generated `C:/data/dataset_tiled/data.yaml`.

---

## Step 2 — Train (`train_detector.py`)

### Basic usage
```bash
python train_detector/train_detector.py --input /path/to/data.yaml --name my_detector_v1
```

### Windows paths work as-is
```bash
python train_detector/train_detector.py --input c:\Users\Me\Dataset\data.yaml --name my_detector_v1
```

### Override any auto-calculated value
```bash
python train_detector/train_detector.py --input c:\Users\Me\Dataset\data.yaml --name my_detector_v1 --model yolo26x.pt --batch 8
```

### Resume a crashed or interrupted run
```bash
# Resume specific run by name
python train_detector/train_detector.py --resume --name my_detector_v1

# Resume most recent run automatically
python train_detector/train_detector.py --resume
```

### All arguments

| Argument | Default | Description |
|---|---|---|
| `--input` | required | Path to `data.yaml` file |
| `--name` | `detector_v1` | Name for this training run |
| `--resume` | off | Resume from last checkpoint |
| `--model` | auto | Override model: `yolo26m/l/x.pt` |
| `--imgsz` | auto | Override image size in pixels |
| `--batch` | auto | Override batch size |
| `--workers` | auto | Override dataloader worker count |
| `--epochs` | `600` | Max training epochs |
| `--patience` | `100` | Early stopping patience |
| `--device` | `0` | `0` for GPU, `cpu` for CPU |
| `--probe-vram` | off | Measure or refresh local VRAM estimates; exits when `--input` is omitted |
| `--fix-label-structure` | off | Repair a segment dataset that mixes a few bbox-only labels into polygons in place (`.label_repair_backup/` + `.label_repair_manifest.json`); refused unless `segment` dominates every split at ≥95% |

### Mixed annotation types

Ultralytics reads YOLO label files **line-by-line, per-annotation**, so a `Segmentation`-configured run (e.g. a `-seg` checkpoint) that hits a bbox-only line parses it as `class + 4 scalars` in a segment tensor and **silently drops every polygon in the dataset**. `train_detector.py` therefore refuses mixed-structure datasets outright at validation rather than train a broken segmenter.

When a dataset is segment-dominant but a small bbox-only minority slipped in (common with Roboflow exports that mix rectangle and polygon tools), `--fix-label-structure` expands each 5-field bbox line into its exact 4-corner polygon ring in place:

- Originals are preserved under `<dataset>/.label_repair_backup/` (the first original wins on repeat runs) and every change is recorded in `<dataset>/.label_repair_manifest.json`.
- The repair is refused unless `segment` is both ≥95% of all label lines **and** a strict majority within every non-empty split; any unparseable line, foreign same-named artifact, or a `.bak` that no longer matches its recorded hash aborts the run and restores every touched file.
- Repaired runs delete the stale `<split>/labels.cache` so Ultralytics re-indexes the new polygons, then re-scan strictly (post-repair the dataset must be 100% segment) before training.
- Coordinates use exact decimal math (no binary-float drift) and shortest round-trip formatting, so untouched values keep their original precision.

**Repair the source before tiling:** run `--fix-label-structure` on the original dataset, then run `tile_yolo_dataset.py` on the repaired output — tiling clips polygons correctly, but only once every label line is already a polygon.

### Auto-configuration — quality-first policy

The script detects your hardware and dataset, then targets **maximum achievable quality/accuracy** — speed is not prioritized. Every selection stays **probe-verified safe**: dynamic 1-epoch probe runs on a dataset subset measure real GPU/CPU memory, and configs only ship when a probe (or the runtime memory guard) says they fit.

- **Model:** the largest task model is tried first for every dataset size — GPU runs attempt `yolo26x` → `l` → `m` → `n` (task-correct `-seg`/`-pose` suffixes); CPU runs attempt `l` → `m` → `n` (x excluded on CPU as a practicality cap). Dataset size no longer picks the model; dynamic probes decide fit. Smaller models are probe-failure fallbacks, not defaults.
- **imgsz:** native-capped — the largest standard size that does not upscale beyond the dataset's native resolution, then lower standards as fallbacks.
- **Augmentation:** the heaviest combo is attempted first — `mosaic 1.0, copy_paste 0.3, mixup 0.15, multi_scale 0.5` — and the run steps down an augmentation ladder only on measured-unsafe probes. Fixed fresh-run constants: `degrees=180`, `flipud=0.5`, `close_mosaic=60`, `cos_lr=True`, `nbs=64`.
- **Batch ladder:** `8 → 4` (effective batch stays 64 via `nbs` gradient accumulation). Batch is only considered at imgsz/model combos whose probes pass; `MIN_AUTO_BATCH=4` floor as before.
- **Adaptive supervisor:** during training the runtime guard downgrades batch/aug/imgsz/workers on pressure, and a stable run **steps back up** the ladder one rung at a time, each step probe-verified before adoption.
- **Resume:** the checkpoint's own augmentation (including `multi_scale`) is kept when a probe on the current hardware says it is safe; it is reset to `0.0` only when the probe says unsafe or cannot verify (no dataset yaml, CPU device, `--no-memory-probe`).

**Expect slower runs than the old speed-balanced defaults — this is by design.** Safety guarantees are unchanged: `--memory-profile safe` remains the default, guards and adaptive recovery stay active, and nothing trains unverified unless you explicitly pass `--force-unsafe` or `--no-memory-probe`.

**Cache:** measured-safe selections are cached at `train_detector/weights/memory_profiles.json` (schema **v3**, keyed by intent `quality-v1`). The first run after this policy change discards the old v2 cache and recalibrates fully (one round of probe runs).

**Key constraint:** imgsz is only raised when batch size stays ≥ 4. Fresh runs set `nbs=64` so YOLO gradient accumulation keeps the effective batch at 64 even when per-step batch is smaller. The `[Auto Config]` block prints `effective batch` when batch < 64, plus a `[Selection policy]` line stating the quality-first behavior.

Worker count is calculated from CPU cores and RAM, capped at 8 for Windows stability.

### VRAM probe

On CUDA, training automatically measures any missing model VRAM estimates before calculating a batch size. The estimates are cached locally at `train_detector/weights/vram_estimates.json` (gitignored); when probing is unavailable or fails, the toolkit uses built-in FLOPs-scaled estimates instead.

Refresh the local measurements without starting a training run:

```bash
python train_detector/train_detector.py --probe-vram
```

To probe only a selected model, supply `--model`:

```bash
python train_detector/train_detector.py --probe-vram --model yolo26l.pt
```

You can always override any single value while letting the rest auto-calculate:
```bash
python train_detector/train_detector.py --input data.yaml --name run1 --model yolo26x.pt
```

### Output

Training results are saved to `train_detector/runs/detect/<name>/`:

```
train_detector/runs/detect/my_detector_v1/
├── weights/
│   ├── best.pt      ← best model weights (use this)
│   └── last.pt      ← last epoch checkpoint (used for resume)
├── results.png
├── confusion_matrix.png
└── args.yaml
```

After training, the script automatically runs validation and reports:
```
[Metrics]
  mAP@0.50       : 0.8731  (target: > 0.80)
  mAP@0.50:0.95  : 0.6214

  PASS -- model meets target accuracy (mAP@0.50 >= 0.80)
```

---

## Step 3 — Detect (`detect_images.py`)

Run a trained model over a folder of images:

```bash
python detect_images/detect_images.py \
    --images /path/to/images \
    --model  train_detector/runs/detect/my_detector_v1/weights/best.pt
```

### With custom confidence threshold
```bash
python detect_images/detect_images.py --images /path/to/images --model best.pt --conf 0.5
```

### Export detection data as JSON
```bash
python detect_images/detect_images.py --images /path/to/images --model best.pt --export-json
```

Saves `detect_images/detections/<input_folder_name>/<name>.json` for images with detections, plus `labels.txt` mapping class IDs to names.

### JSON-only mode (no annotated image files)
```bash
python detect_images/detect_images.py --images /path/to/images --model best.pt --no-export-annotated-images
```

### Subdirectory scanning (default ON)
By default, `detect_images/detect_images.py` walks all subdirectories of `--images` and processes every supported image it finds.

Outputs are kept flat under `detect_images/detections/<input_folder_name>/`. Files inside subfolders are renamed by prefixing the relative subpath joined with underscores so basenames don't collide. Examples:

- `images/foo.jpg`         -> `detect_images/detections/images/foo.jpg` + `foo.json`
- `images/sub/foo.jpg`     -> `detect_images/detections/images/sub_foo.jpg` + `sub_foo.json`
- `images/sub/a/foo.jpg`   -> `detect_images/detections/images/sub_a_foo.jpg` + `sub_a_foo.json`

Disable with `--no-recursive` to scan only the top-level of `--images`:

```bash
python detect_images/detect_images.py --images /path/to/images --model best.pt --no-recursive
```

### Tiled inference (default ON)

Images are split into overlapping tiles before inference by default. Images that already fit in one tile are processed once. Tile detections are translated back to full-image coordinates and merged with class-wise NMS, which improves detection of small objects in high-resolution images.

```bash
# Default tiled inference: 1024px tiles, 20% overlap, NMS IoU 0.5
python detect_images/detect_images.py --images /path/to/images --model best.pt

# Tune tiling for your image scale
python detect_images/detect_images.py --images /path/to/images --model best.pt \
    --tile-imgsz 1280 --tile-overlap 0.2 --tile-iou 0.5

# Restore legacy whole-image inference
python detect_images/detect_images.py --images /path/to/images --model best.pt --no-tiles
```

### Performance: hardware-aware parallel inference (default ON)

The script measures host hardware at startup and sizes the whole pipeline from
that profile: GPU VRAM, CPU cores and host RAM drive the decode pool, the
inference batch, the post-processing workers, the precision and the device.
A `[Hardware]` block is printed before the `[Config]` block on every run.

1. **Batched GPU inference** (`--batch`) — a single `model([item1, item2, ...])`
   call per chunk. In tiled mode the items are tiles accumulated **across
   source images**; in non-tiled mode they are whole images. Tiles from
   different images are never mixed into one result: per-image merge/NMS still
   runs once per image after its last tile lands.
2. **Threaded post-inference I/O** (`--workers`) — a `ThreadPoolExecutor` runs
   sidecar JSON export, ExifTool metadata extract / copy, and Pillow image
   saves in parallel while the main thread reads the next batch and runs the
   next inference call.
3. **Decode prefetch** (`--decode-workers`) — a small thread pool decodes
   images (`cv2.imread`) and pushes them onto a bounded queue ahead of the
   main thread, overlapping disk reads with inference.

Auto-sizing rules:

- `--workers auto` -> `min(8, cpu_cores - decode_workers + 1)`. Override with
  an integer (e.g. `--workers 4`).
- `--batch auto` -> ladder-clamped from a VRAM budget
  `min(free_now * 0.9, total * profile.gpu)` divided by a measured per-item
  inference cost. Override with an integer (e.g. `--batch 16`).
- `--decode-workers auto` -> `clamp(cpu_count // 6, 1, 6)`.
- `--device auto` -> `cuda:0` when CUDA is available, else `cpu`.
- `--half auto` -> `on` for CUDA, `off` for CPU. `--half off` restores FP32.
- `--memory-profile` -> `balanced` (GPU 0.75, CPU 0.45 of totals); `safe`
  (0.68/0.38) and `aggressive` (0.85/0.55) are selectable. `--vram-frac`
  overrides the GPU fraction.

Safety ceilings:

| Resource | Ceiling | Guard |
|---|---|---|
| VRAM | `min(free_now * 0.9, total * profile.gpu)` | warmup probe sizes the first batch; mid-run free-VRAM check downshifts; OOM halving ladder (max 4); hard fail at batch 1 |
| Host RAM | `min(total * profile.cpu, available * 0.5)` | byte budget acquired before decode, released after post-processing |
| CPU | `decode_workers + workers <= cpu_count + 2` | explicit caps; `cv2.setNumThreads(1)` when decode workers > 1; `torch.set_num_threads()` only on CPU |

To reproduce the previous fully sequential, single-image-at-a-time behaviour
byte-for-byte:

```bash
python detect_images/detect_images.py --images /path/to/images --model best.pt --workers 1 --batch 1 --half off
```

Notes:

- Inference itself stays single-threaded on one GPU; threads accelerate the
  surrounding I/O work, which is dominated by ExifTool subprocess startup when
  annotated images are exported.
- Per-image progress lines may print slightly out of order across worker
  threads, but the `[i/N]` index and final summary remain accurate.
- Memory cost grows with batch size: roughly `batch * imgsz^2 * 3 * 4 bytes`
  on host plus a similar amount of GPU RAM.
- `--half auto` may shift boxes by small amounts vs FP32; `--half off`
  restores exact numerics.
- `--only-detections` was a no-op flag and is now rejected by argparse.

JSON format:
```json
{
  "image": {
    "file_name": "sample.jpg",
    "source_path": "D:/data/sample.jpg",
    "width": 1920,
    "height": 1080
  },
  "metadata": {
    "EXIF:DateTimeOriginal": "2026:04:24 15:26:16"
  },
  "detections": [
    {
      "class_id": 0,
      "class_name": "screw",
      "confidence": 0.95,
      "pixel": {"x1": 120, "y1": 80, "x2": 340, "y2": 520},
      "yolo": {"cx": 0.273438, "cy": 0.317188, "bw": 0.21875, "bh": 0.4375}
    }
  ]
}
```

- `metadata` — full ExifTool metadata when available (`-j -a -u -G1` output), with Pillow EXIF fallback. When ExifTool is not used, `ortho_tag_sidecar.merge_pillow_gps_exif_into_metadata` adds `GPS:*` and basic `ExifIFD:*` keys from Pillow so Ortho-Tag can read geolocation and focal length; DJI XMP (`XMP-drone-dji:*` yaw, pitch, relative altitude) still requires ExifTool on the image.
- `detections[*].pixel` — absolute (x1, y1, x2, y2) in image coordinates
- `detections[*].yolo` — normalized (cx, cy, bw, bh) / (W, H) for YOLO training/export

### Ortho-Tag / B3DM (detection sidecars)

Ortho-Tag B3DM (`detect_to_3d`) expects `metadata` keys in **ExifTool `-G1` form** (for example `GPS:GPSLatitude`, `XMP-drone-dji:FlightYawDegree`, `ExifIFD:FocalLength`). Export with ExifTool available (PATH, `--exiftool`, or bundled `./exiftool/`) so sidecar JSON matches that layout.

After export, confirm georeference fields on a sample file:

```bash
python ortho_tag_sidecar/ortho_tag_sidecar.py path/to/detect_images/detections/<folder>/some_image.json
```

Exit code `0` means latitude and longitude are present in the expected keys; the script also lists recommended fields (altitude, heading, pitch, focal length) that often come from DJI XMP via ExifTool.

Run detection with strict verification (fails the run if any written sidecar lacks lat/lon):

```bash
python detect_images/detect_images.py --images ... --model ... --verify-b3dm
```

### All arguments

| Argument | Default | Description |
|---|---|---|
| `--images` | required | Path to folder of input images |
| `--model` | required | Path to trained `.pt` model file |
| `--conf` | `0.25` | Minimum confidence threshold (0.0–1.0) |
| `--export-json` | `True` | Export detection JSON + labels.txt in detect_images/detections/ |
| `--export-annotated-images` | `True` | Save annotated detection images |
| `--no-export-annotated-images` | `False` | Disable annotated image save and write JSON-only outputs |
| `--exiftool` | auto | Optional path to ExifTool executable for full metadata transfer |
| `--allow-missing-exiftool` | `False` | Allow saving with Pillow-only metadata copy when ExifTool is unavailable |
| `--verify-b3dm` | `False` | After each JSON sidecar write, verify Ortho-Tag georeference keys; exit `1` if any file fails |
| `--recursive` | `True` | Walk subdirectories of `--images` for input files; outputs stay flat with relative-subpath prefixes |
| `--no-recursive` | `False` | Only scan the top-level of `--images` |
| `--workers` | `auto` | Worker threads for post-inference I/O (JSON + ExifTool + image save). `auto` = hardware-aware; pass an integer or `1` to disable threading |
| `--batch` | `auto` | Inference items per `model()` call (tiles across images in tiled mode, whole images otherwise). `auto` = VRAM-budget ladder-clamped; pass an integer or `1` to disable batching |
| `--decode-workers` | `auto` | Threads for `cv2.imread` + tile prep. `auto` = `clamp(cpu_count // 6, 1, 6)` |
| `--device` | `auto` | Inference device: `auto` \| `cpu` \| `0`. `auto` picks `cuda:0` when CUDA is available |
| `--half` | `auto` | FP16 precision: `auto` \| `on` \| `off`. `auto` = `on` for CUDA, `off` for CPU. `--half off` restores exact FP32 numerics |
| `--memory-profile` | `balanced` | Memory ceiling profile: `safe` (0.68/0.38) \| `balanced` (0.75/0.45) \| `aggressive` (0.85/0.55) of total GPU/host RAM |
| `--vram-frac` | unset | Advanced: override the GPU memory fraction used by the memory profile |
| `--exiftool-stay-open` | `False` | Run ExifTool as persistent `-stay_open` daemons (`min(4, workers)`) instead of spawning one process per image |
| `--tiles` | `True` | Tile images before inference; tile detections are merged with class-wise NMS |
| `--no-tiles` | `False` | Disable tiled inference and process each whole image |
| `--tile-imgsz` | `1024` | Tile size in pixels |
| `--tile-overlap` | `0.2` | Overlap fraction between adjacent tiles |
| `--tile-iou` | `0.5` | NMS IoU threshold for merging tiled detections |

Annotated images are saved to `detect_images/detections/<input_folder_name>/` when enabled — originals are never modified.
By default, full metadata transfer requires ExifTool when an output image is actually being saved.
ExifTool lookup order is: explicit `--exiftool`, then `PATH`, then repo-local `./exiftool/` (`exiftool.exe` / `exiftool`), then bundled Perl runtime (`./exiftool/exiftool_files/perl.exe` + `exiftool.pl`).
`exiftool(-k).exe` is intentionally ignored because it is interactive (`-- press ENTER --`) and can block batch runs.
If ExifTool is missing at save time, the script exits with guidance to download it from [https://exiftool.org/](https://exiftool.org/) into `./exiftool/` or pass `--exiftool`.
Use `--allow-missing-exiftool` only if you accept limited Pillow-only metadata copy.

Supported formats: `.jpg` `.jpeg` `.png` `.bmp` `.tiff` `.tif` `.webp`

### Windows batch helper (detect)

`detect_images/_Run_detect_images_template.bat` is a fill-in template that wraps the
above flags. Copy it to `detect_images/_Run_detect_images_personal.bat` (gitignored)
and edit the `set` lines:

| Variable | Default | Flag |
|---|---|---|
| `imagesDirPath` | `_YOUR_IMAGES_DIR_HERE_` | `--images` |
| `modelPath` | `_YOUR_MODEL_PATH_HERE_` | `--model` |
| `confidence` | `0.3` | `--conf` |
| `exportAnnotatedImages` | `true` | `--no-export-annotated-images` when `false` |
| `recursive` | `true` | `--no-recursive` when `false` |
| `workers` | `auto` | `--workers` |
| `batch` | `auto` | `--batch` |
| `decodeWorkers` | `auto` | `--decode-workers` |
| `device` | `auto` | `--device` |
| `half` | `auto` | `--half` |
| `memoryProfile` | `balanced` | `--memory-profile` |
| `useTiles` | `true` | `--no-tiles` when `false` |
| `exiftoolStayOpen` | `false` | `--exiftool-stay-open` when `true` |

---

## Step 4 — ExifTool helper batch (Windows)

Use `exiftool/_Run_exiftool.bat` to inspect metadata for a single image and save full output to:
`./exiftool/outputs/`.

The helper expects ExifTool files in `./exiftool/`. If that directory is missing, the batch script exits with instructions to download ExifTool from [https://exiftool.org/](https://exiftool.org/) and place it there.

---

## Related — X-AnyLabel-toolkit

The PT → ONNX + X-AnyLabeling export script lives in the separate [X-AnyLabel-toolkit](https://github.com/NiRo-2/X-AnyLabel-toolkit) repo (`scripts/yolov8_pt_to_xanylabeling_onnx/`). Use it after training when you want **[X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling)** auto-labelling with your own detector.

```bash
# From a clone of X-AnyLabel-toolkit
python scripts/yolov8_pt_to_xanylabeling_onnx/yolov8_pt_to_xanylabeling_onnx.py \
    /path/to/yolov-toolkit/train_detector/runs/detect/my_detector_v1/weights/best.pt
```

Default output: `scripts/yolov8_pt_to_xanylabeling_onnx/<stem>_xanylabeling/`. See that repo for flags, batch templates, and [Load Custom Model](https://github.com/CVHub520/X-AnyLabeling/blob/main/docs/en/custom_model.md) usage.

---

## Dataset Format

All scripts produce and consume datasets in **YOLO format** with a `data.yaml` file:

```yaml
train: ../train/images
val: ../val/images

nc: 1
names: ['my_object']
```

Both `vlm_yolo_prep.py` and `voc_to_yolo.py` generate this file automatically. [Roboflow](https://roboflow.com) also exports directly in this format and is recommended for manual labelling and dataset management.

---

## Tips

- **Poor detections from the VLM?** Lower `--confidence` to `0.5` and review the `preview/` folder. The VLM may need a more descriptive object name.
- **VOC class names don't match what you want?** Use `--classes` to rename or reorder them when converting.
- **mAP@0.50 < 0.80 after training?** Add more labelled images — dataset size is the biggest factor in accuracy.
- **OOM crash during training?** Add `--batch 8` or `--batch 4` to override the auto-calculated batch size. With `nbs=64`, batch 4 still yields an effective batch of 64 via gradient accumulation.
- **PC crashed mid-training?** Run with `--resume` — picks up from the last saved epoch automatically.
- **Specific variants not detected?** (e.g. rusted bolts, painted-over screws) Collect photos of those specific types and add them to the dataset.
- **Roboflow export mixes rectangles and polygons?** `train_detector.py` refuses mixed-structure labels (Ultralytics would silently drop every segment). Use `--fix-label-structure` to normalise the bbox-only lines to polygons first, then tile.

---

## License

MIT — use freely, no restrictions.
