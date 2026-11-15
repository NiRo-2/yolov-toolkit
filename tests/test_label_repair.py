import argparse
import hashlib
import json
from pathlib import Path

import pytest

from train_detector.train_detector import (
    LABEL_REPAIR_BACKUP_DIRNAME,
    LABEL_REPAIR_MANIFEST_NAME,
    _fmt_coord,
    analyze_label_lines,
    bbox_line_to_polygon,
    derive_task_with_repair,
    repair_label_structures,
    repairability,
    scan_yolo_labels,
)


class FakeLogger:
    def __init__(self):
        self.lines = []

    def log(self, msg):
        self.lines.append(msg)


def make_args(fix=False, yes=False):
    return argparse.Namespace(fix_label_structure=fix, yes=yes)


# A valid 4-vertex segment line (9 tokens) used to build segment-dominant fixtures.
SEG_LINE = "0 0.1 0.1 0.3 0.1 0.3 0.3 0.1 0.3"


def seg_file(count):
    return "\n".join(SEG_LINE for _ in range(count)) + "\n"


def make_ds(tmp, mapping):
    """mapping: role -> {filename: content}. Returns (backup_root, role_pairs)."""
    role_pairs = []
    for role, files in mapping.items():
        ldir = tmp / role / "labels"
        ldir.mkdir(parents=True, exist_ok=True)
        for name, content in files.items():
            # Write exact bytes so newlines are deterministic (no Windows CRLF translation).
            (ldir / name).write_bytes(content.encode("utf-8"))
        role_pairs.append((role, ldir))
    return tmp, role_pairs


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


# -- bbox -> polygon math ------------------------------------------------------------

def test_bbox_to_polygon_exact_coordinates():
    res = bbox_line_to_polygon(0, 0.49296875, 0.5390625, 0.36640625, 0.921875)
    assert res["coords"] == [
        0.309765625, 0.078125, 0.676171875, 0.078125,
        0.676171875, 1.0, 0.309765625, 1.0,
    ]
    # y2 is exactly 1.0, so nothing was clamped.
    assert res["clamped"] is False
    assert res["zero_area"] is False


def test_bbox_to_polygon_clamps_to_range():
    res = bbox_line_to_polygon(0, 0.9, 0.5, 0.4, 0.2)  # x2 = 1.1 -> clamp
    assert res["clamped"] is True
    assert res["coords"][2] == 1.0  # x2 clamped
    assert res["coords"][4] == 1.0  # x2 clamped (second occurrence)
    assert res["zero_area"] is False


def test_bbox_to_polygon_zero_area_drop():
    # Zero-width box: x1 == x2 -> should be dropped.
    res = bbox_line_to_polygon(0, 0.5, 0.5, 0.0, 0.2)
    assert res["zero_area"] is True


def test_polygon_vertex_order_is_segment_parseable():
    # After conversion a 5-field bbox becomes a 9-token line that re-reads as segment.
    from train_detector.train_detector import classify_label_tokens
    vals = [float(v) for v in bbox_line_to_polygon(0, 0.5, 0.5, 0.2, 0.2)["coords"]]
    task, _ = classify_label_tokens(vals)
    assert task == "segment"


# -- formatting keeps round-trip precision ------------------------------------------

def test_fmt_coord_round_trip_precision():
    assert _fmt_coord(0.5566254640625) == "0.5566254640625"  # 13-decimal export value survives
    assert _fmt_coord(1.0) == "1"
    assert _fmt_coord(0.0) == "0"
    assert _fmt_coord(0.309765625) == "0.309765625"


def test_fmt_coord_sci_notation_fallback():
    assert _fmt_coord(1e-6) == "0.000001"
    assert _fmt_coord(1e-5) == "0.00001"
    # repr round-trip for a float that does not round cleanly stays intact.
    v = 0.1 + 0.2
    s = _fmt_coord(v)
    assert float(s) == v


# -- guards -------------------------------------------------------------------------

def test_global_ratio_guard_rejects_90pct(tmp_path):
    _root, pairs = make_ds(tmp_path, {"train": {"a.txt": "\n".join([SEG_LINE] * 9 + ["0 0.5 0.5 0.2 0.2"]) + "\n"}})
    analysis = analyze_label_lines(pairs, tmp_path)
    ok, reason = repairability(analysis)
    assert ok is False
    assert "90.00%" in reason and "95" in reason


def test_split_majority_guard_rejects_bbox_test_split(tmp_path):
    root, pairs = make_ds(tmp_path, {
        "train": {"tr.txt": seg_file(1000)},
        "val": {"v.txt": seg_file(1000)},
        "test": {"t.txt": "\n".join(["0 0.5 0.5 0.2 0.2"] * 100) + "\n"},
    })
    analysis = analyze_label_lines(pairs, root)
    # Global ratio 2000/2100 = 95.2% passes; the all-bbox test split must still be refused.
    assert analysis["dominant_ratio"] >= 0.95
    assert analysis["split_majority_ok"] is False
    ok, reason = repairability(analysis)
    assert ok is False
    assert "test" in reason


def test_refuse_polygon_minority_in_detect_dataset(tmp_path):
    root, pairs = make_ds(tmp_path, {"train": {
        "bboxes.txt": "\n".join(["0 0.5 0.5 0.2 0.2"] * 10) + "\n",
        "poly.txt": SEG_LINE + "\n",
    }})
    analysis = analyze_label_lines(pairs, root)
    ok, reason = repairability(analysis)
    assert ok is False
    assert "segment" in reason


def test_refuse_pose_segment_mix(tmp_path):
    pose_line = "0 0.1 0.1 0.2 0.2 0.3 2 1"  # 8 fields -> pose
    root, pairs = make_ds(tmp_path, {"train": {
        "seg.txt": seg_file(100),
        "mix.txt": SEG_LINE + "\n" + pose_line + "\n",
    }})
    analysis = analyze_label_lines(pairs, root)
    ok, reason = repairability(analysis)
    assert ok is False
    assert "pose" in reason


def test_refuse_non_numeric_line(tmp_path):
    root, pairs = make_ds(tmp_path, {"train": {
        "seg.txt": seg_file(100),
        "bad.txt": SEG_LINE + "\n" + "0 hello world\n",
    }})
    analysis = analyze_label_lines(pairs, root)
    ok, reason = repairability(analysis)
    assert ok is False
    assert "unparseable" in reason


def test_refuse_invalid_class_id(tmp_path):
    root, pairs = make_ds(tmp_path, {"train": {
        "seg.txt": seg_file(100),
        "bad.txt": SEG_LINE + "\n" + "-1 0.5 0.5 0.2 0.2\n",
    }})
    analysis = analyze_label_lines(pairs, root)
    ok, reason = repairability(analysis)
    assert ok is False
    assert "unparseable" in reason


def test_refuse_differing_keypoint_counts(tmp_path):
    # Two pose files with different keypoint counts fail the strict scan and are not repairable.
    pose1 = "0 0.1 0.1 0.2 0.2 0.3 2 1"          # 8 fields -> pose kpt 1
    pose2 = "0 0.1 0.1 0.2 0.2 0.3 2 1 0.4 2 1"  # 11 fields w/ out-of-range -> pose kpt 2
    root, pairs = make_ds(tmp_path, {"train": {"p1.txt": pose1 + "\n", "p2.txt": pose2 + "\n"}})
    with pytest.raises(SystemExit):
        derive_task_with_repair(pairs, root, make_args(fix=True), FakeLogger())


# -- acceptance fixtures from the real dataset --------------------------------------

CL478 = "CL33_png.rf.478ec90c79bdda7f656146d2b6d3085c.txt"
CL5BC = "CL33_png.rf.5bc745465f9f4056a759b4796ac9c1fe.txt"
Z078 = "000078_jpg.rf.e61748ecea87cd67d486f3acec55a1b7.txt"
Z078_LINE1 = "0 0.5566254640625 0 0.7394366203125 0.1875 0.992957746875 0.34375 1 0.344039359375 1 0 0.5566254640625 0"
Z078_LINE2 = "0 0.2921875 0.978125 0.0171875 0.0046875"

EXP_478 = "0 0.309765625 0.078125 0.676171875 0.078125 0.676171875 1 0.309765625 1"
EXP_5BC = "0 0.309765625 0 0.676171875 0 0.676171875 1 0.309765625 1"
EXP_078 = "0 0.28359375 0.97578125 0.30078125 0.97578125 0.30078125 0.98046875 0.28359375 0.98046875"


def build_real_mixed_dataset(tmp):
    return make_ds(tmp, {
        "train": {
            CL478: "0 0.49296875 0.5390625 0.36640625 0.921875\n",
            CL5BC: "0 0.49296875 0.5 0.36640625 1\n",
            Z078: Z078_LINE1 + "\n" + Z078_LINE2 + "\n",
            "seg_many.txt": seg_file(1000),
        },
        "val": {"v.txt": seg_file(500)},
        "test": {"t.txt": seg_file(500)},
    })


def test_real_mixed_dataset_repairs_to_three_polygons(tmp_path):
    root, pairs = build_real_mixed_dataset(tmp_path)
    task, _k, _mc, had = derive_task_with_repair(pairs, root, make_args(fix=True), FakeLogger())
    assert task == "segment"
    assert had is True

    assert (root / "train" / "labels" / CL478).read_text(encoding="utf-8") == EXP_478 + "\n"
    assert (root / "train" / "labels" / CL5BC).read_text(encoding="utf-8") == EXP_5BC + "\n"
    assert (root / "train" / "labels" / Z078).read_text(encoding="utf-8") == (
        Z078_LINE1 + "\n" + EXP_078 + "\n"
    )

    manifest = json.loads((root / LABEL_REPAIR_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["format"] == "label_repair"
    assert manifest["dominant_task"] == "segment"
    assert manifest["files_changed"] == 3
    assert manifest["lines_converted"] == 3
    assert manifest["clamped"] == 0
    assert manifest["dropped"] == 0
    assert len(manifest["entries"]) == 3
    assert all(e["split"] == "train" for e in manifest["entries"])

    # Backups preserved, originals reachable via .bak.
    assert sha_bytes((root / LABEL_REPAIR_BACKUP_DIRNAME / "train" / "labels" / (CL478 + ".bak")).read_bytes()) \
        == sha_bytes(b"0 0.49296875 0.5390625 0.36640625 0.921875\n")


def test_intra_file_mixed_bbox_converted_polygon_unchanged(tmp_path):
    root, pairs = build_real_mixed_dataset(tmp_path)
    derive_task_with_repair(pairs, root, make_args(fix=True), FakeLogger())
    content = (root / "train" / "labels" / Z078).read_text(encoding="utf-8")
    lines = content.splitlines()
    # The existing polygon line must remain byte-identical.
    assert lines[0] == Z078_LINE1
    # Only the bbox line was converted.
    assert lines[1] == EXP_078


# -- repairs write to disk / idempotency / artifacts --------------------------------

def test_clean_dataset_writes_nothing(tmp_path):
    root, pairs = make_ds(tmp_path, {
        "train": {"a.txt": seg_file(100)},
        "val": {"b.txt": seg_file(50)},
        "test": {"c.txt": seg_file(50)},
    })
    task, _k, _mc, _had = derive_task_with_repair(pairs, root, make_args(), FakeLogger())
    assert task == "segment"
    assert not (root / LABEL_REPAIR_BACKUP_DIRNAME).exists()
    assert not (root / LABEL_REPAIR_MANIFEST_NAME).exists()


def test_repaired_dataset_second_pass_writes_nothing(tmp_path):
    root, pairs = build_real_mixed_dataset(tmp_path)
    derive_task_with_repair(pairs, root, make_args(fix=True), FakeLogger())
    manifest_before = json.loads((root / LABEL_REPAIR_MANIFEST_NAME).read_text(encoding="utf-8"))
    backups_before = sorted((root / LABEL_REPAIR_BACKUP_DIRNAME).rglob("*.bak"))

    # Second pass: labels are now uniform -> clean path, no new artifacts.
    task, _k, _mc, _had = derive_task_with_repair(pairs, root, make_args(fix=True), FakeLogger())
    assert task == "segment"
    manifest_after = json.loads((root / LABEL_REPAIR_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest_after["entries"] == manifest_before["entries"]
    assert sorted((root / LABEL_REPAIR_BACKUP_DIRNAME).rglob("*.bak")) == backups_before


def test_existing_backup_is_never_overwritten(tmp_path):
    root, pairs = build_real_mixed_dataset(tmp_path)
    derive_task_with_repair(pairs, root, make_args(fix=True), FakeLogger())

    bak = root / LABEL_REPAIR_BACKUP_DIRNAME / "train" / "labels" / (CL478 + ".bak")
    pristine = bak.read_bytes()

    # Re-corrupt the repaired file with a fresh bbox mix.
    (root / "train" / "labels" / CL478).write_text("0 0.11 0.11 0.2 0.2\n", encoding="utf-8")

    repair_label_structures(analyze_label_lines(pairs, root), make_args(fix=True), FakeLogger())

    # The backup still holds the pristine first original, not the re-mixed bytes.
    assert bak.read_bytes() == pristine
    repaired = (root / "train" / "labels" / CL478).read_text(encoding="utf-8")
    assert repaired.startswith("0 ") and repaired.count(" ") > 5


def test_edited_backup_mismatch_aborts_with_rollback(tmp_path):
    root, pairs = build_real_mixed_dataset(tmp_path)
    derive_task_with_repair(pairs, root, make_args(fix=True), FakeLogger())

    # Re-corrupt CL478 so a live bbox minority exists again.
    original_text = "0 0.49296875 0.5390625 0.36640625 0.921875\n"
    (root / "train" / "labels" / CL478).write_text(original_text, encoding="utf-8")

    bak = root / LABEL_REPAIR_BACKUP_DIRNAME / "train" / "labels" / (CL478 + ".bak")
    # Poison the backup so it no longer matches the manifest original_sha256.
    bak.write_bytes(b"tampered\n")

    analysis = analyze_label_lines(pairs, root)
    with pytest.raises(SystemExit):
        repair_label_structures(analysis, make_args(fix=True), FakeLogger())

    # Repair refused before writing; the tampered backup was not overwritten.
    assert bak.read_bytes() == b"tampered\n"
    # The label file was never rewritten during the aborted run.
    assert (root / "train" / "labels" / CL478).read_text(encoding="utf-8") == original_text


def test_repair_deletes_adjacent_label_cache(tmp_path):
    root, pairs = build_real_mixed_dataset(tmp_path)
    cache = root / "train" / "labels.cache"
    cache.write_text("stale", encoding="utf-8")
    derive_task_with_repair(pairs, root, make_args(fix=True), FakeLogger())
    assert not cache.exists()


def test_foreign_manifest_refused_nothing_written(tmp_path):
    root, pairs = build_real_mixed_dataset(tmp_path)
    (root / LABEL_REPAIR_MANIFEST_NAME).write_text('{"format": "other"}', encoding="utf-8")
    analysis = analyze_label_lines(pairs, root)
    with pytest.raises(SystemExit):
        repair_label_structures(analysis, make_args(fix=True), FakeLogger())
    # Nothing repaired, no backups.
    assert (root / "train" / "labels" / CL478).read_text(encoding="utf-8").startswith("0 0.49296875")
    assert not (root / LABEL_REPAIR_BACKUP_DIRNAME).exists()


def test_backup_dir_without_manifest_refused(tmp_path):
    root, pairs = build_real_mixed_dataset(tmp_path)
    (root / LABEL_REPAIR_BACKUP_DIRNAME).mkdir(parents=True, exist_ok=True)
    analysis = analyze_label_lines(pairs, root)
    with pytest.raises(SystemExit):
        repair_label_structures(analysis, make_args(fix=True), FakeLogger())
    assert (root / "train" / "labels" / CL478).read_text(encoding="utf-8").startswith("0 0.49296875")


# -- clamped + dropped records on disk ------------------------------------------------

def test_clamped_bbox_repaired_and_recorded(tmp_path):
    root, pairs = make_ds(tmp_path, {
        "train": {
            "seg.txt": seg_file(1000),
            "clamp.txt": "0 0.9 0.5 0.4 0.2\n",
        },
        "val": {"v.txt": seg_file(200)},
    })
    derive_task_with_repair(pairs, root, make_args(fix=True), FakeLogger())
    assert (root / "train" / "labels" / "clamp.txt").read_text(encoding="utf-8") == \
        "0 0.7 0.4 1 0.4 1 0.6 0.7 0.6\n"
    manifest = json.loads((root / LABEL_REPAIR_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["clamped"] == 1
    entry = [e for e in manifest["entries"] if e["relpath"].endswith("clamp.txt")][0]
    assert entry["clamped"] is True


def test_zero_area_bbox_dropped_not_converted(tmp_path):
    root, pairs = make_ds(tmp_path, {
        "train": {
            "seg.txt": seg_file(1000),
            "drop.txt": SEG_LINE + "\n" + "0 0.5 0.5 0.0 0.2\n",  # second line is zero-area bbox
        },
        "val": {"v.txt": seg_file(200)},
    })
    derive_task_with_repair(pairs, root, make_args(fix=True), FakeLogger())
    # The polygon line stays; the zero-area bbox line is dropped.
    assert (root / "train" / "labels" / "drop.txt").read_text(encoding="utf-8") == SEG_LINE + "\n"
    manifest = json.loads((root / LABEL_REPAIR_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["dropped"] == 1
    assert manifest["lines_converted"] == 0


def test_no_minority_no_repair_path(tmp_path):
    # Pure segment dataset: scan is uniform, derive never analyses or repairs.
    root, pairs = make_ds(tmp_path, {"train": {"a.txt": seg_file(10)}})
    _tasks, _ksets, bad, _mc, _n = scan_yolo_labels([d for _r, d in pairs])
    assert bad == []
    task, _k, _mc, _had = derive_task_with_repair(pairs, root, make_args(), FakeLogger())
    assert task == "segment"
    assert not (root / LABEL_REPAIR_MANIFEST_NAME).exists()
