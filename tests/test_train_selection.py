"""Pure-function tests for the quality-first auto selection policy.

No GPU, no training, no probes: these lock the selection ladder semantics
(model ladder, augmentation ladder, config/selection signatures, cache schema).
"""

import pytest

from train_detector.train_detector import model_candidates


EXPECTED_GPU_CHAIN = [
    "yolo26x.pt",
    "yolo26l.pt",
    "yolo26m.pt",
    "yolo26n.pt",
]


@pytest.mark.parametrize("image_count,vram_gb", [
    (10, 8.0),        # tiny dataset on a modest GPU still starts at x
    (1000, 12.0),     # mid-size dataset: no 'm'/'l' size up-shift anymore
    (100000, 8.0),    # big dataset under 16 GB VRAM no longer gates x out
])
def test_gpu_model_candidates_are_largest_first_for_every_dataset_size(image_count, vram_gb):
    assert model_candidates(image_count, vram_gb, for_cpu=False) == EXPECTED_GPU_CHAIN


@pytest.mark.parametrize("task,suffix", [
    ("detect", ""),
    ("segment", "-seg"),
    ("pose", "-pose"),
])
def test_gpu_model_candidates_carry_task_suffixes(task, suffix):
    assert model_candidates(100000, 24.0, for_cpu=False, task=task) == [
        f"yolo26x{suffix}.pt",
        f"yolo26l{suffix}.pt",
        f"yolo26m{suffix}.pt",
        f"yolo26n{suffix}.pt",
    ]


@pytest.mark.parametrize("image_count", [10, 1000, 100000])
def test_cpu_model_candidates_start_at_l_and_exclude_x(image_count):
    assert model_candidates(image_count, None, for_cpu=True) == [
        "yolo26l.pt",
        "yolo26m.pt",
        "yolo26n.pt",
    ]
