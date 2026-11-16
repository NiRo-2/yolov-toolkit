"""Pytest bootstrap: make sibling packages importable when tests run from anywhere.

The repo uses namespace packages (no __init__.py) and cross-module imports
(e.g. ``from train_detector.train_detector import ...`` and
``from tile_yolo_dataset.tile_geometry import ...``). This conftest injects
the repo root onto sys.path so plain ``pytest`` works regardless of cwd.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))