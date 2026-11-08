"""Helpers for importing icefall's librispeech zipformer recipe code."""

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

RECIPE_SUBDIR = Path("egs/librispeech/ASR/zipformer")


def icefall_root() -> Path:
    """Return `$ICEFALL_ROOT` or `<repo>/external/icefall`."""
    env = os.environ.get("ICEFALL_ROOT")
    if env:
        return Path(env)
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "external" / "icefall"


def setup_icefall_paths() -> Path:
    """Put the icefall root and recipe dir on `sys.path` (idempotent).

    Returns the recipe directory.
    """
    root = icefall_root()
    recipe_dir = root / RECIPE_SUBDIR
    # Insert root first, then recipe dir at 0, so the recipe dir ends up first.
    for p in (str(root), str(recipe_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)
    return recipe_dir


def load_recipe_module(filename: str) -> ModuleType:
    """Load a recipe file (possibly hyphenated, e.g. `export-onnx-streaming-ctc.py`)."""
    recipe_dir = setup_icefall_paths()
    path = recipe_dir / filename
    name = path.stem.replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
