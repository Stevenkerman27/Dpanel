"""Test isolation for the reference audit.

``jax_enable_x64`` must be set before any array is built: the digests and the
source-strength identity are only meaningful in double precision.

``tmp_path`` is overridden because the DSH sandbox cannot enumerate the process
temp directory (``PermissionError: [WinError 5]`` while listing
``pytest-of-*``), so pytest cannot create its own base temporary directory.
"""
import os
import pathlib
import shutil
import uuid

import jax
import pytest

jax.config.update("jax_enable_x64", True)

# Route B option A makes torch the production default. The recorded baselines in this
# suite were measured on the NumPy/SciPy oracle, so the suite pins it; tests that
# exercise the torch path pass backend='torch' explicitly.
os.environ.setdefault("PANEL_P1_BACKEND", "numpy")

_ROOT = pathlib.Path(__file__).resolve().parent


@pytest.fixture
def tmp_path():
    path = _ROOT / "_tmp" / uuid.uuid4().hex[:12]
    path.mkdir(parents=True, exist_ok=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)