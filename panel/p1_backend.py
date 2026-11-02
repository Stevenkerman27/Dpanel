"""Backend selection for the production P1 solve (Route B, slice S9).

The user chose option A: torch becomes the production default, while the NumPy/SciPy
chain stays reachable as the explicit evidence generator (`backend='numpy'`), keeps
running in the environment that has no torch, and remains the oracle the frozen bundles
in docs/issue92/references/ were produced with.

The default is resolved rather than assumed: `backend=None` means "torch when this
interpreter can import it, NumPy otherwise", and every result records both the backend
that ran and whether the caller asked for it (`backend_source='default'|'explicit'`).
An explicit backend that cannot run raises; nothing is switched behind the caller's back.

`PANEL_P1_BACKEND=numpy|torch` pins the resolution for a whole process. The recorded
test suite pins the oracle that way so its numeric baselines keep running on the path
they were recorded on; a caller that passes `backend=` always wins.
"""
from __future__ import annotations

import os

DEFAULT = "torch"
ORACLE = "numpy"
CHOICES = (ORACLE, DEFAULT)


def torch_available():
    """True when this interpreter can import torch (the only requirement of the path)."""
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def resolve_backend(backend=None):
    """Return `(name, source)` for a user-facing backend argument.

    >>> resolve_backend('numpy')
    ('numpy', 'explicit')
    """
    if backend is None:
        pinned = os.environ.get("PANEL_P1_BACKEND")
        if pinned:
            if pinned not in CHOICES:
                raise ValueError("PANEL_P1_BACKEND must be 'numpy' or 'torch'")
            return pinned, "environment"
        return ((DEFAULT if torch_available() else ORACLE), "default")
    if backend not in CHOICES:
        raise ValueError("backend must be None, 'numpy' or 'torch'")
    if backend == DEFAULT and not torch_available():
        raise ValueError("backend='torch' requested but torch is not importable")
    return backend, "explicit"
