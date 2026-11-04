"""Backend selection for the production P1 solve (Route B, slices S9 and S10).

S9 moved the default to torch with the user's option A: the NumPy/SciPy chain stayed
reachable as the explicit evidence generator while the decision was still open.
S10 closes that: the NumPy solver is retired, so this module now reports the one
production backend and rejects the retired one with a message that says where the
answers went.

What a caller may pass:

* `backend=None` (default) -> `('torch', 'default')`;
* `backend='torch'` -> `('torch', 'explicit')`, or a `ValueError` when this interpreter
  cannot import torch - nothing is switched silently;
* `backend='numpy'` -> `ValueError`, because the NumPy solve chain that produced the
  S1-S8 baselines is preserved at tag `numpy-path-frozen-s8` and its answers for the
  eight frozen fixtures are in `docs/issue92/references/`.

`PANEL_P1_BACKEND` may still pin a process; only `torch` is a live choice.
"""
from __future__ import annotations

import os

DEFAULT = "torch"
RETIRED = "numpy"
CHOICES = (DEFAULT,)
ARCHIVE_TAG = "numpy-path-frozen-s8"
ARCHIVE_EVIDENCE = "docs/issue92/references/"


def torch_available():
    """True when this interpreter can import torch (the only requirement of the path)."""
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def _retired(backend):
    return ValueError(
        "backend=%r was retired with the NumPy solve chain; its code is tag %s and its "
        "recorded answers are %s" % (backend, ARCHIVE_TAG, ARCHIVE_EVIDENCE))


def resolve_backend(backend=None):
    """Return `(name, source)` for a user-facing backend argument.

    >>> resolve_backend()
    ('torch', 'default')
    """
    if backend is None:
        pinned = os.environ.get("PANEL_P1_BACKEND")
        if pinned:
            if pinned == RETIRED:
                raise _retired(pinned)
            if pinned not in CHOICES:
                raise ValueError("PANEL_P1_BACKEND must be 'torch'")
            if not torch_available():
                raise ValueError("PANEL_P1_BACKEND='torch' but torch is not importable")
            return pinned, "environment"
        if not torch_available():
            raise ValueError(
                "the P1 solve chain requires torch; the retired NumPy chain is tag %s and "
                "its recorded answers are %s" % (ARCHIVE_TAG, ARCHIVE_EVIDENCE))
        return DEFAULT, "default"
    if backend == RETIRED:
        raise _retired(backend)
    if backend != DEFAULT:
        raise ValueError("backend must be None or 'torch'")
    if not torch_available():
        raise ValueError("backend='torch' requested but torch is not importable")
    return backend, "explicit"
