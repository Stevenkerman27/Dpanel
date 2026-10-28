"""Tessellation plan for OpenVSP wings (ported from DEP-OPT)."""

from __future__ import annotations

from typing import Iterable, Sequence


def next_tess_value(x: float) -> int:
    """DEP-OPT's `next_tess_value`: round a requested chordwise count up to 5+4n."""
    n = max((x - 6) // 4 + 1, 0)
    return int(5 + n * 4)


def make_wing_tess_plan(spanlist: Sequence[float], chordlist: Sequence[float],
                        tess_int: float) -> dict:
    """Chordwise / spanwise tessellation counts for a multi-section wing.

    `tess_int` is the target panel size in metres; `chordlist` holds one more
    entry than `spanlist` (root chord, then the tip chord of every section).
    """
    if len(chordlist) != len(spanlist) + 1:
        raise ValueError("Wing chord list must contain one more value than the span list")
    if tess_int <= 0:
        raise ValueError("tess_int must be positive")
    tess_w = next_tess_value(int(chordlist[0] / tess_int * 2))
    section_tess_u = [
        round(spanlist[i - 1] / ((chordlist[i] / chordlist[0]) * tess_int)) + 1
        for i in range(1, len(spanlist) + 1)
    ]
    return {"Tess_W": tess_w, "SectTess_U": section_tess_u}


def wing_area(spanlist: Iterable[float], chordlist: Sequence[float]) -> float:
    """Reference area of a (half or full) trapezoidal wing: sum s_i (c_i + c_{i+1}).

    This is DEP-OPT's convention and, for a symmetric model, already the
    full-aircraft area because `spanlist` spans the semi-span only.
    """
    spans = list(spanlist)
    if len(chordlist) != len(spans) + 1:
        raise ValueError("Wing chord list must contain one more value than the span list")
    return float(sum(s * (chordlist[i] + chordlist[i + 1]) for i, s in enumerate(spans)))
