"""Opt-in profiling counters for the regularized wake-velocity kernels (issue #84, P0).

Every call site reads one module-level Boolean before doing anything, so a normal
production run pays no measurable cost and no numerical result changes. Counters are
deliberately flat (name -> number) so that a profile can be diffed between two heads.

Intended use, after the PR #91 review: measure how many targets, triangles, fan
sub-triangles and per-stage seconds an actual B2 iteration consumes, instead of
extrapolating a single-point cost by a nonlinear-solver iteration count.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

_ENABLED = False
_COUNTERS = {}
_SECONDS = {}
_STARTS = {}


def enable(flag=True):
    """Turn profiling on or off; returns the previous state."""
    global _ENABLED
    previous, _ENABLED = _ENABLED, bool(flag)
    return previous


def enabled():
    return _ENABLED


def reset():
    _COUNTERS.clear()
    _SECONDS.clear()
    _STARTS.clear()


def add(name, n=1):
    if _ENABLED:
        _COUNTERS[name] = _COUNTERS.get(name, 0) + n


def begin(name):
    if _ENABLED:
        _STARTS[name] = time.perf_counter()


def end(name):
    if _ENABLED:
        start = _STARTS.pop(name, None)
        if start is not None:
            _SECONDS[name] = _SECONDS.get(name, 0.0) + (time.perf_counter() - start)


class stage:
    """Context manager form of begin/end, for stages that can raise."""

    __slots__ = ("name",)

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        begin(self.name)
        return self

    def __exit__(self, *exc):
        end(self.name)
        return False


def peak_rss_mb():
    """Peak resident set size of this process in MiB, or None when unavailable."""
    try:
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]

        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        handle = ctypes.c_void_p(ctypes.windll.kernel32.GetCurrentProcess())
        # K32GetProcessMemoryInfo is the supported entry point; psapi.dll's alias
        # silently returns 0 on current Windows builds.
        for name in ("K32GetProcessMemoryInfo",):
            try:
                query = getattr(ctypes.windll.kernel32, name)
            except AttributeError:
                continue
            if query(handle, ctypes.byref(counters), ctypes.sizeof(Counters)):
                return counters.PeakWorkingSetSize/1048576.0
        try:
            query = ctypes.windll.psapi.GetProcessMemoryInfo
        except Exception:
            return None
        if not query(handle, ctypes.byref(counters), ctypes.sizeof(Counters)):
            return None
        return counters.PeakWorkingSetSize/1048576.0
    except Exception:
        pass
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return float(line.split()[1])/1024.0
    except Exception:
        return None
    return None


def snapshot():
    """Counters and accumulated seconds as plain JSON-safe values."""
    return {"counters": {k: int(v) for k, v in sorted(_COUNTERS.items())},
            "seconds": {k: round(v, 6) for k, v in sorted(_SECONDS.items())}}


def write(path, **extra):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = snapshot()
    payload.update(extra)
    payload["peak_rss_mb"] = peak_rss_mb()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")
    return payload
