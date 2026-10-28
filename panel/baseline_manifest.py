"""Frozen baseline manifests for issue #49 Phase 0 (read-only diagnostics).

A manifest binds one physical case to the exact bytes that produced it: the
half-surface fixture digests, wall/wake/TE topology counts, the solver state
digest, the loads, the linear residual, the code digest of ``panel/*.py`` and
the runtime versions.  Nothing here solves, meshes or changes a default; a
manifest is a checksummed receipt so a later A/B cannot silently compare
different geometry, different code or a different reference area.

Schema ``dpanel.baseline_manifest.v1``.  ``verify_manifest`` re-derives every
digest from live objects and reports mismatches instead of trusting the file.
"""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import jax
import numpy as np

SCHEMA = "dpanel.baseline_manifest.v1"

# Every fixture field that changes the discrete problem, in canonical order.
FIXTURE_FIELDS = (
    "coordinates", "faces", "wake_faces", "trailing_edges",
    "upper_te_faces", "lower_te_faces", "gamma_stations",
    "gamma_station_dof", "component_ids", "cut_edges",
    "freestream", "moment_reference", "reference_area", "reference_chord",
    "density",
)


def file_digest(path):
    """SHA-256 of the file bytes exactly as written."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def array_digest(value):
    """Canonical digest: dtype, shape and little-endian C-order payload.

    Integers are compared as int64 and reals as float64 so a fixture written
    with another dtype cannot pass as the same geometry.
    """
    a = np.asarray(value)
    if a.dtype.kind in "iu":
        a = a.astype("<i8")
    elif a.dtype.kind == "f":
        a = a.astype("<f8")
    elif a.dtype.kind == "b":
        a = a.astype(np.uint8)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def surface_digests(fixture):
    """Per-field digests of a ``dpanel.half_planar_cut.v1`` fixture."""
    out = {}
    for key in FIXTURE_FIELDS:
        if key in fixture:
            out[key] = array_digest(fixture[key])
    return out


def solve_state_digest(result):
    """Digest of the three physical field arrays, never of intermediates."""
    return {
        "mu_half": array_digest(result["mu_half"]),
        "gamma_half": array_digest(result["gamma_half"]),
        "sigma_half": array_digest(result["sigma_half"]),
        "Cp_raw_half": array_digest(result["Cp_raw_half"]),
    }


def code_digest(code_root=None):
    """Digest over every ``panel/*.py`` file: sorted (name, sha256) pairs."""
    here = Path(code_root) if code_root else Path(__file__).resolve().parent
    entries = [(path.name, file_digest(path)) for path in sorted(here.glob("*.py"))]
    h = hashlib.sha256()
    for name, digest in entries:
        h.update(name.encode())
        h.update(digest.encode())
    return {"digest": h.hexdigest(), "files": len(entries)}


def environment_digest():
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "jax": jax.__version__,
        "jax_x64": bool(jax.config.x64_enabled),
        "platform": platform.platform(),
    }


def _float(value):
    return float(np.asarray(value))


def _vector(value):
    return [float(v) for v in np.asarray(value).reshape(-1)]


def loads_section(result):
    by_component = {}
    for tag, force, moment in zip((10, 20, 30), result["component_forces"],
                                  result["component_moments"]):
        by_component[str(int(tag))] = {
            "force": _vector(force), "moment": _vector(moment)}
    return {
        "CL": _float(result["CL"]),
        "CD_pressure": _float(result["CD_pressure"]),
        "CM": _vector(result["CM"]),
        "force": _vector(result["force"]),
        "moment": _vector(result["moment"]),
        "component": by_component,
    }


def pressure_section(result):
    cp = np.asarray(result["Cp_raw_half"], dtype=float)
    return {
        "Cp_raw": {
            "count": int(cp.size),
            "min": float(cp.min()),
            "max": float(cp.max()),
            "mean": float(cp.mean()),
            "rms": float(np.sqrt(np.mean(cp * cp))),
            "digest": array_digest(cp),
        },
        "panel_areas_half": {
            "count": int(np.size(result["panel_areas_half"])),
            "sum": float(np.sum(result["panel_areas_half"])),
        },
    }


def build_manifest(*, surface_path, fixture, result, residual_absolute,
                   residual_relative, metrics=None, code_root=None,
                   case_id=None, extra=None):
    """Assemble one manifest dict; the caller writes it next to its evidence."""
    flow = np.asarray(fixture["freestream"], dtype=float)
    speed = float(np.linalg.norm(flow))
    faces = np.asarray(fixture["faces"]).reshape(-1, 3)
    topology = {
        "wall_faces": int(faces.shape[0]),
        "wake_faces": int(np.asarray(fixture["wake_faces"]).reshape(-1, 3).shape[0]),
        "te_edges": int(np.asarray(fixture["trailing_edges"]).reshape(-1, 2).shape[0]),
        "vertices": int(np.asarray(fixture["coordinates"]).reshape(-1, 3).shape[0]),
        "circulation_unknowns": int(np.asarray(result["gamma_half"]).size),
        "independent_unknowns": int(np.asarray(result["mu_half"]).size
                                    + np.asarray(result["gamma_half"]).size),
    }
    manifest = {
        "schema": SCHEMA,
        "case_id": case_id or Path(surface_path).parent.name,
        "surface_path": str(surface_path),
        "source": {
            "fixture_digest": file_digest(surface_path),
            "surface_fields": surface_digests(fixture),
        },
        "flow": {
            "freestream": _vector(flow),
            "speed": speed,
            "alpha_deg": float(np.degrees(np.arctan2(flow[2], flow[0]))),
            "density": float(fixture["density"]),
            "reference_area": float(fixture["reference_area"]),
            "reference_chord": float(fixture["reference_chord"]),
            "moment_reference": _vector(fixture["moment_reference"]),
        },
        "topology": topology,
        "solve": {
            "linear_residual_absolute": float(residual_absolute),
            "linear_residual_relative": float(residual_relative),
            "state": solve_state_digest(result),
            "loads": loads_section(result),
            "pressure": pressure_section(result),
        },
        "code": code_digest(code_root),
        "environment": environment_digest(),
        "metrics": metrics,
        "physical_accuracy_accepted": False,
    }
    if extra:
        manifest.update(extra)
    return manifest


def verify_manifest(manifest, *, surface_path, fixture, result):
    """Re-derive digests from live objects; return the list of mismatches."""
    problems = []
    if manifest.get("schema") != SCHEMA:
        problems.append("schema:" + str(manifest.get("schema")))
    if file_digest(surface_path) != manifest["source"]["fixture_digest"]:
        problems.append("fixture_digest")
    for key, value in surface_digests(fixture).items():
        if manifest["source"]["surface_fields"].get(key) != value:
            problems.append("surface_field:" + key)
    for key, value in solve_state_digest(result).items():
        if manifest["solve"]["state"].get(key) != value:
            problems.append("state:" + key)
    return problems


def main():
    import argparse

    p = argparse.ArgumentParser(description="Write a frozen baseline manifest")
    p.add_argument("--surface", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True,
                   help="report.json of a finished run_half_native solve")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    fixture = json.loads(Path(args.surface).read_text(encoding="utf-8"))
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    if report.get("status") != "ok":
        raise ValueError("Only a finished report can be frozen")
    state = np.load(Path(args.report).parent / "state.npz")
    result = {name: state[name] for name in state.files}
    manifest = build_manifest(
        surface_path=args.surface, fixture=fixture, result=result,
        residual_absolute=report["linear_residual_absolute"],
        residual_relative=report["linear_residual_relative"],
        case_id=report.get("case_id"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=1, sort_keys=True),
                        encoding="utf-8")
    print(json.dumps({"out": str(args.out),
                      "code": manifest["code"]["digest"][:12],
                      "CL": manifest["solve"]["loads"]["CL"]}, indent=1))


if __name__ == "__main__":
    main()