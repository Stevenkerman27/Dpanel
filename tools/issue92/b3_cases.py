"""Issue #92 B3: one station table per case, shared by DPanel and VSPAERO.

The station table is the geometry contract. For the CAD cases it is measured
from the same surface.json that DPanel solves (chord and leading-edge x at
every declared trailing-edge station, plus the section thickness ratio). The
VSPAERO side rebuilds the wing from that table, so the two codes see the same
planform up to spanwise discretisation; the manifest records the measured
numbers and the hash of the table, not a promise.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("PANEL_ROOT", Path.cwd()))
SURFACE_ROOT = ROOT / "cases/geom_robust/surfaces"

ALPHAS = (0.0, 2.0, 4.0, 6.0)
DPANEL_LEVELS = {"coarse": dict(chord_panels=4, span_panels=6),
                 "fine": dict(chord_panels=8, span_panels=10)}


def _read_surface(name):
    path = SURFACE_ROOT / name / "surface.json"
    return path, json.loads(path.read_text(encoding="utf-8"))


def _stations_from_surface(name, *, slab=1e-6):
    """Measure (y, chord, le_x, thickness ratio) at every trailing-edge station."""
    path, data = _read_surface(name)
    xyz = np.asarray(data["coordinates"], float)
    faces = np.asarray(data["faces"], int)
    used = np.unique(faces)
    te_ids = np.unique(np.asarray(data["trailing_edges"], int).ravel())
    te_ids = te_ids[te_ids < len(xyz)]
    te = xyz[te_ids]
    order = np.argsort(te[:, 1])
    stations = []
    for index in order:
        y = float(te[index, 1])
        mask = np.abs(xyz[used, 1] - y) <= slab
        if mask.sum() < 4:
            continue
        block = xyz[used][mask]
        le_x = float(block[:, 0].min())
        te_x = float(te[index, 0])
        chord = te_x - le_x
        if chord <= 1e-9:
            continue
        thick = float(block[:, 2].max() - block[:, 2].min()) / chord
        station = dict(y=round(y, 9), chord=round(chord, 9), le_x=round(le_x, 9),
                       twist_deg=0.0, thickness=round(min(max(thick, 0.0), 0.6), 9))
        if stations and abs(station["y"] - stations[-1]["y"]) < 1e-9:
            continue
        stations.append(station)
    return {
        "kind": "cad_half",
        "surface": str(path.relative_to(ROOT)).replace("\\", "/"),
        "surface_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "sref": float(data["reference_area"]),
        "cref": float(data["reference_chord"]),
        "moment_reference": [float(v) for v in data["moment_reference"]],
        "stations": stations,
    }


def _rect_case():
    """Analytic rectangular wing: DPanel make_wing and VSPAERO see the same box."""
    half_span, chord = 2.0, 1.0
    return {
        "kind": "rect",
        "make_wing": dict(chord_panels=4, span_panels=6, thickness=.12, camber=0.,
                          span=4., chord=1., wake_panels=2),
        "sref": 4.0 * 1.0,
        "cref": 1.0,
        "moment_reference": [0.25, 0.0, 0.0],
        "stations": [dict(y=0.0, chord=chord, le_x=0.0, twist_deg=0.0, thickness=0.12),
                     dict(y=half_span, chord=chord, le_x=0.0, twist_deg=0.0, thickness=0.12)],
    }


def cases():
    return {
        "rect": _rect_case(),
        "swept30": _stations_from_surface("swept30_s075"),
        "elliptic": _stations_from_surface("elliptic_s075"),
        "bwb": _stations_from_surface("bwb_s055"),
    }


def geometry_hash(spec):
    payload = {key: spec[key] for key in ("kind", "sref", "cref", "moment_reference", "stations")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def manifest(out_path):
    doc = {"schema": "dpanel.issue92.b3.cases.v1", "alphas_deg": list(ALPHAS),
           "dpanel_levels": DPANEL_LEVELS, "cases": {}}
    for name, spec in cases().items():
        spec = dict(spec)
        spec["geometry_sha256"] = geometry_hash(spec)
        spec["symmetry"] = 0 if spec["kind"] == "rect" else 1
        spec["n_stations"] = len(spec["stations"])
        spec["chord_root"] = spec["stations"][0]["chord"]
        spec["chord_tip"] = spec["stations"][-1]["chord"]
        spec["half_span"] = spec["stations"][-1]["y"]
        doc["cases"][name] = spec
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return doc


def main(argv=None):
    parser = argparse.ArgumentParser(description="B3 station tables and manifests")
    parser.add_argument("--out", type=Path, default=ROOT / "docs/issue92/b3_cases.json")
    args = parser.parse_args(argv)
    doc = manifest(args.out)
    for name, spec in doc["cases"].items():
        print("%-8s stations=%-3d half_span=%.3f chord %.4f->%.4f sref=%.4f sha=%s" % (
            name, spec["n_stations"], spec["half_span"], spec["chord_root"],
            spec["chord_tip"], spec["sref"], spec["geometry_sha256"][:12]))
    return doc


if __name__ == "__main__":
    main()
