"""Issue #92 Phase 0: immutable benchmark fixtures with SHA256 manifests.

Fixtures are geometry + topology only; they do not embed solver settings, so the
same file can be replayed by the NumPy/SciPy reference, the PyTorch CPU backend
and the CUDA backend.  Each fixture is a dict of named arrays; the manifest stores
the sha256 of every array (raw little-endian bytes, C order) plus one digest over
the sorted (name, sha256) pairs.

Usage:
  python -m tools.issue92.fixtures --out docs/issue92/fixtures
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("PANEL_ROOT", Path.cwd()))

import numpy as np

from panel.p1_lifting_g1 import make_wing, from_dpanel_half_surface
from panel.p1_lifting_g2 import combine_wings, MultiLiftingMesh

WING_KEYS = ("xyz", "faces", "dof_face", "face_bank", "wake_xyz", "wake_faces",
             "gamma_y", "gamma_hat", "te_pairs", "te_pair_y", "te_geo")
MULTI_KEYS = ("xyz", "faces", "dof_face", "face_bank", "face_group", "ndof_mu",
              "reference_area", "reference_chord", "reference_point")


def _wing_arrays(wing):
    out = {}
    for key in WING_KEYS:
        value = getattr(wing, key, None)
        if value is None:
            continue
        out[key] = np.ascontiguousarray(np.asarray(value))
    out["ndof_mu"] = np.asarray(int(wing.ndof_mu))
    out["reference_area"] = np.asarray(float(wing.reference_area))
    out["chord"] = np.asarray(float(wing.chord))
    out["reference_point"] = np.ascontiguousarray(np.asarray(wing.reference_point, float))
    return out


def _multi_arrays(mesh: MultiLiftingMesh):
    out = {key: np.ascontiguousarray(np.asarray(getattr(mesh, key))) for key in MULTI_KEYS}
    for group in mesh.groups:
        prefix = "group:" + group.name
        for key in ("face_ids", "te_pairs", "te_xyz", "te_pair_y", "wake_xyz",
                    "wake_faces", "gamma_y", "gamma_hat"):
            value = getattr(group, key, None)
            if value is None:
                continue
            out[prefix + ":" + key] = np.ascontiguousarray(np.asarray(value))
        out[prefix + ":chord"] = np.asarray(float(group.chord))
        out[prefix + ":reference_area"] = np.asarray(float(group.reference_area))
    return out


def _digest(arrays):
    per_array = {}
    for name in sorted(arrays):
        value = arrays[name]
        if value.dtype == object:
            raise ValueError("object arrays are forbidden in fixtures: " + name)
        per_array[name] = hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
    joined = "".join(name + ":" + per_array[name] + "\n" for name in per_array)
    return per_array, hashlib.sha256(joined.encode("ascii")).hexdigest()


SURFACE_ROOTS = (ROOT, ROOT.parent, ROOT.parent.parent)


def _first_existing(rel):
    for root in SURFACE_ROOTS:
        candidate = Path(root) / rel
        if candidate.exists():
            return candidate
    raise FileNotFoundError("none of the fixture sources exist: " +
                            ", ".join(str(Path(root) / rel) for root in SURFACE_ROOTS))


def _half_surface(rel):
    path = _first_existing(rel)
    return from_dpanel_half_surface(path), dict(source=str(path).replace("\\", "/"),
                                                source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def builders():
    def g1_wing_rect():
        return _wing_arrays(make_wing(chord_panels=4, span_panels=6, thickness=.12,
                                      camber=0., span=4., chord=1., wake_panels=2, wake_length=5.)), \
               dict(kind="wing", alpha_deg=5., kutta="pressure", chord_panels=4,
                    span_panels=6, thickness=.12, camber=0., span=4., chord=1., wake_panels=2)

    def g1_wing_camber():
        return _wing_arrays(make_wing(chord_panels=4, span_panels=6, thickness=.12,
                                      camber=.02, span=4., chord=1., wake_panels=2, wake_length=5.)), \
               dict(kind="wing", alpha_deg=5., kutta="pressure", chord_panels=4,
                    span_panels=6, thickness=.12, camber=.02, span=4., chord=1., wake_panels=2)

    def g2_conventional():
        mesh = combine_wings([
            make_wing(chord_panels=4, span_panels=6, span=4., chord=1., camber=.02, wake_panels=2, wake_length=5.),
            make_wing(chord_panels=4, span_panels=4, span=1.8, chord=.55, camber=0., wake_panels=2, wake_length=5.)],
            origins=[(0., 0., 0.), (1.6, 0., .35)], names=["main_wing", "raised_tail"])
        return _multi_arrays(mesh), dict(kind="multi", alpha_deg=5., kutta="pressure",
                                         tail_origin=[1.6, 0., .35])

    def g2_tail_low():
        mesh = combine_wings([
            make_wing(chord_panels=4, span_panels=6, span=4., chord=1., camber=.02, wake_panels=2, wake_length=5.),
            make_wing(chord_panels=4, span_panels=4, span=1.8, chord=.55, camber=0., wake_panels=2, wake_length=5.)],
            origins=[(0., 0., 0.), (1.6, 0., .24)], names=["main_wing", "raised_tail"])
        return _multi_arrays(mesh), dict(kind="multi", alpha_deg=5., kutta="pressure",
                                         tail_origin=[1.6, 0., .24])

    def g1_swept30():
        wing, meta = _half_surface("cases/geom_robust/surfaces/swept30_s075/surface.json")
        return _wing_arrays(wing), dict(meta, kind="wing", alpha_deg=5., kutta="pressure")

    def g1_elliptic():
        wing, meta = _half_surface("cases/geom_robust/surfaces/elliptic_s075/surface.json")
        return _wing_arrays(wing), dict(meta, kind="wing", alpha_deg=5., kutta="pressure")

    def g1_wingbody_cad():
        wing, meta = _half_surface("runs/cases/wingbody/half_cad_wingbody_v1/surface.json")
        return _wing_arrays(wing), dict(meta, kind="wing", alpha_deg=5., kutta="pressure")

    def g1_wingbody_cad_3x():
        wing, meta = _half_surface("runs/cases/wingbody/half_cad_wingbody_3x/surface.json")
        return _wing_arrays(wing), dict(meta, kind="wing", alpha_deg=5., kutta="pressure")

    return {
        "g1_wing_rect": g1_wing_rect,
        "g1_wing_camber": g1_wing_camber,
        "g2_conventional": g2_conventional,
        "g2_tail_low": g2_tail_low,
        "g1_swept30": g1_swept30,
        "g1_elliptic": g1_elliptic,
        "g1_wingbody_cad": g1_wingbody_cad,
        "g1_wingbody_cad_3x": g1_wingbody_cad_3x,
    }


def build(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "dpanel.issue92.fixtures.v1", "arrays_dir": out_dir.name,
                "fixtures": {}}
    for name, make in builders().items():
        try:
            arrays, meta = make()
        except Exception as exc:  # a fixture that cannot be built must be reported, not hidden
            manifest["fixtures"][name] = {"status": "unavailable", "reason": repr(exc)}
            continue
        per_array, digest = _digest(arrays)
        np.savez_compressed(out_dir / (name + ".npz"), **arrays)
        file_sha = hashlib.sha256((out_dir / (name + ".npz")).read_bytes()).hexdigest()
        n_face = int(len(arrays["faces"]))
        manifest["fixtures"][name] = {
            "status": "ok", "meta": meta, "array_sha256": per_array,
            "digest": digest, "npz_sha256": file_sha, "n_wall_faces": n_face,
            "n_wall_nodes": int(len(arrays["xyz"])),
            "n_wake_triangles": int(len(arrays.get("wake_faces", []))) if "wake_faces" in arrays else None,
            "bytes": int((out_dir / (name + ".npz")).stat().st_size),
        }
    (out_dir.parent / "fixtures_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def manifest_path(arrays_dir=None):
    arrays_dir = Path(arrays_dir or (ROOT / "docs/issue92/fixtures"))
    return arrays_dir.parent / "fixtures_manifest.json"


def load_manifest(arrays_dir=None):
    return json.loads(manifest_path(arrays_dir).read_text(encoding="utf-8"))


def available(name, arrays_dir=None):
    """True when the fixture file exists and the manifest records it as built."""
    arrays_dir = Path(arrays_dir or (ROOT / "docs/issue92/fixtures"))
    if not (arrays_dir / (name + ".npz")).exists():
        return False
    entry = load_manifest(arrays_dir)["fixtures"].get(name, {})
    return entry.get("status") == "ok"


def load(name, arrays_dir=None):
    """Load one fixture into a dict of arrays; raises if the manifest digest drifts."""
    arrays_dir = Path(arrays_dir or (ROOT / "docs/issue92/fixtures"))
    entry = load_manifest(arrays_dir)["fixtures"][name]
    if entry["status"] != "ok":
        raise ValueError("fixture unavailable: " + name + ": " + entry["reason"])
    with np.load(arrays_dir / (name + ".npz")) as data:
        arrays = {key: data[key] for key in data.files}
    _, digest = _digest(arrays)
    if digest != entry["digest"]:
        raise ValueError("fixture digest drift for " + name + ": " + digest + " != " + entry["digest"])
    return arrays, entry


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "docs/issue92/fixtures")
    args = parser.parse_args(argv)
    manifest = build(args.out)
    for name, entry in manifest["fixtures"].items():
        if entry["status"] == "ok":
            print("%-18s faces=%-5d nodes=%-5d digest=%s" % (
                name, entry["n_wall_faces"], entry["n_wall_nodes"], entry["digest"][:16]))
        else:
            print("%-18s UNAVAILABLE %s" % (name, entry["reason"]))
    return manifest


if __name__ == "__main__":
    main()
