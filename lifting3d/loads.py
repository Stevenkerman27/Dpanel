"""Integrate raw pressure forces and moments, retaining the explicit reference scales."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def pressure_loads(coordinates, cp, normals, weights, tags, center, reference_area, reference_length):
    """Fluid-outward normals give the pressure force on the solid as Cp*n_fluid."""
    force_samples = cp[:, :, None]*normals*weights[:, :, None]
    lever = coordinates-np.asarray(center)
    moment_samples = np.cross(lever, force_samples)
    components = {}
    for tag in np.unique(tags[np.isin(tags, [10, 20, 30])]):
        selected = tags == tag
        components[str(int(tag))] = dict(
            force_coefficient=(force_samples[selected].sum(axis=(0, 1))/reference_area).tolist(),
            moment_coefficient=(moment_samples[selected].sum(axis=(0, 1))/(reference_area*reference_length)).tolist(),
            surface_area=float(weights[selected].sum()))
    wall = np.isin(tags, [10, 20, 30])
    return dict(force_coefficient=(force_samples[wall].sum(axis=(0, 1))/reference_area).tolist(),
                moment_coefficient=(moment_samples[wall].sum(axis=(0, 1))/(reference_area*reference_length)).tolist(),
                components=components)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--center", type=float, nargs=3, help="Moment center in saved mesh coordinates; default root quarter chord")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists(), "Preserve previous pressure load reports"
    result = json.loads((args.run/"result.json").read_text())
    metadata = json.loads(args.mesh.with_name("mesh.json").read_text())
    field_path = args.run/"field.npz"
    field = np.load(field_path, allow_pickle=False)
    assert result["converged"], "Require a converged lifting result"
    mesh_hash = hashlib.sha256(args.mesh.read_bytes()).hexdigest()
    assert result["mesh_sha256"] == mesh_hash
    metadata_current = metadata["mesh_sha256"] == mesh_hash
    assert metadata_current or metadata.get("transform_source_mesh_sha256") == metadata["mesh_sha256"]
    assert np.array_equal(field["points"], np.load(args.mesh, allow_pickle=False)["points"])
    origin = np.asarray(result["normalization_origin"])
    chord = float(metadata["chord"])
    sref = float(metadata["reference_area"])
    assert chord == result["normalization_length"]
    assert np.array_equal(origin, metadata.get("normalization_origin", [0., 0., 0.]))
    center = np.asarray(args.center) if args.center is not None else origin+np.array([.25*chord, 0., 0.])
    assert np.isfinite(center).all() and np.isfinite([chord, sref]).all() and chord > 0. and sref > 0.
    for key in ["surface_points", "cp", "surface_normal", "surface_weight"]:
        assert np.isfinite(field[key]).all(), f"Nonfinite raw data: {key}"
    assert np.all(field["surface_weight"] > 0.)
    loads = pressure_loads(field["surface_points"], field["cp"], field["surface_normal"],
                           field["surface_weight"], field["tags"], center, sref, chord)
    force_error = float(np.max(abs(np.array(loads["force_coefficient"])-result["pressure_force_coefficient"])))
    assert force_error < 1e-10, "Raw pressure forces disagree with the original result"
    component_force_error = float(np.max(abs(np.sum(
        [entry["force_coefficient"] for entry in loads["components"].values()], axis=0)-loads["force_coefficient"])))
    component_moment_error = float(np.max(abs(np.sum(
        [entry["moment_coefficient"] for entry in loads["components"].values()], axis=0)-loads["moment_coefficient"])))
    assert max(component_force_error, component_moment_error) < 1e-10
    loads.update(moment_center=center.tolist(), reference_area=sref, reference_length=chord,
                 normalization_origin=origin.tolist(), force_roundtrip_max_error=force_error,
                 component_force_sum_max_error=component_force_error,
                 component_moment_sum_max_error=component_moment_error,
                 force_definition="CF = integral Cp*n_fluid*dS / Sref",
                 moment_definition="CM = integral (x-center) cross (Cp*n_fluid)*dS / (Sref*root_chord)",
                 axes="Right-handed global x,y,z; moment components about these axes, without aircraft sign relabeling",
                 scope="Raw pressure integration only; does not certify pressure accuracy",
                 mesh_metadata_hash_relation="current_mesh" if metadata_current else "declared_transform_source_mesh",
                 mesh_sha256=mesh_hash, field_sha256=hashlib.sha256(field_path.read_bytes()).hexdigest(),
                 source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(loads, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(loads, indent=2))
