"""Create additional imported-CAD contact fixtures without running the builder."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root.parent / "audit" / "runtime" / "python"))
import gmsh

out = root / "results" / "cad_contact_inputs_v1"
out.mkdir(parents=True, exist_ok=True)

fixture_root = root / "results" / "cad_inputs_v1"
wing0012_source = fixture_root / "naca0012_wing" / "wing.step"
wing4412_source = fixture_root / "naca4412_wing_body" / "wing.step"
te_source = fixture_root / "naca0012_wing" / "trailing_edge.json"
te4412_source = fixture_root / "naca4412_wing_body" / "trailing_edge.json"
assert hashlib.sha256(te_source.read_bytes()).digest() == hashlib.sha256(
    te4412_source.read_bytes()).digest()

cases = [
    {
        "name": "two_offset_bodies_crossing_te",
        "wing_source": wing0012_source,
        "wing_family": "NACA 0012",
        "body_centers": [(-0.5, -0.8, 0.0), (-0.5, 0.8, 0.0)],
        "body_semiaxes": [2.0, 0.25, 0.25],
        "purpose": "Two span-separated body/wake intersections with the TE at x=1.",
    },
    {
        "name": "aft_body_max_section_after_te",
        "wing_source": wing0012_source,
        "wing_family": "NACA 0012",
        "body_centers": [(1.2, 0.0, 0.0)],
        "body_semiaxes": [2.0, 0.35, 0.35],
        "purpose": "Aft body endpoint x=3.2; wake/body contact widens downstream of TE.",
    },
    {
        "name": "body_ahead_of_te_no_contact",
        "wing_source": wing0012_source,
        "wing_family": "NACA 0012",
        "body_centers": [(-0.7, 0.0, 0.0)],
        "body_semiaxes": [1.5, 0.35, 0.35],
        "purpose": "Body aft endpoint x=0.8; no wall/wake contact is expected downstream of TE x=1.",
    },
    {
        "name": "thick_4412_wing_body",
        "wing_source": wing4412_source,
        "wing_family": "NACA 4412",
        "body_centers": [(-0.5, 0.0, 0.0)],
        "body_semiaxes": [2.0, 0.6, 0.6],
        "purpose": "Thick NACA 4412 wing with enlarged body radius.",
    },
]

gmsh.initialize()
gmsh.option.setNumber("General.Verbosity", 0)
gmsh.option.setNumber("General.NumThreads", 8)
fixture_records = []

for case in cases:
    case_dir = out / case["name"]
    if case_dir.exists():
        expected_files = [
            case_dir / "wing.step",
            case_dir / "body.step",
            case_dir / "trailing_edge.json",
            case_dir / "metadata.json",
        ]
        assert all(path.is_file() for path in expected_files), (
            f"Preserve incomplete existing fixture without modifying it: {case_dir}"
        )
        saved_record = json.loads(
            (case_dir / "metadata.json").read_text(encoding="utf-8"))
        assert all(
            hashlib.sha256((case_dir / filename).read_bytes()).hexdigest()
            == saved_record["files"][key]["sha256"]
            for key, filename in [
                ("wing_step", "wing.step"),
                ("body_step", "body.step"),
                ("trailing_edge", "trailing_edge.json"),
            ]
        ), f"Existing fixture hash mismatch; preserve for inspection: {case_dir}"
        fixture_records.append(saved_record)
        continue
    case_dir.mkdir()
    wing_path = case_dir / "wing.step"
    body_path = case_dir / "body.step"
    te_path = case_dir / "trailing_edge.json"
    shutil.copy2(case["wing_source"], wing_path)
    shutil.copy2(te_source, te_path)

    gmsh.clear()
    gmsh.model.add(case["name"] + "_body")
    occ = gmsh.model.occ
    body_half_solids = []
    for center_x, center_y, center_z in case["body_centers"]:
        body = occ.addSphere(center_x, center_y, center_z, 1.0)
        occ.dilate([(3, body)], center_x, center_y, center_z,
                   *case["body_semiaxes"])
        body_copy = occ.copy([(3, body)])[0][1]
        left_box = occ.addBox(
            center_x - 10.0, center_y - 10.0, center_z - 10.0,
            10.0, 20.0, 20.0)
        right_box = occ.addBox(
            center_x, center_y - 10.0, center_z - 10.0,
            10.0, 20.0, 20.0)
        left_halves, _ = occ.intersect(
            [(3, body)], [(3, left_box)], removeObject=True, removeTool=True)
        right_halves, _ = occ.intersect(
            [(3, body_copy)], [(3, right_box)], removeObject=True, removeTool=True)
        body_half_solids.extend(
            entity for entity in left_halves + right_halves if entity[0] == 3)

    occ.synchronize()
    assert len(body_half_solids) == 2 * len(case["body_centers"])
    expected_volume = (
        len(case["body_centers"]) * 4.0 / 3.0 * np.pi
        * np.prod(case["body_semiaxes"]))
    measured_volume = sum(occ.getMass(3, tag) for _, tag in body_half_solids)
    relative_volume_error = abs(measured_volume - expected_volume) / expected_volume
    assert relative_volume_error < 2e-3
    gmsh.write(str(body_path))

    gmsh.clear()
    gmsh.model.add(case["name"] + "_body_reimport")
    imported = gmsh.model.occ.importShapes(
        str(body_path.resolve()), highestDimOnly=True)
    imported_solids = [entity for entity in imported if entity[0] == 3]
    gmsh.model.occ.synchronize()
    imported_volume = sum(
        gmsh.model.occ.getMass(3, tag) for _, tag in imported_solids)
    imported_relative_error = abs(
        imported_volume - expected_volume) / expected_volume
    assert len(imported_solids) == 2 * len(case["body_centers"])
    assert imported_relative_error < 2e-3

    te_data = json.loads(te_path.read_text(encoding="utf-8"))
    wing_metadata = json.loads(
        (case["wing_source"].parent / "metadata.json").read_text(encoding="utf-8"))
    record = {
        "name": case["name"],
        "status": "fixtures_created_and_reimported",
        "builder_executed": False,
        "purpose": case["purpose"],
        "wing_family": case["wing_family"],
        "files": {
            "wing_step": {
                "path": "wing.step",
                "sha256": hashlib.sha256(wing_path.read_bytes()).hexdigest(),
            },
            "body_step": {
                "path": "body.step",
                "sha256": hashlib.sha256(body_path.read_bytes()).hexdigest(),
            },
            "trailing_edge": {
                "path": "trailing_edge.json",
                "sha256": hashlib.sha256(te_path.read_bytes()).hexdigest(),
            },
        },
        "wing_reference": {
            "chord": float(te_data["chord"]),
            "reference_area": float(te_data["reference_area"]),
            "span": float(wing_metadata["geometry"]["span_native"]),
            "normalization_origin": te_data.get("origin", [0.0, 0.0, 0.0]),
            "coordinate_units": "source chord units; chord is 1.0",
            "te_point_count": len(te_data["points"]),
        },
        "body": {
            "centers": [list(center) for center in case["body_centers"]],
            "semiaxes": list(case["body_semiaxes"]),
            "closed_half_solid_count": len(imported_solids),
            "expected_volume": float(expected_volume),
            "measured_volume_before_export": float(measured_volume),
            "measured_volume_after_reimport": float(imported_volume),
            "relative_volume_error_before_export": float(relative_volume_error),
            "relative_volume_error_after_reimport": float(imported_relative_error),
            "multipatch_method": (
                "Each ellipsoid is duplicated and intersected with left/right "
                "half-space boxes at its center-x plane, matching cad_inputs_v1."
            ),
        },
        "source_files": {
            "wing_step_sha256": hashlib.sha256(
                case["wing_source"].read_bytes()).hexdigest(),
            "trailing_edge_sha256": hashlib.sha256(
                te_source.read_bytes()).hexdigest(),
            "generator_sha256": hashlib.sha256(
                Path(__file__).read_bytes()).hexdigest(),
        },
    }
    (case_dir / "metadata.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8")
    fixture_records.append(record)

gmsh.finalize()
manifest = {
    "status": "fixtures_created",
    "builder_executed": False,
    "threads": 8,
    "reference_fixture_root": str(fixture_root),
    "fixtures": fixture_records,
}
manifest_path = out / "manifest.json"
if manifest_path.exists():
    manifest_path = out / "manifest_attempt_002.json"
manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps({
    "status": manifest["status"],
    "builder_executed": manifest["builder_executed"],
    "fixture_names": [row["name"] for row in fixture_records],
    "body_solid_counts": [
        row["body"]["closed_half_solid_count"] for row in fixture_records],
    "maximum_reimport_relative_volume_error": max(
        row["body"]["relative_volume_error_after_reimport"]
        for row in fixture_records),
    "output": str(out),
}, indent=2))
