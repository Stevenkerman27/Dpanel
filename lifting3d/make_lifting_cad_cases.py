"""Create closed, reusable STEP fixtures for full-volume lifting CAD tests."""
import hashlib
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "audit" / "runtime" / "python"))
import gmsh

OUT = ROOT / "results" / "cad_inputs_v1"
parser = argparse.ArgumentParser()
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()
assert not OUT.exists() or args.resume, f"Preserve existing CAD fixtures: {OUT}"
OUT.mkdir(parents=True, exist_ok=args.resume)
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
NCPU = 8
SPAN = 4.0
ROOT_CHORD = 1.0
BODY_LENGTH = 4.0
BODY_RADIUS = 0.35
BODY_AFT = 1.5
BODY_CENTER_X = BODY_AFT - BODY_LENGTH / 2.0
BODY_AXES = [BODY_LENGTH / 2.0, BODY_RADIUS, BODY_RADIUS]
SECTION_X = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, 81)))

CASES = [
    dict(name="naca0012_wing", camber=0.00, position=0.00, taper=1.00, sweep=0.0, body=False),
    dict(name="naca4412_wing", camber=0.04, position=0.40, taper=1.00, sweep=0.0, body=False),
    dict(name="naca0012_wing_body", camber=0.00, position=0.00, taper=1.00, sweep=0.0, body=True),
    dict(name="naca4412_wing_body", camber=0.04, position=0.40, taper=1.00, sweep=0.0, body=True),
    dict(name="naca4412_scaled_translated", camber=0.04, position=0.40, taper=1.00, sweep=0.0,
         body=False, scale=1000.0, translation=[100.0, -200.0, 50.0]),
    dict(name="naca4412_taper060_sweep15", camber=0.04, position=0.40, taper=0.60, sweep=15.0, body=False),
]


def profile_coordinates(camber, position):
    x = SECTION_X
    thickness = 5.0 * 0.12 * (
        0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2
        + 0.2843 * x**3 - 0.1036 * x**4
    )
    mean = np.zeros_like(x)
    slope = np.zeros_like(x)
    if camber > 0.0:
        forward = x < position
        mean[forward] = camber / position**2 * (2.0 * position * x[forward] - x[forward]**2)
        slope[forward] = 2.0 * camber / position**2 * (position - x[forward])
        aft = ~forward
        mean[aft] = camber / (1.0 - position)**2 * (
            (1.0 - 2.0 * position) + 2.0 * position * x[aft] - x[aft]**2
        )
        slope[aft] = 2.0 * camber / (1.0 - position)**2 * (position - x[aft])
    angle = np.arctan(slope)
    upper = np.column_stack([x - thickness * np.sin(angle), mean + thickness * np.cos(angle)])
    lower = np.column_stack([x + thickness * np.sin(angle), mean - thickness * np.cos(angle)])
    upper[0] = lower[0] = [0.0, 0.0]
    upper[-1] = lower[-1] = [1.0, 0.0]
    return upper, lower


def add_section_wire(occ, y, chord, leading_x, camber, position):
    upper, lower = profile_coordinates(camber, position)
    upper_tags = [occ.addPoint(float(leading_x + chord * x), float(y), float(chord * z))
                  for x, z in upper]
    lower_tags = [occ.addPoint(float(leading_x + chord * x), float(y), float(chord * z))
                  for x, z in lower[::-1]]
    upper_edge = occ.addSpline(upper_tags)
    lower_edge = occ.addSpline(lower_tags)
    return occ.addWire([upper_edge, lower_edge])


def begin_model(name):
    gmsh.clear()
    gmsh.model.add(name)
    gmsh.option.setNumber("General.NumThreads", NCPU)


def validate_step(path, expected_solids):
    begin_model("validate_" + path.stem)
    imported = gmsh.model.occ.importShapes(str(path.resolve()), highestDimOnly=True)
    gmsh.model.occ.synchronize()
    solids = [entity for entity in imported if entity[0] == 3]
    assert len(solids) == expected_solids, (
        f"{path.name}: expected {expected_solids} closed STEP solids, imported {len(solids)}"
    )
    volumes = [float(gmsh.model.occ.getMass(*entity)) for entity in solids]
    assert np.isfinite(volumes).all() and min(volumes) > 0.0
    face_counts, edge_counts = [], []
    for solid in solids:
        faces = gmsh.model.getBoundary([solid], combined=False, oriented=False)
        assert faces and all(dim == 2 for dim, _ in faces)
        face_counts.append(len(faces))
        edges = gmsh.model.getBoundary(faces, combined=False, oriented=False)
        edge_counts.append(len(edges))
        assert edges and all(dim == 1 for dim, _ in edges)
    return dict(
        imported_closed_solid_count=len(solids),
        imported_solid_volumes=volumes,
        boundary_face_counts=face_counts,
        boundary_edge_occurrence_counts=edge_counts,
        positive_volume=True,
        status="step_reimport_passed",
    )


records = []
manifest = dict(
    status="running",
    ncpu_requested=NCPU,
    case_order=[item["name"] for item in CASES],
    cases=records,
    source_sha256=SOURCE_SHA256,
)
gmsh.initialize()
gmsh.option.setNumber("General.NumThreads", NCPU)
try:
    for case in CASES:
        started = time.perf_counter()
        target = OUT / case["name"]
        target.mkdir(exist_ok=args.resume)
        saved_metadata = target / "metadata.json"
        if args.resume and saved_metadata.is_file() and (target / "trailing_edge.json").is_file():
            record = json.loads(saved_metadata.read_text(encoding="utf-8"))
            old_metadata = target / "metadata_attempt_001.json"
            if not old_metadata.exists():
                shutil.copy2(saved_metadata, old_metadata)
            record["wing_step_validation"] = validate_step(target / "wing.step", 1)
            if case["body"]:
                record["body_step_validation"] = validate_step(target / "body.step", 2)
                expected_body_volume = 4.0 / 3.0 * np.pi * np.prod(BODY_AXES)
                measured_body_volume = sum(record["body_step_validation"]["imported_solid_volumes"])
                record["body_step_validation"]["expected_ellipsoid_volume"] = float(expected_body_volume)
                record["body_step_validation"]["relative_ellipsoid_volume_error"] = float(
                    abs(measured_body_volume - expected_body_volume) / expected_body_volume)
                assert record["body_step_validation"]["relative_ellipsoid_volume_error"] < 0.002
                old_assembly = target / "assembly_attempt_001.step"
                assembly_path = target / "assembly.step"
                if assembly_path.is_file() and not old_assembly.exists():
                    shutil.copy2(assembly_path, old_assembly)
                begin_model("assembly_" + case["name"])
                gmsh.model.occ.importShapes(
                    str((target / "wing.step").resolve()), highestDimOnly=True)
                gmsh.model.occ.importShapes(
                    str((target / "body.step").resolve()), highestDimOnly=True)
                gmsh.model.occ.synchronize()
                gmsh.write(str(assembly_path))
                record["assembly_step_validation"] = validate_step(assembly_path, 3)
                assembly_body_volumes = record["assembly_step_validation"]["imported_solid_volumes"][1:]
                assembly_body_volume_error = abs(
                    sum(assembly_body_volumes) - expected_body_volume) / expected_body_volume
                record["assembly_step_validation"]["relative_ellipsoid_volume_error"] = float(
                    assembly_body_volume_error)
                assert assembly_body_volume_error < 0.002
                record["files"]["body_step"]["sha256"] = hashlib.sha256(
                    (target / "body.step").read_bytes()).hexdigest()
                record["files"]["assembly_step"]["sha256"] = hashlib.sha256(
                    assembly_path.read_bytes()).hexdigest()
            record["files"]["wing_step"]["sha256"] = hashlib.sha256(
                (target / "wing.step").read_bytes()).hexdigest()
            record["trailing_edge_sha256"] = hashlib.sha256(
                (target / "trailing_edge.json").read_bytes()).hexdigest()
            record["source_sha256"] = SOURCE_SHA256
            saved_metadata.write_text(json.dumps(record, indent=2), encoding="utf-8")
            records.append(record)
            continue
        previous_failure = target / "failure.json"
        archived_failure = target / "failure_attempt_001.json"
        if args.resume and previous_failure.is_file() and not archived_failure.exists():
            shutil.copy2(previous_failure, archived_failure)
        scale = float(case.get("scale", 1.0))
        translation = np.asarray(case.get("translation", [0.0, 0.0, 0.0]), dtype=float)
        reference_origin = translation.tolist()
        span_native = SPAN * scale
        root_chord_native = ROOT_CHORD * scale
        reference_area_native = SPAN * ROOT_CHORD * (1.0 + case["taper"]) / 2.0 * scale**2
        try:
            begin_model("wing_" + case["name"])
            occ = gmsh.model.occ
            if case["taper"] == 1.0 and case["sweep"] == 0.0:
                section = add_section_wire(occ, -SPAN / 2.0, ROOT_CHORD,
                                           0.0, case["camber"], case["position"])
                profile = occ.addPlaneSurface([section])
                wing_entities = occ.extrude([(2, profile)], 0.0, SPAN, 0.0)
                wing_solids = [entity for entity in wing_entities if entity[0] == 3]
            else:
                sections = []
                for y in [-SPAN / 2.0, 0.0, SPAN / 2.0]:
                    fraction = abs(2.0 * y / SPAN)
                    chord = ROOT_CHORD * (1.0 + (case["taper"] - 1.0) * fraction)
                    leading_x = abs(y) * np.tan(np.deg2rad(case["sweep"])) + 0.25 * (ROOT_CHORD - chord)
                    sections.append(add_section_wire(occ, y, chord, leading_x,
                                                      case["camber"], case["position"]))
                loft = occ.addThruSections(sections, makeSolid=True, makeRuled=True)
                wing_solids = [entity for entity in loft if entity[0] == 3]
            assert len(wing_solids) == 1, f"Wing OCC construction returned {len(wing_solids)} solids"
            if case.get("scale", 1.0) != 1.0 or np.any(translation != 0.0):
                occ.dilate(wing_solids, 0.0, 0.0, 0.0, scale, scale, scale)
                occ.translate(wing_solids, *translation)
            gmsh.model.occ.synchronize()
            wing_path = target / "wing.step"
            gmsh.write(str(wing_path))
            wing_validation = validate_step(wing_path, 1)

            body_path = None
            body_validation = None
            assembly_validation = None
            if case["body"]:
                begin_model("body_" + case["name"])
                occ = gmsh.model.occ
                body = occ.addSphere(BODY_CENTER_X, 0.0, 0.0, 1.0)
                occ.dilate([(3, body)], BODY_CENTER_X, 0.0, 0.0, *BODY_AXES)
                body_copy = occ.copy([(3, body)])[0][1]
                left_box = occ.addBox(-10.0, -10.0, -10.0,
                                      BODY_CENTER_X + 10.0, 20.0, 20.0)
                right_box = occ.addBox(BODY_CENTER_X, -10.0, -10.0, 10.0, 20.0, 20.0)
                left_halves, _ = occ.intersect(
                    [(3, body)], [(3, left_box)], removeObject=True, removeTool=True)
                right_halves, _ = occ.intersect(
                    [(3, body_copy)], [(3, right_box)], removeObject=True, removeTool=True)
                body_solids = [entity for entity in left_halves + right_halves if entity[0] == 3]
                assert len(body_solids) == 2, (
                    f"Multipatch ellipsoid split produced {len(body_solids)} solids"
                )
                gmsh.model.occ.synchronize()
                body_path = target / "body.step"
                gmsh.write(str(body_path))
                body_validation = validate_step(body_path, 2)
                expected_body_volume = 4.0 / 3.0 * np.pi * np.prod(BODY_AXES)
                measured_body_volume = sum(body_validation["imported_solid_volumes"])
                body_validation["expected_ellipsoid_volume"] = float(expected_body_volume)
                body_validation["relative_ellipsoid_volume_error"] = float(
                    abs(measured_body_volume - expected_body_volume) / expected_body_volume)
                assert body_validation["relative_ellipsoid_volume_error"] < 0.002

                begin_model("assembly_" + case["name"])
                wing_import = gmsh.model.occ.importShapes(str(wing_path.resolve()), highestDimOnly=True)
                body_import = gmsh.model.occ.importShapes(str(body_path.resolve()), highestDimOnly=True)
                gmsh.model.occ.synchronize()
                assert len([entity for entity in wing_import + body_import if entity[0] == 3]) == 3
                assembly_path = target / "assembly.step"
                gmsh.write(str(assembly_path))
                assembly_validation = validate_step(assembly_path, 3)

            trailing_points = []
            for y in np.linspace(-SPAN / 2.0, SPAN / 2.0, 41):
                fraction = abs(2.0 * y / SPAN)
                chord = ROOT_CHORD * (1.0 + (case["taper"] - 1.0) * fraction)
                leading_x = abs(y) * np.tan(np.deg2rad(case["sweep"])) + 0.25 * (ROOT_CHORD - chord)
                point = np.array([leading_x + chord, y, 0.0]) * scale + translation
                trailing_points.append(point.tolist())
            trailing_data = dict(
                points=trailing_points,
                chord=root_chord_native,
                reference_area=reference_area_native,
                origin=reference_origin,
            )
            (target / "trailing_edge.json").write_text(
                json.dumps(trailing_data, indent=2), encoding="utf-8")

            files = dict(wing_step=dict(path="wing.step",
                                        sha256=hashlib.sha256(wing_path.read_bytes()).hexdigest()))
            if body_path is not None:
                files["body_step"] = dict(
                    path="body.step", sha256=hashlib.sha256(body_path.read_bytes()).hexdigest())
                files["assembly_step"] = dict(
                    path="assembly.step",
                    sha256=hashlib.sha256((target / "assembly.step").read_bytes()).hexdigest())
            record = dict(
                name=case["name"],
                status="fixtures_created_and_reimported",
                files=files,
                wing_step_validation=wing_validation,
                body_step_validation=body_validation,
                assembly_step_validation=assembly_validation,
                trailing_edge_sha256=hashlib.sha256(
                    (target / "trailing_edge.json").read_bytes()).hexdigest(),
                geometry=dict(
                    airfoil="NACA 0012" if case["camber"] == 0.0 else "NACA 4412",
                    airfoil_construction="81 cosine-spaced analytic NACA section samples, OCC interpolating splines; closed TE x/c=1,z/c=0",
                    wing_solid="Closed finite-thickness section extruded or three-section ruled loft",
                    span_native=span_native,
                    root_chord_native=root_chord_native,
                    tip_chord_native=root_chord_native * case["taper"],
                    thickness_ratio=0.12,
                    camber_ratio=case["camber"],
                    camber_position=case["position"],
                    taper_ratio=case["taper"],
                    quarter_chord_sweep_deg=case["sweep"],
                    body=(dict(shape="two half-ellipsoid closed OCC solids, split at x=center for multipatch STEP",
                               length=BODY_LENGTH * scale, radius=BODY_RADIUS * scale,
                               aft_x=BODY_AFT * scale + translation[0],
                               center=[BODY_CENTER_X * scale + translation[0],
                                       translation[1], translation[2]],
                               semi_axes=(np.asarray(BODY_AXES) * scale).tolist())
                           if case["body"] else None),
                    reference_area_native=reference_area_native,
                    normalization_origin_native=reference_origin,
                    native_transform=dict(scale=scale, translation=translation.tolist()),
                    components_exported_separately=bool(case["body"]),
                    assembly_step_keeps_components_unfused=bool(case["body"]),
                    ncpu_requested=NCPU,
                ),
                source_sha256=SOURCE_SHA256,
                elapsed_seconds=time.perf_counter() - started,
            )
            (target / "metadata.json").write_text(
                json.dumps(record, indent=2), encoding="utf-8")
        except Exception as error:
            record = dict(
                name=case["name"], status="failed_preserved",
                exception_type=type(error).__name__, message=str(error),
                existing_files=[path.name for path in target.iterdir()],
                source_sha256=SOURCE_SHA256,
                elapsed_seconds=time.perf_counter() - started,
            )
            (target / "failure.json").write_text(
                json.dumps(record, indent=2), encoding="utf-8")
        records.append(record)
        manifest = dict(
            status="running",
            ncpu_requested=NCPU,
            case_order=[item["name"] for item in CASES],
            cases=records,
            source_sha256=SOURCE_SHA256,
        )
        (OUT / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8")
finally:
    if gmsh.isInitialized():
        gmsh.finalize()

manifest["status"] = (
    "all_step_fixtures_reimport_passed"
    if len(records) == len(CASES)
    and all(row["status"] == "fixtures_created_and_reimported" for row in records)
    else "completed_with_preserved_failures"
)
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps(manifest, indent=2))
