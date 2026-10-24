"""Round-trip checks for the saved wall VTU exports."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

import export_surface

root = Path(__file__).resolve().parent
run_dir = root / "results" / "wingbody_initial" / "a5"
mesh_path = root / "meshes" / "wingbody02_curv32" / "mesh.npz"
out_dir = root / "results" / "export_v1"
if not out_dir.exists():
    export_surface.export_surface(run_dir, mesh_path, out_dir)

field = np.load(run_dir / "field.npz")
result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
metadata = json.loads((out_dir / "export_metadata.json").read_text(encoding="utf-8"))
surface_path = out_dir / "wall_surface.vtu"
samples_path = out_dir / "raw_gauss_samples.vtu"


def read_vtu(path):
    print(f"Reading {path.name}", flush=True)
    root_attributes = {}
    piece_attributes = {}
    arrays = {}
    for event, element in ET.iterparse(path, events=("start", "end")):
        if event == "start" and element.tag == "VTKFile":
            root_attributes = element.attrib.copy()
        if event == "start" and element.tag == "Piece":
            piece_attributes = element.attrib.copy()
        if event == "end" and element.tag == "DataArray":
            components = int(element.attrib.get("NumberOfComponents", "1"))
            dtype = np.int64 if element.attrib["type"] in ["Int64", "Int32", "UInt8"] else np.float64
            values = np.fromstring(element.text or "", sep=" ", dtype=dtype)
            if components > 1:
                values = values.reshape(-1, components)
            arrays[element.attrib.get("Name", "Points")] = values
            element.clear()
    print(f"Read {path.name}: {len(arrays)} arrays", flush=True)
    return root_attributes, piece_attributes, arrays


surface_root, surface_piece, surface_data = read_vtu(surface_path)
surface_point_data = surface_data
surface_points = surface_data["Points"]
surface_cells = surface_data
surface_cell_data = surface_data
surface_point_values = surface_points.reshape(-1, 3)

assert surface_root["type"] == "UnstructuredGrid"
source_face_ids = surface_cell_data["source_face_id"]
source_point_ids = surface_point_data["source_point_id"]
source_faces = field["faces"][source_face_ids]
assert np.array_equal(surface_point_values, field["points"][source_point_ids])
assert np.array_equal(source_point_ids, np.unique(source_faces.ravel()))
assert np.array_equal(
    source_point_ids[surface_cells["connectivity"].reshape(-1, 6)],
    source_faces)
assert np.array_equal(surface_cells["offsets"], 6 * np.arange(1, len(source_faces) + 1))
assert np.array_equal(surface_cells["types"], np.full(len(source_faces), 22))
assert np.array_equal(surface_cell_data["tag"], field["tags"][source_face_ids])
assert set(np.unique(surface_cell_data["tag"]).tolist()).issubset({10, 20, 30})
assert 100 not in surface_cell_data["tag"]
assert not np.isin(surface_cell_data["tag"], [201, 202]).any()

sample_root, sample_piece, sample_data = read_vtu(samples_path)
sample_point_data = sample_data
sample_cells = sample_data
sample_point_values = sample_data["Points"].reshape(-1, 3)
assert sample_root["type"] == "UnstructuredGrid"
assert np.array_equal(sample_cells["connectivity"], np.arange(len(sample_point_values)))
assert np.array_equal(sample_cells["offsets"], np.arange(1, len(sample_point_values) + 1))
assert np.array_equal(sample_cells["types"], np.ones(len(sample_point_values), dtype=np.int64))

selected = np.isin(field["tags"], [10, 20, 30])
expected_points = field["surface_points"][selected].reshape(-1, 3)
expected_cp = field["cp"][selected].reshape(-1)
expected_velocity = field["surface_velocity"][selected].reshape(-1, 3)
expected_normal = field["surface_normal"][selected].reshape(-1, 3)
expected_weight = field["surface_weight"][selected].reshape(-1)
expected_tags = np.repeat(field["tags"][selected], field["cp"].shape[1])
expected_face_ids = np.repeat(np.flatnonzero(selected), field["cp"].shape[1])
expected_gauss_ids = np.tile(np.arange(field["cp"].shape[1]), np.count_nonzero(selected))
assert np.array_equal(sample_point_values, expected_points)
assert np.array_equal(sample_point_data["rawCp"], expected_cp)
assert np.array_equal(sample_point_data["velocity"], expected_velocity)
assert np.array_equal(sample_point_data["normal"], expected_normal)
assert np.array_equal(sample_point_data["areaweight"], expected_weight)
assert np.array_equal(sample_point_data["tag"], expected_tags)
assert np.array_equal(sample_point_data["source_face_id"], expected_face_ids)
assert np.array_equal(sample_point_data["gauss_index"], expected_gauss_ids)

sample_force_errors = {}
for tag in [10, 20, 30]:
    tag_samples = sample_point_data["tag"] == tag
    if not np.any(tag_samples):
        assert str(tag) not in result["component_loads"]
        continue
    raw_integral = np.sum(
        sample_point_data["rawCp"][tag_samples, None]
        * sample_point_data["normal"][tag_samples]
        * sample_point_data["areaweight"][tag_samples, None], axis=0)
    force_coefficient = raw_integral / metadata["force_reference_area"]
    expected_force = np.asarray(result["component_loads"][str(tag)]["force_coefficient"])
    error = float(np.max(np.abs(force_coefficient - expected_force)))
    assert error <= 1.0e-12
    sample_force_errors[str(tag)] = error

cell_force = surface_cell_data["integral_Cp_n_dS"].reshape(-1, 3)
cell_cp_mean = surface_cell_data["cellcp_area_mean"]
cell_vn_rms = surface_cell_data["raw_vn_rms"]
face_cp = field["cp"][source_face_ids]
face_normal = field["surface_normal"][source_face_ids]
face_weight = field["surface_weight"][source_face_ids]
face_area = face_weight.sum(axis=1)
expected_mean = np.sum(face_cp * face_weight, axis=1) / face_area
expected_vn_rms = np.sqrt(np.sum(
    field["surface_normal_velocity"][source_face_ids]**2 * face_weight,
    axis=1) / face_area)
expected_integral = np.sum(
    face_cp[:, :, None] * face_normal * face_weight[:, :, None], axis=1)
assert np.max(np.abs(cell_cp_mean - expected_mean)) < 1.0e-14
assert np.max(np.abs(cell_vn_rms - expected_vn_rms)) < 1.0e-14
assert np.max(np.abs(cell_force - expected_integral)) < 1.0e-14

validation = {
    "status": "passed",
    "surface_tri6_cells": int(len(source_faces)),
    "surface_unique_points": int(len(source_point_ids)),
    "raw_gauss_vertex_cells": int(len(sample_point_values)),
    "rawCp_exact_roundtrip": True,
    "tri6_connectivity_exact_roundtrip": True,
    "cell_force_integral_max_abs_error": float(np.max(np.abs(
        cell_force - expected_integral))),
    "per_component_sampling_force_max_abs_error": sample_force_errors,
    "output_sha256": {
        "wall_surface.vtu": hashlib.sha256(surface_path.read_bytes()).hexdigest(),
        "raw_gauss_samples.vtu": hashlib.sha256(samples_path.read_bytes()).hexdigest(),
    },
}
(out_dir / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
print(json.dumps(validation, indent=2))

scaled_run_dir = root / "results" / "invariance_v1" / "large_shifted" / "solve"
scaled_mesh_path = root / "results" / "invariance_v1" / "large_shifted" / "mesh.npz"
scaled_out_dir = root / "results" / "export_scaled_v1"
if not scaled_out_dir.exists():
    export_surface.export_surface(scaled_run_dir, scaled_mesh_path, scaled_out_dir)

scaled_field = np.load(scaled_run_dir / "field.npz")
scaled_result = json.loads(
    (scaled_run_dir / "result.json").read_text(encoding="utf-8"))
scaled_metadata = json.loads(
    (scaled_out_dir / "export_metadata.json").read_text(encoding="utf-8"))
scaled_mesh_metadata = json.loads(
    scaled_mesh_path.with_name("mesh.json").read_text(encoding="utf-8"))
scaled_origin = np.asarray(scaled_mesh_metadata["normalization_origin"], dtype=float)
scaled_chord = float(scaled_mesh_metadata["chord"])
scaled_reference_area = float(scaled_mesh_metadata["reference_area"])
assert scaled_metadata["mesh_chord"] == scaled_chord
assert np.array_equal(scaled_metadata["normalization_origin"], scaled_origin)
assert "not chord-normalized" in scaled_metadata["coordinate_units"]
assert "squared mesh coordinate units" in scaled_metadata["areaweight_units"]
assert scaled_reference_area == 4.0e6

base_field = np.load(
    root / "results" / "invariance_v1" / "base" / "solve" / "field.npz")
assert np.array_equal(scaled_field["tags"], base_field["tags"])
assert np.allclose(
    (scaled_field["surface_points"] - scaled_origin) / scaled_chord,
    base_field["surface_points"], rtol=0.0, atol=1.0e-10)
assert np.allclose(
    scaled_field["surface_weight"] / scaled_chord**2,
    base_field["surface_weight"], rtol=1.0e-10, atol=1.0e-12)

scaled_surface_path = scaled_out_dir / "wall_surface.vtu"
scaled_samples_path = scaled_out_dir / "raw_gauss_samples.vtu"
scaled_surface_root, _, scaled_surface_data = read_vtu(scaled_surface_path)
scaled_sample_root, _, scaled_sample_data = read_vtu(scaled_samples_path)
assert scaled_surface_root["type"] == scaled_sample_root["type"] == "UnstructuredGrid"
scaled_source_face_ids = scaled_surface_data["source_face_id"]
scaled_source_faces = scaled_field["faces"][scaled_source_face_ids]
scaled_source_point_ids = scaled_surface_data["source_point_id"]
assert np.array_equal(
    scaled_surface_data["Points"].reshape(-1, 3),
    scaled_field["points"][scaled_source_point_ids])
assert np.array_equal(
    scaled_source_point_ids[
        scaled_surface_data["connectivity"].reshape(-1, 6)],
    scaled_source_faces)
scaled_selected = np.isin(scaled_field["tags"], [10, 20, 30])
assert np.array_equal(
    scaled_sample_data["rawCp"], scaled_field["cp"][scaled_selected].reshape(-1))
assert np.array_equal(
    scaled_sample_data["Points"].reshape(-1, 3),
    scaled_field["surface_points"][scaled_selected].reshape(-1, 3))
assert np.array_equal(
    scaled_sample_data["areaweight"],
    scaled_field["surface_weight"][scaled_selected].reshape(-1))

scaled_force_errors = {}
for tag in [10, 20, 30]:
    selected_samples = scaled_sample_data["tag"] == tag
    if not np.any(selected_samples):
        assert str(tag) not in scaled_result["component_loads"]
        continue
    direct_integral = np.sum(
        scaled_sample_data["rawCp"][selected_samples, None]
        * scaled_sample_data["normal"][selected_samples]
        * scaled_sample_data["areaweight"][selected_samples, None], axis=0)
    expected_integral = (
        np.asarray(scaled_result["component_loads"][str(tag)]["force_coefficient"])
        * scaled_reference_area)
    force_error = float(np.max(np.abs(direct_integral - expected_integral)))
    assert force_error <= 1.0e-12 * scaled_reference_area
    scaled_force_errors[str(tag)] = {
        "integral_force_max_abs_error_mesh_units_squared": force_error,
        "force_coefficient_max_abs_error": force_error / scaled_reference_area,
    }

scaled_validation = {
    "status": "passed",
    "case": "invariance_v1/large_shifted",
    "mesh_chord": scaled_chord,
    "normalization_origin": scaled_origin.tolist(),
    "reference_area_mesh_units_squared": scaled_reference_area,
    "coordinates_match_base_after_normalization": True,
    "areaweights_match_base_after_chord_squared_normalization": True,
    "rawCp_exact_roundtrip": True,
    "tri6_connectivity_exact_roundtrip": True,
    "per_component_force_checks": scaled_force_errors,
    "output_sha256": {
        "wall_surface.vtu": hashlib.sha256(
            scaled_surface_path.read_bytes()).hexdigest(),
        "raw_gauss_samples.vtu": hashlib.sha256(
            scaled_samples_path.read_bytes()).hexdigest(),
    },
}
(scaled_out_dir / "scaled_validation.json").write_text(
    json.dumps(scaled_validation, indent=2), encoding="utf-8")
print(json.dumps(scaled_validation, indent=2))
