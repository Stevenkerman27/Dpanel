"""Export saved P2 surface fields to ParaView-readable ASCII VTU files."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

root = Path(__file__).resolve().parent


def array_text(values, integer=False):
    values = np.asarray(values)
    buffer = []
    if values.ndim == 1:
        values = values[:, None]
    for row in values:
        if integer:
            buffer.append(" ".join(str(int(value)) for value in row))
        else:
            buffer.append(" ".join(format(float(value), ".17g") for value in row))
    return "\n".join(buffer)


def add_data_array(parent, name, values, vtk_type, components=1, integer=False):
    element = ET.SubElement(parent, "DataArray", {
        "type": vtk_type,
        "Name": name,
        "NumberOfComponents": str(components),
        "format": "ascii",
    })
    element.text = array_text(values, integer=integer)
    return element


def write_unstructured_grid(path, points, connectivity, offsets, cell_types,
                            point_data, cell_data):
    vtk = ET.Element("VTKFile", {
        "type": "UnstructuredGrid",
        "version": "0.1",
        "byte_order": "LittleEndian",
    })
    grid = ET.SubElement(vtk, "UnstructuredGrid")
    piece = ET.SubElement(grid, "Piece", {
        "NumberOfPoints": str(len(points)),
        "NumberOfCells": str(len(cell_types)),
    })
    point_section = ET.SubElement(piece, "Points")
    add_data_array(point_section, "Points", points, "Float64", components=3)
    cell_section = ET.SubElement(piece, "Cells")
    add_data_array(cell_section, "connectivity", connectivity, "Int64", integer=True)
    add_data_array(cell_section, "offsets", offsets, "Int64", integer=True)
    add_data_array(cell_section, "types", cell_types, "UInt8", integer=True)
    point_section = ET.SubElement(piece, "PointData")
    for name, values, vtk_type, components, integer in point_data:
        add_data_array(point_section, name, values, vtk_type,
                       components=components, integer=integer)
    cell_section = ET.SubElement(piece, "CellData")
    for name, values, vtk_type, components, integer in cell_data:
        add_data_array(cell_section, name, values, vtk_type,
                       components=components, integer=integer)
    ET.indent(vtk, space="  ")
    ET.ElementTree(vtk).write(path, encoding="utf-8", xml_declaration=True)


def export_surface(run_dir, mesh_path, out_dir, include_wake=False):
    run_dir = Path(run_dir)
    mesh_path = Path(mesh_path)
    out_dir = Path(out_dir)
    assert not out_dir.exists(), f"Refusing to overwrite existing output directory: {out_dir}"
    out_dir.mkdir(parents=True)

    field_path = run_dir / "field.npz"
    result_path = run_dir / "result.json"
    metadata_path = mesh_path.with_name("mesh.json")
    field_bytes = field_path.read_bytes()
    result_bytes = result_path.read_bytes()
    mesh_bytes = mesh_path.read_bytes()
    exporter_bytes = Path(__file__).read_bytes()
    field_sha256 = hashlib.sha256(field_bytes).hexdigest()
    result_sha256 = hashlib.sha256(result_bytes).hexdigest()
    mesh_sha256 = hashlib.sha256(mesh_bytes).hexdigest()
    exporter_sha256 = hashlib.sha256(exporter_bytes).hexdigest()

    field = np.load(field_path)
    result = json.loads(result_bytes)
    mesh_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    mesh_chord = float(mesh_metadata["chord"])
    normalization_origin = np.asarray(
        mesh_metadata.get("normalization_origin", [0.0, 0.0, 0.0]), dtype=float)
    assert np.isfinite(mesh_chord) and mesh_chord > 0.0
    assert normalization_origin.shape == (3,) and np.isfinite(normalization_origin).all()
    assert result.get("converged", False)
    assert result["mesh_sha256"] == mesh_sha256
    mesh_metadata_sha256 = mesh_metadata["mesh_sha256"]
    transformed_mesh_metadata = (
        mesh_metadata.get("transform_source_mesh_sha256") == mesh_metadata_sha256)
    assert mesh_metadata_sha256 == mesh_sha256 or transformed_mesh_metadata
    assert np.array_equal(field["points"], np.load(mesh_path)["points"])

    faces = field["faces"]
    tags = field["tags"]
    surface_points = field["surface_points"]
    cp = field["cp"]
    velocity = field["surface_velocity"]
    normal = field["surface_normal"]
    area_weight = field["surface_weight"]
    raw_vn = field["surface_normal_velocity"]
    assert faces.ndim == 2 and faces.shape[1] == 6
    assert tags.shape == (len(faces),)
    assert surface_points.shape[:2] == cp.shape == raw_vn.shape == area_weight.shape
    assert surface_points.shape[:2] == velocity.shape[:2] == normal.shape[:2]
    assert surface_points.shape[2] == velocity.shape[2] == normal.shape[2] == 3
    assert np.isfinite(surface_points).all() and np.isfinite(cp).all()
    assert np.isfinite(velocity).all() and np.isfinite(normal).all()
    assert np.isfinite(area_weight).all() and np.all(area_weight > 0.0)
    assert np.max(np.abs(np.linalg.norm(normal, axis=2) - 1.0)) < 1.0e-12
    reconstructed_vn = np.einsum("fqi,fqi->fq", velocity, normal)
    assert np.max(np.abs(reconstructed_vn - raw_vn)) < 1.0e-12

    wall_mask = np.isin(tags, [10, 20, 30])
    selected_mask = wall_mask.copy()
    if include_wake:
        selected_mask |= np.isin(tags, [201, 202])
    selected_indices = np.flatnonzero(selected_mask)
    selected_faces = faces[selected_indices]
    selected_tags = tags[selected_indices]
    selected_cp = cp[selected_indices]
    selected_velocity = velocity[selected_indices]
    selected_normal = normal[selected_indices]
    selected_area_weight = area_weight[selected_indices]
    selected_raw_vn = raw_vn[selected_indices]

    face_area = selected_area_weight.sum(axis=1)
    cell_cp_area_mean = np.sum(selected_cp * selected_area_weight, axis=1) / face_area
    cell_raw_vn_rms = np.sqrt(
        np.sum(selected_raw_vn**2 * selected_area_weight, axis=1) / face_area)
    face_force_integral = np.sum(
        selected_cp[:, :, None] * selected_normal * selected_area_weight[:, :, None], axis=1)

    reference_area = float(mesh_metadata["reference_area"])
    force_checks = {}
    for tag in [10, 20, 30]:
        selected_tag = selected_tags == tag
        if not np.any(selected_tag):
            assert str(tag) not in result["component_loads"]
            continue
        force_integral = face_force_integral[selected_tag].sum(axis=0)
        force_coefficient = force_integral / reference_area
        result_force = np.asarray(result["component_loads"][str(tag)]["force_coefficient"])
        max_error = float(np.max(np.abs(force_coefficient - result_force)))
        assert max_error <= 1.0e-12, f"Tag {tag} force differs from result.json by {max_error:.3e}"
        force_checks[str(tag)] = {
            "surface_integral_Cp_n_dS": force_integral.tolist(),
            "reference_area": reference_area,
            "integral_over_reference_area": force_coefficient.tolist(),
            "result_json_force_coefficient": result_force.tolist(),
            "max_abs_error": max_error,
        }

    selected_nodes = np.unique(selected_faces.ravel())
    node_map = np.full(len(field["points"]), -1, dtype=np.int64)
    node_map[selected_nodes] = np.arange(len(selected_nodes), dtype=np.int64)
    surface_connectivity = node_map[selected_faces].reshape(-1)
    surface_offsets = 6 * np.arange(1, len(selected_faces) + 1, dtype=np.int64)
    surface_types = np.full(len(selected_faces), 22, dtype=np.uint8)
    surface_point_data = [
        ("source_point_id", selected_nodes, "Int64", 1, True),
    ]
    surface_cell_data = [
        ("cellcp_area_mean", cell_cp_area_mean, "Float64", 1, False),
        ("tag", selected_tags, "Int32", 1, True),
        ("raw_vn_rms", cell_raw_vn_rms, "Float64", 1, False),
        ("integral_Cp_n_dS", face_force_integral, "Float64", 3, False),
        ("source_face_id", selected_indices, "Int64", 1, True),
    ]
    surface_path = out_dir / "wall_surface.vtu"
    write_unstructured_grid(
        surface_path, field["points"][selected_nodes], surface_connectivity,
        surface_offsets, surface_types, surface_point_data, surface_cell_data)

    sample_points = surface_points[selected_indices].reshape(-1, 3)
    sample_count = len(sample_points)
    sample_connectivity = np.arange(sample_count, dtype=np.int64)
    sample_offsets = np.arange(1, sample_count + 1, dtype=np.int64)
    sample_types = np.ones(sample_count, dtype=np.uint8)
    sample_tags = np.repeat(selected_tags, cp.shape[1])
    sample_face_ids = np.repeat(selected_indices, cp.shape[1])
    sample_gauss_ids = np.tile(np.arange(cp.shape[1], dtype=np.int32), len(selected_indices))
    sample_point_data = [
        ("rawCp", selected_cp.reshape(-1), "Float64", 1, False),
        ("velocity", selected_velocity.reshape(-1, 3), "Float64", 3, False),
        ("normal", selected_normal.reshape(-1, 3), "Float64", 3, False),
        ("areaweight", selected_area_weight.reshape(-1), "Float64", 1, False),
        ("tag", sample_tags, "Int32", 1, True),
        ("source_face_id", sample_face_ids, "Int64", 1, True),
        ("gauss_index", sample_gauss_ids, "Int32", 1, True),
    ]
    sample_path = out_dir / "raw_gauss_samples.vtu"
    write_unstructured_grid(
        sample_path, sample_points, sample_connectivity, sample_offsets,
        sample_types, sample_point_data, [])

    sample_force_checks = {}
    for tag in [10, 20, 30]:
        selected_tag = sample_tags == tag
        if not np.any(selected_tag):
            continue
        sample_integral = np.sum(
            sample_point_data[0][1][selected_tag, None]
            * sample_point_data[2][1][selected_tag]
            * sample_point_data[3][1][selected_tag, None], axis=0)
        sample_force = sample_integral / reference_area
        result_force = np.asarray(result["component_loads"][str(tag)]["force_coefficient"])
        sample_error = float(np.max(np.abs(sample_force - result_force)))
        assert sample_error <= 1.0e-12
        sample_force_checks[str(tag)] = {
            "integral_Cp_n_dS": sample_integral.tolist(),
            "integral_over_reference_area": sample_force.tolist(),
            "result_json_force_coefficient": result_force.tolist(),
            "max_abs_error": sample_error,
        }

    metadata = {
        "status": "exported_and_integrals_verified",
        "run": str(run_dir),
        "source_mesh": str(mesh_path),
        "source_field": str(field_path),
        "source_result": str(result_path),
        "mesh_sha256": mesh_sha256,
        "field_sha256": field_sha256,
        "result_json_sha256": result_sha256,
        "exporter_sha256": exporter_sha256,
        "mesh_metadata_sha256": mesh_metadata_sha256,
        "mesh_metadata_hash_relation": (
            "current_mesh" if mesh_metadata_sha256 == mesh_sha256
            else "declared_transform_source_mesh"
        ),
        "include_wake": bool(include_wake),
        "selected_tags": sorted(int(tag) for tag in np.unique(selected_tags)),
        "excluded_tags": sorted(int(tag) for tag in np.unique(tags[~selected_mask])),
        "normal_direction": "Outward from fluid domain, as stored by the oriented P2 surface evaluator; on solid walls this points into the solid.",
        "mesh_chord": mesh_chord,
        "normalization_origin": normalization_origin.tolist(),
        "coordinate_units": (
            f"Saved mesh coordinate units; coordinates are exported unchanged, "
            f"not chord-normalized. chord={mesh_chord:.17g} mesh units; "
            f"normalization_origin={normalization_origin.tolist()} mesh units."
        ),
        "cp_units": "Dimensionless pressure coefficient Cp, exported unchanged.",
        "velocity_units": "Saved nondimensional velocity, exported unchanged.",
        "normal_units": "Unit outward fluid-domain normal.",
        "areaweight_units": (
            f"Saved quadrature dS in squared mesh coordinate units; "
            f"mesh chord={mesh_chord:.17g} mesh units."
        ),
        "cellcp_area_mean": "Area-weighted Cp mean for visualization only; not used to construct the force integral.",
        "integral_Cp_n_dS": "Per-face sum_q Cp_q * n_q * areaweight_q; three components, independently summed from raw quadrature.",
        "force_reference_area": reference_area,
        "surface_cells": int(len(selected_faces)),
        "surface_points": int(len(selected_nodes)),
        "gauss_vertex_cells": int(sample_count),
        "per_component_force_checks": force_checks,
        "raw_sample_force_checks": sample_force_checks,
        "outputs": {
            "wall_surface_vtu": {
                "file": surface_path.name,
                "sha256": hashlib.sha256(surface_path.read_bytes()).hexdigest(),
            },
            "raw_gauss_samples_vtu": {
                "file": sample_path.name,
                "sha256": hashlib.sha256(sample_path.read_bytes()).hexdigest(),
            },
        },
    }
    (out_dir / "export_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path,
                        default=root / "results" / "wingbody_initial" / "a5")
    parser.add_argument("--mesh", type=Path,
                        default=root / "meshes" / "wingbody02_curv32" / "mesh.npz")
    parser.add_argument("--out", type=Path, default=root / "results" / "export_v1")
    parser.add_argument("--include-wake", action="store_true")
    args = parser.parse_args()
    metadata = export_surface(args.run, args.mesh, args.out, args.include_wake)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
