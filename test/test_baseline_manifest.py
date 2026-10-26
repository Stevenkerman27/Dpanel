"""Baseline manifest digests: canonical, dtype-stable and tamper-evident."""
import json
from pathlib import Path

import numpy as np

from panel.baseline_manifest import (
    array_digest, build_manifest, code_digest, file_digest, loads_section,
    pressure_section, solve_state_digest, surface_digests, verify_manifest,
)


def fake_fixture():
    return {
        "coordinates": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
                        + [[0.5, 0.0, 0.5]],
        "faces": [[0, 1, 2]],
        "wake_faces": [[0, 1, 3]],
        "trailing_edges": [[0, 1]],
        "component_ids": [30],
        "freestream": [1.0, 0.0, 0.0],
        "moment_reference": [0.0, 0.0, 0.0],
        "reference_area": 1.0,
        "reference_chord": 1.0,
        "density": 1.0,
    }


def fake_result():
    return {
        "mu_half": np.array([0.1, 0.2]),
        "gamma_half": np.array([0.3]),
        "sigma_half": np.array([-0.4]),
        "Cp_raw_half": np.array([-0.5, 0.25]),
        "panel_areas_half": np.array([0.5]),
        "component_forces": [np.array([0.0, 0.0, 0.0]),
                             np.array([0.0, 0.0, 0.0]),
                             np.array([0.01, 0.0, 0.02])],
        "component_moments": [np.zeros(3), np.zeros(3), np.zeros(3)],
        "force": np.array([0.01, 0.0, 0.02]),
        "moment": np.zeros(3),
        "CL": 0.5,
        "CD_pressure": 0.01,
        "CM": np.zeros(3),
    }


def test_array_digest_is_dtype_normalized_but_shape_sensitive():
    integer32 = np.array([1, 2, 3], dtype=np.int32)
    integer64 = np.array([1, 2, 3], dtype=np.int64)
    assert array_digest(integer32) == array_digest(integer64)
    # Exactly representable reals: the float32 payload widens without change.
    assert array_digest(np.array([0.5, 1.5], np.float32)) == array_digest(
        np.array([0.5, 1.5], np.float64))
    assert array_digest(np.array([1.0, 2.0])) != array_digest(
        np.array([[1.0], [2.0]]))
    assert array_digest(np.array([1.0, 2.0])) != array_digest(
        np.array([1.0, 2.5]))


def test_surface_digests_cover_present_fields_only():
    digests = surface_digests(fake_fixture())
    assert "coordinates" in digests and "faces" in digests
    assert "cut_edges" not in digests and "upper_te_faces" not in digests
    # A changed node invalidates the geometry digest only.
    changed = fake_fixture()
    changed["coordinates"][0][2] = 1e-9
    other = surface_digests(changed)
    assert other["coordinates"] != digests["coordinates"]
    assert other["faces"] == digests["faces"]


def test_file_digest_reads_bytes(tmp_path):
    path = tmp_path / "surface.json"
    path.write_text("{}", encoding="utf-8")
    first = file_digest(path)
    path.write_text("{} ", encoding="utf-8")
    assert file_digest(path) != first


def test_code_digest_is_stable_and_counts_panel_sources():
    first = code_digest()
    assert first == code_digest()
    assert first["files"] >= 5 and len(first["digest"]) == 64


def test_manifest_round_trip_and_tamper_detection(tmp_path):
    surface_path = tmp_path / "surface.json"
    fixture = fake_fixture()
    surface_path.write_text(json.dumps(fixture), encoding="utf-8")
    result = fake_result()
    manifest = build_manifest(
        surface_path=surface_path, fixture=fixture, result=result,
        residual_absolute=1e-15, residual_relative=1e-16, case_id="unit")
    assert manifest["schema"] == "dpanel.baseline_manifest.v1"
    assert manifest["topology"]["wall_faces"] == 1
    assert manifest["topology"]["circulation_unknowns"] == 1
    assert manifest["topology"]["independent_unknowns"] == 3
    assert manifest["flow"]["alpha_deg"] == 0.0
    assert manifest["physical_accuracy_accepted"] is False
    assert verify_manifest(manifest, surface_path=surface_path,
                           fixture=fixture, result=result) == []
    # A different solution state is caught.
    tampered = fake_result()
    tampered["mu_half"] = tampered["mu_half"] + 1e-12
    problems = verify_manifest(manifest, surface_path=surface_path,
                               fixture=fixture, result=tampered)
    assert problems == ["state:mu_half"]
    # A different fixture field is caught even with the same file.
    moved = fake_fixture()
    moved["reference_area"] = 2.0
    problems = verify_manifest(manifest, surface_path=surface_path,
                               fixture=moved, result=result)
    assert problems == ["surface_field:reference_area"]
    # Rewriting the file bytes is caught by the whole-file digest.
    surface_path.write_text(json.dumps(fixture) + "\n", encoding="utf-8")
    assert "fixture_digest" in verify_manifest(
        manifest, surface_path=surface_path, fixture=fixture, result=result)


def test_loads_section_reports_every_component():
    loads = loads_section(fake_result())
    assert set(loads["component"]) == {"10", "20", "30"}
    assert loads["CL"] == 0.5
    assert pressure_section(fake_result())["Cp_raw"]["count"] == 2
    state = solve_state_digest(fake_result())
    assert set(state) == {"mu_half", "gamma_half", "sigma_half", "Cp_raw_half"}