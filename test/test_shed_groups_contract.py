"""Contract tests F0-F8 for the v2 multi-shed-group half model.

The fixture is the accepted v1 half wing duplicated at z=0.4, so the two groups
are geometrically independent closed slabs with their own horizontal wakes.
Nothing here relaxes a physical gate; the module never claims accuracy.
"""
import json
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from panel.half_native import prepare_half_topology, solve_half_fields
from panel.shed_groups import (ShedGroup, assemble_half_groups,
                               solve_half_groups, validate_shed_groups)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "cases" / "panel_half" / "thick_wing_two_strip.json"
SHIFT = 0.4


def _raw():
    return json.loads(FIXTURE.read_text())


def _wall_only():
    raw = _raw()
    x = np.asarray(raw["coordinates"], dtype=float)
    f = np.asarray(raw["faces"], dtype=np.int32)
    tags = np.asarray(raw["component_ids"], dtype=np.int32)
    return raw, x, f, tags


def _group(raw, node_offset=0, face_offset=0, dz=0.0, gid="wing_z0",
           root="symmetry_plane", role=None):
    return ShedGroup(
        group_id=gid,
        wall_faces=np.asarray(raw["faces"], dtype=np.int32) + face_offset,
        wake_faces=np.asarray(raw["wake_faces"], dtype=np.int32) + node_offset,
        shed_edges=np.asarray(raw["trailing_edges"], dtype=np.int32) + node_offset,
        upper_te=np.asarray(raw["upper_te_faces"], dtype=np.int32) + face_offset,
        lower_te=np.asarray(raw["lower_te_faces"], dtype=np.int32) + face_offset,
        stations=np.asarray(raw["gamma_stations"], dtype=float),
        station_dof=np.asarray(raw["gamma_station_dof"], dtype=np.int32),
        role=role,
        root_condition=root,
        wake_z=0.0 + dz,
    )


def _two_group_case():
    raw, x, f, tags = _wall_only()
    n = len(x)
    shifted = x.copy()
    shifted[:, 2] += SHIFT
    vertices = np.concatenate((x, shifted), axis=0)
    faces = np.concatenate((f, f + n), axis=0)
    wake = np.concatenate((np.asarray(raw["wake_faces"], dtype=np.int32),
                           np.asarray(raw["wake_faces"], dtype=np.int32) + n), axis=0)
    component_ids = np.concatenate((tags, tags), axis=0)
    groups = [_group(raw, 0, 0, 0.0, "wing_low"),
              _group(raw, n, len(f), SHIFT, "wing_high")]
    return raw, vertices, faces, wake, component_ids, groups


def _solve(raw, vertices, faces, component_ids, groups, *, contact_cuts=(),
           flow=None):
    topology = validate_shed_groups(vertices, faces, component_ids, groups)
    return solve_half_groups(
        vertices, topology,
        np.asarray(raw["freestream"] if flow is None else flow, dtype=float),
        float(raw["density"]), float(raw["reference_area"]),
        float(raw["reference_chord"]),
        np.asarray(raw["moment_reference"], dtype=float),
        contact_cuts=contact_cuts)


def _v1_solve(raw, vertices, station_y, faces, wake, shed, up, lo, tags, cuts):
    topology = prepare_half_topology(
        vertices, faces, wake, shed, up, lo,
        np.asarray(station_y, dtype=float),
        np.asarray(raw["gamma_station_dof"], dtype=np.int32), tags,
        stencil_rings=2, cut_edges=np.asarray(cuts, dtype=np.int32))
    return solve_half_fields(
        jnp.asarray(vertices, dtype=float), jnp.asarray(np.asarray(station_y, dtype=float)),
        topology, jnp.asarray(np.asarray(raw["freestream"], dtype=float)),
        float(raw["density"]), float(raw["reference_area"]),
        float(raw["reference_chord"]),
        jnp.asarray(np.asarray(raw["moment_reference"], dtype=float)))


def _rel(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    scale = max(float(np.max(np.abs(b))), 1e-30)
    return float(np.max(np.abs(a - b)) / scale)


def test_f0_zero_groups_closed_body_solves():
    raw, x, f, tags = _wall_only()
    topology = validate_shed_groups(x, f, tags, [])
    assert topology.audit["groups"] == []
    out = _solve(raw, x, f, tags, [])
    assert out["unknowns"] == len(f)
    residual = np.asarray(out["linear_residual_half"], dtype=float)
    assert np.linalg.norm(residual) / np.linalg.norm(np.asarray(out["rhs_half"])) < 1e-12
    assert out["group_rows"] == []
    assert out["physical_accuracy_accepted"] is False
    assert float(np.max(np.abs(np.asarray(out["mu_half"])))) > 0.0


def test_f1_single_group_matches_v1():
    raw, x, f, tags = _wall_only()
    groups = [_group(raw)]
    grouped = _solve(raw, x, f, tags, groups)
    v1 = _v1_solve(raw, x, raw["gamma_stations"], f,
                   np.asarray(raw["wake_faces"], dtype=np.int32),
                   np.asarray(raw["trailing_edges"], dtype=np.int32),
                   np.asarray(raw["upper_te_faces"], dtype=np.int32),
                   np.asarray(raw["lower_te_faces"], dtype=np.int32),
                   tags, raw["cut_edges"])
    assert _rel(grouped["mu_half"], v1["mu_half"]) < 1e-13
    assert _rel(grouped["gamma_groups"][0], v1["gamma_half"]) < 1e-13
    assert _rel(grouped["Cp_raw_half"], v1["Cp_raw_half"]) < 1e-12
    assert abs(float(grouped["CD_pressure"]) - float(v1["CD_pressure"])) < 1e-12
    assert abs(float(grouped["CL"]) - float(v1["CL"])) < 1e-12
    assert grouped["unknowns"] == len(f) + int(raw["gamma_station_dof"].__len__()) - 1


def test_f2_known_forcing_subproblem_reproduces_full_solution():
    raw, vertices, faces, wake, tags, groups = _two_group_case()
    topology = validate_shed_groups(vertices, faces, tags, groups)
    assembled = assemble_half_groups(vertices, topology,
                                     np.asarray(raw["freestream"], dtype=float))
    matrix = np.asarray(assembled["matrix"], dtype=float)
    rhs = np.asarray(assembled["rhs"], dtype=float)
    full = np.linalg.solve(matrix, rhs)
    wall = assembled["wall"]
    g0 = assembled["group_slices"][0]
    g1 = assembled["group_slices"][1]
    keep = np.concatenate((np.arange(*wall), np.arange(*g0)))
    drop = np.arange(*g1)
    reduced = matrix[np.ix_(keep, keep)]
    forcing = rhs[keep] - matrix[np.ix_(keep, drop)] @ full[drop]
    part = np.linalg.solve(reduced, forcing)
    assert _rel(part, full[keep]) < 1e-12


def test_f3_true_removal_differs_by_known_term():
    raw, vertices, faces, wake, tags, groups = _two_group_case()
    topology = validate_shed_groups(vertices, faces, tags, groups)
    assembled = assemble_half_groups(vertices, topology,
                                     np.asarray(raw["freestream"], dtype=float))
    matrix = np.asarray(assembled["matrix"], dtype=float)
    rhs = np.asarray(assembled["rhs"], dtype=float)
    full = np.linalg.solve(matrix, rhs)
    wall = assembled["wall"]
    g0 = assembled["group_slices"][0]
    g1 = assembled["group_slices"][1]
    keep = np.concatenate((np.arange(*wall), np.arange(*g0)))
    drop = np.arange(*g1)
    reduced = matrix[np.ix_(keep, keep)]
    removed = np.linalg.solve(reduced, rhs[keep])
    term = np.zeros((len(keep),), dtype=float)
    term[:wall[1] - wall[0]] = matrix[np.ix_(np.arange(*wall), drop)] @ full[drop]
    expected = np.linalg.solve(reduced, term)
    assert _rel(removed - full[keep], expected) < 1e-12
    # The naive claim "removing a group leaves the others unchanged" is false.
    assert _rel(removed, full[keep]) > 1e-6


def test_f4_group_at_other_z_matches_v1():
    raw, x, f, tags = _wall_only()
    shifted = x.copy()
    shifted[:, 2] += SHIFT
    group = _group(raw, 0, 0, SHIFT, "wing_high")
    grouped = _solve(raw, shifted, f, tags, [group])
    v1 = _v1_solve(raw, shifted, raw["gamma_stations"], f,
                   np.asarray(raw["wake_faces"], dtype=np.int32),
                   np.asarray(raw["trailing_edges"], dtype=np.int32),
                   np.asarray(raw["upper_te_faces"], dtype=np.int32),
                   np.asarray(raw["lower_te_faces"], dtype=np.int32),
                   tags, raw["cut_edges"])
    assert _rel(grouped["mu_half"], v1["mu_half"]) < 1e-13
    assert _rel(grouped["gamma_groups"][0], v1["gamma_half"]) < 1e-13
    assert abs(float(grouped["CL"]) - float(v1["CL"])) < 1e-12


def test_f5_contract_rejections():
    raw, x, f, tags = _wall_only()
    n = len(x)
    shifted = x.copy()
    shifted[:, 2] += SHIFT
    vertices = np.concatenate((x, shifted), axis=0)
    faces = np.concatenate((f, f + n), axis=0)
    component_ids = np.concatenate((tags, tags), axis=0)
    g0 = _group(raw, 0, 0, 0.0, "a")
    g1 = _group(raw, n, len(f), SHIFT, "b")
    # Duplicate shed edge across groups
    dup = _group(raw, 0, 0, 0.0, "b")
    with pytest.raises(ValueError, match="Duplicate shed edge across groups"):
        validate_shed_groups(vertices, faces, component_ids, [g0, dup])
    # Buried root declared on the symmetry plane
    buried = _group(raw, 0, 0, 0.0, "a", root="buried_in_solid")
    buried.buried_inside_component = int(tags[0])
    with pytest.raises(ValueError, match="Buried root must not lie on the symmetry plane"):
        validate_shed_groups(vertices, faces, component_ids, [buried])
    # Role contradiction against declared component roles
    contradictory = _group(raw, 0, 0, 0.0, "a", role="lifting")
    contradictory.source_component_ids = (int(tags[0]),)
    with pytest.raises(ValueError, match="Shed group role contradicts"):
        validate_shed_groups(vertices, faces, component_ids, [contradictory],
                             component_roles={int(tags[0]): "non_lifting"})
    # Tilted wake is not representable
    tilt = vertices.copy()
    tilt[np.asarray(raw["wake_faces"]), 2] += 0.01
    with pytest.raises(ValueError,
                       match="Only horizontal constant-z wakes|not planar at its declared"):
        validate_shed_groups(tilt, faces, component_ids, [g0, g1])
    # Shed edge must be shared by the group wake
    off = _group(raw, 0, 0, 0.0, "a")
    off.wake_faces = np.asarray(raw["wake_faces"], dtype=np.int32)[[0]] + n
    with pytest.raises(ValueError, match="shared by the group wake"):
        validate_shed_groups(vertices, faces, component_ids, [off])


def test_f6_two_groups_residual_and_rows():
    raw, vertices, faces, wake, tags, groups = _two_group_case()
    out = _solve(raw, vertices, faces, tags, groups)
    matrix = np.asarray(out["matrix_half"], dtype=float)
    state = np.concatenate([np.asarray(out["mu_half"], dtype=float)]
                           + [np.asarray(g, dtype=float) for g in out["gamma_groups"]])
    residual = matrix @ state - np.asarray(out["rhs_half"], dtype=float)
    assert np.linalg.norm(residual) / np.linalg.norm(np.asarray(out["rhs_half"], dtype=float)) < 1e-12
    assert out["unknowns"] == len(faces) + 4
    rows = out["group_rows"]
    assert [r["group_id"] for r in rows] == ["wing_low", "wing_high"]
    for row in rows:
        assert row["circulation_unknowns"] == 2
        assert row["te_jump_rms"] > 0.0
        assert row["kutta_residual_rms"] < 1e-10
        assert row["wake_z"] == pytest.approx(0.0 if row["group_id"] == "wing_low" else SHIFT)
    # Two unequal groups: the lower wing sits on the symmetry plane, the upper
    # one does not, so their circulations must not be interchangeable.
    assert abs(rows[0]["gamma_sum"]) > 1e-6
    assert abs(rows[1]["gamma_sum"]) > 1e-6
    assert _rel(rows[1]["gamma"], rows[0]["gamma"]) > 1e-3
    # The pair is being solved together: zeroing group two's wake must change
    # group one's wall doublets, i.e. there is real mutual influence.
    alone = _solve(raw, vertices, faces, tags, groups[:1] + [])
    assert _rel(alone["mu_half"][: len(faces) // 2],
                np.asarray(out["mu_half"], dtype=float)[: len(faces) // 2]) > 1e-3


def test_f7_groups_couple_through_wall_doublets():
    raw, vertices, faces, wake, tags, groups = _two_group_case()
    base = _solve(raw, vertices, faces, tags, groups)
    flipped = [_group(raw, 0, 0, 0.0, "wing_low"),
               _group(raw, len(vertices) // 2, len(faces) // 2, SHIFT, "wing_high")]
    flipped[1].upper_te = groups[1].lower_te
    flipped[1].lower_te = groups[1].upper_te
    perturbed = _solve(raw, vertices, faces, tags, flipped)
    g0_base, g1_base = (np.asarray(g, dtype=float) for g in base["gamma_groups"])
    g0_new, g1_new = (np.asarray(g, dtype=float) for g in perturbed["gamma_groups"])
    # Flipping one group's Kutta rows must move *both* groups: the wall row
    # couples them through mu, so the untouched group is not frozen.
    assert _rel(g0_new, g0_base) > 1e-3
    assert _rel(g1_new, g1_base) > 1e-2
    assert abs(float(perturbed["CL"]) - float(base["CL"])) > 1e-9
    residual = np.asarray(perturbed["linear_residual_half"], dtype=float)
    assert np.linalg.norm(residual) / np.linalg.norm(np.asarray(perturbed["rhs_half"], dtype=float)) < 1e-12


def test_f8_adjoint_pairing_holds():
    raw, vertices, faces, wake, tags, groups = _two_group_case()
    topology = validate_shed_groups(vertices, faces, tags, groups)
    assembled = assemble_half_groups(vertices, topology,
                                     np.asarray(raw["freestream"], dtype=float))
    matrix = np.asarray(assembled["matrix"], dtype=float)
    rhs = np.asarray(assembled["rhs"], dtype=float)
    rng = np.random.default_rng(20261026)
    seed = rng.standard_normal(len(rhs))
    adjoint = rng.standard_normal(len(rhs))
    x = np.linalg.solve(matrix, rhs)
    y = np.linalg.solve(matrix.T, adjoint)
    left = float(y @ rhs)
    right = float(adjoint @ x)
    assert abs(left - right) <= 1e-10 * max(abs(left), abs(right), 1.0)
