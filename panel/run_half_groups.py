"""Research entry point for the v2 multi-shed-group half model.

This module is intentionally separate from library::panel.run_half_native.  It
never changes a default mode and it never claims physical accuracy: it exists so
that the grouped system can be driven, hashed and reported reproducibly, with
the small-fixture battery (F0-F8) available as --self-check.

Usage
    python -m panel.run_half_groups --surface runs/x/surface.json --out r.json
    python -m panel.run_half_groups --self-check --out r.json
    python -m panel.run_half_groups --surface s.json --groups g.json --out r.json --full-oracle

Exit codes: 0 on success, 2 on a refused input (ValueError/MemoryError/etc.,
the traceback is written into the report), 1 on an unexpected crash.
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from .differentiable import linear_solve
from .half_native import (_half_loads, _mirror_triangles, prepare_half_topology,
                         solve_half_fields)
from .kernel import triangle_potential
from .shed_groups import (ShedGroup, assemble_half_groups,
                          solve_half_groups, validate_shed_groups)

SCHEMA = "dpanel.half_group_report.v1"
ROOT = Path(__file__).resolve().parents[1]
CODE_FILES = ("panel/shed_groups.py", "panel/run_half_groups.py",
              "panel/half_native.py", "panel/kernel.py")


def _sha256_file(path):
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_manifest():
    import platform

    out = dict(python=platform.python_version(), numpy=np.__version__,
               jax=jax.__version__, platform=platform.platform(), files={})
    for rel in CODE_FILES:
        path = ROOT / rel
        if path.exists():
            out["files"][rel] = dict(sha256=_sha256_file(path), bytes=path.stat().st_size)
    return out


# --------------------------------------------------------------------------
# surface / group declarations
# --------------------------------------------------------------------------
REQUIRED_SURFACE_KEYS = ("coordinates", "faces", "component_ids", "freestream",
                         "density", "reference_area", "reference_chord",
                         "moment_reference")


def load_surface(path):
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED_SURFACE_KEYS if k not in raw]
    if missing:
        raise ValueError(f"Surface is missing keys: {missing}")
    surface = dict(
        vertices=np.asarray(raw["coordinates"], dtype=float),
        faces=np.asarray(raw["faces"], dtype=np.int32),
        component_ids=np.asarray(raw["component_ids"], dtype=np.int32),
        freestream=np.asarray(raw["freestream"], dtype=float),
        density=float(raw["density"]),
        reference_area=float(raw["reference_area"]),
        reference_chord=float(raw["reference_chord"]),
        reference_point=np.asarray(raw["moment_reference"], dtype=float),
        raw=raw,
    )
    return path, surface


def single_group_from_surface(surface, group_id="main"):
    """Wrap a v1 single-wake surface as one shed group (the v1 fallback shape)."""
    raw = surface["raw"]
    for key in ("wake_faces", "trailing_edges", "upper_te_faces",
                "lower_te_faces", "gamma_stations", "gamma_station_dof"):
        if key not in raw:
            raise ValueError(f"Surface has no {key}; declare --groups instead")
    wake = np.asarray(raw["wake_faces"], dtype=np.int32)
    if wake.size == 0:
        raise ValueError("Surface declares no wake sheet; pass --groups for the "
                         "v2 declaration or use a G=0 surface directly")
    wake_z = float(np.max(np.abs(surface["vertices"][wake][:, :, 2])))
    return ShedGroup(
        group_id=group_id,
        wall_faces=surface["faces"],
        wake_faces=wake,
        shed_edges=np.asarray(raw["trailing_edges"], dtype=np.int32),
        upper_te=np.asarray(raw["upper_te_faces"], dtype=np.int32),
        lower_te=np.asarray(raw["lower_te_faces"], dtype=np.int32),
        stations=np.asarray(raw["gamma_stations"], dtype=float),
        station_dof=np.asarray(raw["gamma_station_dof"], dtype=np.int32),
        wake_z=wake_z,
    )


def groups_from_spec(path, surface):
    """Read a dpanel.half_shed_groups.v2 declaration into ShedGroup objects."""
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    if spec.get("schema") not in (None, "dpanel.half_shed_groups.v2"):
        raise ValueError(f"Unsupported shed-group schema: {spec.get('schema')!r}")
    groups = []
    for item in spec.get("groups", []):
        missing = [k for k in ("group_id", "wake_faces", "shed_edges", "upper_te",
                               "lower_te", "stations", "station_dof")
                   if k not in item]
        if missing:
            raise ValueError(f"Shed group {item.get('group_id')!r} is missing {missing}")
        wall = item.get("wall_faces")
        groups.append(ShedGroup(
            group_id=str(item["group_id"]),
            wall_faces=(surface["faces"] if wall is None
                        else np.asarray(wall, dtype=np.int32)),
            wake_faces=np.asarray(item["wake_faces"], dtype=np.int32),
            shed_edges=np.asarray(item["shed_edges"], dtype=np.int32),
            upper_te=np.asarray(item["upper_te"], dtype=np.int32),
            lower_te=np.asarray(item["lower_te"], dtype=np.int32),
            stations=np.asarray(item["stations"], dtype=float),
            station_dof=np.asarray(item["station_dof"], dtype=np.int32),
            kind=item.get("kind", "attached_lifting_TE"),
            role=item.get("role"),
            root_condition=item.get("root_condition", "symmetry_plane"),
            source_component_ids=tuple(int(v) for v in item.get("source_component_ids", ())),
            buried_inside_component=(None if item.get("buried_inside_component") is None
                                     else int(item["buried_inside_component"])),
            wake_z=None if item.get("wake_z") is None else float(item["wake_z"]),
        ))
    if not groups:
        raise ValueError("Shed-group declaration contains no groups")
    return groups, spec


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------
def check_block_dimensions(assembled):
    sizes = assembled["sizes"]
    nf = assembled["wall_panels"]
    matrix = np.asarray(assembled["matrix"], dtype=float)
    expected = nf + sum(sizes)
    ok = matrix.shape == (expected, expected)
    unknowns = expected
    return dict(performed=True, ok=bool(ok), shape=list(matrix.shape),
                expected=unknowns, unknowns=unknowns,
                wall_unknowns=int(nf),
                group_unknowns=[int(s) for s in sizes],
                group_slices=[list(map(int, s)) for s in assembled["group_slices"]],
                kutta_rows=int(sum(sizes)),
                finite=bool(np.isfinite(matrix).all()))


def check_circulation_mass_rows(assembled):
    """Recheck M_g = mass_g / measure_g from the TE quadrature, independently.

    Two separate facts are reported because they failed for different reasons in
    the v1 derivation:
      * the Gauss quadrature really integrates B_i B_j and B_i (self-consistency);
      * M_g = raw_mass / measure_g and sum_j raw_mass_ij = integral of
        B_i * (sum_j B_j).  That equals measure_i only where the station basis is
        a partition of unity; at the tapered tip station (dof -1) the basis sums
        to less than one, so the normalised row sum is published instead of
        being silently forced to 1.
    """
    worst = 0.0
    worst_quad = 0.0
    rows = []
    for op in assembled["operators"]:
        basis = np.asarray(op["te_basis"], dtype=float)
        weights = np.asarray(op["weights"], dtype=float)
        m = np.asarray(op["mass"], dtype=float)
        measure = np.asarray(op["measure"], dtype=float)
        raw = np.asarray(op["raw_mass"], dtype=float)
        integrals = np.asarray(op["integrals"], dtype=float)
        mass_quad = np.einsum("eq,eqi,eqj->ij", weights, basis, basis)
        integral_quad = np.einsum("eq,eqi->ei", weights, basis)
        quad_gap = max(float(np.max(np.abs(raw - mass_quad))),
                       float(np.max(np.abs(integrals - integral_quad))))
        normalise_gap = float(np.max(np.abs(m - raw / measure[:, None])))
        partition = basis.sum(axis=-1)
        row_sum = m.sum(axis=1)
        worst = max(worst, normalise_gap)
        worst_quad = max(worst_quad, quad_gap)
        rows.append(dict(
            group_size=int(m.shape[0]), min_measure=float(np.min(measure)),
            quadrature_gap=quad_gap, normalisation_gap=normalise_gap,
            basis_partition_max_abs_error=float(np.max(np.abs(partition - 1.0))),
            quadrature_points_with_partial_basis=int(np.count_nonzero(partition < 1.0 - 1e-12)),
            row_sums=[float(v) for v in row_sum],
            row_sums_where_partition_of_unity=[
                float(v) for v, p in zip(row_sum, np.all(partition >= 1.0 - 1e-12, axis=1))
                if p]))
    return dict(performed=True, max_normalisation_gap=worst, max_quadrature_gap=worst_quad,
                note="M_g = raw_mass/measure_g is exact by construction; sum_j M_ij = 1 "
                     "holds only on rows whose TE quadrature has a partition-of-unity basis, "
                     "which the tapered tip station (dof -1) does not",
                groups=rows)


def check_kutta_rows(out):
    rows = [dict(group_id=r["group_id"],
                 kutta_residual_rms=r["kutta_residual_rms"],
                 circulation_unknowns=r["circulation_unknowns"])
            for r in out["group_rows"]]
    worst = max((r["kutta_residual_rms"] for r in rows), default=0.0)
    return dict(performed=True, max_rms=float(worst), groups=rows)


def _wall_tags(surface, topology, contact_cuts=()):
    from .shed_groups import pressure_neighbors

    neighbors = pressure_neighbors(surface["vertices"], topology, contact_cuts)
    return dict(faces=jnp.asarray(np.asarray(topology.wall_faces, dtype=np.int32)),
                component_ids=jnp.asarray(np.asarray(topology.component_ids, dtype=np.int32)),
                neighbor_ghost_ids=jnp.asarray(np.asarray(neighbors, dtype=np.int32)))


def _lift_of_alpha(alpha, surface, topology, tags):
    """CL as a differentiable function of the angle of attack (radians)."""
    speed = float(np.linalg.norm(surface["freestream"]))
    zero = jnp.zeros((), dtype=alpha.dtype)
    flow = speed * jnp.stack((jnp.cos(alpha), zero, jnp.sin(alpha)))
    assembled = assemble_half_groups(surface["vertices"], topology, flow)
    state = linear_solve(assembled["matrix"], assembled["rhs"])
    loads = _half_loads(jnp.asarray(surface["vertices"], dtype=float), tags, flow,
                        surface["density"], surface["reference_area"],
                        surface["reference_chord"],
                        jnp.asarray(surface["reference_point"]),
                        state[:assembled["wall_panels"]])
    return loads["CL"]


def check_lift_slope_fd_vs_ad(surface, topology, *, steps=(1e-4, 1e-5), contact_cuts=()):
    """Compare dCL/dalpha from forward-mode AD with central differences."""
    from types import SimpleNamespace

    tags = SimpleNamespace(**_wall_tags(surface, topology, contact_cuts))
    alpha0 = float(np.arctan2(surface["freestream"][2], surface["freestream"][0]))
    ad = float(jax.grad(_lift_of_alpha)(jnp.asarray(alpha0), surface, topology, tags))
    rows = []
    for h in steps:
        plus = float(_lift_of_alpha(jnp.asarray(alpha0 + h), surface, topology, tags))
        minus = float(_lift_of_alpha(jnp.asarray(alpha0 - h), surface, topology, tags))
        fd = (plus - minus) / (2.0 * h)
        rows.append(dict(step=h, central=fd,
                         relative_difference=abs(fd - ad) / max(abs(ad), 1e-30)))
    return dict(performed=True, alpha_rad=alpha0, dcl_dalpha_ad=ad,
                dcl_dalpha_per_deg=ad * np.pi / 180.0, finite_difference=rows,
                max_relative_difference=float(max(r["relative_difference"] for r in rows)))


def check_operator_mirror(assembled, surface, topology):
    """Operator-level mirror check.

    The folded half operator must equal the documented formula: the wall wall
    block is the interior-diagonal doublet plus its mirrored image (no interior
    diagonal on the image), and every group wake operator is rebuilt from its own
    triangles plus mirrored images.  This is a wiring check of the mirror map, not
    a physical proof; the field-level proof is the full-mirror oracle.
    """
    vertices = np.asarray(surface["vertices"], dtype=float)
    reflect = np.asarray([1.0, -1.0, 1.0])
    centers = np.asarray(assembled["centers"], dtype=float)
    tri = vertices[np.asarray(topology.wall_faces, dtype=np.int32)]
    mirrored = (tri * reflect)[:, ::-1, :]
    folded = np.asarray(assembled["doublet"], dtype=float)
    _, d_orig = triangle_potential(centers, jnp.asarray(tri), interior_diagonal=True)
    # Only the wall wall self term gets the interior-diagonal convention; the
    # image triangles are distinct panels even where they coincide geometrically.
    _, d_mirror = triangle_potential(centers, jnp.asarray(mirrored))
    explicit = np.asarray(d_orig, dtype=float) + np.asarray(d_mirror, dtype=float)
    # The diagonal is the panel-on-itself solid angle convention (-1/2), not a
    # mirror property, so the mirror identity is checked off the diagonal only.
    off = ~np.eye(folded.shape[0], dtype=bool)
    gap = float(np.max(np.abs((folded - explicit)[off])))
    peak = max(float(np.max(np.abs(folded[off]))), 1e-300)
    rows = []
    worst_wake = 0.0
    for g, op in zip(topology.groups, assembled["operators"]):
        wake = vertices[np.asarray(g.wake_faces, dtype=np.int32)]
        basis = np.asarray(op["basis"], dtype=float)
        slope = np.asarray(op["slope"], dtype=float)
        dw, my = triangle_potential(centers, jnp.asarray(wake), source=False,
                                    linear_y=True)
        di, mi = triangle_potential(centers, _mirror_triangles(jnp.asarray(wake)),
                                    source=False, linear_y=True)
        rebuilt = ((np.asarray(dw, float) + np.asarray(di, float)) @ basis
                   + (np.asarray(my, float) - np.asarray(mi, float)) @ slope)
        gap_g = float(np.max(np.abs(np.asarray(op["wake_operator"], float) - rebuilt)))
        peak_g = max(float(np.max(np.abs(rebuilt))), 1e-300)
        worst_wake = max(worst_wake, gap_g / peak_g)
        rows.append(dict(group_id=g.group_id,
                         wake_operator_shape=list(map(int, np.shape(op["wake_operator"]))),
                         wake_faces=int(len(g.wake_faces)),
                         wake_operator_max_abs_gap=gap_g,
                         wake_operator_relative_to_peak=gap_g / peak_g))
    return dict(performed=True,
                notes="doublet block compared off the diagonal; every group wake "
                      "operator rebuilt from its triangles plus mirrored images",
                doublet_max_abs_gap=gap,
                doublet_relative_to_peak=gap / peak,
                worst_wake_operator_relative=worst_wake,
                groups=rows)


def _relative(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1e-300))


def _v1_topology(surface, topology):
    group = topology.groups[0]
    return prepare_half_topology(
        surface["vertices"], topology.wall_faces, group.wake_faces,
        group.shed_edges, group.upper_te, group.lower_te, group.stations,
        group.station_dof, topology.component_ids, stencil_rings=2,
        cut_edges=np.zeros((0, 2), dtype=np.int32))


def check_v1_equivalence(surface, topology, out):
    if len(topology.groups) != 1:
        return dict(performed=False, reason="v1 has exactly one wake sheet")
    half = _v1_topology(surface, topology)
    v1 = solve_half_fields(jnp.asarray(surface["vertices"], dtype=float),
                           jnp.asarray(topology.groups[0].stations, dtype=float), half,
                           jnp.asarray(surface["freestream"], dtype=float),
                           surface["density"], surface["reference_area"],
                           surface["reference_chord"],
                           jnp.asarray(surface["reference_point"]))
    return dict(performed=True,
                mu_relative_to_peak=_relative(out["mu_half"], v1["mu_half"]),
                gamma_relative_to_peak=_relative(out["gamma_groups"][0], v1["gamma_half"]),
                cp_relative_to_peak=_relative(out["Cp_raw_half"], v1["Cp_raw_half"]),
                cl_absolute=abs(float(out["CL"]) - float(v1["CL"])),
                cd_absolute=abs(float(out["CD_pressure"]) - float(v1["CD_pressure"])))


def check_full_oracle(surface, topology, out, *, max_panels=400):
    """Explicit full-mirror solve through the verified v1 path; G=1 only."""
    from .differentiable import solve_fields
    from .verify_half_mirror import make_full_mirror_topology, mirrored_full_input

    if len(topology.groups) != 1:
        return dict(performed=False,
                    reason="the full-mirror oracle only knows one wake sheet; "
                           "a multi-wake full model does not exist yet")
    nf = int(len(topology.wall_faces))
    if nf > max_panels:
        raise MemoryError(f"Full oracle refused: {nf} half panels above --full-oracle-panels")
    group = topology.groups[0]
    half = dict(topology=_v1_topology(surface, topology),
                vertices=np.asarray(surface["vertices"], dtype=float),
                station_y=np.asarray(group.stations, dtype=float),
                freestream=np.asarray(surface["freestream"], dtype=float),
                density=surface["density"],
                reference_area=surface["reference_area"],
                reference_chord=surface["reference_chord"],
                reference_point=np.asarray(surface["reference_point"], dtype=float))
    plan = make_full_mirror_topology(half, group.station_dof)
    full = jax.jit(solve_fields)(**mirrored_full_input(half, plan))
    return dict(performed=True, full_panels=int(2 * nf),
                half_circulation_unknowns=int(group.ng),
                full_circulation_unknowns=int(len(np.asarray(full["gamma"]))),
                note="the full model appends an image copy of every non-root "
                     "circulation DOF; only its first ng entries are the half DOFs",
                mu_relative_to_peak=_relative(np.asarray(full["mu"], float)[:nf], out["mu_half"]),
                gamma_relative_to_peak=_relative(
                    np.asarray(full["gamma"], float)[: group.ng], out["gamma_groups"][0]),
                physical_accuracy_accepted=False)


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def run(surface_path, *, groups_path=None, full_oracle=False,
        full_oracle_panels=400, gradient=True, v1_check=True,
        max_panels=4000, contact_cuts=()):
    """Drive one grouped half solve and return the report dictionary."""
    started = time.perf_counter()
    surface_path, surface = load_surface(surface_path)
    if groups_path is None:
        # A surface without a wake sheet is the G=0 pure-Dirichlet configuration;
        # forcing a "single group" onto it would index an empty wake array.
        declared = np.asarray(surface["raw"].get("wake_faces") or [], dtype=np.int32)
        groups = [] if declared.size == 0 else [single_group_from_surface(surface)]
        spec = None
    else:
        groups, spec = groups_from_spec(groups_path, surface)
    nf = len(surface["faces"])
    if nf > max_panels:
        raise MemoryError(f"Refused: {nf} wall panels above --max-panels={max_panels}")
    topology = validate_shed_groups(surface["vertices"], surface["faces"],
                                    surface["component_ids"], groups)
    assembled = assemble_half_groups(surface["vertices"], topology, surface["freestream"])
    started_solve = time.perf_counter()
    out = solve_half_groups(surface["vertices"], topology, surface["freestream"],
                            surface["density"], surface["reference_area"],
                            surface["reference_chord"], surface["reference_point"],
                            contact_cuts=contact_cuts)
    hot = time.perf_counter() - started_solve
    residual = np.asarray(out["linear_residual_half"], dtype=float)
    rnorm = float(np.linalg.norm(residual) /
                  max(float(np.linalg.norm(np.asarray(out["rhs_half"], dtype=float))), 1e-300))
    cp = np.asarray(out["Cp_raw_half"], dtype=float)
    report = dict(
        schema=SCHEMA,
        status="numerical_solve_completed_not_physical_validation",
        physical_accuracy_accepted=False,
        inputs=dict(
            surface=str(surface_path),
            surface_sha256=_sha256_file(surface_path),
            groups=None if groups_path is None else str(groups_path),
            groups_sha256=None if groups_path is None else _sha256_file(groups_path),
            groups_schema=None if spec is None else spec.get("schema"),
            surface_audit=surface["raw"].get("audit"),
            surface_provenance=surface["raw"].get("provenance"),
        ),
        code=code_manifest(),
        groups=[dict(group_id=g.group_id, kind=g.kind, role=g.role,
                     root_condition=g.root_condition,
                     wake_faces=int(len(g.wake_faces)),
                     shed_edges=int(len(g.shed_edges)),
                     circulation_unknowns=int(len(g.stations)),
                     wake_z=g.wake_z) for g in topology.groups],
        validation=dict(topology.audit),
        solve=dict(unknowns=int(out["unknowns"]), wall_panels=int(out["wall_panels"]),
                   group_slices=[list(map(int, s)) for s in out["group_slices"]],
                   linear_residual_relative=rnorm, hot_seconds=hot,
                   CL=float(out["CL"]), CD_pressure=float(out["CD_pressure"]),
                   CM=[float(v) for v in np.asarray(out["CM"], dtype=float)],
                   Cp_min=float(np.min(cp)), Cp_max=float(np.max(cp)),
                   physical_accuracy_accepted=False),
        group_rows=[dict(group_id=r["group_id"],
                         circulation_unknowns=r["circulation_unknowns"],
                         gamma=[float(v) for v in np.asarray(r["gamma"], dtype=float)],
                         gamma_sum=r["gamma_sum"], gamma_max_abs=r["gamma_max_abs"],
                         te_jump_rms=r["te_jump_rms"], te_jump_max_abs=r["te_jump_max_abs"],
                         kutta_residual_rms=r["kutta_residual_rms"],
                         wake_z=r["wake_z"], wake_faces=r["wake_faces"],
                         shed_edges=r["shed_edges"]) for r in out["group_rows"]],
        checks={},
    )
    checks = report["checks"]
    checks["block_dimensions"] = check_block_dimensions(assembled)
    checks["circulation_mass_rows"] = check_circulation_mass_rows(assembled)
    checks["kutta_rows"] = check_kutta_rows(out)
    checks["operator_mirror"] = check_operator_mirror(assembled, surface, topology)
    checks["v1_equivalence"] = (check_v1_equivalence(surface, topology, out)
                                if v1_check else dict(performed=False, reason="disabled"))
    checks["full_oracle"] = (check_full_oracle(surface, topology, out,
                                               max_panels=full_oracle_panels)
                             if full_oracle else dict(performed=False, reason="not requested"))
    checks["lift_slope"] = (check_lift_slope_fd_vs_ad(surface, topology,
                                                      contact_cuts=contact_cuts)
                            if gradient and topology.groups else
                            dict(performed=False,
                                 reason=("disabled" if not gradient
                                         else "no circulation unknowns: a G=0 pure "
                                              "Dirichlet solve has CL ~ 0")))
    report["elapsed_seconds"] = time.perf_counter() - started
    return report


def self_check(out_path, *, shift=0.4, full_oracle_panels=400):
    """Small-fixture battery: G=0 / G=1 / G=2, oracle, gradient, elevated tail."""
    from .group_fixture import as_surface_json, group_specs, two_layer_case

    out_path = Path(out_path)
    workdir = out_path.parent
    workdir.mkdir(parents=True, exist_ok=True)
    report = dict(schema=SCHEMA, status="self_check", physical_accuracy_accepted=False,
                  code=code_manifest(), cases=[], checks={})
    cases = {}
    for label, ngroups, dz in (("G0", 0, 0.0), ("G1", 1, 0.0),
                               ("G2", 2, shift), ("G2_tail_z024", 2, 0.24)):
        case, raw, vertices, faces, tags, groups = two_layer_case(None, dz, groups=ngroups)
        surface = as_surface_json(raw, vertices, faces, tags, groups)
        surface_path = workdir / f"surface_{label}.json"
        surface_path.write_text(json.dumps(surface), encoding="utf-8")
        groups_path = None
        if groups:
            groups_path = workdir / f"groups_{label}.json"
            groups_path.write_text(json.dumps(group_specs(groups)), encoding="utf-8")
        entry = dict(label=label, groups=ngroups, shift=dz,
                     surface=str(surface_path), surface_sha256=_sha256_file(surface_path),
                     groups_path=None if groups_path is None else str(groups_path),
                     groups_sha256=None if groups_path is None else _sha256_file(groups_path))
        try:
            r = run(surface_path, groups_path=groups_path,
                    full_oracle=(label == "G1"),
                    full_oracle_panels=full_oracle_panels, gradient=True,
                    v1_check=bool(groups))
            entry.update(exit_code=0, unknowns=r["solve"]["unknowns"],
                         CL=r["solve"]["CL"], CD_pressure=r["solve"]["CD_pressure"],
                         CM_y=r["solve"]["CM"][1],
                         linear_residual_relative=r["solve"]["linear_residual_relative"],
                         gamma_sum=[row["gamma_sum"] for row in r["group_rows"]],
                         checks=r["checks"])
            cases[label] = r
        except Exception as exc:  # noqa: BLE001 - the traceback is the evidence
            entry.update(exit_code=2, error=f"{type(exc).__name__}: {exc}",
                         traceback=traceback.format_exc())
        report["cases"].append(entry)
    if all(k in cases for k in ("G1", "G2")):
        g1, g2 = cases["G1"], cases["G2"]
        report["checks"]["g1_vs_g2_gamma"] = [
            dict(group=row1["group_id"], gamma_sum_g1=row1["gamma_sum"],
                 gamma_sum_g2=row2["gamma_sum"])
            for row1, row2 in zip(g1["group_rows"], g2["group_rows"])]
        report["checks"]["g1_vs_g2_notes"] = (
            "two groups are coupled through the doublet block, so the G=1 solve is "
            "not a subset of G=2; both mu and every gamma change")
    report["cases"].sort(key=lambda e: ["G0", "G1", "G2", "G2_tail_z024"].index(e["label"]))
    out_path.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    return report


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="v2 shed-group research entry")
    parser.add_argument("--surface")
    parser.add_argument("--groups")
    parser.add_argument("--out", required=True)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--full-oracle", action="store_true")
    parser.add_argument("--full-oracle-panels", type=int, default=400)
    parser.add_argument("--max-panels", type=int, default=4000)
    parser.add_argument("--no-gradient", action="store_true")
    parser.add_argument("--shift", type=float, default=0.4)
    args = parser.parse_args(argv)
    out_path = Path(args.out)
    if out_path.exists():
        raise FileExistsError(f"Do not overwrite a previous group report: {out_path}")
    try:
        if args.self_check:
            report = self_check(out_path, shift=args.shift,
                                full_oracle_panels=args.full_oracle_panels)
        else:
            if not args.surface:
                raise ValueError("--surface is required unless --self-check is given")
            report = run(args.surface, groups_path=args.groups,
                         full_oracle=args.full_oracle,
                         full_oracle_panels=args.full_oracle_panels,
                         gradient=not args.no_gradient,
                         max_panels=args.max_panels)
            out_path.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        out_path.write_text(json.dumps(dict(
            schema=SCHEMA, status="refused", exit_code=2,
            error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc(),
            physical_accuracy_accepted=False), indent=2), encoding="utf-8")
        raise
    print(f"OK {out_path} unknowns={report.get('solve', {}).get('unknowns')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
