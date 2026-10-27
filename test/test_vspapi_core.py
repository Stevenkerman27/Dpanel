"""Pure-Python tests for vspapi (no OpenVSP install needed).

Run from the repository root::

    .conda_env\\python.exe -m pytest test/test_vspapi_core.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vspapi.config import SweepConfig
from vspapi.health import check_polar_health, cluster_risk
from vspapi.mesh import (VspGeom, measure, mesh_quality, mirror_about_y0,
                         open_boundary_edges, read_vspgeom, validate_mesh,
                         write_vspgeom)
from vspapi.results import read_history, read_lod, read_polar, read_tecplot
from vspapi.tessellate import make_wing_tess_plan, next_tess_value, wing_area

SCRATCH = Path(__file__).resolve().parent / "_vspapi_tmp"


@pytest.fixture()
def scratch():
    SCRATCH.mkdir(exist_ok=True)
    return SCRATCH


def _plateau_wing() -> VspGeom:
    """Two quads forming a half-span flat plate: y in [0, 1], x in [0, 0.2]."""
    pts = np.array([
        [0.0, 0.0, 0.0], [0.2, 0.0, 0.0],
        [0.0, 0.5, 0.0], [0.2, 0.5, 0.0],
        [0.0, 1.0, 0.0], [0.2, 1.0, 0.0],
    ])
    panels = [np.array([0, 1, 3, 2]), np.array([2, 3, 5, 4])]
    return VspGeom(points=pts, panels=panels, symmetry=1)


def test_next_tess_value_matches_dep_opt():
    # DEP-OPT: n = max((x - 6) // 4 + 1, 0); 5 + 4 n
    assert next_tess_value(0) == 5
    assert next_tess_value(6) == 9
    assert next_tess_value(10) == 13
    assert next_tess_value(105) == 105
    assert next_tess_value(106) == 109


def test_tessellation_plan_for_trapezoid():
    plan = make_wing_tess_plan([1.0], [0.2, 0.1], 0.0038)
    assert plan["Tess_W"] == 105
    assert plan["SectTess_U"] == [527]


def test_wing_area_is_full_aircraft_area_for_semispan():
    assert wing_area([1.0], [0.2, 0.1]) == pytest.approx(0.30, abs=1e-15)
    with pytest.raises(ValueError):
        wing_area([1.0], [0.2])


def test_sweep_config_round_trip(scratch):
    cfg = SweepConfig(alpha=[0.0, 2.5, 5.0])
    text = cfg.to_text()
    assert text.endswith("\r\n")
    assert "WakeRelax = 0.5" in text
    assert "Symmetry = 1" in text
    again = SweepConfig.from_text(text)
    assert again.alpha == [0.0, 2.5, 5.0]
    assert again.sref == pytest.approx(0.30)
    assert again.wake_relax == pytest.approx(0.5)
    assert again.wake_iters == 20
    path = scratch / "case.vspaero"
    cfg.write(path)
    assert SweepConfig.from_text(path.read_text()).alpha == cfg.alpha


def test_sweep_config_rejects_bad_input():
    with pytest.raises(ValueError):
        SweepConfig(alpha=[])
    with pytest.raises(ValueError):
        SweepConfig(sref=0.0)
    with pytest.raises(ValueError):
        SweepConfig(wake_iters=0)


def test_vspgeom_round_trip_and_measure(scratch):
    geom = _plateau_wing()
    assert geom.n_panels == 2
    assert geom.panel_types() == {4: 2}
    assert measure(geom)["area"] == pytest.approx(0.2, abs=1e-12)

    path = write_vspgeom(geom, scratch / "plate.vspgeom")
    again = read_vspgeom(path)
    assert again.n_points == geom.n_points
    assert again.n_panels == 2
    assert again.symmetry == 1
    assert np.allclose(again.points, geom.points)
    assert measure(again)["area"] == pytest.approx(0.2, abs=1e-12)


def test_vspgeom_accepts_355_echoed_panel_count(scratch):
    """VSPAERO 3.55 repeats the panel count on its own line before the records."""
    geom = _plateau_wing()
    base = write_vspgeom(geom, scratch / "plate355.vspgeom").read_text().splitlines()
    out = base[:3 + geom.n_points] + ["2"] + base[3 + geom.n_points:]
    path = scratch / "plate355_echo.vspgeom"
    path.write_text("\n".join(out) + "\n")
    again = read_vspgeom(path)
    assert again.n_panels == 2
    assert np.allclose(again.points, geom.points)
    assert measure(again)["area"] == pytest.approx(0.2, abs=1e-12)


def test_vspgeom_rejects_broken_files(scratch):
    bad = scratch / "bad.vspgeom"
    bad.write_text("# vspgeom v3 | 1\n1\n3 1 1\n0 0 0\n1 0 0\n0 1 0\n4 1 2 3 9\n")
    with pytest.raises(ValueError):
        read_vspgeom(bad)
    other = scratch / "other.vspgeom"
    other.write_text("not a vspgeom\n")
    with pytest.raises(ValueError):
        read_vspgeom(other)


def test_open_boundary_edges_of_half_plate():
    edges = open_boundary_edges(_plateau_wing())
    # outer loop of a 2-quad strip: 6 edges (4 outer + 2 at the split are shared)
    assert edges.shape[0] == 6


def test_mirror_welds_plane_and_flips_winding():
    half = _plateau_wing()
    full = mirror_about_y0(half)
    assert full.n_panels == 4
    assert full.n_points == 6 + 4          # two nodes sit on y = 0
    assert measure(full)["area"] == pytest.approx(2 * 0.2, abs=1e-12)
    mirrored = full.points[full.panels[2]]
    # mirrored panel normal points the other way (z flip after winding reversal)
    n_half = np.cross(half.points[half.panels[0]][1] - half.points[half.panels[0]][0],
                      half.points[half.panels[0]][2] - half.points[half.panels[0]][0])
    n_mir = np.cross(mirrored[1] - mirrored[0], mirrored[2] - mirrored[0])
    assert np.sign(n_half[2]) == np.sign(n_mir[2])


def test_mesh_quality_and_validation(scratch):
    geom = _plateau_wing()
    quality = mesh_quality(geom)
    assert quality["max_aspect"] == pytest.approx(0.5 / 0.2 * 0.5 / 0.5, rel=1e-9)
    assert validate_mesh(geom)["degenerate_panels"] == 0

    flat = VspGeom(points=np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]], float),
                   panels=[np.array([0, 1, 2, 3])])
    with pytest.raises(RuntimeError):
        validate_mesh(flat)


POLAR = """VSPAERO 3.52
Surface Integration Forces and Moments -->
Beta Mach AoA Re/1e6 CLo CLi CLtot CDo CDi CDtot L/D E CLwtot CDwtot
0.0 0.001 0.0 10.0 -0.0003 27.309 27.309 0.0099 0.1198 0.1297 148.6 9.1 0.1762 0.0105
0.0 0.001 5.0 10.0 -0.0015 34.249 34.248 0.0124 3.0755 3.0879 9.10 4.6 0.6611 0.0216
"""


def test_read_polar_by_column_name(scratch):
    path = scratch / "case.polar"
    path.write_text(POLAR)
    rows = read_polar(path)
    assert len(rows) == 2
    assert rows[1]["CLi"] == pytest.approx(34.249)
    assert rows[1]["CLwtot"] == pytest.approx(0.6611)


def test_polar_health_flags_wild_cli():
    path_rows = [
        {"AoA": 0.0, "CLi": 27.309, "CLwtot": 0.1762, "CDi": 0.1198, "CDo": 0.0099,
         "CDtot": 0.1297},
        {"AoA": 5.0, "CLi": 0.6346, "CLwtot": 0.6544, "CDi": 0.0109, "CDo": 0.0205,
         "CDtot": 0.0314},
    ]
    health = check_polar_health(path_rows)
    assert health["healthy"] is False
    assert len(health["issues"]) == 1
    assert "alpha=0" in health["issues"][0]
    assert health["rows"][0]["cli_trusted"] is False
    assert health["rows"][1]["cli_trusted"] is True
    assert health["rows"][1]["cd_tot_additive"] is True
    assert "CLwtot" in health["trusted_columns"]


def test_cluster_risk_thresholds():
    assert cluster_risk(0.9, 0.8)["risk"] == "ok"
    assert cluster_risk(0.25, 0.25)["risk"] == "high"
    assert cluster_risk(0.9, 0.5)["risk"] == "marginal"
    assert cluster_risk(0.25, 0.8, 1.255)["risk"] == "ok"
    assert cluster_risk(0.25, 0.8)["max_growth_relevant"] is False


LOD = """Iter VortexSheet TrailVort Xavg Yavg Zavg dSpan SoverB Chord dArea Cl Cd
1 1 1 0.1 0.25 0.0 0.5 0.25 0.15 0.075 0.4 -0.01
1 1 2 0.1 0.75 0.0 0.5 0.75 0.15 0.075 0.6 -0.02
1 1 1 0.1 0.25 0.0 0.5 0.25 0.15 0.075 0.5 -0.03
1 1 2 0.1 0.75 0.0 0.5 0.75 0.15 0.075 0.7 -0.04
"""


def test_read_lod_splits_blocks_per_alpha(scratch):
    path = scratch / "case.lod"
    path.write_text(LOD)
    blocks = read_lod(path, 2)
    assert len(blocks) == 2
    assert blocks[0]["strips"] == 2
    assert blocks[0]["sum_cl_dspan"] == pytest.approx(0.5 * 0.4 + 0.5 * 0.6)
    assert blocks[1]["sum_cl_dspan"] == pytest.approx(0.6)
    assert read_lod(path)[0]["sum_cl_dspan"] == pytest.approx(0.5)


HISTORY = """Iter Mach AoA Beta CLo CLi CLtot CDo CDi CDtot
1 0.001 5.0 0.0 -0.0015 2.5 2.5 0.0124 0.0921 0.1045
2 0.001 5.0 0.0 -0.0015 26.7 26.7 0.0124 2.4240 2.4364
"""


def test_read_history(scratch):
    path = scratch / "case.history"
    path.write_text(HISTORY)
    rows = read_history(path)
    assert [r["CLi"] for r in rows] == [2.5, 26.7]


def test_read_tecplot_block(scratch):
    tri = "1 2 3\n1 3 4\n"
    lines = [
        "VARIABLES = x,y,z,Cp",
        'ZONE N=4, E=2, T="wall"',
        "ZONETYPE=FETRIANGLE, DATAPACKING=BLOCK",
        "SOLUTIONTIME = 0., STRANDID = 0",
        "0.0", "1.0", "0.0", "1.0",
        "0.0", "0.0", "1.0", "1.0",
        "0.0", "0.0", "0.0", "0.0",
        "-1.0", "-2.0",
        tri.rstrip("\n"),
    ]
    path = scratch / "case.tec.dat"
    path.write_text("\n".join(lines) + "\n")
    out = read_tecplot(path)
    assert out["points"].shape == (4, 3)
    assert np.allclose(out["cell_values"], [-1.0, -2.0])
    assert out["faces"].shape == (2, 3)
    assert out["faces"][0].tolist() == [0, 1, 2]