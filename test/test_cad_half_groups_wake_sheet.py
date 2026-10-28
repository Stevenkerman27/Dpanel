"""End-to-end contract test: OCC half domain with two independent wake sheets.

Gated on gmsh because the fixture is a real STEP solid: the point of this
test is that two *separate* horizontal wake sheets (z=0 and z=0.24) produced
by the OCC builder survive into the grouped solver as two independent
circulation blocks.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("gmsh")

from panel.run_half_groups import run  # noqa: E402

CASE = ROOT / "cases" / "panel_half" / "two_layer_wing"
SCRATCH = ROOT / "runs" / "cad_group_tests"


def _mesh(size, name):
    out = SCRATCH / name
    if (out / "surface.json").exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "panel.cad_half_groups",
               "--wing-cad", str(CASE / "two_layer.step"),
               "--wake-groups", str(CASE / "groups.json"),
               "--out", str(out), "--chord", "1.0",
               "--reference-area", "4.8", "--size", "%g" % size,
               "--far-size", "1.5", "--radius", "5", "--wake-strips", "6",
               "--alpha-deg", "5", "--ncpu", "4"]
    completed = subprocess.run(command, cwd=str(ROOT), capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert (out / "surface.json").exists()
    return out


def test_two_independent_wake_sheets_reach_the_solver():
    out = _mesh(0.3, "mesh_s030")
    surface = json.loads((out / "surface.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    declaration = json.loads((out / "groups.json").read_text(encoding="utf-8"))

    assert [g["wake_z"] for g in manifest["groups"]] == [
        0.0, pytest.approx(0.24, abs=1e-12)]
    first = declaration["groups"][0]["wake_faces"]
    second = declaration["groups"][1]["wake_faces"]
    assert not set(first) & set(second), "wake sheets must not share triangles"
    assert len(first) + len(second) == len(surface["wake_faces"])
    import numpy as np
    coordinates = np.asarray(surface["coordinates"], dtype=float)
    wake = np.asarray(surface["wake_faces"], dtype=int)
    upper_z = coordinates[wake[second]][:, :, 2]
    assert float(np.min(upper_z)) == pytest.approx(0.24, abs=1e-9)

    report = run(out / "surface.json", groups_path=out / "groups.json",
                 gradient=False, v1_check=False, max_panels=4000)
    widths = [int(b - a) for a, b in report["solve"]["group_slices"]]
    expected = [int(g["circulation_unknowns"]) for g in manifest["groups"]]
    assert widths == expected
    assert report["solve"]["unknowns"] == report["solve"]["wall_panels"] + sum(widths)
    assert report["solve"]["linear_residual_relative"] < 1e-10
    assert report["checks"]["block_dimensions"]["ok"]
    assert report["checks"]["kutta_rows"]["max_rms"] < 1e-12
    assert [round(float(g["wake_z"]), 12) for g in report["groups"]] == [0.0, 0.24]
    # The v1 oracle does not apply to a two-sheet surface, and it must say so
    # instead of silently comparing only the first sheet.
    assert report["checks"]["v1_equivalence"]["performed"] is False
    assert report["checks"]["full_oracle"]["performed"] is False
    assert report["physical_accuracy_accepted"] is False


def test_builder_rejects_an_empty_declaration_and_records_the_failure():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    empty = SCRATCH / "empty_groups.json"
    empty.write_text(json.dumps(dict(schema="dpanel.half_wake_groups_declaration.v1",
                                     groups=[])), encoding="utf-8")
    out = SCRATCH / "mesh_empty"
    if out.exists():
        import shutil
        shutil.rmtree(out)
    command = [sys.executable, "-m", "panel.cad_half_groups",
               "--wing-cad", str(CASE / "two_layer.step"),
               "--wake-groups", str(empty), "--out", str(out),
               "--chord", "1.0", "--reference-area", "4.8", "--size", "0.4",
               "--far-size", "1.5", "--radius", "5", "--wake-strips", "4",
               "--alpha-deg", "5", "--ncpu", "4"]
    completed = subprocess.run(command, cwd=str(ROOT), capture_output=True, text=True)
    assert completed.returncode != 0
    assert not (out / "surface.json").exists()
    failure = json.loads((out / "failure.json").read_text(encoding="utf-8"))
    assert "contains no groups" in failure["message"]