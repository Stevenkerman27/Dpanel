"""Compare the optimized validator and geometry derivatives with a Git baseline.

  python -m tools.verify_shed_group_optimization --baseline 5ca04e8 --surface runs/web/20261105-141850-a135fd/mesh/surface.json --out runs/verification/web_performance/equivalence.json

The baseline module is loaded in memory, without another checkout or source copy.
Geometry gradients are compared to the baseline, not certified against physics.
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time
from types import ModuleType

import jax
import jax.numpy as jnp
import numpy as np

from panel import shed_groups as current
from panel.differentiable import linear_solve
from panel.half_native import _half_loads
from panel.run_half_groups import load_surface, single_group_from_surface


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--baseline", required=True)
parser.add_argument("--surface", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
source = subprocess.run(["git", "show", args.baseline + ":panel/shed_groups.py"],
                        cwd=root, check=True, capture_output=True, text=True).stdout
baseline = ModuleType("panel._baseline_shed_groups")
baseline.__package__ = "panel"
sys.modules[baseline.__name__] = baseline
exec(compile(source, "<baseline shed_groups.py>", "exec"), baseline.__dict__)
_, surface = load_surface(args.surface)
_, small = load_surface(root / "cases/panel_half/thick_wing_two_strip.json")
cases = [("original_mesh", surface, 20000), ("small", small, 20000)]

# Extra non-adjacent wall panels exercise both exact intersection rejection and
# broad-phase overlaps that must not be mistaken for intersections.
for label, triangle, budget in [
    ("piercing", [[2., .15, -.2], [2., .35, .2], [2., .15, .2]], 20000),
    ("clear", [[2., .15, .8], [2., .35, 1.2], [2., .15, 1.2]], 20000),
    ("touching", [[2., .15, 0.], [2., .35, .2], [2., .15, .2]], 20000),
    ("near_tolerance", [[2., .15, 1e-10], [2., .35, .2], [2., .15, .2]], 20000),
    ("outside_tolerance", [[2., .15, 1e-8], [2., .35, .2], [2., .15, .2]], 20000),
    ("budget", [[2., .15, 0.], [2., .35, .2], [2., .15, .2]], 0),
]:
    case = copy.deepcopy(small)
    n = len(case["vertices"])
    case["vertices"] = np.concatenate([case["vertices"], triangle])
    case["faces"] = np.concatenate([case["faces"], [[n, n + 1, n + 2]]])
    case["component_ids"] = np.append(case["component_ids"], 20)
    cases.append((label, case, budget))

report = dict(baseline=args.baseline, surface=str(args.surface), validation=[], geometry=[])
for label, case, budget in cases:
    results = []
    for module in (baseline, current):
        calls = []
        segment_test = module._segment_triangle_t

        def record_segment(p, q, triangle, tol):
            calls.append(np.concatenate([p, q, triangle.ravel()]).tolist())
            return segment_test(p, q, triangle, tol)

        module._segment_triangle_t = record_segment
        started = time.perf_counter()
        try:
            topology = module.validate_shed_groups(
                case["vertices"], case["faces"], case["component_ids"],
                [single_group_from_surface(case)], max_wake_wall_tests=budget)
            result = dict(status="accepted", audit=topology.audit)
        except ValueError as error:
            result = dict(status="rejected", error=str(error))
        finally:
            module._segment_triangle_t = segment_test
        result.update(seconds=time.perf_counter() - started, segment_calls=calls)
        results.append(result)
    before, after = results
    equal = {k: v for k, v in before.items() if k != "seconds"} == {
        k: v for k, v in after.items() if k != "seconds"}
    report["validation"].append(dict(case=label, equal=equal, baseline=before, optimized=after))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not equal:
        raise AssertionError("Validation/candidate mismatch: " + label)
    print("%s: identical %s; %.3f -> %.3f seconds" %
          (label, after["status"], before["seconds"], after["seconds"]), flush=True)

for dihedral in (0., 10.):
    vertices = small["vertices"].copy()
    vertices[:, 2] += vertices[:, 1] * np.tan(np.deg2rad(dihedral))
    values, gradients = [], []
    for module in (baseline, current):
        group = single_group_from_surface(small)
        group.wake_z = None
        topology = module.validate_shed_groups(vertices, small["faces"],
                                               small["component_ids"], [group])
        neighbors = module.pressure_neighbors(vertices, topology)
        tags = module._WallTags(topology.wall_faces, topology.component_ids,
                               jnp.asarray(neighbors))

        def objective(scales):
            x = jnp.asarray(vertices) * scales
            assembled = module.assemble_half_groups(x, topology, small["freestream"])
            state = linear_solve(assembled["matrix"], assembled["rhs"])
            loads = _half_loads(x, tags, jnp.asarray(small["freestream"]),
                                small["density"], small["reference_area"],
                                small["reference_chord"],
                                jnp.asarray(small["reference_point"]),
                                state[:assembled["wall_panels"]])
            return loads["CL"]

        value, derivative = jax.value_and_grad(objective)(jnp.ones(3))
        values.append(float(value))
        gradients.append(np.asarray(derivative))
    np.testing.assert_allclose(values[0], values[1], rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(gradients[0], gradients[1], rtol=1e-11, atol=1e-12)
    assert np.isfinite(values).all() and np.isfinite(gradients).all()
    report["geometry"].append(dict(
        dihedral_deg=dihedral, baseline_cl=values[0], optimized_cl=values[1],
        baseline_gradient=gradients[0].tolist(), optimized_gradient=gradients[1].tolist(),
        max_gradient_difference=float(np.max(np.abs(gradients[0] - gradients[1])))))
report["correct"] = True
args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
print("Geometry scale gradients at 0/10 degrees match the baseline.", flush=True)
