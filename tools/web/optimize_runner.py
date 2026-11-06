"""SLSQP section optimization on the current GRASS fixed-topology mesh.

Reads the mesh written by tools.web.grass_mesh_runner, rebuilds the differentiable
P1 fixed-wake model for every candidate design and differentiates CL/CD/CDi with
respect to (chord, twist, 17 CST weights) of every section.  Leading-edge
coordinates and the last CST value (sharp trailing edge) stay fixed so the
topology is preserved.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize

from panel.p1_cst_gradient import fixed_wake_geometry, grass_section_geometry

parser = argparse.ArgumentParser()
parser.add_argument("--surface", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--alpha", type=float, default=4.0)
parser.add_argument("--objective", choices=("pressure", "induced", "lift"),
                    default="induced")
parser.add_argument("--cl-min", type=float, default=None)
parser.add_argument("--keep-area", action="store_true")
parser.add_argument("--iterations", type=int, default=20)
parser.add_argument("--max-memory-gb", type=float, default=10.0)
args = parser.parse_args()

data = json.loads(args.surface.read_text(encoding="utf-8"))
panels = len(data["faces"])
# Measured on this machine (2 sections, GRASS fixed topology, 38 variables):
# 460 faces -> 1.8 GB peak / 3.5 s per evaluation, 1720 faces -> 16 GB / 40 s.
# The assembly and its graph scale like O(faces * unknowns) with a large constant,
# so a 0.05 m mesh needs tens of GB and takes the process down before evaluation 1.
memory_gb = 5.7e-6 * panels * panels
print("网格 %d 面，预估峰值内存 %.1f GB" % (panels, memory_gb), flush=True)
if memory_gb > args.max_memory_gb:
    raise SystemExit(
        "网格 %d 面预估峰值内存 %.1f GB，超过上限 %.1f GB；请把网格尺寸放大到 0.2 m 以上"
        "（约 460 面）或减少截面后再优化（--max-memory-gb 可强制放开）"
        % (panels, memory_gb, args.max_memory_gb))
sections0 = data["sections"]
n = len(sections0["chord"])
chord_panels = int(data["chord_panels"])
span_panels = int(data["span_panels_per_segment"])
wake_panels = int(data.get("wake_panels", 2))
wake_length = float(data.get("wake_length", 10.0))

leading = torch.tensor(np.asarray(sections0["leading_edge_xyz"], dtype=float), dtype=torch.float64)
chord0 = np.asarray(sections0["chord"], dtype=float)
twist0 = np.asarray(sections0["twist_radians"], dtype=float)
cst0 = np.asarray(sections0["cst"], dtype=float)[:, :17]
y = np.asarray(sections0["leading_edge_xyz"], dtype=float)[:, 1]
area0 = float(2 * np.trapezoid(chord0, y))

d0 = np.column_stack((chord0, twist0, cst0))
x0 = d0.ravel()
span = np.r_[chord0 * 0.25, np.full(n, np.deg2rad(3.0)), np.full((n, 17), 0.04).ravel()]

history = []
best = {"value": float("inf"), "x": x0.copy()}
failures = [0]
cache = {}
last_good = dict(CL=float("nan"), CD=float("nan"), CDi=float("nan"), area=area0, value=0.0)


def value_from(values):
    """Objective value; pressure drag can go slightly negative on a coarse mesh,
    so a lift-to-drag objective is deliberately not offered."""
    return {"pressure": values[1], "induced": values[2], "lift": -values[0]}[args.objective]


def jac_from(values, jac):
    return {"pressure": jac[1], "induced": jac[2], "lift": -jac[0]}[args.objective]


def evaluate(x):
    if "x" in cache and np.array_equal(x, cache["x"]):
        return cache
    variable = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    matrix = variable.reshape(n, 19)
    geometry = dict(leading_edge_xyz=leading, chord=matrix[:, 0],
                    twist_radians=matrix[:, 1],
                    cst=torch.cat((matrix[:, 2:], torch.zeros((n, 1), dtype=torch.float64)), dim=1))
    try:
        wing, xyz, wake, area = grass_section_geometry(
            geometry, chord_panels=chord_panels, span_panels_per_segment=span_panels,
            wake_panels=wake_panels, wake_length=wake_length)
        values, info = fixed_wake_geometry(wing, xyz, wake, area, args.alpha, 160,
                                           include_ideal_drag=True)
        derivatives = [torch.autograd.grad(value, variable, retain_graph=True)[0].detach().numpy()
                       for value in (values[0], values[1], values[2], area)]
        jac = derivatives[:3]
        record = dict(evaluation=len(history) + 1,
                      objective=float(value_from(values).detach()),
                      CL=float(values[0].detach()), CD=float(values[1].detach()),
                      CDi=float(values[2].detach()), area=float(area.detach()),
                      stationarity=float(info["stationarity"]))
        history.append(record)
        with (args.out.parent / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print("eval %d  CL %.6f  CD %.6f  CDi %.6f  目标 %.6f" %
              (record["evaluation"], record["CL"], record["CD"], record["CDi"],
               record["objective"]), flush=True)
        cache.clear()
        cache.update(x=x.copy(), value=record["objective"], jac=jac_from(values, jac),
                     CL=record["CL"], CD=record["CD"], CDi=record["CDi"],
                     area=record["area"], area_jac=derivatives[3], ok=True)
        if record["objective"] < best["value"]:
            best.update(value=record["objective"], x=x.copy())
        last_good.update(CL=record["CL"], CD=record["CD"], CDi=record["CDi"],
                         area=record["area"], value=record["objective"])
        failures[0] = 0
        return cache
    except Exception as exc:  # keep the optimizer alive on a bad design point
        failures[0] += 1
        message = "%s: %s" % (type(exc).__name__, exc)
        with (args.out.parent / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(error=message, x=x.tolist()), allow_nan=False) + "\n")
        print("eval 目标点失败 (%d/5): %s" % (failures[0], message[:200]), flush=True)
        if failures[0] >= 5:
            raise RuntimeError("连续 5 个设计点无法建立可微模型: " + message)
        cache.clear()
        cache.update(x=x.copy(), value=last_good["value"] + 10.0 * failures[0],
                     jac=np.zeros_like(x), ok=False, CL=last_good["CL"],
                     CD=last_good["CD"], CDi=last_good["CDi"],
                     area=last_good["area"], area_jac=np.zeros_like(x))
        return cache


def fun(x):
    return evaluate(x)["value"]


# SLSQP is solved in scaled variables z in [-1, 1]: x = x0 + z * span.  The raw
# design vector mixes metres, radians and CST weights, which ruins the QP model.
def to_x(z):
    return x0 + np.asarray(z, dtype=float) * span


def fun(z):
    return evaluate(to_x(z))["value"]


def fun_jac(z):
    return evaluate(to_x(z))["jac"] * span


constraints = []
if args.cl_min is not None:
    constraints.append(dict(
        type="ineq",
        fun=lambda z: evaluate(to_x(z))["CL"] - args.cl_min,
        jac=lambda z: evaluate(to_x(z))["jac"][0] * span))
if args.keep_area:
    constraints.append(dict(
        type="eq",
        fun=lambda z: evaluate(to_x(z))["area"] - area0,
        jac=lambda z: evaluate(to_x(z))["area_jac"] * span))

constraint_text = "升力下限 %.4f" % args.cl_min if args.cl_min is not None else "无"
if args.keep_area:
    constraint_text += "，参考面积固定"
print("设计变量 %d 个（%d 截面 × 弦长/扭转/17 个 CST），目标=%s，约束=%s，迭代=%d" %
      (len(x0), n, args.objective, constraint_text, args.iterations), flush=True)
if args.objective != "lift" and args.cl_min is None:
    print("警告：目标是最小阻力但没有升力下限，最优解会退化为零升力"
          "（实测 0.1 m 网格 CL 0.349 → 0.060）；建议设置升力下限", flush=True)
baseline = dict(evaluate(x0))
base_value = baseline["value"]
result = minimize(fun, np.zeros_like(x0), jac=fun_jac, bounds=[(-1.0, 1.0)] * len(x0),
                  constraints=constraints, method="SLSQP",
                  options=dict(maxiter=int(args.iterations), ftol=1e-8, disp=False))

final_x = to_x(result.x)
final_point = evaluate(final_x)
matrix = final_x.reshape(n, 19)
final_sections = dict(leading_edge_xyz=sections0["leading_edge_xyz"],
                      chord=matrix[:, 0].tolist(),
                      twist_radians=matrix[:, 1].tolist(),
                      cst=np.column_stack((matrix[:, 2:], np.zeros(n))).tolist())
report = dict(
    success=bool(result.success), status=int(result.status), message=str(result.message),
    iterations=int(getattr(result, "nit", 0)), evaluations=len(history),
    objective=args.objective, cl_min=args.cl_min, keep_area=bool(args.keep_area),
    alpha=args.alpha, sections_initial=sections0, sections_final=final_sections,
    initial=dict(CL=baseline["CL"], CD=baseline["CD"], CDi=baseline["CDi"],
                 area=baseline["area"], objective=base_value),
    final=dict(CL=final_point["CL"], CD=final_point["CD"], CDi=final_point["CDi"],
               area=final_point["area"], objective=final_point["value"]),
    bounds="弦长 ±25%，扭转 ±3°，CST 权重 ±0.04",
    reference_area=area0, panel_count=len(data["faces"]),
    physical_accuracy_accepted=False)
args.out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
print(json.dumps(dict(success=report["success"], message=report["message"],
                      initial=report["initial"], final=report["final"])), flush=True)
