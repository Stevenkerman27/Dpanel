"""SLSQP section optimization on the current GRASS fixed-topology mesh.

Reads the mesh written by tools.web.grass_mesh_runner, rebuilds the differentiable
P1 fixed-wake model for every candidate design and differentiates CL/CD/CDi with
respect to (chord, twist, 17 CST weights) of every section.  Leading-edge
coordinates and the last CST value (sharp trailing edge) stay fixed so the
topology is preserved.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize

from panel.p1_cst_gradient import fixed_wake_geometry, grass_section_geometry

STARTED = time.time()


def elapsed_text():
    seconds = int(time.time() - STARTED)
    if seconds < 60:
        return "%d 秒" % seconds
    return "%d 分 %02d 秒" % (seconds // 60, seconds % 60)

parser = argparse.ArgumentParser()
parser.add_argument("--surface", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--alpha", type=float, default=4.0)
parser.add_argument("--objective", choices=("pressure", "induced", "lift"),
                    default="induced")
parser.add_argument("--cl-min", type=float, default=None)
parser.add_argument("--cl-target", type=float, default=None)
parser.add_argument("--consistency-tolerance", type=float, default=0.0)
parser.add_argument("--cl-gamma-min", type=float, default=None)
parser.add_argument("--ideal-drag", choices=("quadrature", "exact"),
                    default="exact")
parser.add_argument("--keep-area", action="store_true")
parser.add_argument("--kutta", choices=("pressure", "potential_jump"), default="pressure")
parser.add_argument("--iterations", type=int, default=20)
parser.add_argument("--max-memory-gb", type=float, default=10.0)
args = parser.parse_args()
need_cl_gamma = True  # the UI always shows circulation lift next to pressure lift

data = json.loads(args.surface.read_text(encoding="utf-8"))
panels = len(data["faces"])
# Measured on this machine (2 sections, GRASS fixed topology, 38 variables) after the
# per-block checkpoint fix in panel/p1_assembly_torch.py: 1720 faces -> 1.84 GB peak /
# 25 s per evaluation, 3400 faces -> 3.85 GB.  Before that fix the same meshes needed
# 15 GB and 30+ GB and the process died mid-run, which is why the guard exists at all.
memory_gb = max(0.3, 1.2e-3 * panels - 0.2)
print("网格 %d 面，预估峰值内存 %.1f GB" % (panels, memory_gb), flush=True)
if memory_gb > args.max_memory_gb:
    raise SystemExit(
        "网格 %d 面预估峰值内存 %.1f GB，超过上限 %.1f GB；请放大网格尺寸"
        "或减少截面后再优化（--max-memory-gb 可强制放开）"
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


OBJECTIVE_ROW = {"pressure": "CD", "induced": "CDi", "lift": "CL"}
# A row is only worth a backward pass if SLSQP reads it: the objective, the lift
# constraints, the circulation-lift constraint, the area constraint.  CD and CLi are
# reported but never constrained, and each row costs a full assembly plus adjoint solve.
NEEDED_ROWS = {OBJECTIVE_ROW[args.objective]}
if args.cl_min is not None or args.cl_target is not None:
    NEEDED_ROWS.add("CL")
if args.cl_gamma_min is not None:
    NEEDED_ROWS.add("CL_gamma")
if args.keep_area:
    NEEDED_ROWS.add("area")


shape = {}


def metrics(point):
    # Aerodynamic numbers the UI shows next to the objective.
    cl, cd, cdi = point["CL"], point["CD"], point["CDi"]
    clg = point.get("CL_gamma") or point.get("cl_gamma")
    ar = shape.get("aspect_ratio")
    e_far = (clg ** 2 / (np.pi * ar * cdi)) if (ar and clg and cdi > 0) else None
    rel = ((cd - cdi) / cdi) if cdi else None
    return dict(e_far=e_far, near_far_rel=rel)


def evaluate(x):
    if "x" in cache and np.array_equal(x, cache["x"]):
        return cache
    print("eval %d 开始（已用 %s）" % (len(history) + 1, elapsed_text()), flush=True)
    variable = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    matrix = variable.reshape(n, 19)
    geometry = dict(leading_edge_xyz=leading, chord=matrix[:, 0],
                    twist_radians=matrix[:, 1],
                    cst=torch.cat((matrix[:, 2:], torch.zeros((n, 1), dtype=torch.float64)), dim=1))
    try:
        wing, xyz, wake, area = grass_section_geometry(
            geometry, chord_panels=chord_panels, span_panels_per_segment=span_panels,
            wake_panels=wake_panels, wake_length=wake_length)
        shape["aspect_ratio"] = (float(np.ptp(np.asarray(wing.gamma_y, float))) ** 2
                               / float(area.detach()))
        values, info = fixed_wake_geometry(wing, xyz, wake, area, args.alpha, 160, kutta=args.kutta,
                                           include_ideal_drag=True,
                                           include_circulation_lift=need_cl_gamma,
                                           ideal_drag=args.ideal_drag)
        outputs = {"CL": values[0], "CD": values[1], "CDi": values[2], "area": area}
        if need_cl_gamma:
            outputs["CL_gamma"] = values[3]
        derivatives = {}
        for name, output in outputs.items():
            if name in NEEDED_ROWS:
                derivatives[name] = torch.autograd.grad(
                    output, variable, retain_graph=True)[0].detach().numpy()
            else:
                derivatives[name] = np.zeros_like(x)
        jac = [derivatives["CL"], derivatives["CD"], derivatives["CDi"]]
        record = dict(evaluation=len(history) + 1,
                      objective=float(value_from(values).detach()),
                      CL=float(values[0].detach()), CD=float(values[1].detach()),
                      CDi=float(values[2].detach()), area=float(area.detach()),
                      CL_gamma=(float(values[3].detach()) if need_cl_gamma else None),
                      stationarity=float(info["stationarity"]))
        record.update(metrics(record))
        if (args.consistency_tolerance > 0 and record["near_far_rel"] is not None
                and abs(record["near_far_rel"]) > args.consistency_tolerance):
            raise ValueError("近场阻力与远场诱导阻力相差 %.1f%%，超过容差 %.1f%%；"                             "该设计点不可信（多半是弦向网格太粗，前缘吸力没分辨出来）"                             % (100 * record["near_far_rel"], 100 * args.consistency_tolerance))
        history.append(record)
        with (args.out.parent / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print("eval %d  CL %.6f  CD %.6f  CDi %.6f  近远差 %s  目标 %.6f  (用时 %s)" %
              (record["evaluation"], record["CL"], record["CD"], record["CDi"],
               "-" if record["near_far_rel"] is None else "%+.1f%%" % (100 * record["near_far_rel"]),
               record["objective"], elapsed_text()), flush=True)
        cache.clear()
        cache.update(x=x.copy(), value=record["objective"], jac=jac_from(values, jac),
                     CL=record["CL"], CD=record["CD"], CDi=record["CDi"],
                     area=record["area"], area_jac=derivatives["area"],
                     cl_jac=jac[0].copy(), cl_gamma=record["CL_gamma"],
                     cl_gamma_jac=(derivatives["CL_gamma"] if need_cl_gamma
                                   else np.zeros_like(x)),
                     ok=True)
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
                     area=last_good["area"], area_jac=np.zeros_like(x),
                     cl_jac=np.zeros_like(x), cl_gamma=None,
                     cl_gamma_jac=np.zeros_like(x))
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
if args.cl_target is not None:
    constraints.append(dict(
        type="eq",
        fun=lambda z: evaluate(to_x(z))["CL"] - args.cl_target,
        jac=lambda z: evaluate(to_x(z))["cl_jac"] * span))
if args.cl_min is not None:
    constraints.append(dict(
        type="ineq",
        fun=lambda z: evaluate(to_x(z))["CL"] - args.cl_min,
        jac=lambda z: evaluate(to_x(z))["cl_jac"] * span))
if args.cl_gamma_min is not None:
    constraints.append(dict(
        type="ineq",
        fun=lambda z: evaluate(to_x(z))["cl_gamma"] - args.cl_gamma_min,
        jac=lambda z: evaluate(to_x(z))["cl_gamma_jac"] * span))
if args.keep_area:
    constraints.append(dict(
        type="eq",
        fun=lambda z: evaluate(to_x(z))["area"] - area0,
        jac=lambda z: evaluate(to_x(z))["area_jac"] * span))

constraint_text = ""
if args.cl_min is not None:
    constraint_text += "升力下限 %.4f" % args.cl_min
if args.cl_target is not None:
    constraint_text += "升力等式 %.4f" % args.cl_target
if args.cl_gamma_min is not None:
    constraint_text += "，环量升力下限 %.4f" % args.cl_gamma_min
if args.keep_area:
    constraint_text += "，参考面积固定"
if not constraint_text:
    constraint_text = "无"
print("设计变量 %d 个（%d 截面 × 弦长/扭转/17 个 CST），目标=%s，约束=%s，迭代=%d" %
      (len(x0), n, args.objective, constraint_text, args.iterations), flush=True)
if args.objective != "lift" and args.cl_min is None:
    print("警告：目标是最小阻力但没有升力下限，最优解会退化为零升力"
          "（实测 0.1 m 网格 CL 0.349 → 0.060）；建议设置升力下限", flush=True)
baseline = dict(evaluate(x0))
base_value = baseline["value"]
iteration = [0]


def iteration_callback(z):
    iteration[0] += 1
    point = evaluate(to_x(z))
    print("SLSQP 迭代 %d：目标 %.6f  CL %.6f  CDi %.6f  (已用 %s)" %
          (iteration[0], point["value"], point["CL"], point["CDi"], elapsed_text()),
          flush=True)


result = minimize(fun, np.zeros_like(x0), jac=fun_jac, bounds=[(-1.0, 1.0)] * len(x0),
                  constraints=constraints, method="SLSQP", callback=iteration_callback,
                  options=dict(maxiter=int(args.iterations), ftol=1e-8, disp=False))

final_x = to_x(result.x)
final_point = evaluate(final_x)
matrix = final_x.reshape(n, 19)
final_sections = dict(leading_edge_xyz=sections0["leading_edge_xyz"],
                      chord=matrix[:, 0].tolist(),
                      twist_radians=matrix[:, 1].tolist(),
                      cst=np.column_stack((matrix[:, 2:], np.zeros(n))).tolist())
feasible = True
if args.cl_target is not None and abs(final_point["CL"] - args.cl_target) > 1e-6:
    feasible = False
if args.cl_min is not None and final_point["CL"] < args.cl_min - 1e-6:
    feasible = False
if args.cl_gamma_min is not None and final_point.get("cl_gamma", 0.0) < args.cl_gamma_min - 1e-6:
    feasible = False
if args.keep_area and abs(final_point["area"] - area0) > 1e-6 * abs(area0):
    feasible = False
report = dict(
    success=bool(result.success), feasible=bool(feasible),
    status=int(result.status), message=str(result.message),
    iterations=int(getattr(result, "nit", 0)), evaluations=len(history),
    objective=args.objective, cl_min=args.cl_min, cl_target=args.cl_target,
    cl_gamma_min=args.cl_gamma_min, consistency_tolerance=args.consistency_tolerance,
    history=history,
    ideal_drag=args.ideal_drag,
    keep_area=bool(args.keep_area),
    alpha=args.alpha, sections_initial=sections0, sections_final=final_sections,
    initial=dict(CL=baseline["CL"], CD=baseline["CD"], CDi=baseline["CDi"],
                 area=baseline["area"], objective=base_value,
                 CL_gamma=baseline.get("cl_gamma"), **metrics(baseline)),
    final=dict(CL=final_point["CL"], CD=final_point["CD"], CDi=final_point["CDi"],
               area=final_point["area"], objective=final_point["value"],
               CL_gamma=final_point.get("cl_gamma"), **metrics(final_point)),
    bounds="弦长 ±25%，扭转 ±3°，CST 权重 ±0.04",
    reference_area=area0, panel_count=len(data["faces"]),
    physical_accuracy_accepted=False)
args.out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
print(json.dumps(dict(success=report["success"], message=report["message"],
                      initial=report["initial"], final=report["final"])), flush=True)
