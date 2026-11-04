"""Job pipeline: parameters -> STEP -> half surface -> solve -> summary."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path

from . import geometry
from . import params as params_module

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs" / "web"
ENV = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")


class Cancelled(RuntimeError):
    """Raised when the user cancels a running job."""


class Job:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.log = []
        self.state = "built"
        self.result = None
        self.error = None
        self.build = None
        self.cancel = threading.Event()
        self.proc = None
        self.started = time.time()
        self.finished = None

    def say(self, message):
        self.log.append("[%s] %s" % (time.strftime("%H:%M:%S"), message))
        if len(self.log) > 500:
            del self.log[:250]

    def check(self):
        if self.cancel.is_set():
            raise Cancelled("用户取消")


def new_job():
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS / (stamp + "-" + uuid.uuid4().hex[:6])
    run_dir.mkdir(parents=True, exist_ok=False)
    return Job(run_dir)


def _plain(value):
    """Convert numpy scalars / arrays and nested containers to JSON-safe values."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, "tolist"):
        return _plain(value.tolist())
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, int):
        return int(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return number if number == number and abs(number) != float("inf") else None


def build(job, raw, count=41):
    """Write STEP + TE declaration and return the display mesh payload."""
    params, clamped = params_module.normalise(raw)
    if clamped:
        job.say("参数超出范围，已截断: " + ", ".join(clamped))
    step_path = job.run_dir / "wing.step"
    te_path = job.run_dir / "trailing_edge.json"
    job.say("构建 OCC 实体 ...")
    # gmsh 只能在主线程 initialize（它装 SIGINT handler），而这里跑在 HTTP 线程，
    # 所以建体走子进程。子进程 stdout 同样落文件（沙箱禁止管道 stdio）。
    params_path = job.run_dir / "params.json"
    params_path.write_text(json.dumps(raw), encoding="utf-8")
    log_path = job.run_dir / "build.log"
    command = [sys.executable, "-m", "tools.web.step_builder",
               "--params", str(params_path), "--out", str(job.run_dir),
               "--count", str(int(count))]
    with open(log_path, "w", encoding="utf-8", errors="replace") as fh:
        job.proc = subprocess.Popen(command, cwd=str(ROOT),
                                    env=dict(ENV, PYTHONIOENCODING="utf-8"),
                                    stdout=fh, stderr=subprocess.STDOUT)
        job.proc.wait()
        code = job.proc.returncode
    job.proc = None
    if code != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
        raise RuntimeError("建体失败 (exit %s): %s" % (code, " | ".join(tail[-6:])))
    payload = json.loads((job.run_dir / "build.json").read_text(encoding="utf-8"))
    payload["clamped"] = clamped or payload.get("clamped") or []
    info = payload["step"]
    job.say("STEP 已写出: 体积 %.4f m^3, %d 个截面" % (info["volume"], info["sections"]))
    mesh = geometry.display_mesh(params, count=count)
    payload = _plain(dict(payload, mesh=mesh))
    (job.run_dir / "build.json").write_text(json.dumps(payload), encoding="utf-8")
    job.build = payload
    job.state = "built"
    return payload


def _forward(job, log_path, emitted):
    """Relay freshly written log lines of the mesh subprocess into the job log."""
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return
    while emitted[0] < len(lines):
        line = lines[emitted[0]].strip()
        emitted[0] += 1
        # Gmsh dumps one "Label 'Shapes/...'" line per STEP entity; that is pure noise.
        if line and "Label 'Shapes/" not in line and "Open CASCADE STEP translator" not in line:
            job.say("  " + line[:160])


def mesh_surface(job, params, size, alpha_deg, ncpu=8):
    """Run `panel.cad_half` on the built STEP solid.

    `panel/cad_half.py` works on geometry normalized by the reference chord, so
    the physical mesh size / far-field radius have to be divided by that chord
    before they are passed on the command line.
    """
    out = job.run_dir / "mesh"
    if out.exists():
        shutil.rmtree(out)
    chord = params_module.reference(params)["chord"]
    te = params_module.trailing_edge_points(params)
    reach = max(max(abs(point[0]), abs(point[1])) for point in te)
    size_norm = float(size) / chord
    radius_norm = 1.5 * reach + 4.0 * size_norm
    far_norm = 0.3 * radius_norm
    command = [sys.executable, "-m", "panel.cad_half",
               "--wing-cad", str(job.run_dir / "wing.step"),
               "--trailing-edge", str(job.run_dir / "trailing_edge.json"),
               "--out", str(out), "--size", "%g" % size_norm,
               "--far-size", "%g" % far_norm, "--radius", "%g" % radius_norm,
               "--wake-length", "30",
               "--alpha-deg", "%g" % float(alpha_deg), "--ncpu", str(int(ncpu))]
    job.say("网格划分: size=%g m (%g 弦), alpha=%g, 远场半径=%g 弦"
            % (float(size), size_norm, float(alpha_deg), radius_norm))
    # 子进程的 stdout 必须落到文件: 本机沙箱禁止管道 stdio（spawn 会 EPERM）。
    log_path = job.run_dir / "mesh.log"
    emitted = [0]
    with open(log_path, "w", encoding="utf-8", errors="replace") as fh:
        job.proc = subprocess.Popen(command, cwd=str(ROOT),
                                    env=dict(ENV, PYTHONIOENCODING="utf-8"),
                                    stdout=fh, stderr=subprocess.STDOUT)
        while job.proc.poll() is None:
            if job.cancel.is_set():
                job.proc.kill()
                job.proc.wait()
                job.proc = None
                raise Cancelled("用户取消")
            _forward(job, log_path, emitted)
            time.sleep(0.5)
        code = job.proc.returncode
    job.proc = None
    _forward(job, log_path, emitted)
    lines = [line for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    tail = lines[-8:]
    if code != 0:
        raise RuntimeError("cad_half 失败 (exit %s): %s" % (code, " | ".join(tail[-6:])))
    surface = out / "surface.json"
    if not surface.exists():
        raise RuntimeError("cad_half 未写出 surface.json: " + " | ".join(tail[-6:]))
    data = json.loads(surface.read_text(encoding="utf-8"))
    panels = len(data.get("faces") or [])
    wake = len(data.get("wake_faces") or [])
    job.say("网格完成: 壁面 %d 面板, 尾迹 %d 面板" % (panels, wake))
    return surface, dict(panels=panels, wake_panels=wake,
                         vertices=len(data.get("coordinates") or []))


def _solve_fixed(job, surface):
    from panel.run_half_groups import run as run_half
    job.say("稠密参考解（半域, 固定尾迹） ...")
    report = run_half(surface, gradient=False, v1_check=False, max_panels=20000)
    solve = report["solve"]
    rows = []
    for row in report["group_rows"]:
        gamma = [float(v) for v in row["gamma"]]
        n = len(gamma)
        rows.append(dict(group=row["group_id"],
                         gamma=[[i / max(n - 1, 1), g] for i, g in enumerate(gamma)],
                         gamma_sum=row["gamma_sum"],
                         kutta_residual_rms=row["kutta_residual_rms"]))
    return dict(mode="fixed", CL=solve["CL"], CD=solve["CD_pressure"], CM=solve["CM"],
                cp=[solve["Cp_min"], solve["Cp_max"]],
                residual=solve["linear_residual_relative"],
                seconds=solve["hot_seconds"], unknowns=solve["unknowns"],
                panels=solve["wall_panels"], group_rows=rows, history=None)


def _solve_free(job, surface, alpha, iterations, reference):
    from panel.p1_lifting_g1 import from_dpanel_half_surface
    from panel.p1_lifting_g2 import combine_wings
    from panel.p1_steady_free_wake import solve_steady_free_wake
    job.say("装载半域面板 ...")
    wing = from_dpanel_half_surface(surface)
    model = combine_wings([wing], reference_area=reference["area"],
                          reference_chord=reference["chord"],
                          reference_point=tuple(reference["point"]))
    job.say("自由尾迹: alpha=%g, iterations=%d, 每轮同时重解 mu/Gamma" % (alpha, iterations))

    def save(iteration, mesh, state, record):
        job.check()
        cl = record.get("CL")
        jump = record.get("jump_rms")
        job.say("iter %d: CL=%s jump_rms=%s" % (
            iteration,
            "-" if cl is None else "%+.6f" % float(cl),
            "-" if jump is None else "%.2e" % float(jump)))

    value = solve_steady_free_wake(model, alpha, kutta="pressure",
                                   iterations=int(iterations), relaxation=0.25,
                                   save_callback=save)
    history = []
    for row in value["history"]:
        history.append(dict(iteration=row.get("iteration"), CL=row.get("CL"),
                            CD=row.get("CD_pressure"), jump_rms=row.get("jump_rms")))
    final = value["state"]
    rows = []
    for result in final.get("group_results", []):
        gamma = [float(v) for v in result["gamma"]]
        n = len(gamma)
        rows.append(dict(group=result["name"],
                         gamma=[[i / max(n - 1, 1), g] for i, g in enumerate(gamma)],
                         gamma_sum=result.get("Gamma_integral"),
                         CL_gamma=result.get("CL_gamma")))
    last = history[-1] if history else {}
    return dict(mode="free", CL=last.get("CL"), CD=last.get("CD"),
                CM=final.get("CM_y"), cp=None, residual=None, seconds=value.get("seconds"),
                converged=bool(value.get("converged")),
                geometry_stable=bool(value.get("geometry_stable")),
                iterations=value.get("iterations"), history=history,
                group_rows=rows, group_results=_plain(final.get("group_results")))


def run_job(job, options):
    """Mesh and solve; called on a worker thread."""
    alpha = float(options.get("alpha", 4.0))
    wake = str(options.get("wake", "fixed"))
    device = str(options.get("device", "cpu"))
    size = float(options.get("size", 0.3))
    iterations = int(options.get("iterations", 12))
    try:
        job.state = "meshing"
        surface, mesh_info = mesh_surface(job, (job.build or {}).get("params") or
                                          params_module.normalise({})[0], size, alpha)
        job.check()
        job.state = "solving"
        if wake == "free":
            result = _solve_free(job, surface, alpha, iterations,
                                 (job.build or {}).get("reference") or params_module.reference(
                                     (job.build or {}).get("params") or params_module.normalise({})[0]))
        else:
            result = _solve_fixed(job, surface)
        result = _plain(result)
        if wake == "free" and not (result.get("converged") and result.get("geometry_stable")):
            history = result.get("history") or []
            first = history[0]["CL"] if history else None
            last = result.get("CL")
            if first is not None and last is not None:
                result["warning"] = ("自由尾迹未收敛（几何不稳定）：CL 由 %+.3f 降到 %+.3f。"
                                     "这是已知的尾迹塌陷问题，数值仅供诊断，不可当作气动结论。"
                                     % (first, last))
            else:
                result["warning"] = "自由尾迹未收敛：数值仅供诊断。"
            job.say(result["warning"])
        result["options"] = dict(alpha=alpha, wake=wake, size=size, iterations=iterations,
                                 device_requested=device, device_used="cpu")
        if device != "cpu":
            result["device_note"] = ("当前两条求解路径（半域稠密参考解与自由尾迹）"
                                     "只暴露 CPU 设备; GPU 仅在 p1 单翼 torch 路径可选。")
        result["mesh_info"] = mesh_info
        result["alpha"] = alpha
        result["seconds_total"] = time.time() - job.started
        (job.run_dir / "summary.json").write_text(json.dumps(result), encoding="utf-8")
        job.result = result
        job.state = "done"
        job.say("完成: CL=%s" % ("-" if result.get("CL") is None else "%+.6f" % result["CL"]))
    except Cancelled as exc:
        job.state = "cancelled"
        job.say(str(exc))
    except Exception as exc:  # noqa: BLE001 - reported to the browser
        job.state = "error"
        job.error = "%s: %s" % (type(exc).__name__, exc)
        if "Non-manifold" in str(exc):
            job.error += "（网格过粗或展向站数过少：请减小网格尺寸或增加展向站数后重试）"
        job.say(job.error)
        (job.run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        job.finished = time.time()
        if job.proc is not None:
            try:
                job.proc.kill()
            except Exception:
                pass
            job.proc = None
