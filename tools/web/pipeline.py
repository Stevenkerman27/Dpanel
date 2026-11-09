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
SOLVE_THREADS = min(8, os.cpu_count() or 1)


class Cancelled(RuntimeError):
    """Raised when the user cancels a running job."""


class Job:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.log = []
        self.state = "built"
        self.result = None
        self.mesh_info = None
        self.mesh_options = None
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


def split_parts(payload):
    """Split the request into one lifting wing and any number of fuselages.

    Backwards compatible: a bare parameter dictionary (the original API) is one
    wing part with no bodies.
    """
    payload = payload or {}
    parts = payload.get("parts") if isinstance(payload, dict) else None
    if not parts:
        return (payload.get("params") if isinstance(payload, dict) else payload) or {}, []
    wings = [part for part in parts if str(part.get("kind", "wing")) == "wing"]
    bodies = [part for part in parts if str(part.get("kind")) == "body"]
    unknown = [part for part in parts
               if str(part.get("kind", "wing")) not in ("wing", "body")]
    if unknown:
        raise ValueError("未知的部件类型: " + ", ".join(str(p.get("kind")) for p in unknown))
    if len(wings) != 1:
        raise ValueError("需要且只需要一个升力面部件（求解器只有一套尾迹系统），"
                         "现在有 %d 个" % len(wings))
    return (wings[0].get("params") or wings[0] or {},
            [body.get("params") or body for body in bodies])


def build(job, payload, count=41):
    """Write STEP + TE declaration and return the display mesh payload."""
    raw, raw_bodies = split_parts(payload)
    params, clamped = params_module.normalise(raw)
    bodies = []
    for item in raw_bodies:
        body, body_clamped = params_module.normalise_body(item)
        bodies.append(body)
        clamped.extend(body_clamped)
    if clamped:
        job.say("参数超出范围，已截断: " + ", ".join(clamped))
    job.say("构建 OCC 实体 ...")
    # gmsh 只能在主线程 initialize（它装 SIGINT handler），而这里跑在 HTTP 线程，
    # 所以建体走子进程。子进程 stdout 同样落文件（沙箱禁止管道 stdio）。
    params_path = job.run_dir / "params.json"
    params_path.write_text(json.dumps(raw), encoding="utf-8")
    bodies_path = job.run_dir / "bodies.json"
    bodies_path.write_text(json.dumps(bodies), encoding="utf-8")
    log_path = job.run_dir / "build.log"
    command = [sys.executable, "-m", "tools.web.step_builder",
               "--params", str(params_path), "--bodies", str(bodies_path),
               "--out", str(job.run_dir), "--count", str(int(count))]
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
    mesh = payload.get("mesh") or geometry.display_mesh(params, count=count)
    if bodies:
        body_mesh = geometry.body_display_mesh(bodies)
        offset = len(mesh["vertices"])
        mesh["vertices"].extend(body_mesh["vertices"])
        mesh["faces"].extend([[index + offset for index in face] for face in body_mesh["faces"]])
        mesh["bodies"] = len(bodies)
    payload = _plain(dict(payload, parts=dict(wing=params, bodies=bodies), mesh=mesh))
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
    if "grass_sections" in params and not (job.run_dir / "body.step").exists():
        out.mkdir(parents=True)
        source = job.run_dir / "grass_mesh_input.json"
        source.write_text(json.dumps(params, allow_nan=False), encoding="utf-8")
        surface = out / "surface.json"
        command = [sys.executable, "-m", "tools.web.grass_mesh_runner",
                   "--input", str(source), "--out", str(surface), "--size", str(size)]
        job.say("GRASS 固定拓扑网格: size=%g m，前缘余弦加密" % float(size))
        code, log = _spawn(job, command, job.run_dir / "mesh.log")
        if code != 0 or not surface.is_file():
            lines = log.read_text(encoding="utf-8", errors="replace").strip().splitlines()
            raise RuntimeError("GRASS 网格失败: " + " | ".join(lines[-6:]))
        data = json.loads(surface.read_text(encoding="utf-8"))
        return surface, dict(panels=len(data["faces"]), wake_panels=len(data["wake_faces"]),
                             vertices=len(data["coordinates"]), mesh_type="grass_fixed_topology")
    chord = params_module.reference(params)["chord"]
    te = (job.build or {}).get("trailing_edge") or geometry.trailing_edge_points(params)
    reach = max(max(abs(point[0]), abs(point[1])) for point in te)
    size_norm = float(size) / chord
    radius_norm = 1.5 * reach + 4.0 * size_norm
    far_norm = 0.3 * radius_norm
    command = [sys.executable, "-m", "panel.cad_half",
               "--wing-cad", str(job.run_dir / "wing.step"),
               "--trailing-edge", str(job.run_dir / "trailing_edge.json"),
               "--mesh-features", str(job.run_dir / "mesh_features.json"),
               "--out", str(out), "--size", "%g" % size_norm,
               "--far-size", "%g" % far_norm, "--radius", "%g" % radius_norm,
               "--wake-length", "10",
               "--alpha-deg", "%g" % float(alpha_deg), "--ncpu", str(int(ncpu))]
    body_step = job.run_dir / "body.step"
    if body_step.exists():
        command[command.index("--alpha-deg"):command.index("--alpha-deg")] = [
            "--body-cad", str(body_step)]
    job.say("网格划分: size=%g m (%g 弦), alpha=%g, 远场半径=%g 弦"
            % (float(size), size_norm, float(alpha_deg), radius_norm))
    job.say("前缘局部加密: 目标尺寸 0.5 倍，影响半径 0.12c")
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


CRASH_HINT = ("求解进程发生原生异常，已自动重试；"
              "若仍失败，请保留运行日志并检查数值运行库。")
# Exit codes seen when the solver's native code takes the interpreter down.
CRASH_CODES = (-1073741819, 3221225477, -1073741571, 3221225725, 139)


def _exit_text(code):
    """Readable name for the abnormal exit codes seen on Windows."""
    if code in (-1073741819, 3221225477):
        return "0xC0000005 访问冲突"
    if code in (-1073741571, 3221225725):
        return "0xC00000FD 栈溢出"
    if code is not None and code < 0:
        return "0x%08X" % (code & 0xFFFFFFFF)
    return str(code)


def _spawn(job, command, log_path):
    """Run one child process, forwarding its log; return (exit code, log path)."""
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
            time.sleep(0.3)
        code = job.proc.returncode
    job.proc = None
    _forward(job, log_path, emitted)
    return code, log_path


def solve(job, surface, wake, alpha, iterations, reference, attempts=6,
          ncpu=SOLVE_THREADS, wall_formulation="galerkin"):
    """Run the solve in a child process, retrying native crashes.

    Isolation keeps the HTTP server alive when a native dependency crashes.
    Each retry starts from the same surface file; deterministic runtime defects
    still require repairing the dependency (see README).
    """
    last = None
    for attempt in range(attempts):
        if attempt:
            job.say("求解进程崩溃（%s），第 %d 次重试" % (_exit_text(last), attempt))
            time.sleep(0.5)
        directory = job.run_dir / ("solve" if attempt == 0 else "solve_retry%d" % attempt)
        log_path = job.run_dir / ("solve.log" if attempt == 0 else "solve_retry%d.log" % attempt)
        command = [sys.executable, "-m", "tools.web.solve_runner",
                   "--surface", str(surface), "--out", str(directory),
                   "--wake", wake, "--alpha", "%g" % float(alpha),
                   "--iterations", str(int(iterations)),
                   "--reference", json.dumps(reference or {}),
                   "--wall-formulation", wall_formulation, "--ncpu", str(ncpu)]
        if attempt == 0:
            job.say("求解进程启动: wake=%s, alpha=%g, 设备=CPU, 线程=%d"
                    % (wake, float(alpha), ncpu))
        code, log_path = _spawn(job, command, log_path)
        result_path = directory / "result.json"
        if code == 0 and result_path.exists():
            return json.loads(result_path.read_text(encoding="utf-8"))
        lines = [line.strip() for line in
                 log_path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
        if code not in CRASH_CODES:
            raise RuntimeError("求解进程异常退出 (exit %s %s)：%s。%s"
                               % (code, _exit_text(code), " | ".join(lines[-4:]), CRASH_HINT))
        last = code
    raise RuntimeError("求解进程连续 %d 次崩溃 (exit %s %s)，已放弃。%s"
                       % (attempts, last, _exit_text(last), CRASH_HINT))


def run_mesh(job, options):
    """Generate and save the solver mesh without starting the solve."""
    alpha = float(options.get("alpha", 4.0))
    size = float(options.get("size", 0.3))
    try:
        job.mesh_info = None
        job.mesh_options = None
        job.result = None
        job.state = "meshing"
        _, mesh_info = mesh_surface(job, (job.build or {}).get("params") or
                                    params_module.normalise({})[0], size, alpha)
        job.check()
        job.mesh_info = mesh_info
        job.mesh_options = dict(size=size, alpha=alpha)
        job.state = "meshed"
        job.say("网格已生成，可以开始求解")
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


def run_grass(job, options):
    """Differentiate the current production fixed-topology mesh in a child."""
    try:
        job.state = "solving"
        source = job.run_dir / "mesh" / "surface.json"
        if not source.is_file() or not job.mesh_info or not job.mesh_options:
            raise ValueError("请先生成当前 GRASS 计算网格")
        if job.mesh_info.get("mesh_type") != "grass_fixed_topology":
            raise ValueError("当前网格不是 GRASS 固定拓扑网格，机身/CAD 网格暂不支持此梯度")
        if float(options.get("alpha", 4.)) != job.mesh_options["alpha"]:
            raise ValueError("迎角已变化，请重新生成网格")
        job.say("使用当前正式求解网格计算 GRASS 固定尾迹梯度")
        output = job.run_dir / "grass_result.json"
        code, log = _spawn(job, [sys.executable, "-m", "tools.web.grass_runner",
                                 "--input", str(source), "--out", str(output),
                                 "--alpha", str(job.mesh_options["alpha"])],
                           job.run_dir / "grass.log")
        if code != 0 or not output.is_file():
            lines = log.read_text(encoding="utf-8", errors="replace").strip().splitlines()
            raise RuntimeError("GRASS 截面求解失败: " + " | ".join(lines[-5:]))
        job.result = json.loads(output.read_text(encoding="utf-8"))
        job.state = "done"
        job.say("梯度求解完成；结果仅作离散验证，气动精度尚未验收")
    except Cancelled as exc:
        job.state = "cancelled"
        job.say(str(exc))
    except Exception as exc:
        job.state = "error"
        job.error = "%s: %s" % (type(exc).__name__, exc)
        job.say(job.error)
    finally:
        job.finished = time.time()


def run_optimize(job, options):
    """Run the SLSQP section optimization on the current fixed-topology mesh."""
    try:
        source = job.run_dir / "mesh" / "surface.json"
        if not source.is_file() or not job.mesh_info or not job.mesh_options:
            raise ValueError("请先生成当前 GRASS 计算网格")
        if job.mesh_info.get("mesh_type") != "grass_fixed_topology":
            raise ValueError("当前网格不是 GRASS 固定拓扑网格，机身/CAD 网格暂不支持截面优化")
        alpha = float(job.mesh_options["alpha"])
        if float(options.get("alpha", alpha)) != alpha:
            raise ValueError("迎角已变化，请重新生成网格")
        objective = str(options.get("objective", "induced"))
        if objective not in ("induced", "pressure", "lift"):
            raise ValueError("未知优化目标")
        iterations = int(options.get("iterations", 20))
        if not 1 <= iterations <= 200:
            raise ValueError("迭代次数必须在 1 到 200 之间")
        command = [sys.executable, "-m", "tools.web.optimize_runner",
                   "--surface", str(source), "--out", str(job.run_dir / "opt_result.json"),
                   "--alpha", "%g" % alpha, "--objective", objective,
                   "--iterations", str(iterations)]
        cl_min = options.get("cl_min")
        cl_target = options.get("cl_target")
        for name, value in (("升力下限", cl_min), ("升力等式", cl_target)):
            if value not in (None, "") and not -2.0 < float(value) < 4.0:
                raise ValueError(name + "超出合理范围")
        if cl_min not in (None, ""):
            command += ["--cl-min", "%g" % float(cl_min)]
        if cl_target not in (None, ""):
            command += ["--cl-target", "%g" % float(cl_target)]
        tolerance = options.get("consistency_tolerance", 0)
        tolerance = 0.0 if tolerance in (None, "") else float(tolerance)
        if not 0.0 <= tolerance < 1.0:
            raise ValueError("近场/远场一致性容差必须在 0 到 1 之间")
        if tolerance > 0:
            command += ["--consistency-tolerance", "%g" % tolerance]
        if options.get("keep_area"):
            command.append("--keep-area")
        job.state = "solving"
        if cl_target not in (None, ""):
            lift_text = "等式 %g" % float(cl_target)
        elif cl_min not in (None, ""):
            lift_text = "下限 %g" % float(cl_min)
        else:
            lift_text = "-"
        job.say("截面优化启动: 目标=%s, 迭代=%d, 升力约束=%s, 保面积=%s, 一致性容差=%s"
                % (objective, iterations, lift_text,
                   "是" if options.get("keep_area") else "否",
                   "-" if tolerance <= 0 else "%.1f%%" % (100 * tolerance)))
        output = job.run_dir / "opt_result.json"
        code, log = _spawn(job, command, job.run_dir / "opt.log")
        if code != 0 or not output.is_file():
            lines = log.read_text(encoding="utf-8", errors="replace").strip().splitlines()
            raise RuntimeError("截面优化失败: " + " | ".join(lines[-5:]))
        report = json.loads(output.read_text(encoding="utf-8"))
        report["mesh_size"] = job.mesh_options.get("size")
        report["alpha"] = alpha
        (job.run_dir / "opt_summary.json").write_text(json.dumps(report), encoding="utf-8")
        job.result = report
        job.state = "done"
        job.say("优化结束: %s，目标 %s → %s"
                % (report.get("message", ""),
                   "%.6f" % report.get("initial", {}).get("objective", float("nan")),
                   "%.6f" % report.get("final", {}).get("objective", float("nan"))))
    except Cancelled as exc:
        job.state = "cancelled"
        job.say(str(exc))
    except Exception as exc:  # noqa: BLE001 - reported to the browser
        job.state = "error"
        job.error = "%s: %s" % (type(exc).__name__, exc)
        job.say(job.error)
    finally:
        job.finished = time.time()


def run_job(job, options):
    """Solve using the previously generated mesh."""
    alpha = float(options.get("alpha", 4.0))
    wake = str(options.get("wake", "fixed"))
    wall_formulation = str(options.get("wall_formulation", "galerkin"))
    device = str(options.get("device", "cpu"))
    size = float(options.get("size", 0.3))
    iterations = int(options.get("iterations", 12))
    ncpu = int(options.get("ncpu", SOLVE_THREADS))
    try:
        if wake not in ("fixed", "free", "free_filament"):
            raise ValueError("未知尾迹方法")
        if wall_formulation not in ("galerkin", "collocation"):
            raise ValueError("未知壁面离散方法")
        if wake == "free_filament" and wall_formulation != "galerkin":
            raise ValueError("涡线自由尾迹目前只支持 P1 弱形式壁面")
        surface = job.run_dir / "mesh" / "surface.json"
        if not surface.is_file() or not job.mesh_info or not job.mesh_options:
            raise ValueError("请先生成计算网格")
        if job.mesh_options.get("size") != size or job.mesh_options.get("alpha") != alpha:
            raise ValueError("网格尺寸或迎角已变化，请重新生成网格")
        mesh_info = job.mesh_info
        job.check()
        job.state = "solving"
        reference = (job.build or {}).get("reference") or params_module.reference(
            (job.build or {}).get("params") or params_module.normalise({})[0])
        result = solve(job, surface, wake, alpha, iterations, reference, ncpu=ncpu,
                       wall_formulation=wall_formulation)
        result = _plain(result)
        if wake in ("free", "free_filament") and (not result.get("converged") or result.get("geometry_stable") is False):
            history = result.get("history") or []
            first = history[0]["CL"] if history else None
            last = result.get("CL")
            if first is not None and last is not None:
                result["warning"] = ("自由尾迹未收敛（状态：%s）：CL 从 %+.3f 变化至 %+.3f。"
                                     "结果仅供诊断，不可作为已验收的气动结论。"
                                     % (result.get("status","未知"), first, last))
            else:
                result["warning"] = "自由尾迹未收敛：数值仅供诊断。"
            job.say(result["warning"])
        result["options"] = dict(alpha=alpha, wake=wake, wall_formulation=wall_formulation,
                                 size=size, iterations=iterations,
                                 ncpu=ncpu,
                                 device_requested=device, device_used="cpu")
        if device != "cpu":
            result["device_note"] = "Web 固定和自由尾迹求解目前均使用 CPU。"
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
