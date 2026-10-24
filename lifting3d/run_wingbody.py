"""Sequential robust wing-body geometry and fixed-mesh symmetry matrix."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

root = Path(__file__).resolve().parent
out = root/"results/wingbody_matrix_v1"
assert not out.exists(), "Preserve prior wingbody_matrix_v1 evidence"
out.mkdir(parents=True)
protocol_text = (root/"WINGBODY_RESULTS.md").read_text(encoding="utf-8")
(out/"WINGBODY_RESULTS.md.protocol_snapshot").write_text(protocol_text, encoding="utf-8")
protocol_sha256 = hashlib.sha256(protocol_text.encode("utf-8")).hexdigest()
runner_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
started = time.perf_counter()
commands = []
geometry_records = []

geometry_cases = [
    ("t012_r020_a090", .12, .20, .90),
    ("t024_r035_a090", .24, .35, .90),
    ("t036_r035_a090", .36, .35, .90),
    ("t024_r015_a090", .24, .15, .90),
    ("t024_r050_a090", .24, .50, .90),
    ("t024_r035_a099", .24, .35, .99),
]
reference_geometry = "t024_r035_a090"

for name, thickness, body_radius, body_aft in geometry_cases:
    case_root = out/name
    case_root.mkdir()
    mesh_root = case_root/"mesh"
    mesh_command = [
        sys.executable, str(root/"build_mesh.py"), "--size", ".2",
        "--curvature", "32", "--wake-strips", "24",
        "--body-length", "3", "--body-radius", str(body_radius),
        "--body-aft", str(body_aft), "--thickness", str(thickness),
        "--radius", "5", "--far-size", str(5./3.), "--out", str(mesh_root)]
    tick = time.perf_counter()
    mesh_process = subprocess.run(mesh_command, capture_output=True, text=True)
    mesh_seconds = time.perf_counter()-tick
    (case_root/"mesh.log").write_text(mesh_process.stdout+mesh_process.stderr, encoding="utf-8")
    geometry_record = dict(
        case=name, thickness=thickness, body_length=3., body_radius=body_radius,
        body_aft=body_aft, span=4., chord=1., mach=0., alpha_deg=5.,
        size=.2, curvature=32, wake_strips=24, far_radius=5., far_size=5./3.,
        mesh_command=mesh_command, mesh_returncode=mesh_process.returncode,
        mesh_elapsed_seconds=mesh_seconds, solver_cases=[], status="mesh_failed")
    commands.append(dict(case=name, stage="mesh", command=mesh_command,
                         returncode=mesh_process.returncode, seconds=mesh_seconds))
    geometry_records.append(geometry_record)
    (out/"progress.json").write_text(
        json.dumps(dict(runner_sha256=runner_sha256, commands=commands,
                        geometries=geometry_records), indent=2), encoding="utf-8")
    if mesh_process.returncode:
        continue

    mesh_path = mesh_root/"mesh.npz"
    mesh_test_command = [sys.executable, str(root/"test_mesh.py"), "--mesh", str(mesh_path)]
    tick = time.perf_counter()
    mesh_test_process = subprocess.run(mesh_test_command, capture_output=True, text=True)
    mesh_test_seconds = time.perf_counter()-tick
    (case_root/"mesh_test.log").write_text(
        mesh_test_process.stdout+mesh_test_process.stderr, encoding="utf-8")
    tests_path = mesh_root/"tests.json"
    mesh_tests = json.loads(tests_path.read_text(encoding="utf-8")) if tests_path.exists() else []
    geometry_record.update(mesh_test_returncode=mesh_test_process.returncode,
                           mesh_test_seconds=mesh_test_seconds, mesh_tests=mesh_tests)
    commands.append(dict(case=name, stage="mesh_test", command=mesh_test_command,
                         returncode=mesh_test_process.returncode, seconds=mesh_test_seconds,
                         checks=mesh_tests))
    if mesh_test_process.returncode:
        geometry_record["status"] = "mesh_test_failed"
        (out/"progress.json").write_text(
            json.dumps(dict(runner_sha256=runner_sha256, commands=commands,
                            geometries=geometry_records), indent=2), encoding="utf-8")
        continue

    mesh_metadata = json.loads((mesh_root/"mesh.json").read_text(encoding="utf-8"))
    geometry_record.update(
        status="mesh_ready", mesh_sha256=hashlib.sha256(mesh_path.read_bytes()).hexdigest(),
        mesh_metadata_sha256=hashlib.sha256((mesh_root/"mesh.json").read_bytes()).hexdigest(),
        points=mesh_metadata["points"], tets=mesh_metadata["tets"],
        boundary_tags=mesh_metadata.get("cad_surface_components", mesh_metadata.get("cad_component_surfaces")),
        minimum_dense_jacobian=mesh_metadata["min_dense_jacobian"])
    if name == reference_geometry:
        geometry_record["reference_mesh"] = str(mesh_path.relative_to(root))

    solver_cases = [("m0_a5", 5., 0.)]
    if name == reference_geometry:
        solver_cases.extend([("m0_a0", 0., 0.), ("m0_am5", -5., 0.),
                             ("m02_a5", 5., .2)])
    for solve_name, alpha, mach in solver_cases:
        run_root = case_root/solve_name
        solve_command = [
            sys.executable, str(root/"solve.py"), "--mesh", str(mesh_path),
            "--alpha", str(alpha), "--mach", str(mach), "--ncpu", "8",
            "--kutta", "weak", "--out", str(run_root)]
        tick = time.perf_counter()
        timed_out = False
        timeout_child_terminated = False
        try:
            solve_process = subprocess.run(solve_command, capture_output=True, text=True,
                                           timeout=240)
            solve_stdout = solve_process.stdout
            solve_stderr = solve_process.stderr
            solve_returncode = solve_process.returncode
        except subprocess.TimeoutExpired as exception:
            timed_out = True
            timeout_child_terminated = True
            solve_stdout = exception.stdout or ""
            solve_stderr = exception.stderr or ""
            if isinstance(solve_stdout, bytes):
                solve_stdout = solve_stdout.decode(errors="replace")
            if isinstance(solve_stderr, bytes):
                solve_stderr = solve_stderr.decode(errors="replace")
            solve_returncode = None
        solve_seconds = time.perf_counter()-tick
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root/"run.log").write_text(solve_stdout+solve_stderr, encoding="utf-8")
        result_path = run_root/"result.json"
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
        field_path = run_root/"field.npz"
        analysis_path = run_root/"analysis.json"
        analysis_returncode = None
        analysis = {}
        if result_path.exists() and field_path.exists():
            analyze_command = [sys.executable, str(root/"analyze.py"),
                               "--run", str(run_root), "--mesh", str(mesh_path)]
            analyze_tick = time.perf_counter()
            analyze_process = subprocess.run(analyze_command, capture_output=True, text=True)
            analyze_seconds = time.perf_counter()-analyze_tick
            (run_root/"analyze.log").write_text(
                analyze_process.stdout+analyze_process.stderr, encoding="utf-8")
            analysis_returncode = analyze_process.returncode
            commands.append(dict(case=name, stage=solve_name+"_analyze",
                                 command=analyze_command, returncode=analysis_returncode,
                                 seconds=analyze_seconds))
            if analysis_path.exists():
                analysis = json.loads(analysis_path.read_text(encoding="utf-8"))

        finite_record = dict(value_count=0, finite_count=0, nonfinite_count=None,
                             all_finite=False, minimum=None, maximum=None)
        if field_path.exists():
            field = np.load(field_path)
            cp = np.asarray(field["cp"])
            finite_count = int(np.isfinite(cp).sum())
            finite_record = dict(value_count=int(cp.size), finite_count=finite_count,
                                 nonfinite_count=int(cp.size-finite_count),
                                 all_finite=bool(finite_count == cp.size),
                                 minimum=float(np.nanmin(cp)) if finite_count else None,
                                 maximum=float(np.nanmax(cp)) if finite_count else None,
                                 source_sha256=hashlib.sha256(field_path.read_bytes()).hexdigest())
        component_cl_sum = None
        component_force_sum = None
        component_sum_error = analysis.get("component_sum_error")
        if result.get("component_loads"):
            component_force_sum = np.sum(
                [entry["force_coefficient"] for entry in result["component_loads"].values()],
                axis=0).tolist()
            component_cl_sum = float(sum(entry["cl"]
                                         for entry in result["component_loads"].values()))
        solve_record = dict(
            name=solve_name, alpha_deg=alpha, mach=mach, kutta="weak", ncpu_requested=8,
            command=solve_command, returncode=solve_returncode, timeout_seconds=240,
            timed_out=timed_out, timeout_child_terminated=timeout_child_terminated,
            elapsed_wall_seconds=solve_seconds,
            solver_elapsed_seconds=result.get("elapsed_seconds"),
            status=result.get("status", "timeout" if timed_out else "missing_result"),
            converged=result.get("converged", False),
            result_path=str(result_path.relative_to(root)) if result_path.exists() else None,
            result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest()
            if result_path.exists() else None,
            mesh_sha256=result.get("mesh_sha256"),
            solver_sha256=result.get("solver_sha256"),
            p2_sha256=result.get("p2_sha256"),
            nodes=result.get("nodes"), tets=result.get("elements"),
            cl_pressure=result.get("cl_pressure"), cl_gamma=result.get("cl_jump"),
            cd_pressure=result.get("cd_pressure"),
            pressure_force_coefficient=result.get("pressure_force_coefficient"),
            component_loads=result.get("component_loads"),
            component_cl_sum=component_cl_sum,
            component_force_sum=component_force_sum,
            component_sum_error=component_sum_error,
            raw_cp=finite_record,
            raw_mass_by_tag=result.get("raw_mass_by_tag"),
            weak_mass_residual_norm=result.get("weak_mass_residual_norm"),
            wall_normal_rms=result.get("wall_normal_rms"),
            wall_normal_max=result.get("wall_normal_max"),
            cp_min=result.get("cp_min"), cp_max=result.get("cp_max"),
            wall_regions=result.get("wall_regions"),
            wake_diagnostics=result.get("wake_diagnostics"),
            analysis_path=str(analysis_path.relative_to(root)) if analysis_path.exists() else None,
            analysis_sha256=hashlib.sha256(analysis_path.read_bytes()).hexdigest()
            if analysis_path.exists() else None,
            analysis_returncode=analysis_returncode,
            analysis=analysis,
            final_residual=result.get("history", [{}])[-1].get("residual_norm")
            if result.get("history") else None,
            min_beta=result.get("min_beta"), max_local_mach=result.get("max_local_mach"))
        solve_record["component_sum_passed"] = bool(
            component_sum_error is not None and component_sum_error < 1.e-10)
        solve_record["raw_cp_finite_passed"] = finite_record["all_finite"]
        solve_record["case_passed"] = bool(
            solve_returncode == 0 and result.get("converged")
            and solve_record["component_sum_passed"]
            and solve_record["raw_cp_finite_passed"])
        geometry_record["solver_cases"].append(solve_record)
        geometry_record["status"] = "solved" if solve_record["case_passed"] else "solve_failed"
        commands.append(dict(case=name, stage=solve_name, command=solve_command,
                             returncode=solve_returncode, timed_out=timed_out,
                             timeout_child_terminated=timeout_child_terminated,
                             seconds=solve_seconds))
        (out/"progress.json").write_text(
            json.dumps(dict(runner_sha256=runner_sha256, commands=commands,
                            geometries=geometry_records), indent=2, allow_nan=False),
            encoding="utf-8")

# The solve records above retain complete array provenance in field.npz; load gamma
# directly for exact same-mesh zero-load and positive/negative symmetry gates.
for geometry_record in geometry_records:
    if geometry_record["case"] != reference_geometry or len(geometry_record["solver_cases"]) < 3:
        continue
    base_root = out/geometry_record["case"]
    zero_case = next(row for row in geometry_record["solver_cases"] if row["name"] == "m0_a0")
    negative_case = next(row for row in geometry_record["solver_cases"] if row["name"] == "m0_am5")
    positive_case = next(row for row in geometry_record["solver_cases"] if row["name"] == "m0_a5")
    zero_field_path = base_root/"m0_a0"/"field.npz"
    negative_field_path = base_root/"m0_am5"/"field.npz"
    positive_field_path = base_root/"m0_a5"/"field.npz"
    symmetry = dict(zero_load_errors=None, positive_negative_errors=None,
                    zero_load_passed=False, positive_negative_passed=False,
                    evidence_available=False)
    if zero_field_path.exists() and negative_field_path.exists() and positive_field_path.exists():
        zero_field = np.load(zero_field_path)
        negative_field = np.load(negative_field_path)
        positive_field = np.load(positive_field_path)
        symmetry.update(
            evidence_available=True,
            zero_load_errors=dict(
                cl_pressure=abs(zero_case["cl_pressure"]),
                cl_gamma=abs(zero_case["cl_gamma"]),
                gamma_max=float(np.max(abs(zero_field["gamma"])))),
            positive_negative_errors=dict(
                cl_pressure=abs(positive_case["cl_pressure"]+negative_case["cl_pressure"]),
                cl_gamma=abs(positive_case["cl_gamma"]+negative_case["cl_gamma"]),
                gamma_max=float(np.max(abs(positive_field["gamma"]+negative_field["gamma"])))))
        symmetry["zero_load_passed"] = all(value < 1.e-9
                                            for value in symmetry["zero_load_errors"].values())
        symmetry["positive_negative_passed"] = all(value < 1.e-9
                                                    for value in symmetry["positive_negative_errors"].values())
    symmetry["all_cases_raw_cp_finite"] = all(
        row["raw_cp_finite_passed"] for row in geometry_record["solver_cases"])
    symmetry["all_cases_component_sum_passed"] = all(
        row["component_sum_passed"] for row in geometry_record["solver_cases"])
    symmetry["mach02_status"] = next(
        row["status"] for row in geometry_record["solver_cases"] if row["name"] == "m02_a5")
    geometry_record["fixed_mesh_gates"] = symmetry
    geometry_record["status"] = "completed_with_gate_results"

(out/"progress.json").write_text(
    json.dumps(dict(runner_sha256=runner_sha256, commands=commands,
                    geometries=geometry_records), indent=2, allow_nan=False),
    encoding="utf-8")

report = [
    "## 执行结果",
    "",
    "| Case | 厚度 | 机身半径 | aft | Mesh 状态 | points/tets | mesh checks |",
    "|---|---:|---:|---:|---|---:|---:|",
]
for geometry in geometry_records:
    report.append(
        f"| {geometry['case']} | {geometry['thickness']:.2f} | {geometry['body_radius']:.2f} | "
        f"{geometry['body_aft']:.2f} | {geometry['status']} | "
        f"{geometry.get('points', 'NA')}/{geometry.get('tets', 'NA')} | "
        f"{sum(row['passed'] for row in geometry.get('mesh_tests', []))}/"
        f"{len(geometry.get('mesh_tests', []))} |")
report.extend(["", "## 各例结果", ""])
for geometry in geometry_records:
    report.extend([f"### {geometry['case']}", ""])
    if geometry.get("mesh_sha256"):
        report.extend([
            f"- mesh SHA256 `{geometry['mesh_sha256']}`; dense minimum Jacobian "
            f"`{geometry['minimum_dense_jacobian']}`; mesh metadata "
            f"`results/wingbody_matrix_v1/{geometry['case']}/mesh/mesh.json`.",
            ""])
    elif geometry["status"] == "mesh_failed":
        failure_path = out/geometry["case"]/"mesh"/"failure.json"
        report.extend([
            f"- Mesh/CAD failure: returncode={geometry['mesh_returncode']}; "
            f"log=`results/wingbody_matrix_v1/{geometry['case']}/mesh.log`; "
            f"failure metadata exists={failure_path.exists()}.",
            ""])
    for case in geometry["solver_cases"]:
        report.extend([
            f"- **{case['name']}** (`M={case['mach']}`, `alpha={case['alpha_deg']}°`, weak): "
            f"status=`{case['status']}`, converged=`{case['converged']}`, "
            f"wall={case['elapsed_wall_seconds']:.2f}s, timed_out={case['timed_out']}, "
            f"timeout child terminated={case['timeout_child_terminated']}.",
            f"  - nodes/tets={case['nodes']}/{case['tets']}; CL pressure/Gamma="
            f"{case['cl_pressure']}/{case['cl_gamma']}; CD pressure={case['cd_pressure']}; "
            f"raw Cp finite={case['raw_cp']['finite_count']}/{case['raw_cp']['value_count']} "
            f"(min/max={case['raw_cp']['minimum']}/{case['raw_cp']['maximum']}).",
            f"  - wall RMS/max={case['wall_normal_rms']}/{case['wall_normal_max']}; "
            f"global raw normal peak={case['analysis'].get('global_normal_max')}; "
            f"global Cp minimum={case['analysis'].get('global_cp_min')}.",
            f"  - junction/full-wall regions=`{json.dumps(case['analysis'].get('regions', {}), ensure_ascii=False)}`.",
            f"  - component loads=`{json.dumps(case['component_loads'], ensure_ascii=False)}`; "
            f"component sum error={case['component_sum_error']} "
            f"(pass={case['component_sum_passed']}).",
            f"  - raw mass by tag=`{json.dumps(case['raw_mass_by_tag'], ensure_ascii=False)}`; "
            f"wake diagnostics=`{json.dumps(case['analysis'].get('raw_wake_flux_jump_rms'), ensure_ascii=False)}` "
            f"and `{json.dumps(case['wake_diagnostics'], ensure_ascii=False)}`.",
            f"  - result=`{case['result_path']}`; analysis=`{case['analysis_path']}`.",
            ""])
    if geometry.get("fixed_mesh_gates"):
        gates = geometry["fixed_mesh_gates"]
        report.extend([
            "**固定网格门禁**",
            f"- alpha=0 误差 `{json.dumps(gates['zero_load_errors'])}`; passed=`{gates['zero_load_passed']}`.",
            f"- ±5° 对称误差 `{json.dumps(gates['positive_negative_errors'])}`; "
            f"passed=`{gates['positive_negative_passed']}`.",
            f"- 所有变体 raw Cp 有限=`{gates['all_cases_raw_cp_finite']}`；"
            f"组件力闭合=`{gates['all_cases_component_sum_passed']}`；"
            f"M=.2 状态=`{gates['mach02_status']}`.",
            ""])
report.extend([
    "## 解释边界",
    "",
    "求解器返回收敛只说明离散非线性方程达到自身残差条件。组件载荷闭合、有限 Cp、壁面无穿透、wake raw 通量及压力跳跃是不同诊断。报告保留完整 wall 区域、全域峰值和原始 mass flux；不通过裁剪极值或几何处理美化结果。冻结尾迹仍是模型限制；本矩阵只评估指定网格/几何下的鲁棒性，不能单独证明厚翼身 Cp 或整体气动精度。",
    "",
    f"Runner SHA256 `{runner_sha256}`；总经过时间 {time.perf_counter()-started:.2f} s。"])
final_report = protocol_text.replace(
    "待运行。执行命令由 [run_wingbody.py](run_wingbody.py) 记录；完整输入、日志、mesh、field、result 与分析输出保存在 `results/wingbody_matrix_v1/`。",
    "执行结果已附于本协议之后。完整输入、日志、mesh、field、result 与分析输出保存在 `results/wingbody_matrix_v1/`。")
(root/"WINGBODY_RESULTS.md").write_text(
    final_report+"\n"+"\n".join(report)+"\n", encoding="utf-8")
summary = dict(
    protocol_path="WINGBODY_RESULTS.md",
    protocol_sha256=protocol_sha256,
    runner_sha256=runner_sha256, elapsed_wall_seconds=time.perf_counter()-started,
    sequential=True, ncpu_requested=8, per_solve_timeout_seconds=240,
    geometries=geometry_records, commands=commands)
(out/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
print(json.dumps(dict(elapsed_wall_seconds=summary["elapsed_wall_seconds"],
                      geometries=[dict(case=row["case"], status=row["status"],
                                       points=row.get("points"), tets=row.get("tets"),
                                       cases=len(row["solver_cases"]))
                                  for row in geometry_records]), indent=2))
