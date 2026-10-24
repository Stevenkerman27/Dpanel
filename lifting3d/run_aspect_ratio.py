"""Sequential AR8/AR20 trend cases against the existing lifting-line models."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, default=root/"results/aspect_ratio_v1")
args = parser.parse_args()
args.out = args.out.resolve()
assert args.out == (root/"results/aspect_ratio_v1").resolve(), "Output is restricted to results/aspect_ratio_v1"
assert not args.out.exists(), "Preserve prior aspect_ratio_v1 evidence"
args.out.mkdir(parents=True)
started = time.perf_counter()
commands = []
case_records = []

for aspect_ratio, span, radius, far_size, strips in [
        (8, 8., 8., 5./3., 48),
        (20, 20., 15., 2., 120)]:
    ar_root = args.out/("ar"+str(aspect_ratio))
    ar_root.mkdir()
    mesh_root = ar_root/"mesh"
    mesh_command = [
        sys.executable, str(root/"build_mesh.py"), "--size", ".2",
        "--span", str(span), "--curvature", "32", "--wake-strips", str(strips),
        "--radius", str(radius), "--far-size", str(far_size), "--out", str(mesh_root)]
    tick = time.perf_counter()
    mesh_process = subprocess.run(mesh_command, capture_output=True, text=True)
    mesh_seconds = time.perf_counter()-tick
    (ar_root/"mesh.log").write_text(mesh_process.stdout+mesh_process.stderr, encoding="utf-8")
    mesh_record = dict(aspect_ratio=aspect_ratio, stage="mesh", command=mesh_command,
                       returncode=mesh_process.returncode, seconds=mesh_seconds)
    commands.append(mesh_record)
    (args.out/"progress.json").write_text(
        json.dumps(dict(commands=commands, cases=case_records), indent=2), encoding="utf-8")
    if mesh_process.returncode:
        case_records.append(dict(aspect_ratio=aspect_ratio, status="mesh_failed"))
        continue

    mesh_path = mesh_root/"mesh.npz"
    test_command = [sys.executable, str(root/"test_mesh.py"), "--mesh", str(mesh_path)]
    tick = time.perf_counter()
    test_process = subprocess.run(test_command, capture_output=True, text=True)
    test_seconds = time.perf_counter()-tick
    (ar_root/"mesh_test.log").write_text(test_process.stdout+test_process.stderr, encoding="utf-8")
    tests_path = mesh_root/"tests.json"
    mesh_tests = json.loads(tests_path.read_text(encoding="utf-8")) if tests_path.exists() else []
    test_record = dict(aspect_ratio=aspect_ratio, stage="mesh_test", command=test_command,
                       returncode=test_process.returncode, seconds=test_seconds,
                       checks=mesh_tests)
    commands.append(test_record)
    (args.out/"progress.json").write_text(
        json.dumps(dict(commands=commands, cases=case_records), indent=2), encoding="utf-8")
    if test_process.returncode:
        case_records.append(dict(aspect_ratio=aspect_ratio, status="mesh_test_failed",
                                 mesh_tests=mesh_tests))
        continue

    ar_cases = []
    for kutta in ["node", "weak"]:
        case_root = ar_root/kutta
        solve_command = [
            sys.executable, str(root/"solve.py"), "--mesh", str(mesh_path),
            "--alpha", "5", "--mach", "0", "--ncpu", "8",
            "--kutta", kutta, "--out", str(case_root)]
        tick = time.perf_counter()
        timed_out = False
        timeout_terminated = False
        try:
            solve_process = subprocess.run(solve_command, capture_output=True, text=True,
                                           timeout=240)
            solve_stdout = solve_process.stdout
            solve_stderr = solve_process.stderr
            solve_returncode = solve_process.returncode
        except subprocess.TimeoutExpired as exception:
            timed_out = True
            solve_stdout = exception.stdout or ""
            solve_stderr = exception.stderr or ""
            if isinstance(solve_stdout, bytes):
                solve_stdout = solve_stdout.decode(errors="replace")
            if isinstance(solve_stderr, bytes):
                solve_stderr = solve_stderr.decode(errors="replace")
            solve_returncode = None
            timeout_terminated = True
        solve_seconds = time.perf_counter()-tick
        case_root.mkdir(parents=True, exist_ok=True)
        (case_root/"run.log").write_text(solve_stdout+solve_stderr, encoding="utf-8")
        result_path = case_root/"result.json"
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
        case_record = dict(
            aspect_ratio=aspect_ratio, span=span, chord=1., reference_area=span,
            mach=0., alpha_deg=5., kutta=kutta, ncpu_requested=8,
            command=solve_command, returncode=solve_returncode,
            timeout_seconds=240, timed_out=timed_out,
            timeout_child_terminated=timeout_terminated,
            elapsed_wall_seconds=solve_seconds,
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
            wall_normal_rms=result.get("wall_normal_rms"),
            wall_normal_max=result.get("wall_normal_max"),
            cp_min=result.get("cp_min"), cp_max=result.get("cp_max"),
            weak_mass_residual_norm=result.get("weak_mass_residual_norm"),
            wake_diagnostics=result.get("wake_diagnostics"),
            solver_elapsed_seconds=result.get("elapsed_seconds"),
            final_residual=result.get("history", [{}])[-1].get("residual_norm")
            if result.get("history") else None)
        ar_cases.append(case_record)
        commands.append(dict(aspect_ratio=aspect_ratio, stage=kutta, command=solve_command,
                             returncode=solve_returncode, timed_out=timed_out,
                             timeout_child_terminated=timeout_terminated,
                             seconds=solve_seconds))
        case_records = [row for row in case_records if row["aspect_ratio"] != aspect_ratio]
        case_records.append(dict(aspect_ratio=aspect_ratio, status="running",
                                 mesh_sha256=json.loads((mesh_root/"mesh.json").read_text())["mesh_sha256"],
                                 mesh_tests=mesh_tests, cases=ar_cases))
        (args.out/"progress.json").write_text(
            json.dumps(dict(commands=commands, cases=case_records), indent=2), encoding="utf-8")

    # Compare the stored 3D near-midspan Cp samples with the original 2D P2 surface data.
    reference_root = root.parent/"airfoil2d/results/general_v1/20_n12_02_a5_m0"
    reference = np.load(reference_root/"field.npz")
    reference_points = reference["surface_point"]
    reference_cp = reference["cp"]
    reference_upper = reference_points[:, :, 1] > 0.
    reference_lower = reference_points[:, :, 1] < 0.
    reference_x_upper = reference_points[:, :, 0][reference_upper]
    reference_cp_upper = reference_cp[reference_upper]
    reference_x_lower = reference_points[:, :, 0][reference_lower]
    reference_cp_lower = reference_cp[reference_lower]
    comparison_rows = []
    for case in ar_cases:
        field_path = ar_root/case["kutta"]/"field.npz"
        if not case["converged"] or not field_path.exists():
            continue
        field = np.load(field_path)
        physical_wall = field["tags"] == 10
        section = (physical_wall[:, None]
                   & (np.abs(field["surface_points"][:, :, 1]) < .1))
        x3 = field["surface_points"][:, :, 0][section]
        z3 = field["surface_points"][:, :, 2][section]
        cp3 = field["cp"][section]
        upper3 = z3 > 0.
        lower3 = z3 < 0.
        for side, mask, x2, cp2 in [
                ("upper", upper3, reference_x_upper, reference_cp_upper),
                ("lower", lower3, reference_x_lower, reference_cp_lower)]:
            order = np.argsort(x3[mask])
            x_side = x3[mask][order]
            cp_side = cp3[mask][order]
            unique_x, unique_index = np.unique(x_side, return_index=True)
            cp_side = cp_side[unique_index]
            inside = (x2 >= unique_x.min()) & (x2 <= unique_x.max())
            interpolated = np.interp(x2[inside], unique_x, cp_side)
            for x_value, cp_value_2d, cp_value_3d in zip(
                    x2[inside], cp2[inside], interpolated):
                comparison_rows.append((case["kutta"], side, x_value,
                                        cp_value_2d, cp_value_3d, cp_value_3d-cp_value_2d))
    if comparison_rows:
        with (ar_root/"midspan_cp_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["aspect_ratio", "kutta", "side", "x_over_chord",
                             "Cp_2D", "Cp_3D_interpolated", "Cp_3D_minus_2D"])
            for row in comparison_rows:
                writer.writerow([aspect_ratio, row[0], row[1], *row[2:]])
    (args.out/"progress.json").write_text(
        json.dumps(dict(commands=commands, cases=case_records), indent=2), encoding="utf-8")

ll_results = (root/"LL_RESULTS.md").read_text(encoding="utf-8")
report = [
    "# 高展弦比矩形 NACA0012 一致性算例",
    "",
    "本批只运行 AR=8、20 的指定工况；M=0、alpha=+5°，各 AR 共用一张 P1/P2 曲面网格，先 node 后 weak 顺序求解，NCPU=8。每个求解器子进程上限 240 秒；超时保留日志并记录终止状态。此为高 AR 趋势与 2D 极限一致性检查，不是绝对 3D 精度认证。",
    "",
    "几何为 build_mesh.py 生成的有限厚度、封闭后缘 NACA 多项式矩形翼。弦向 x、展向 y、厚度 z。AR8: span=8, chord=1, h=.2, curvature=32, wake strips=48, R=8, far-size=5/3。AR20: span=20, chord=1, h=.2, curvature=32, wake strips=120, R=15, far-size=2。网格生成过程中曲率细化可能使实际元素尺度小于 h。",
    "",
    "## LLT 同展弦比预测",
    "",
    "以下来自 `LL_RESULTS.md` 的 N=64 Prandtl 升力线结果。两种二维 section slope 是不同闭合假设，不将任何一个当作精确真值。",
    "",
    "| AR | slope 模型 | CL | CDi |",
    "|---:|---|---:|---:|",
]
for aspect_ratio, model, cl, cdi in [
        (8, "薄翼 2pi/rad", .422169241, .007570878),
        (8, "二维 NACA0012 P2 slope", .452636769, .008650571),
        (20, "薄翼 2pi/rad", .483828951, .004328219),
        (20, "二维 NACA0012 P2 slope", .523634992, .005020622)]:
    report.append(f"| {aspect_ratio} | {model} | {cl:.9f} | {cdi:.9f} |")
report.extend(["", "## 运行记录", ""])
for case in case_records:
    report.extend([f"### AR={case['aspect_ratio']}", ""])
    if case.get("mesh_sha256"):
        report.extend([
            f"- 网格 SHA256: `{case['mesh_sha256']}`",
            f"- 网格检查: {sum(row['passed'] for row in case.get('mesh_tests', []))}/{len(case.get('mesh_tests', []))} passed",
            f"- 节点/四面体: `{json.loads((args.out/('ar'+str(case['aspect_ratio']))/'mesh'/'mesh.json').read_text())['points']}` / `{json.loads((args.out/('ar'+str(case['aspect_ratio']))/'mesh'/'mesh.json').read_text())['tets']}`",
            ""])
        for check in case.get("mesh_tests", []):
            report.append(f"  - `{check['test']}`: {'PASS' if check['passed'] else 'FAIL'}")
        report.append("")
    for row in case.get("cases", []):
        report.extend([
            f"- **{row['kutta']}**: status=`{row['status']}`, converged=`{row['converged']}`, "
            f"wall time={row['elapsed_wall_seconds']:.2f}s, solver time={row['solver_elapsed_seconds']}, "
            f"returncode={row['returncode']}, timeout terminated={row['timeout_child_terminated']}.",
            f"  - mesh nodes/tets={row['nodes']}/{row['tets']}; CL pressure/Gamma={row['cl_pressure']}/{row['cl_gamma']}; "
            f"CD pressure={row['cd_pressure']}; wall-normal RMS/max={row['wall_normal_rms']}/{row['wall_normal_max']}; "
            f"Cp min/max={row['cp_min']}/{row['cp_max']}.",
            f"  - Raw wake diagnostics: `{json.dumps(row['wake_diagnostics'], ensure_ascii=False)}`",
            f"  - Final nonlinear residual={row['final_residual']}; result=`{row['result_path']}`.",
            ""])
    cp_path = args.out/("ar"+str(case["aspect_ratio"]))/"midspan_cp_comparison.csv"
    if cp_path.exists():
        report.extend([
            f"中剖面原始 Cp 对照文件：`{cp_path.relative_to(root)}`。3D 使用物理表面求积点 `|y|<0.1`，按 `z>0/<0` 分组；2D 使用保存的原始 Cp 上下表面点，在共同 x 范围线性插值到 3D 样本。它是采样/插值诊断，不是独立验证。",
            ""])
    else:
        report.extend(["中剖面 Cp 对照未生成（需要 node 与 weak 解均成功并具有共同有效 x 区间）；本批仅比较可用的 spanwise Gamma/CL 与 LLT 整体量。",
                       ""])
report.extend([
    "## 资格说明",
    "",
    "LLT 的 CL 是整翼积分预测，CDi 是升力线诱导阻力；求解器的 `cl_pressure`、`cl_jump` 和 `cd_pressure` 定义不同。需分别看压力积分、环量积分、raw wall normal 与 wake diagnostics，不能只用接近某个 LLT 数值来宣称验证。有限厚度几何、固定尾迹 cut、离散网格及不同 2D slope 均造成模型差异。本组结果只提供指定网格上的 AR 趋势和中剖面 2D 极限线索。",
    "",
    f"Runner SHA256: `{hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}`；总 wall time={time.perf_counter()-started:.2f}s。"])
(root/"ASPECT_RATIO_RESULTS.md").write_text("\n".join(report)+"\n", encoding="utf-8")
summary = dict(scope="High-AR trend and midspan 2D-limit consistency only; not an absolute 3D accuracy certification",
               runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               elapsed_wall_seconds=time.perf_counter()-started,
               commands=commands, cases=case_records)
(args.out/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
