"""Run or collect the complete frozen-input B3 experiment with one command."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--workers", type=int, default=8)
parser.add_argument("--collect-only", action="store_true")
parser.add_argument("--timeout", type=float, default=18000.)
args = parser.parse_args()
if args.workers < 3:
    raise ValueError("At least three workers are needed for the three frozen baselines")
root = Path(__file__).resolve().parents[1]
cases = ("rect", "swept30", "elliptic")
os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                  MKL_NUM_THREADS="1", PYTHONIOENCODING="utf-8")
started = time.time()
processes, commands = [], []
if not args.collect_only:
    if args.out.exists():
        raise FileExistsError("Preserve previous evidence; choose a fresh output directory")
    args.out.mkdir(parents=True)
    for case in cases:
        folder = args.out/case
        folder.mkdir()
        command = [sys.executable, "-m", "tools.wake_sensitivity_baseline", "--case", case,
                   "--iterations", "60", "--out", str(args.out)]
        with (folder/"baseline.log").open("w") as stdout, (folder/"baseline_error.log").open("w") as stderr:
            process = subprocess.Popen(command, cwd=root, stdout=stdout, stderr=stderr)
        processes.append(process)
        commands.append(dict(command=command, pid=process.pid))
    pending = set(cases)
    while pending:
        if time.time()-started > args.timeout:
            raise TimeoutError("Frozen baseline readiness timeout; preserve child logs")
        for process in processes:
            if process.poll() not in (None, 0):
                raise RuntimeError("Baseline/E1 process failed: pid="+str(process.pid))
        for case in list(pending):
            path = args.out/case/"manifest.json"
            if not path.exists():
                continue
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if len(manifest["rows"]) < 2:
                continue
            if sum(p.poll() is None for p in processes) >= args.workers:
                continue
            command = [sys.executable, "-m", "tools.wake_position_study", "--phase", "e1",
                       "--case", case, "--out", str(args.out)]
            folder = args.out/case
            with (folder/"e1.log").open("w") as stdout, (folder/"e1_error.log").open("w") as stderr:
                process = subprocess.Popen(command, cwd=root, stdout=stdout, stderr=stderr)
            processes.append(process)
            commands.append(dict(command=command, pid=process.pid))
            pending.remove(case)
            print(case, "frozen first step ready; E1 started", flush=True)
        if pending:
            time.sleep(5)
    command = [sys.executable, "-m", "tools.wake_sensitivity_queue", "--out", str(args.out),
               "--workers", str(args.workers), "--external-pids", *[str(p.pid) for p in processes]]
    with (args.out/"queue.log").open("w") as stdout, (args.out/"queue_error.log").open("w") as stderr:
        queue_process = subprocess.Popen(command, cwd=root, stdout=stdout, stderr=stderr)
    commands.append(dict(command=command, pid=queue_process.pid))
last_update = 0.
while True:
    if time.time()-started > args.timeout:
        raise TimeoutError("Experiment collection timeout; inspect queue.json and preserved logs")
    try:
        queue = json.loads((args.out/"queue.json").read_text(encoding="utf-8"))
        complete = not queue["pending"] and not queue["running"] and not queue["external_pids"]
        terminal_jobs = {r["label"] for r in queue["completed"] if "exit_code" in r}
        for case in cases:
            baseline = json.loads((args.out/case/"manifest.json").read_text(encoding="utf-8"))
            complete &= baseline["status"] == "completed"
            # Original interactive runs may include workers outside queue.json.
            for path in (args.out/case).glob("e3_*/history.json"):
                if not path.parent.name.endswith("_failure"):
                    state = json.loads(path.read_text(encoding="utf-8"))["status"]
                    complete &= state != "running" or case+"/"+path.parent.name in terminal_jobs
    except (FileNotFoundError, json.JSONDecodeError):
        complete = False
    for process in processes:
        if process.poll() not in (None, 0):
            raise RuntimeError("Baseline/E1 worker failed; inspect preserved logs")
    if complete:
        break
    if time.time()-last_update > 60:
        print("waiting for bounded numerical jobs; elapsed %.0fs" % (time.time()-started), flush=True)
        last_update = time.time()
    time.sleep(5)
for case in cases:
    for module in ("tools.wake_sensitivity_audit", "tools.wake_first_step_decomposition"):
        command = [sys.executable, "-m", module, "--case", case, "--out", str(args.out)]
        result = subprocess.run(command, cwd=root, timeout=1800)
        commands.append(dict(command=command, exit_code=result.returncode))
        result.check_returncode()
for module in ("tools.wake_material_kinematics", "tools.wake_sensitivity_plot", "tools.wake_sensitivity_report"):
    command = [sys.executable, "-m", module, "--out", str(args.out)]
    result = subprocess.run(command, cwd=root, timeout=1800)
    commands.append(dict(command=command, exit_code=result.returncode))
    result.check_returncode()
for process, record in zip(processes, commands):
    record["exit_code"] = process.poll()
if not args.collect_only:
    commands[len(processes)]["exit_code"] = queue_process.wait(timeout=15)
(args.out/"driver_receipt.json").write_text(
    json.dumps(dict(command=[sys.executable, *sys.argv], child_commands=commands,
                    seconds=time.time()-started, collect_only=args.collect_only), indent=2), encoding="utf-8")
manifest = json.loads((args.out/"manifest.json").read_text(encoding="utf-8"))
manifest["driver_receipt"] = dict(path="driver_receipt.json",
    sha256=hashlib.sha256((args.out/"driver_receipt.json").read_bytes()).hexdigest())
(args.out/"manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
if not manifest["execution_complete"]:
    raise RuntimeError("Collection incomplete: "+str(manifest["missing_completed_runs_or_audits"]))
print("all numerical attempts collected; physical accuracy remains unaccepted", flush=True)
