"""Run the remaining issue-113 experiments with bounded process concurrency."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--external-pids", type=int, nargs="*", default=[])
parser.add_argument("--workers", type=int, default=8)
parser.add_argument("--timeout", type=float, default=7200.)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                  MKL_NUM_THREADS="1", PYTHONIOENCODING="utf-8")
jobs = [
    ("e3", "rect", "freeze_1", 60), ("e2", "rect", "rect_96", 8),
    ("e3", "swept30", "freeze_1", 60), ("e3", "swept30", "freeze_2", 60),
    ("e3", "elliptic", "freeze_1", 60),
    ("e2", "swept30", "length_5", 8), ("e2", "swept30", "length_10", 8),
    ("e2", "swept30", "length_20", 8), ("e2", "swept30", "streamwise", 8),
    ("e2", "swept30", "spanwise", 8),
    ("e3", "swept30", "freeze_3", 60), ("e3", "swept30", "smooth", 60),
    ("e3", "elliptic", "freeze_2", 60), ("e3", "elliptic", "freeze_3", 60),
    ("e3", "elliptic", "smooth", 60),
    ("e3", "rect", "freeze_2", 60), ("e3", "rect", "freeze_3", 60),
    ("e3", "rect", "smooth", 60),
    ("e2", "elliptic", "length_5", 8), ("e2", "elliptic", "streamwise", 8),
    ("e2", "elliptic", "spanwise", 8),
    ("e4", "rect", "", 0), ("e4", "swept30", "", 0), ("e4", "elliptic", "", 0),
]
active, receipts = [], []
external = set(args.external_pids)
while jobs or active or external:
    if external:
        process_list = subprocess.run(["tasklist", "/FO", "CSV", "/NH"], check=True,
                                      capture_output=True, text=True).stdout
        running_pids = {int(row[1]) for row in csv.reader(io.StringIO(process_list))
                        if len(row) >= 2 and row[1].isdigit()}
        external.intersection_update(running_pids)
    for job in list(active):
        process = job["process"]
        code = process.poll()
        timed_out = time.time()-job["started"] > args.timeout
        if code is None and timed_out:
            process.kill()
            code = process.wait(timeout=10)
        if code is not None:
            job["stdout"].close()
            job["stderr"].close()
            receipt = {k: v for k, v in job.items() if k not in ("process", "stdout", "stderr")}
            receipt.update(exit_code=code, timed_out=timed_out, seconds=time.time()-job["started"])
            receipts.append(receipt)
            active.remove(job)
            print("finished %s exit=%s timeout=%s" % (job["label"], code, timed_out), flush=True)
    while jobs and len(active)+len(external) < args.workers:
        phase, case, variant, iterations = jobs[0]
        if phase == "e4" and not (args.out / case / "iteration_060.npz").exists():
            break
        jobs.pop(0)
        label = phase + ("_" + variant if variant else "")
        result = args.out / case / (label + "_conditioning.json" if phase == "e4" else label + "/history.json")
        if result.exists():
            receipts.append(dict(label=case+"/"+label, preserved_existing=str(result)))
            continue
        module = "tools.wake_position_study" if phase == "e4" else "tools.wake_intervention_study"
        command = [sys.executable, "-m", module, "--phase", phase, "--case", case, "--out", str(args.out)]
        if variant:
            command += ["--variant", variant, "--iterations", str(iterations)]
        stdout = (args.out / case / (label+".log")).open("w", encoding="utf-8")
        stderr = (args.out / case / (label+"_error.log")).open("w", encoding="utf-8")
        process = subprocess.Popen(command, cwd=root, stdout=stdout, stderr=stderr)
        active.append(dict(label=case+"/"+label, command=command, started=time.time(),
                           process=process, pid=process.pid, stdout=stdout, stderr=stderr))
        print("started %s/%s pid=%d" % (case, label, process.pid), flush=True)
    status = dict(completed=receipts, external_pids=sorted(external), pending=jobs,
                  running=[{k: v for k, v in j.items() if k not in ("process", "stdout", "stderr")}
                           for j in active])
    (args.out / "queue.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    if jobs or active or external:
        time.sleep(5)
print("queue complete", flush=True)
