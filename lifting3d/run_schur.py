"""Compare two exact linear backends on frozen lifting meshes; retain resource samples."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import psutil

root = Path(__file__).resolve().parent
out = root/"results"/"schur_v1"
assert not out.exists(), "Preserve earlier backend evidence"
out.mkdir(parents=True)
source_paths = [root/"solve.py", root/"schur.py", root.parent/"volume"/"p2.py", root/"SCHUR_PROTOCOL.md"]
source_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
(out/"preflight.json").write_text(json.dumps(dict(source_hashes=source_hashes), indent=2), encoding="utf-8")
(out/"driver.py.snapshot").write_bytes(Path(__file__).read_bytes())
rows, checks = [], []
for name, mesh in [
    ("wing", root/"results"/"planforms_v1"/"combined"/"mesh"/"mesh.npz"),
    ("wingbody", root/"results"/"wingbody_matrix_v1"/"t024_r035_a090"/"mesh"/"mesh.npz"),
]:
    target = out/name
    target.mkdir()
    for backend in ["augmented", "schur"]:
        run_dir = target/backend
        command = [sys.executable, str(root/"solve.py"), "--mesh", str(mesh), "--alpha", "5",
                   "--mach", "0", "--kutta", "weak", "--ncpu", "8",
                   "--linear-solver", backend, "--out", str(run_dir)]
        tick = time.perf_counter()
        samples, peak_rss, peak_private = 0, 0, 0
        with (target/(backend+".log")).open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            monitored = psutil.Process(process.pid)
            resource_rejected = False
            while process.poll() is None:
                try:
                    memory = monitored.memory_info()
                    peak_rss = max(peak_rss, memory.rss)
                    peak_private = max(peak_private, getattr(memory, "private", memory.vms))
                    samples += 1
                except psutil.NoSuchProcess:
                    break
                if time.perf_counter()-tick > 600. or psutil.virtual_memory().available < 2.*1024**3:
                    process.terminate()
                    resource_rejected = True
                    break
                try:
                    process.wait(timeout=.5)
                except subprocess.TimeoutExpired:
                    pass
            process.wait()
        result_path = run_dir/"result.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        row = dict(case=name, backend=backend, command=command, returncode=process.returncode,
                   resource_rejected=resource_rejected, seconds=time.perf_counter()-tick,
                   sampled_peak_working_set_bytes=peak_rss, sampled_peak_private_bytes=peak_private,
                   resource_samples=samples, converged=result.get("converged", False),
                   cl=result.get("cl_pressure"), history=result.get("history"))
        rows.append(row)
        checks.append(dict(test=name+"_"+backend+"_converged",
                            passed=bool(process.returncode == 0 and result.get("converged", False))))
        (out/"progress.json").write_text(json.dumps(dict(rows=rows, checks=checks), indent=2), encoding="utf-8")
    if not all(row["converged"] for row in rows[-2:]):
        continue
    augmented = json.loads((target/"augmented"/"result.json").read_text())
    eliminated = json.loads((target/"schur"/"result.json").read_text())
    assert augmented["solver_sha256"] == eliminated["solver_sha256"] == source_hashes[str(root/"solve.py")]
    with np.load(target/"augmented"/"field.npz") as reference, np.load(target/"schur"/"field.npz") as candidate:
        for key in ["phi", "gamma", "surface_velocity", "cp"]:
            error = float(np.max(abs(reference[key]-candidate[key])))
            checks.append(dict(test=name+"_"+key, max_abs_error=error, limit=1e-9, passed=error < 1e-9))
    error = float(np.max(abs(np.array(augmented["pressure_force_coefficient"])-eliminated["pressure_force_coefficient"])))
    checks.append(dict(test=name+"_pressure_force", max_abs_error=error, limit=1e-9, passed=error < 1e-9))
summary = dict(rows=rows, checks=checks, passed=sum(check["passed"] for check in checks),
               total=len(checks), source_hashes=source_hashes,
               sampling_note="RSS/private sampled at 0.5 s; sampled maxima are not guaranteed absolute peaks")
(out/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
assert len(checks) == 14 and all(check["passed"] for check in checks)
