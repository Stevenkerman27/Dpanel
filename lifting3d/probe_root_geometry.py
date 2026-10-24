"""Retain builder locals at a near-tangent CAD/station failure, without changing the builder."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys

import numpy as np

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--aft", type=float, default=1.0001)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
assert not args.out.exists(), "Use a new diagnostic directory"
args.out.mkdir(parents=True)
sys.argv = [str(root/"build_mesh.py"), "--size", ".25", "--thickness", ".24",
            "--curvature", "16", "--wake-strips", "16", "--body-length", "4",
            "--body-radius", ".35", "--body-aft", str(args.aft),
            "--out", str(args.out/"mesh")]
source = (root/"build_mesh.py").read_bytes()
try:
    runpy.run_path(str(root/"build_mesh.py"), run_name="__main__")
    report = dict(status="builder_passed")
except Exception as error:
    frame = error.__traceback__
    while frame.tb_next is not None:
        frame = frame.tb_next
    state = frame.tb_frame.f_locals
    report = dict(status="captured_builder_failure", exception=type(error).__name__,
                  message=str(error), line=frame.tb_lineno)
    for name in ["stations", "root_interval", "exposed_stations"]:
        if name in state:
            report[name] = np.asarray(state[name]).tolist()
    if "te_nodes" in state:
        coordinates = state["points"][state["te_nodes"]]
        report["matched_TE_coordinates"] = coordinates.tolist()
        if "exposed_stations" in state and len(coordinates) == len(state["exposed_stations"]):
            report["matched_TE_station_error"] = (coordinates[:, 1]-state["exposed_stations"]).tolist()
    if "cad_original_points" in state:
        interval = state["root_interval"]
        raw_points = state["cad_original_points"]
        selected = ((abs(raw_points[:, 2]) < 1e-9) & (abs(raw_points[:, 0]-1.) < 1e-5)
                    & (abs(raw_points[:, 1]) < 2.*interval[1]))
        report["original_near_root_TE_points"] = raw_points[selected].tolist()
report.update(builder_sha256=hashlib.sha256(source).hexdigest(), command=sys.argv)
(args.out/"probe.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
(args.out/"probe_root_geometry.py.snapshot").write_bytes(Path(__file__).read_bytes())
(args.out/"build_mesh.py.snapshot").write_bytes(source)
print(json.dumps(report, indent=2))
