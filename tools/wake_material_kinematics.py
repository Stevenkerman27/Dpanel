"""Check the unchanged B3 material map in a prescribed uniform axial flow."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from panel.p1_lifting_g2 import combine_wings
from panel.p1_steady_free_wake import _one_wake_target, prepare_wake_material
from tools.issue92.b3_dpanel import _wing
from tools.wake_sensitivity_geometry import attachment_rows

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
report = dict(schema="dpanel.wake_sensitivity.uniform_axial.v1",
              command=[sys.executable, *sys.argv],
              source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in (Path(__file__), root / "panel/p1_steady_free_wake.py")},
              imposed_velocity=[1., 0., 0.], min_axial=.08, freeze=True,
              interpretation="kinematic identity check only; no aerodynamic solve or physical accuracy acceptance",
              cases={}, physical_accuracy_accepted=False)
for case in ("rect", "swept30", "elliptic"):
    folder = args.out / case
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    spec = manifest["spec"]
    wing = _wing(spec)
    with np.load(folder / "initial_state.npz") as data:
        for key in ("xyz", "faces", "wake_xyz", "wake_faces", "gamma_hat"):
            np.testing.assert_array_equal(getattr(wing, key), data[key])
    mesh = combine_wings([wing], reference_chord=spec["cref"], reference_area=spec["sref"],
                         reference_point=wing.reference_point)
    group = mesh.groups[0]
    top = attachment_rows(mesh)
    material = prepare_wake_material(group)
    velocity = np.zeros_like(group.wake_xyz)
    velocity[:, 0] = 1.
    target, frozen = _one_wake_target(group, material, velocity, min_axial=.08, freeze=True)
    displacement = np.linalg.norm(target-group.wake_xyz, axis=1)/mesh.reference_chord
    nodes = top["used"][material.predecessor[top["used"]] >= 0]
    predecessor = material.predecessor[nodes]
    offset = (group.wake_xyz[nodes]-group.wake_xyz[predecessor])[:, 1:]
    offset_norm = np.linalg.norm(offset, axis=1)/mesh.reference_chord
    off_axis = nodes[offset_norm > 1e-10]
    attached_moving = top["attached"][displacement[top["attached"]] > 1e-10]
    result = dict(
        frozen_initial_state_sha256=manifest["initial_state_sha256"],
        wake_faces=len(group.wake_faces), used_nodes=len(top["used"]),
        attached_ids=top["attached"].tolist(), graph_roots=material.roots.tolist(),
        moving_attachment_ids=attached_moving.tolist(),
        uniform_flow_max_raw_displacement_chord=float(displacement[top["used"]].max()),
        uniform_flow_attachment_max_raw_displacement_chord=float(displacement[top["attached"]].max()),
        uniform_flow_relaxed_attachment_displacement_chord=float(.3*displacement[top["attached"]].max()),
        uniform_flow_identity_passed=bool(displacement[top["used"]].max() <= 1e-10),
        off_axis_predecessor_count=len(off_axis), off_axis_predecessor_ids=off_axis.tolist(),
        max_original_crossstream_predecessor_offset_chord=float(offset_norm.max()),
        axial_guard_frozen_ids=frozen,
        target_formula="target_yz[node]=target_yz[pred]+.5*original_dx*(Vyz[pred]/Vx[pred]+Vyz[node]/Vx[node]); original transverse edge offset is absent")
    report["cases"][case] = result
    print("%s roots=%d attached=%d moving_attached=%d uniform_raw=%.6gc attach_raw=%.6gc" % (
        case, len(material.roots), len(top["attached"]), len(attached_moving),
        result["uniform_flow_max_raw_displacement_chord"],
        result["uniform_flow_attachment_max_raw_displacement_chord"]))
(args.out / "uniform_axial_kinematics.json").write_text(json.dumps(report, indent=2, allow_nan=False),
                                                       encoding="utf-8")
