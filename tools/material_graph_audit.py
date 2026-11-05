"""Export the frozen material graph of the original G2 T5 CAD case.

python -m tools.material_graph_audit --case g2 --surface runs/cases/wingbody/half_cad_wingbody_v1/surface.json --out verification/wake_material_graph/g2_t5/material_graph.json

Uses the same case builder, material preparation and audit as the coupled
solver. No flow solve, wake update, freezing or topology change is performed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

from panel.p1_coupled_material_wake import material_paths
from panel.p1_steady_free_wake import prepare_wake_material, wake_material_graph_audit
from tools.issue84.wake_node_rejection import build_case


root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--case", choices=("g1", "g2"), default="g2")
parser.add_argument("--surface", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--t5-result", type=Path, help="also compare actual T5 callback records")
args = parser.parse_args()
surface = args.surface if args.surface.is_absolute() else root / args.surface
out = args.out if args.out.is_absolute() else root / args.out
if not surface.is_file():
    parser.error("missing CAD input: " + str(surface))
started = time.perf_counter()
head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                      capture_output=True, text=True).stdout.strip()
status = subprocess.run(["git", "status", "--porcelain"], cwd=root, check=True,
                        capture_output=True, text=True).stdout.splitlines()
sources = ("tools/material_graph_audit.py", "tools/issue84/wake_node_rejection.py",
           "tools/issue84/t5_gate_ab.py",
           "panel/p1_lifting_g1.py", "panel/p1_lifting_g2.py",
           "panel/p1_steady_free_wake.py", "panel/p1_coupled_material_wake.py")
source_hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                 for name in sources}
mesh, alpha = build_case([surface], case=args.case, alpha_deg=5.)
chord = float(mesh.reference_chord)
report = dict(
    schema="dpanel.wake_material_graph.v1", case=args.case, reference_chord=chord,
    alpha_deg=alpha, graph_stage="frozen_pre_iteration", flow_solve_performed=False,
    physical_accuracy_accepted=False,
    case_builder="tools.issue84.wake_node_rejection.build_case",
    generated_tail=(dict(chord_panels=4, span_panels=4, wake_panels=3,
                         span=1.8, chord=.55, origin=[1.6, 0., .35])
                    if args.case == "g2" else None),
    environment=dict(python=sys.version, executable=sys.executable,
                     platform=platform.platform(), numpy=np.__version__),
    git_head=head, worktree_status=status, source_sha256=source_hashes,
    input=dict(path=surface.relative_to(root).as_posix() if surface.is_relative_to(root)
               else str(surface), sha256=hashlib.sha256(surface.read_bytes()).hexdigest()),
    node_id_scope="indices in each group's wake_xyz, including unused wall vertices",
    path_direction="node_to_root", length_units="geometry units; *_chord divides by reference_chord",
    wall_faces=int(len(mesh.faces)), material_graph={}, groups={})

for group in mesh.groups:
    material = prepare_wake_material(group)
    audit = wake_material_graph_audit(material)
    xyz = np.asarray(material.reference_xyz, float)
    faces = np.asarray(group.wake_faces, int)
    pred = np.asarray(material.predecessor, int)
    used = np.asarray(material.used, int)
    distance, roots = material_paths(material)
    edge_faces = {}
    for face_id, (a, b, c) in enumerate(faces):
        for p, q in ((a, b), (b, c), (c, a)):
            edge_faces.setdefault(tuple(sorted((int(p), int(q)))), []).append(face_id)
    links = []
    for node in used[pred[used] >= 0]:
        node, parent = int(node), int(pred[node])
        incident = edge_faces.get(tuple(sorted((node, parent))), [])
        delta = xyz[node] - xyz[parent]
        length = float(np.linalg.norm(delta))
        links.append(dict(node=node, predecessor=parent, length=length,
                          length_chord=length / chord,
                          node_xyz=xyz[node].tolist(), predecessor_xyz=xyz[parent].tolist(),
                          dx=float(delta[0]), dy=float(delta[1]), dz=float(delta[2]),
                          is_mesh_edge=bool(incident), wake_face_ids=incident,
                          boundary_edge=len(incident) == 1,
                          cumulative_path_length=float(distance[node]),
                          root=int(roots[node])))
    longest = []
    for link in sorted(links, key=lambda row: (-row["length"], row["node"]))[:20]:
        node = link["node"]
        neighbors = set()
        for edge in edge_faces:
            if node in edge:
                neighbors.update(edge)
        tolerance = 2e-11 * max(float(np.ptp(xyz[used, 0])), 1.)
        upstream = [p for p in neighbors if xyz[p, 0] < xyz[node, 0] - tolerance]
        candidates = [dict(node=p, xyz=xyz[p].tolist(),
                           material_y_difference=float(abs(xyz[node, 1] - xyz[p, 1])),
                           dx=float(xyz[node, 0] - xyz[p, 0]),
                           length=float(np.linalg.norm(xyz[node] - xyz[p])))
                      for p in sorted(upstream, key=lambda p: (
                          abs(xyz[node, 1] - xyz[p, 1]), xyz[node, 0] - xyz[p, 0], p))]
        longest.append(dict(**link, upstream_candidates=candidates))
    path = audit["max_root_to_tail_path_nodes"]
    path_links = {row["node"]: row for row in links}
    path_length = sum(path_links[node]["length"] for node in path[:-1])
    invalid_edges = [row["node"] for row in links if not row["is_mesh_edge"]]
    checks = dict(all_predecessors_are_mesh_edges=not invalid_edges,
                  invalid_edge_nodes=invalid_edges,
                  no_cycles=audit["predecessor_cycle_count"] == 0,
                  no_nonmonotone_x_edges=audit["nonmonotone_x_edge_count"] == 0,
                  rooted_path_length_matches=bool(np.isclose(
                      path_length, audit["max_root_to_tail_path_length"], rtol=1e-12, atol=1e-12)),
                  all_used_nodes_reach_root=bool(np.isin(roots[used], material.roots).all()))
    report["material_graph"][group.name] = audit
    report["groups"][group.name] = dict(
        vertex_count=len(xyz), used_nodes=used.tolist(),
        unused_nodes=np.setdiff1d(np.arange(len(xyz)), used).tolist(),
        roots=material.roots.tolist(), order=material.order.tolist(),
        reference_xyz=xyz.tolist(), wake_faces=faces.tolist(),
        predecessor=pred.tolist(), gamma_y=np.asarray(group.gamma_y).tolist(),
        links=links, longest_links=longest,
        longest_path=[dict(node=int(node), xyz=xyz[node].tolist(),
                           cumulative_path_length=float(distance[node])) for node in path],
        summary_chord=dict(direct_link_median=audit["direct_link_median"] / chord,
                           direct_link_max=audit["direct_link_max"] / chord,
                           max_root_to_tail_path_length=audit["max_root_to_tail_path_length"] / chord),
        checks=checks)
    if not all(value for key, value in checks.items() if key != "invalid_edge_nodes"):
        raise AssertionError(group.name + ": graph consistency check failed")
    print("%s: used=%d roots=%d direct_max=%.6fc path_max=%.6fc; all links are mesh edges"
          % (group.name, len(used), len(material.roots), audit["direct_link_max"] / chord,
             audit["max_root_to_tail_path_length"] / chord), flush=True)

if args.t5_result is not None:
    t5_path = args.t5_result if args.t5_result.is_absolute() else root / args.t5_result
    t5 = json.loads(t5_path.read_text(encoding="utf-8"))
    if t5["case"] != args.case or t5["material_graph"] != report["material_graph"]:
        raise AssertionError("T5 result does not match the exported material graph")
    if not t5["rows"] or any(row["material_graph"] != report["material_graph"]
                             for row in t5["rows"]):
        raise AssertionError("T5 callback material graph changed or was not saved")
    report["t5_cross_check"] = dict(
        path=t5_path.relative_to(root).as_posix() if t5_path.is_relative_to(root) else str(t5_path),
        sha256=hashlib.sha256(t5_path.read_bytes()).hexdigest(), status=t5["status"],
        condition=t5["condition"], iterations=t5["iterations"], seconds=t5["seconds"],
        every_record_matches=True,
        rows=[dict(iteration=row["iteration"], CL=row["CL"],
                   map_residual_chord=row["map_residual_chord"],
                   accepted=row["accepted"], relaxation=row["relaxation"]) for row in t5["rows"]])
    print("T5 callback graph matches in all %d records." % len(t5["rows"]), flush=True)

report["seconds"] = time.perf_counter() - started
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
print("Wrote " + str(out), flush=True)
