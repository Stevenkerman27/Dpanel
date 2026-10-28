"""Summarize PR47 verification outputs: counts, quality audit, provenance, stencil rank."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\3D\Projects\panel")
OUT = ROOT / "runs" / "pr47_verify"
sys.path.insert(0, str(ROOT / "runs" / "pr47_git"))


def shared_te(data):
    f = np.asarray(data["faces"], np.int64)
    w = np.asarray(data["wake_faces"], np.int64)
    te = np.asarray(data["trailing_edges"], np.int64)
    def edges(faces):
        box = set()
        for face in faces:
            for a, b in ((0, 1), (1, 2), (2, 0)):
                box.add((int(min(face[a], face[b])), int(max(face[a], face[b]))))
        return box
    wall_edges, wake_edges = edges(f), edges(w)
    te_set = {(int(min(a, b)), int(max(a, b))) for a, b in te}
    return dict(te_edges=len(te_set), te_in_wall=len(te_set & wall_edges),
                te_in_wake=len(te_set & wake_edges))


def summary(tag, *, stencil=False):
    out = OUT / tag
    surface = out / "surface.json"
    if not surface.is_file():
        fail = out / "failure.json"
        if fail.is_file():
            data = json.loads(fail.read_text(encoding="utf-8"))
            return dict(tag=tag, status="refused", stage=data.get("stage"),
                        message=data.get("message"), exception=data.get("exception"))
        return dict(tag=tag, status="missing")
    data = json.loads(surface.read_text(encoding="utf-8"))
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    audit = data.get("audit") or {}
    prov = (data.get("provenance") or {}).get("feature_sizing")
    row = dict(tag=tag, status="ok", wall=len(data["faces"]),
               wake=len(data["wake_faces"]), vertices=len(data["coordinates"]),
               gamma_dof=max(data["gamma_station_dof"]) + 1,
               elapsed_s=round(manifest.get("elapsed_seconds", 0.0), 2),
               mesh_quality=audit.get("mesh_quality"),
               max_plane_snap=audit.get("max_plane_snap"),
               no_artificial_symmetry_wall=audit.get("no_artificial_symmetry_wall"),
               wake_wall_contact_edges=audit.get("wake_wall_contact_edges"),
               wall_cut_kinds=audit.get("wall_cut_kinds"),
               symmetry_plane_vertex_count=audit.get("symmetry_plane_vertex_count"),
               feature=None if prov is None else dict(
                   source_sha256=prov.get("source_sha256", "")[:16],
                   feature_names=prov.get("feature_names"),
                   requested=prov.get("requested_features"),
                   global_size=prov.get("global_size"), far_size=prov.get("far_size")))
    row.update(shared_te(data))
    if stencil:
        from panel.run_half_native import load_half_surface
        _, sa = load_half_surface(str(surface), stencil_rings=2, max_stencil_rings=6,
                                  return_stencil_audit=True)
        ratios = np.asarray(sa["initial_singular_ratios"], float)
        row["stencil"] = dict(min_ratio=float(ratios.min()),
                              faces_below_1e_3=int((ratios < 1e-3).sum()),
                              expanded=len(sa["expanded_faces"]),
                              max_depth=int(np.max(sa["selected_ring_depth"])))
    return row


def main():
    tags = sys.argv[1:] or ["base", "feat_le", "feat_lete", "gates_pass"]
    rows = [summary(t, stencil=t in ("base", "feat_le", "feat_lete", "gates_pass")) for t in tags]
    (OUT / "compare.json").write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
    for row in rows:
        if row["status"] != "ok":
            print("%-18s %-8s %s | %s" % (row["tag"], row["status"], row.get("stage"), row.get("message")))
            continue
        w = (row["mesh_quality"] or {}).get("wall", {})
        st = row.get("stencil") or {}
        print("%-18s wall=%4d wake=%4d vert=%4d dof=%3d %5.1fs minang=%5.2f growth=%s shape=%s te=%d/%d/%d stencil=%s below=%s exp=%s" % (
            row["tag"], row["wall"], row["wake"], row["vertices"], row["gamma_dof"],
            row["elapsed_s"], w.get("min_angle_deg", float("nan")),
            ("%.2f" % w["p99_neighbor_area_ratio"]) if w.get("p99_neighbor_area_ratio") else "n/a",
            ("%.3f" % w["min_shape_quality"]) if w.get("min_shape_quality") else "n/a",
            row["te_edges"], row["te_in_wall"], row["te_in_wake"],
            ("%.2e" % st["min_ratio"]) if st else "n/a",
            st.get("faces_below_1e_3", "n/a"), st.get("expanded", "n/a")))


if __name__ == "__main__":
    main()