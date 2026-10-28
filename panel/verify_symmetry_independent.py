"""Independent, JAX-free symmetry audit for `dpanel.planar_cut.v1` fixtures.

Companion to `panel.audit_half_mesh.py`: same question, separate NumPy code
path, so the two implementations can be cross-checked. Nothing here solves a
flow, edits a mesh or relaxes a tolerance; a `status == "ok"` verdict only
says the discrete surface is mirror invariant.

    python -m panel.verify_symmetry_independent --surface cases/panel_cut/single_wing.json
"""
import argparse
import json

from panel import symmetry


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface", required=True, help="dpanel.planar_cut.v1 JSON")
    parser.add_argument("--out", default=None, help="write the report JSON here")
    parser.add_argument("--tol", type=float, default=1e-10,
                        help="relative mirror tolerance (default 1e-10, same as the audit)")
    parser.add_argument("--station-tol", type=float, default=1e-6)
    parser.add_argument("--fail-on-blocked", action="store_true",
                        help="exit 1 when the surface is not mirror invariant (default: report only)")
    args = parser.parse_args(argv)
    if args.tol <= 0:
        parser.error("--tol must be positive")
    with open(args.surface, "r", encoding="utf-8") as handle:
        surface = json.load(handle)
    report = symmetry.analyze_surface(surface, tol=args.tol, station_tol=args.station_tol)
    nodes, faces, wake = report["nodes"], report["faces"], report["wake"]
    te, stations, geometry = report["trailing_edge"], report["stations"], report["geometry"]
    print("surface  %s" % args.surface)
    print("status   %s   blocking: %s" % (report["status"], ", ".join(report["blocking_layers"]) or "-"))
    print("nodes    %d/%d paired, bijective %s, involution %s, max %.3e" % (
        nodes["paired"], nodes["count"], nodes["bijective"], nodes["involution"], nodes["max_residual"]))
    print("wall     %d/%d geometrically verified (%d vertex-set hits, %d spurious, worst %.3e)" % (
        faces["paired"], faces["count"], faces["vertex_set_hits"], faces["spurious_vertex_set_hits"],
        faces["worst_spurious_distance"]))
    print("wake     %d/%d verified (%d vertex-set hits, %d spurious)" % (
        wake["paired"], wake["count"], wake["vertex_set_hits"], wake["spurious_vertex_set_hits"]))
    print("TE       %d edges, %d geometrically mirrored, %d bank-matched, %d mismatched" % (
        te["te_edges"], te["te_edges_geometrically_mirrored"], te["te_edges_paired"], te["bank_mismatches"]))
    print("stations %d/%d paired, involution %s" % (
        stations["paired"], stations["count"], stations["involution"]))
    print("geometry %d plane, %d straddling, %d centroid-near-plane" % (
        geometry["plane_faces"], geometry["straddle_faces"], geometry["centroid_near_plane"]))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=1, default=float)
        print("report   %s" % args.out)
    if args.fail_on_blocked and report["status"] != "ok":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
