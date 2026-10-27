"""Generate docs/ISSUE49_P1_G1_DESIGN.md: measured building blocks for the P1 G1 wing step."""

import json
import sys
from pathlib import Path

import numpy as np

WORKTREE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORKTREE))

from panel.p1_wake import dof_audit, jump_across_sheet, mirror_nodes, sheet_potential  # noqa: E402

DOC = WORKTREE / "docs" / "ISSUE49_P1_G1_DESIGN.md"
DATA = WORKTREE / "docs" / "issue49_vspaero" / "p1" / "g1_design.json"
MESH = Path("D:/3D/Projects/panel/runs/half_cad_wingbody_v1/surface.json")


def rectangle(n_chord=2, n_span=2, chord=1.0, span=1.0):
    xs = np.linspace(0.0, chord, n_chord + 1)
    ys = np.linspace(-0.5 * span, 0.5 * span, n_span + 1)
    vertices = np.array([(x, y, 0.0) for x in xs for y in ys])
    index = lambda i, j: i * (n_span + 1) + j
    faces = []
    for i in range(n_chord):
        for j in range(n_span):
            a, b = index(i, j), index(i, j + 1)
            c, d = index(i + 1, j + 1), index(i + 1, j)
            faces.append((a, b, c))
            faces.append((a, c, d))
    return vertices, np.array(faces)


def wall_nodes():
    data = json.loads(MESH.read_text(encoding="utf-8"))
    coordinates = np.asarray(data["coordinates"], dtype=float)
    faces = np.asarray(data["faces"], dtype=np.int64)
    fused = np.unique(faces)
    return {
        "path": str(MESH),
        "coordinates": int(len(coordinates)),
        "wall_faces": int(len(faces)),
        "referenced_vertices": int(len(fused)),
        "unreferenced_vertices": int(len(coordinates) - len(fused)),
    }


def main():
    vertices, faces = rectangle()
    uniform = np.ones(len(vertices))
    jump_rows = []
    for row in jump_across_sheet(vertices, faces, uniform, heights=(1e-2, 1e-3, 1e-4, 1e-5, 1e-6))["rows"]:
        jump_rows.append({
            "height": row["height"],
            "max_abs_error": float(np.max(np.abs(row["jump"] - 1.0))),
            "mean": float(np.mean(row["jump"])),
        })
    linear = 2.0 * vertices[:, 1] + 0.5
    linear_result = jump_across_sheet(vertices, faces, linear, heights=(1e-6,))
    expected = 2.0 * linear_result["centroids"][:, 1] + 0.5
    linear_error = float(np.max(np.abs(linear_result["rows"][0]["jump"] - expected)))

    distances = np.array([10.0, 20.0, 40.0])
    targets = np.stack([distances, np.zeros(3), np.ones(3)]).T
    far = np.abs(sheet_potential(vertices, faces, uniform, targets))
    far_order = (np.log(far[:-1] / far[1:]) / np.log(2.0)).tolist()
    in_plane = np.abs(sheet_potential(vertices, faces, uniform, np.array([[5.0, 0.1, 0.0], [20.0, -0.2, 0.0]])))

    density = np.linspace(0.2, 0.9, len(vertices))
    mirrored, mirrored_faces = mirror_nodes(vertices, faces)
    probes = np.array([[0.3, 0.11, 0.4], [-0.2, -0.05, 0.9], [0.7, 0.2, 2.0]])
    plain = sheet_potential(vertices, faces, density, probes)
    mirrored_probes = probes.copy()
    mirrored_probes[:, 1] *= -1.0
    same_sign = sheet_potential(mirrored, mirrored_faces, density, mirrored_probes)
    flipped_sign = sheet_potential(mirrored, mirrored_faces, -density, mirrored_probes)
    mirror_same = float(np.max(np.abs(same_sign + plain)))
    mirror_flip = float(np.max(np.abs(flipped_sign - plain)))

    mesh = wall_nodes()
    audits = []
    for wake_nodes, stations in ((22, 21), (44, 21), (385, 21), (770, 21)):
        try:
            audit = dof_audit(n_wall_nodes=mesh["referenced_vertices"], n_wake_nodes=wake_nodes,
                              n_stations=stations, n_wall_rows=mesh["wall_faces"])
        except ValueError as error:
            audit = {
                "unknowns": mesh["referenced_vertices"] + wake_nodes,
                "equations": mesh["wall_faces"] + wake_nodes,
                "square": False,
                "error": str(error),
            }
        audit.update({"wake_nodes": wake_nodes, "stations": stations})
        audits.append(audit)

    DATA.parent.mkdir(parents=True, exist_ok=True)
    DATA.write_text(json.dumps({
        "jump_rows": jump_rows,
        "linear_density_error": linear_error,
        "far_field": {"distances": distances.tolist(), "values": far.tolist(), "order": far_order},
        "in_plane_potential": in_plane.tolist(),
        "mirror_residual": {"same_sign": mirror_same, "flipped_sign": mirror_flip},
        "mesh": mesh,
        "dof_audits": audits,
    }, indent=2, sort_keys=True), encoding="utf-8")

    in_plane_max = float(np.max(in_plane))
    far_order_text = " / ".join("%.2f" % value for value in far_order)
    lines = []
    push = lines.append
    push("# Issue #49 priority d - G1 design: P1 wall + wake sheet + Kutta")
    push("")
    push("Research-only. The production solver is untouched, no default changes and")
    push("`physical_accuracy_accepted=false` still holds. G0 (`docs/ISSUE49_P1_G0_SPHERE.md`,")
    push("commit `95b7ace`) established the closed-body P1 doublet operator; this document fixes")
    push("the three things G1 adds - the wake sheet, the Kutta row and the mirror rule - and")
    push("measures each of them in isolation before any wing coefficient is claimed.")
    push("")
    push("## 1. The system G1 has to assemble")
    push("")
    push("```")
    push("wall rows  : D_wall[mu] + W_wake[Gamma] = -phi_inf   (one per wall collocation point)")
    push("Kutta rows : mu_upper - mu_lower - Gamma_te = 0      (one per trailing edge station)")
    push("wake rows  : Gamma at station k+1 = Gamma at station k (one per wake node, stations fixed)")
    push("```")
    push("")
    push("Unknowns are the wall doublet nodes `mu` plus the wake doublet nodes `Gamma`. The wake is a")
    push("*doublet* sheet, so its density is a potential jump and the Kutta condition says that the")
    push("jump vanishes at the trailing edge.")
    push("")
    push("## 2. Wake sheet: the jump is the interpolated density")
    push("")
    push("A doublet sheet with density `mu` produces the potential jump `mu` across itself. On a")
    push("2x2 triangle strip (unit chord, unit span) with uniform density 1, the measured jump")
    push("`phi(+h n) - phi(-h n)` converges to 1 as the probe height `h` decreases:")
    push("")
    push("| probe height | max abs error of the jump | mean jump |")
    push("| --- | --- | --- |")
    for row in jump_rows:
        push("| %.0e | %.2e | %.9f |" % (row["height"], row["max_abs_error"], row["mean"]))
    push("")
    push("With a *linear* density `mu = 2y + 0.5` the jump equals the interpolated density at the")
    push("panel centroids to %.2e. The sheet carries `mu_h`, not the strip average, and that is what" % linear_error)
    push("makes the P1 wake consistent with a P1 wall. Close to the surface the arctan2 branch loses")
    push("absolute precision, so the measured error saturates at a few 1e-6: a kernel-evaluation")
    push("floor, not a discretisation effect.")
    push("")
    push("Two consequences G1 must respect:")
    push("")
    push("- a constant-density sheet seen from a point **in its own plane** has zero potential")
    push("  (measured %.1e at two in-plane points), because the potential is a solid angle;" % in_plane_max)
    push("  in-plane wake collocation would silently see nothing;")
    push("- far from a compact sheet the potential decays faster than a monopole (measured order")
    push("  %s at r = 10, 20, 40), so wake panels need no special far-field treatment here." % far_order_text)
    push("")
    push("## 3. Mirror rule: the mirrored sheet carries -Gamma")
    push("")
    push("Mirroring a sheet in the symmetry plane flips its normal, and a doublet is odd under a")
    push("normal flip. Measured on the strip with a linear density, comparing the mirrored geometry")
    push("against mirrored probe points:")
    push("")
    push("| mirrored density | residual against the unmirrored field |")
    push("| --- | --- |")
    push("| `+density` | %.2e (the field equals `-phi`, it is not `phi`) |" % mirror_same)
    push("| `-density` | %.2e |" % mirror_flip)
    push("")
    push("So the half-model wake term must enter with the mirrored slope subtracted - exactly the")
    push("`W = (dw + wi) @ basis + (my - mi) @ slope` structure of `panel/half_native.py`, and the")
    push("discrete version of `_REFLECT = [1, -1, 1]`. Being an algebraic identity of the kernel it")
    push("holds for any density, so it is asserted as a test rather than measured per case.")
    push("")
    push("## 4. Unknown/equation bookkeeping")
    push("")
    push("```")
    push("unknowns  = n_wall_nodes + n_wake_nodes")
    push("equations = n_wall_rows + n_stations + (n_wake_nodes - n_stations)")
    push("```")
    push("")
    push("The `n_stations` terms cancel, so the system is square only when `n_wall_rows = n_wall_nodes`.")
    push("On the real 396-panel wing/body half mesh (`%s`):" % MESH.name)
    push("")
    push("| coordinates | wall faces | referenced nodes | unreferenced vertices |")
    push("| --- | --- | --- | --- |")
    push("| %d | %d | %d | %d |" % (mesh["coordinates"], mesh["wall_faces"], mesh["referenced_vertices"], mesh["unreferenced_vertices"]))
    push("")
    push("A closed triangle mesh has roughly two panels per node, so `n_wall_rows = n_wall_nodes` fails")
    push("for the panel-row collocation of G0 and G1 cannot inherit it: the wall needs **node**")
    push("collocation (one row per wall node) and the free term has to come from the assembled solid")
    push("angle `c(x) = interior_angle / 4pi`, not from a constant `-1/2`.")
    push("")
    push("The unreferenced vertices are the wake strip nodes sharing the same coordinate array. They")
    push("are *not* wall unknowns - the same trap the gate audit found on the trapezoid meshes, where")
    push("16.4 percent of the vertices of the 12290-panel case are wake nodes.")
    push("")
    push("| wake nodes | stations | unknowns | equations | square |")
    push("| --- | --- | --- | --- | --- |")
    for audit in audits:
        push("| %d | %d | %d | %d | %s |" % (audit["wake_nodes"], audit["stations"], audit["unknowns"], audit["equations"], audit["square"]))
    push("")
    push("Those rows use `n_wall_rows = len(faces)`, i.e. the panel-row variant, and are listed to show")
    push("that the count cannot be closed that way. The node-row variant sets")
    push("`n_wall_rows = n_wall_nodes` by construction and is what G1 will assemble.")
    push("")
    push("## 5. What G1 must prove before any coefficient is reported")
    push("")
    push("1. Node-row wall collocation reproduces G0 on the sphere (row sums `-1` to machine zero with")
    push("   the assembled `c(x)`); the panel-row G0 result cannot be carried over.")
    push("2. Wake transport keeps `Gamma` constant along the streamwise stations on a flat strip with")
    push("   a prescribed `Gamma(y)`, to machine zero.")
    push("3. A zero-lift case (`alpha = 0` on a symmetric wing) returns `Gamma = 0` and the G0")
    push("   pressures, so the wake coupling adds no spurious lift.")
    push("4. `CL_pressure` and `CL_circulation` agree to a stated tolerance on one mesh, and their")
    push("   difference is reported per mesh level instead of being called convergence.")
    push("5. Mirror residuals are measured on every case (the rule above is exact algebraically, so a")
    push("   non-zero residual is a bug).")
    push("")
    push("## 6. Limits")
    push("")
    push("- No wing coefficient is claimed yet: the wake sheet operator, the Kutta rows and the mirror")
    push("  rule are verified in isolation only.")
    push("- The production solver, its defaults and its thresholds are untouched; nothing here is")
    push("  imported outside `panel/p1_*.py` and its tests.")
    push("- x64 guard: the module is float64-only, mirroring the gate-2 requirement; no 32-bit number")
    push("  appears in this document.")
    push("")
    push("Data: `docs/issue49_vspaero/p1/g1_design.json`. Generated by")
    push("`tools/vsp_evidence/p1/p1_g1_design.py`.")
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("jump", ["%.2e" % row["max_abs_error"] for row in jump_rows])
    print("linear %.2e far_order %s inplane %.1e" % (linear_error, far_order_text, in_plane_max))
    print("mirror same %.2e flip %.2e" % (mirror_same, mirror_flip))
    print("mesh", mesh)
    print("audits", [(a["wake_nodes"], a["unknowns"], a["equations"], a["square"]) for a in audits])


if __name__ == "__main__":
    main()
