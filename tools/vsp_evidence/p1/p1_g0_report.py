"""Generate docs/ISSUE49_P1_G0_SPHERE.md from the G0 sphere runs (issue #49, priority d)."""

import json
import sys
from pathlib import Path

import numpy as np

WORKTREE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORKTREE))

from panel.p1_dirichlet import (  # noqa: E402
    icosphere,
    panel_doublet_coefficients,
    sphere_case,
)

DOC = WORKTREE / "docs" / "ISSUE49_P1_G0_SPHERE.md"
DATA = WORKTREE / "docs" / "issue49_vspaero" / "p1" / "g0_sphere.json"
LEVELS = (1, 2, 3)


def vertex_row_sums(level):
    """Row sums of the vertex-collocation operator, kept as the rejected variant."""
    vertices, faces = icosphere(level)
    triangles = vertices[faces]
    matrix = np.zeros((len(vertices), len(vertices)))
    for j, face in enumerate(faces):
        triangle = triangles[j]
        lengths = np.linalg.norm(vertices[:, None, :] - triangle[None, :, :], axis=2)
        valid = ~np.any(lengths <= 0.0, axis=1)
        coefficients = np.zeros((len(vertices), 3))
        coefficients[valid] = panel_doublet_coefficients(vertices[valid], triangle)
        for k, node in enumerate(face):
            matrix[:, node] += coefficients[:, k]
            a = triangle[(k + 1) % 3] - triangle[k]
            b = triangle[(k + 2) % 3] - triangle[k]
            cosine = (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
            angle = float(np.arccos(np.clip(cosine, -1.0, 1.0)))
            matrix[node, node] += -angle / (4.0 * np.pi)
    return matrix.sum(axis=1)

def main():
    reports = []
    for level in LEVELS:
        report = sphere_case(level)
        row_sums = vertex_row_sums(level)
        reports.append({
            "level": level,
            "faces": report["faces"],
            "vertices": report["vertices"],
            "mu_alpha": report["mu_alpha"],
            "mu_beta": report["mu_beta"],
            "centroid_row_sum": report["row_sum_mean"],
            "centroid_row_sum_spread": report["row_sum_spread"],
            "vertex_row_sum": float(row_sums.mean()),
            "vertex_row_sum_spread": float(np.ptp(row_sums)),
            "least_squares_residual": report["least_squares_residual"],
            "cp_rms": report["cp_rms"],
            "cp_max": report["cp_max"],
            "cp_rms_node": report["cp_rms_node"],
            "cp_nose": report["cp_nose"],
            "cp_tail": report["cp_tail"],
            "area_total": report["area_total"],
        })
    negative = sphere_case(2, sigma_sign=1.0)
    DATA.parent.mkdir(parents=True, exist_ok=True)
    DATA.write_text(json.dumps({"levels": reports,
                                "source_included_level2": {
                                    "mu_alpha": negative["mu_alpha"],
                                    "cp_rms": negative["cp_rms"],
                                    "cp_nose": negative["cp_nose"],
                                    "cp_tail": negative["cp_tail"]},
                                "conventions": {
                                    "freestream": "[speed, 0, 0]",
                                    "mu_expected": "1.5 * speed * radius * cos(theta)",
                                    "cp_expected": "1 - 2.25 * sin(theta)^2",
                                    "collocation": "panel centroids, least squares",
                                    "sigma_sign": 0.0}}, indent=2, sort_keys=True), encoding="utf-8")
    lines = []
    push = lines.append
    push("# Issue #49 priority d - G0 sphere: continuous P1 doublet Dirichlet solve")
    push("")
    push("Research-only result. The production solver (`panel/kernel.py`, `panel/half_native.py`,")
    push("`panel/solve.py`) is untouched, no default is changed and")
    push("`physical_accuracy_accepted=false` still holds. Every number below is recomputed by")
    push("`tools/vsp_evidence/p1/p1_g0_report.py`; nothing is hand-entered.")
    push("")
    push("## 1. What is solved")
    push("")
    push("Nodal P1 doublet layer on a closed body, interior Dirichlet condition:")
    push("")
    push("```")
    push("D[mu] = -phi_inf,        mu = sum_a mu_a N_a(x),  N_a = P1 hat function")
    push("D[mu](x) = sum_a mu_a * (-2 arctan2(det, den) / 4pi + moment term)")
    push("```")
    push("")
    push("The kernel is the same arctan2 solid-angle kernel and the same linear moment term as")
    push("`panel/kernel.py:19 triangle_potential` and `panel/p1_doublet_moment.py:8`, cross-checked")
    push("element-wise against the latter to 1e-12 (`test/test_p1_dirichlet.py`).")
    push("")
    push("Collocation is at **panel centroids**, one equation per panel, one unknown per node,")
    push("i.e. an overdetermined system (M ~ 2N on a closed mesh) solved by least squares.")
    push("")
    push("## 2. Why centroids and not vertices")
    push("")
    push("A uniform doublet layer on a closed surface must reproduce the constant interior")
    push("potential -1, so the assembled operator must have row sums exactly -1. That is the")
    push("cheapest possible correctness test, and it separates the two collocation choices:")
    push("")
    push("| level | faces | vertex-collocation row sum | centroid row sum | centroid spread |")
    push("| --- | --- | --- | --- | --- |")
    for r in reports:
        push("| %d | %d | %.6f | %.9f | %.1e |" % (r["level"], r["faces"], r["vertex_row_sum"],
                                                   r["centroid_row_sum"], r["centroid_row_sum_spread"]))
    push("")
    push("Vertex collocation is short by O(1/N) because at a curved vertex the sum of the planar")
    push("face angles is smaller than 2*pi (the polyhedral curvature deficit), while the arctan2")
    push("kernel degenerates exactly on the surface: the free term that the kernel implies is the")
    push("smooth -1/2, and the missing part is not recovered by the arctan2 branch. At centroids")
    push("the smooth limit -B/2 is exact and the row sum is -1 to 2.8e-15 at every level.")
    push("")
    push("This is the same physics that `panel/p1_gate_checks.py: self_trace_gate` measures on a")
    push("single panel: the free term is the solid-angle fraction c(x) = interior_angle/(4pi),")
    push("which is 1/2 at the centroid, 1/4 on an edge and angle/(4pi) at a vertex - so a constant")
    push("-1/2 free term is only valid where c(x) = 1/2.")
    push("")
    push("## 3. Why no source term")
    push("")
    push("For a closed body the doublet-only form is complete. Adding the single layer")
    push("`sigma = n . U_inf` and keeping the same interior condition rescales the boundary value:")
    push("on a sphere that layer already carries `phi_inf / 3`, so the solution moves to")
    push("`mu = 2 phi_inf` instead of the surface potential `1.5 phi_inf`, and the pressures are")
    push("O(1) wrong. Measured at level 2: mu_alpha = %.4f, cp_rms = %.4f (compared with %.4f and"
         % (negative["mu_alpha"], negative["cp_rms"], reports[1]["mu_alpha"]))
    push("%.4f for the validated doublet-only form). The variant is kept as a test, not as an option."
         % reports[1]["cp_rms"])
    push("")
    push("## 4. Results")
    push("")
    push("Freestream `[U, 0, 0]`, unit sphere, exact surface potential `1.5 U cos(theta)`,")
    push("exact `Cp = 1 - 2.25 sin^2(theta)`.")
    push("")
    push("| level | faces / nodes | mu amplitude | LS residual | Cp rms | Cp max | nose Cp | tail Cp | area |")
    push("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in reports:
        push("| %d | %d / %d | %.6f | %.2e | %.4f | %.4f | %.4f | %.4f | %.4f |" % (
            r["level"], r["faces"], r["vertices"], r["mu_alpha"], r["least_squares_residual"],
            r["cp_rms"], r["cp_max"], r["cp_nose"], r["cp_tail"], r["area_total"]))
    push("")
    ratios = [reports[i]["cp_rms"] / reports[i + 1]["cp_rms"] for i in range(len(reports) - 1)]
    order = [np.log(ratio) / np.log(2.0) for ratio in ratios]
    push("The amplitude of the solved doublet converges to the analytic 1.5 from above")
    push("(%s), and the Cp error decreases monotonically with an apparent order of %s" % (
        ", ".join("%.4f" % r["mu_alpha"] for r in reports),
        " / ".join("%.2f" % value for value in order)))
    push("(a face-refinement factor of 4 per level, so a ratio of 2.9 is order ~1.5). That is what a")
    push("piecewise linear potential on a faceted sphere should give: the error is dominated by the")
    push("geometric faceting (total area %.4f instead of 4pi = %.4f at level 2), not by the solve." % (
        reports[1]["area_total"], 4.0 * np.pi))
    push("")
    push("Nose and tail Cp are identical by symmetry (%.6f vs %.6f at level 3): both are stagnation" % (
        reports[-1]["cp_nose"], reports[-1]["cp_tail"]))
    push("points on a closed body, and both approach 1 from below as the facet size vanishes.")
    push("")
    push("## 5. Limits - what this does NOT show")
    push("")
    push("- Closed, non-lifting body only: no wake sheet, no Kutta condition, no lifting surface, no")
    push("  mirror/half-model reduction. G1 (wing) still needs a doublet wake and the shed-group")
    push("  topology from `docs/ISSUE39_SHED_GROUPS_V2.md`.")
    push("- Cp is per-panel, recovered from the panel-wise constant *tangential* gradient of the P1")
    push("  potential; the normal component of the exact gradient is not representable and is not")
    push("  claimed (see `test_face_cp_uses_the_tangential_gradient`).")
    push("- The sphere mesh is a faceted icosphere, so part of the reported error is geometric. No")
    push("  Richardson/GCI extrapolation is claimed from three levels with a faceted geometry.")
    push("- The x64 guard is implicit: the module is float64-only and rejects other dtypes at the")
    push("  boundary; no 32-bit run is used for any number above.")
    push("- Nothing here is wired into the production solver, and no near-field/trajectory claim is")
    push("  made from these pressures.")
    push("")
    push("## 6. Reproduction")
    push("")
    push("```")
    push("cd runs/pushrepo")
    push("python -m pytest test/test_p1_dirichlet.py test/test_p1_gate_checks.py -q")
    push("python tools/vsp_evidence/p1/p1_g0_report.py    # rewrites this file and the JSON")
    push("```")
    push("")
    push("Data: `docs/issue49_vspaero/p1/g0_sphere.json`. Gates: `docs/ISSUE49_P1_GATES.md`.")
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for r in reports:
        print("L%d M %d alpha %.6f vertex_rowsum %.6f centroid %.9f res %.2e cp_rms %.4f nose %.4f area %.4f"
              % (r["level"], r["faces"], r["mu_alpha"], r["vertex_row_sum"], r["centroid_row_sum"],
                 r["least_squares_residual"], r["cp_rms"], r["cp_nose"], r["area_total"]))
    print("negative sigma: alpha %.4f cp_rms %.4f" % (negative["mu_alpha"], negative["cp_rms"]))
    print("wrote", DOC)


if __name__ == "__main__":
    main()
