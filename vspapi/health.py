"""Sanity gates learned from the thick-panel investigation.

Two failure modes are cheap to detect and very expensive to miss:

1. **Chordwise clustering.**  In panel mode `CLi` is computed by summing
   Kutta-Joukowski circulations along body-fitted vortex strips.  If the
   leading-edge clustering leaves a size jump between neighbouring chordwise
   rows, that sum picks up an O(1/h) singular term.  Measured on the
   trapezoid wing (52x120 panels, alpha = 5 deg, identical case file)::

       LECluster 0.25 -> CLi = 125.45   (TECluster 0.9)
       LECluster 0.50 -> CLi =   5.50
       LECluster 0.80 -> CLi =   0.635  <- CLwtot = 0.654, correct

   `TECluster` and `MaxGrowth` make no difference at all.

2. **Polar rows that cannot be trusted.**  Even with a good mesh, VSPAERO's
   `CDo` contains an additive term in inviscid VLM runs, so `CDtot` and `L/D`
   are meaningless; `CLi` must be consistent with `CLwtot`.

`docs/VSPAERO_PANEL_POLAR_DIAGNOSIS.md` has the full evidence.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

LE_CLUSTER_SAFE = 0.70
LE_CLUSTER_MARGINAL = 0.50


def cluster_risk(te_cluster: float, le_cluster: float,
                 max_growth: Optional[float] = None) -> dict:
    """Classify a chordwise clustering setting before meshing.

    Returns a dict with `risk` in {"ok", "marginal", "high"} plus the evidence
    used, so a caller can raise or just warn.
    """
    if le_cluster >= LE_CLUSTER_SAFE:
        risk, why = "ok", "LE cluster >= 0.70: smooth chordwise growth"
    elif le_cluster >= LE_CLUSTER_MARGINAL:
        risk, why = "marginal", "LE cluster 0.50-0.70: panel-mode CLi may be 5-10x off"
    else:
        risk, why = "high", "LE cluster < 0.50: panel-mode CLi diverges (measured 125x at 0.25)"
    return {
        "risk": risk,
        "reason": why,
        "te_cluster": float(te_cluster),
        "le_cluster": float(le_cluster),
        "max_growth": None if max_growth is None else float(max_growth),
        "max_growth_relevant": False,
    }


def check_polar_health(rows: Sequence[Dict[str, float]], *,
                       cli_ratio_limit: float = 1.5,
                       cd_tot_tolerance: float = 1e-6) -> dict:
    """Flag polar rows whose panel-mode force channels disagree.

    `CLi` is the trailing-edge Kutta-Joukowski lift, `CLwtot` the wake-induced
    lift that VSPAERO also reports; on a healthy mesh they agree to a few
    percent.  `CDo` is non-zero even in inviscid VLM mode, so `CDtot` is only
    usable when it equals `CDi + CDo` to within `cd_tot_tolerance` *and* `CDo`
    is credible (it is not, for VLM); we simply report the split.
    """
    per_row: List[dict] = []
    issues: List[str] = []
    for row in rows:
        cli = row.get("CLi")
        clw = row.get("CLwtot")
        ratio = None
        if cli is not None and clw not in (None, 0.0):
            ratio = cli / clw
        cd_i, cd_o = row.get("CDi"), row.get("CDo")
        cd_tot = row.get("CDtot")
        additive_ok = None
        if None not in (cd_i, cd_o, cd_tot):
            additive_ok = abs((cd_i + cd_o) - cd_tot) <= cd_tot_tolerance * max(1.0, abs(cd_tot))
        bad = ratio is not None and abs(ratio) > cli_ratio_limit
        per_row.append({
            "alpha": row.get("AoA"),
            "cl_i": cli,
            "cl_wtot": clw,
            "cli_over_clwtot": ratio,
            "cli_trusted": not bad,
            "cd_tot_additive": additive_ok,
        })
        if bad:
            issues.append(
                "alpha=%.4g: CLi/CLwtot = %.4g (|ratio| > %.2g) -> CLi/CDi unusable, "
                "use CLwtot and the .lod strips" % (row.get("AoA", float("nan")), ratio,
                                                   cli_ratio_limit))
    return {
        "rows": per_row,
        "issues": issues,
        "healthy": not issues,
        "trusted_columns": ["CLwtot", "CDwtot", "CFxtot", "CFztot"] if issues else ["all"],
        "untrusted_columns": ["CLi", "CDi", "CDtot", "L/D", "E"] if issues else [],
        "cd_tot_note": "CDo is a non-physical additive term in these runs; CDtot and L/D "
                       "must not be compared across codes",
    }
