"""Shared catastrophic-solution screening, not an aerodynamic accuracy certificate.

Wake column support and matrix condition are basis/scale-dependent proxies.
They do not replace a geometric support audit, contact compatibility, or grid
and quadrature convergence. No value is clipped and no solver is retuned here.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class PlausibilityControls:
    max_abs_cl: float = 20.
    max_abs_cd: float = 1.
    max_gamma_to_mu_norm: float = 50.
    max_constrained_condition: float = 1e6
    min_column_to_median: float = .1
    same_case_cd_warning_factor: float = 5.


def assess_solution(state, *, controls=None, reference_cd=None):
    """Return deterministic status/reasons; an optional CD reference is SAME case.

    reference_cd is informational only and must come from the identical geometry,
    incidence, closure and normalization. Comparing different aircraft cannot
    establish a universal acceptable drag value.
    """
    c = PlausibilityControls() if controls is None else controls
    if any(not np.isfinite(v) or v <= 0 for v in asdict(c).values()):
        raise ValueError('Plausibility controls must be positive and finite')
    reasons, warnings = [], []
    cl, cd = float(state.get('CL', 0.)), float(state.get('CD_pressure', 0.))
    mu = float(state.get('mu_norm', np.linalg.norm(state.get('mu', []))))
    gamma = float(state.get('gamma_norm', np.linalg.norm(state.get('gamma', []))))
    cond = state.get('constrained_condition')
    values = [cl, cd, mu, gamma] + ([] if cond is None else [float(cond)])
    if not np.isfinite(values).all():
        reasons.append('nonfinite_solution_or_condition')
    if abs(cl) > c.max_abs_cl:
        reasons.append('excessive_abs_CL')
    if abs(cd) > c.max_abs_cd:
        reasons.append('excessive_abs_CD_pressure')
    if gamma > c.max_gamma_to_mu_norm*max(mu, 1e-30):
        reasons.append('excessive_gamma_to_mu_norm')
    if cond is not None and float(cond) > c.max_constrained_condition:
        reasons.append('excessive_constrained_condition')
    mn, med = state.get('gamma_column_norm_min'), state.get('gamma_column_norm_median')
    if mn is not None and med is not None and float(mn) < c.min_column_to_median*max(float(med), 1e-30):
        # Nonuniform bases or physical tip taper can produce small columns.
        # Report this proxy without calling a finite, converged field invalid.
        warnings.append('weak_gamma_column_support_proxy')
    if reference_cd is not None:
        ref = float(reference_cd)
        if not np.isfinite(ref) or abs(ref) < 1e-15:
            raise ValueError('A finite nonzero same-case CD reference is required')
        if abs(cd) > c.same_case_cd_warning_factor*abs(ref):
            warnings.append('same_case_pressure_drag_regression')
    nonlinear = state.get('nonlinear')
    if nonlinear is not None and not nonlinear.get('success', False):
        status = 'nonlinear_not_converged'
    elif reasons:
        status = 'implausible_solution'
    else:
        status = 'solver_success'
    return dict(solution_status=status, implausible_reasons=reasons,
                diagnostic_warnings=warnings, plausibility_controls=asdict(c),
                physical_accuracy_accepted=False)
