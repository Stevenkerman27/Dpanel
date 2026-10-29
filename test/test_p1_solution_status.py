import ast
from pathlib import Path
import numpy as np
import pytest
from panel.p1_solution_status import assess_solution, PlausibilityControls


def test_extreme_load_not_weak_column_proxy_detects_original_conv_failure():
    state=dict(CL=1327.656,CD_pressure=11664.77,mu_norm=228.4,gamma_norm=297.7,
               constrained_condition=4.70e5,gamma_column_norm_min=4.494e-4,
               gamma_column_norm_median=2.144e-3,nonlinear=None)
    a=assess_solution(state)
    assert a['solution_status']=='implausible_solution'
    assert 'excessive_abs_CL' in a['implausible_reasons']
    assert 'weak_gamma_column_support_proxy' not in a['diagnostic_warnings']


def test_weak_column_is_a_warning_not_a_geometric_support_proof():
    a=assess_solution(dict(CL=.4,CD_pressure=.01,gamma_column_norm_min=.01,
                           gamma_column_norm_median=1.))
    assert a['solution_status']=='solver_success'
    assert a['diagnostic_warnings']==['weak_gamma_column_support_proxy']
    assert a['physical_accuracy_accepted'] is False


def test_same_case_drag_reference_warns_without_altering_the_solution():
    s=dict(CL=.4,CD_pressure=.0999)
    a=assess_solution(s,reference_cd=.01)
    assert 'same_case_pressure_drag_regression' in a['diagnostic_warnings']
    assert s['CD_pressure']==.0999
    assert a['solution_status']=='solver_success'


def test_nonconvergence_precedes_plausibility_and_nonfinite_is_invalid():
    a=assess_solution(dict(CL=1e4,nonlinear=dict(success=False,nfev=80)))
    assert a['solution_status']=='nonlinear_not_converged'
    assert assess_solution(dict(CL=np.nan))['solution_status']=='implausible_solution'
    with pytest.raises(ValueError):
        assess_solution({},controls=PlausibilityControls(max_abs_cl=-1))


def test_shared_screening_matches_runner():
    from tools.issue71.galerkin_cad_sweep import status
    for s in ({'CL':.4,'CD_pressure':.01}, {'CL':21.}, {'CD_pressure':2.}):
        assert status(s,80)==assess_solution(s)['solution_status']


def test_modified_regression_modules_have_no_shadowed_test_names():
    root=Path(__file__).parent
    for filename in ('test_p1_galerkin_cad.py','test_p1_wake_contact.py','test_p1_solution_status.py'):
        tree=ast.parse((root/filename).read_text())
        names=[n.name for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]
        assert len(names)==len(set(names)), filename
