"""Four high-information research checks, not a gate-heavy validation suite."""
import json
import numpy as np
from pathlib import Path
from panel.p1_lifting_g1 import make_wing,from_dpanel_half_surface,solve,alpha_sensitivity


def test_g1_picks_circulation_at_nonzero_incidence_without_hardcoded_cl():
    wing=make_wing(chord_panels=4,span_panels=4,wake_panels=2)
    zero=solve(wing,0.,kutta='pressure')
    positive=solve(wing,5.,kutta='pressure')
    assert abs(zero['CL'])<1e-10 and abs(zero['CL_gamma'])<1e-10
    assert positive['CL']>.15 and positive['CL_gamma']>.15
    assert positive['jump_rms']<1e-10
    assert positive['pressure_kutta_rms']<2e-4


def test_linear_and_nonlinear_kutta_are_separate_closures():
    wing=make_wing(chord_panels=4,span_panels=6)
    a=solve(wing,5.,kutta='signed_speed')
    b=solve(wing,5.,kutta='pressure')
    assert a['CL']>.2 and b['CL']>.2
    assert abs(a['CL']-b['CL'])<.02
    assert a['signed_kutta_rms']<1e-10
    assert b['pressure_kutta_rms']<1e-4
    assert a['jump_rms']<1e-10 and b['jump_rms']<1e-10


def test_alpha_derivative_is_analytic_not_finite_difference_fit():
    wing=make_wing(chord_panels=4,span_panels=4)
    result=alpha_sensitivity(wing,5.)
    h=1e-3
    derivative=(solve(wing,5.+h,kutta='signed_speed')['CL']-solve(wing,5.-h,kutta='signed_speed')['CL'])/(2*np.deg2rad(h))
    assert abs(result['dCL_dalpha_rad']-derivative)<2e-7


def test_half_surface_adapter_runs_full_mirrored_g1(tmp_path):
    w=make_wing(chord_panels=4,span_panels=4,wake_panels=2)
    xyz=np.vstack([w.xyz,w.wake_xyz]);wake=w.wake_faces+len(w.xyz)
    wall_half=w.faces[w.xyz[w.faces][:,:,1].mean(axis=1)>1e-9]
    wake_half=wake[xyz[wake][:,:,1].mean(axis=1)>1e-9]
    used=np.unique(np.r_[wall_half.ravel(),wake_half.ravel(),w.te_geo[2:]])
    ind={int(v):k for k,v in enumerate(used)}
    def map_indices(a):return np.vectorize(ind.__getitem__)(a).tolist()
    te=np.stack((w.te_geo[2:],w.te_geo[2:]),axis=1)
    fixture=dict(coordinates=xyz[used].tolist(),faces=map_indices(wall_half),
                 wake_faces=map_indices(wake_half),trailing_edges=map_indices(te),
                 gamma_stations=w.gamma_y[2:].tolist(),reference_chord=w.chord,
                 reference_area=w.reference_area,reference_point=[0,0,0])
    p=tmp_path/'half_surface.json';p.write_text(json.dumps(fixture))
    full=from_dpanel_half_surface(p)
    result=solve(full,5.,kutta='pressure')
    assert len(full.faces)==len(w.faces)
    assert full.ndof_mu==w.ndof_mu
    assert result['CL']>.15 and result['CL_gamma']>.15
    assert result['jump_rms']<1e-10
