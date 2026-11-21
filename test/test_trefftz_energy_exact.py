import numpy as np
import pytest

from panel.trefftz_energy_exact import exact_trefftz_piecewise_linear as exact


def test_analytic_triangular_circulation_and_lift():
    # Gamma = 1-|y|, y in [-1,1]. Double-log energy = -4*log(2).
    # rho=U=1, Sref=2 => qS=1, CDi = log(2)/pi, CL = 1.
    result = exact([-1, 0, 1], [0, 1, 0], U=1., density=1., reference_area=2.)
    assert result['CDi_ideal'] == pytest.approx(np.log(2)/np.pi, abs=2e-14)
    assert result['CL_circulation'] == pytest.approx(1., abs=2e-14)


def subdivide(y,g,k):
    y=np.asarray(y,float);g=np.asarray(g,float)
    return (np.r_[*[np.linspace(y[i],y[i+1],k+1)[:-1] for i in range(len(y)-1)],y[-1]],
            np.r_[*[np.linspace(g[i],g[i+1],k+1)[:-1] for i in range(len(g)-1)],g[-1]])


@pytest.mark.parametrize('factor',[2,4,8,16])
def test_exact_energy_does_not_change_when_piecewise_linear_segments_split(factor):
    y=np.array([-2.,-1.5,-.4,0.,.4,1.5,2.])
    g=np.array([0.,.11,.35,.4,.35,.11,0.])
    ys,gs=subdivide(y,g,factor)
    a=exact(y,g,U=1,density=1,reference_area=1.7)
    b=exact(ys,gs,U=1,density=1,reference_area=1.7)
    assert b['CDi_ideal']==pytest.approx(a['CDi_ideal'],rel=2e-11,abs=2e-11)
    assert b['CL_circulation']==pytest.approx(a['CL_circulation'],rel=1e-13)


def test_x_and_gamma_scales_density_and_reference_area():
    y=[-1,0,1];g=[0,1,0]
    a=exact(y,g,U=1,density=1,reference_area=2.)
    b=exact(np.array(y)*3+8,np.array(g)*2,U=2,density=3,reference_area=4.)
    assert b['induced_drag_N']==pytest.approx(a['induced_drag_N']*12,abs=1e-13)
    assert b['CDi_ideal']==pytest.approx(a['CDi_ideal']*.5,abs=1e-13)


def test_zero_circulation_exact_zero():
    d=exact([-1.,-.4,0,1],[0,0,0,0],U=1,density=1,reference_area=3)
    assert d['CDi_ideal']==0 and d['CL_circulation']==0


@pytest.mark.parametrize('y,g,extra',[([0,1,2],[1,0,0],{}),
                                   ([0,1,2],[0,1,2],{}),
                                   ([0,0,1],[0,1,0],{}),
                                   ([0,1,2],[0,np.nan,0],{}),
                                   ([0,1,2],[0,1,0],{'max_segments':1}),
                                   ([0,1,2],[0,1,0],{'max_segments':True}),
                                   ([0,1,2],[0,1,0],{'chunk':0}),
                                   ([0,1,2],[0,1,0],{'chunk':True})])
def test_malformed_or_budget_rejected(y,g,extra):
    with pytest.raises(ValueError):
        exact(y,g,U=1,density=1,reference_area=1,**extra)
