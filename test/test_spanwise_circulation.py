import numpy as np
import pytest
from panel.spanwise_circulation import circulation_lift_measures


def tiny():
    y=np.array([0.,1.,2.])
    H=np.array([[1.,0.],[0.,1.],[0.,0.]])
    q=np.array([.5-np.sqrt(3)/6,.5+np.sqrt(3)/6])
    B=np.stack((np.stack((1-q,q),axis=-1),
                np.stack((np.zeros_like(q),1-q),axis=-1)))
    return y,H,B


def edges(sweep):
    return np.array([[[0.,0.,0.],[sweep,1.,0.]],
                     [[sweep,1.,0.],[2*sweep,2.,0.]]])


def run(sweep):
    y,H,B=tiny()
    return circulation_lift_measures(y,H,[.2,.3],edges(sweep),B,
                density=1.225,speed=1.,reference_area=4.)


def test_unswept_arc_and_span_identical():
    r=run(0.)
    assert r['CL_gamma']==pytest.approx(.4)
    assert r['CL_gamma_exposed_TE_span']==pytest.approx(.4)
    assert r['CL_gamma_legacy_TE_arclength']==pytest.approx(.4)


def test_swept_arc_inflates_legacy_lift_not_span_lift():
    r=run(1.)
    assert r['CL_gamma']==pytest.approx(.4)
    assert r['CL_gamma_exposed_TE_span']==pytest.approx(.4)
    assert r['CL_gamma_legacy_TE_arclength']==pytest.approx(.4*np.sqrt(2))


def test_contact_gap_separates_exposed_TE_and_full_span():
    y,H,B=tiny()
    r=circulation_lift_measures(y,H,[.2,.3],edges(1.)[:1],B[:1],
                density=1.,speed=1.,reference_area=4.)
    assert r['CL_gamma']==pytest.approx(.4)
    assert r['CL_gamma_exposed_TE_span']==pytest.approx(.25)
    assert r['max_abs_span_coverage_gap']>0


def test_zero_tip_dof_is_not_renormalized():
    r=run(0.)
    assert r['span_measure']==pytest.approx([.5,1.])


def test_frozen_elliptic_1610_independent_span_check():
    y=np.array([0,.25,.31286893008046185,.5,.6180339887498949,.75,
      .9079809994790936,1,1.1755705045849463,1.25,1.4142135623730951,
      1.5,1.618033988749895,1.75,1.7820130483767358,1.902113032590307,
      1.9753766811902755,2.])
    gamma=np.array([.19063998077274155,.18921480322262801,.188061184179687,
      .1848026586576695,.18092103274859164,.17655110882609948,
      .1692958643275965,.16438099056284045,.15295764317037758,
      .1490548229700595,.13302411374691084,.1245897952495057,
      .11128202556654467,.09144376601294975,.08629421555345333,
      .05976431208063425,.03367464937119798])
    gs=np.r_[gamma,0.]
    correct=4/3.1722548357643863*np.sum(np.diff(y)*.5*(gs[:-1]+gs[1:]))
    assert correct==pytest.approx(.3759193814328078,abs=2e-13)
    assert .394820366478424-correct>0.018


@pytest.mark.parametrize('bad',["reverse","station","empty","basis","nan","area"])
def test_invalid_shapes_and_geometry_rejected(bad):
    y,H,B=tiny();g=np.array([.2,.3]);e=edges(0.);area=4.
    if bad=="reverse":e[0]=e[0,::-1]
    if bad=="station":y[1]=y[0]
    if bad=="empty":e=e[:0]
    if bad=="basis":B=B[:1]
    if bad=="nan":g[0]=np.nan
    if bad=="area":area=0.
    with pytest.raises(ValueError):
        circulation_lift_measures(y,H,g,e,B,density=1.225,speed=1.,
                                  reference_area=area)
