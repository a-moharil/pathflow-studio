import numpy as np
import pytest
from types import SimpleNamespace
from lab.core.fitsettings import dimensions, iterations, FitHistory

def test_limits_and_legacy():
    assert iterations(5000)==5000
    assert dimensions(True)==[8]
    assert dimensions(False)==[]
    assert dimensions([32,8,16,8])==[8,16,32]
    for value in [24,5001,25.5,True]:
        with pytest.raises(ValueError): iterations(value)
    with pytest.raises(ValueError): dimensions([9])

def test_long_history_retains_endpoint_and_real_iterations():
    history=FitHistory(lambda x:(float(x[0]),None),6000)
    for i in range(1,5001): history(np.array([6000-i]))
    rows=history.finish(SimpleNamespace(nit=5000,fun=1000))
    assert len(rows)<=501 and rows[0]['iteration']==0
    assert rows[-1]==dict(iteration=5000,objective=1000)
    assert any(r['iteration']>4000 for r in rows)

def test_all_diagnostic_dimensions_and_metric_integrity():
    import copy
    from test_hyperbolic import example
    from lab.core.metricfit import refine
    from lab.core.hyperbolic import compute
    data=example();before=copy.deepcopy(data)
    design,targets=refine(data,25,[8,16,32])
    report=compute(data,.7,25,design,targets,[8,16,32])
    assert set(data['euclidean']['dimensions'])=={'2','3','8','16','32'}
    assert set(report['dimensions'])=={'3','8','16','32'}
    for key in ['edges','metric','conditional']: assert data[key]==before[key]
    for entries in report['dimensions'].values():
        for e in entries:
            o=e['optimization'];assert o['history'][-1]['iteration']==o['iterations']
            assert o['max_iterations']==25 and 'stop_reason' in o


def test_larger_budgets_reach_both_optimizers():
    from lab.core.metricfit import fit_euclidean,euclidean_distances
    from lab.core.hyperbolic import fit,distances
    import torch
    rng=np.random.default_rng(18);x=rng.normal(size=(9,3))*.2
    pairs=np.array([(i,j) for i in range(9) for j in range(i+1,9)])
    _,e=fit_euclidean(x, pairs,euclidean_distances(x,pairs),5000)
    z=np.column_stack((np.sqrt(1+(x*x).sum(1)),x))
    _,h=fit(9,3,1,pairs,distances(torch.tensor(z),pairs[:,0],pairs[:,1],1).numpy(),x,5000)
    for r in [e,h]:
        assert r['max_iterations']==5000 and r['max_evaluations']==200000
        assert r['history'][-1]['iteration']==r['iterations']
