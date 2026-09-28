"""Local maps preserve native coordinates and the original global normalization."""
import numpy as np
import torch
import pytest
from lab.config import validate
from lab.core.network import Network
from lab.core.pathmath import analyze
from lab.joint.interpret import region_analysis
from lab.joint.tape import Measure
from test_joint10 import fixture

@pytest.mark.parametrize('model',['mlp','cnn'])
def test_source_partition_and_native_shapes(model):
    torch.manual_seed(42)
    c=validate(dict(model=model,widths=[4,3],channels=[2,3],kernel=3,stride=3,padding=1,cnn_head=True,head_widths=[4],activation='tanh',train_size=32))
    m=Network(c);rng=np.random.default_rng(91);x=rng.random((1,28,28));ref=rng.random(x.shape)
    def calc(box):
        m.config['source_region']=box
        return analyze(m,x,ref,dict(scope='selected',a=0,b=1),geometry_enabled=False)
    full=calc([0,0,28,28]);left=calc([0,0,14,28]);right=calc([14,0,28,28]);assert full['source_region']['source_mass']==pytest.approx(1)
    assert left['source_region']['source_mass']+right['source_region']['source_mass']==pytest.approx(1)
    for g,f,a,b in zip(full['layer_fields'],full['source_region']['layers'],left['source_region']['layers'],right['source_region']['layers']):
        key='native_absolute' if len(g['shape'])==3 else 'unit_absolute'
        np.testing.assert_allclose(f[key],g[key],atol=1e-12)
        np.testing.assert_allclose(np.array(a[key])+np.array(b[key]),g[key],atol=1e-12)
        assert np.sum(a[key])==pytest.approx(left['source_region']['source_mass'],abs=1e-12)
        if key=='native_absolute':assert np.shape(g[key])==tuple(g['shape'][-2:])
        else:assert g['native_absolute'] is None
    m.config['source_region']=[0,0,2,2];zero=analyze(m,x,x,dict(a=0,b=1),geometry_enabled=False)
    assert zero['layer_fields'] is None and zero['source_region']['source_mass']==0


def test_residual_local_partition():
    t,s=fixture();m=Measure(t,s);a=region_analysis(m,[0,0,1,2]);b=region_analysis(m,[1,0,2,2]);full=region_analysis(m,[0,0,2,2])
    for x,y,z in zip(a['operations'],b['operations'],full['operations']):
        key='native_absolute' if x['native_absolute'] is not None else 'unit_absolute'
        np.testing.assert_allclose(np.array(x[key])+np.array(y[key]),z[key],atol=1e-12)
        assert np.sum(x[key])==pytest.approx(x['absolute_mass'],abs=1e-12)
