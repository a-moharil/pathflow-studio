import numpy as np
from lab.core.transforms import image,transport,specification,hellinger

def test_identity_and_boundary_transport():
    rng=np.random.default_rng(4);x=rng.random((1,7,7));s=specification()
    np.testing.assert_array_equal(image(x,s),x)
    a=x[0]/x.sum();t=transport(a,s)
    np.testing.assert_allclose(t['field'],a);assert abs(t['lost_mass'])<1e-14
    shift=specification(dict(dx=1));t=transport(a,shift)
    assert abs(t['lost_mass']-a[:,-1].sum())<1e-14
    assert abs(np.sum(t['field'])-1)<1e-14
    assert hellinger(a,a)==0
    assert transport(a,specification(dict(dx=56)))['field'] is None

def test_rotation_scaling_and_zero():
    x=np.zeros((1,7,7));x[0,3,4]=1
    rotated=image(x,specification(dict(angle=90)));assert rotated[0,4,3]>.999
    for s in [specification(dict(scale=.5)),specification(dict(scale=2,dx=2,angle=17))]:
        assert image(np.zeros_like(x),s).sum()==0
        t=transport(x[0],s);assert 0<=t['retained_mass']<=1+1e-14

def test_zero_bias_and_recomputation():
    import torch
    from lab.config import validate
    from lab.core.network import Network
    from lab.core.pathmath import analyze
    torch.set_num_threads(2)
    for architecture in ['mlp','cnn']:
        torch.manual_seed(8);m=Network(validate(dict(model=architecture,widths=[3,3],channels=[2,2],bias=True,activation='tanh')))
        with torch.no_grad():
            for name,v in m.named_parameters():
                if 'bias' in name:v.fill_(.2)
        x=np.random.default_rng(1).random((1,28,28)).astype('float32');zero=np.zeros_like(x);readout=dict(scope='all',a=0,b=1)
        baseline=analyze(m,x,zero,readout,{},False)
        assert np.linalg.norm(baseline['reference_logits'])>0
        assert baseline['residual']<1e-12
        changed=analyze(m,image(x,specification(dict(dx=2,angle=15))),zero,readout,{},False)
        assert changed['residual']<1e-12
        assert not np.allclose(baseline['input_fields']['conditional'],changed['input_fields']['conditional'])
        assert changed['readout']==readout
