import copy
import numpy as np
import torch
from lab.core.hyperbolic import lorentz,distances,move_origin,ball,compute,inspect_pair
from lab.core.pathmath import geometry

def test_lorentz_isometry_and_ball_distance():
    rng=np.random.default_rng(2);k=.7
    z=lorentz(torch.tensor(rng.normal(size=(12,3))),k).numpy()
    np.testing.assert_allclose(-z[:,0]**2+(z[:,1:]**2).sum(1),-1/k,atol=1e-13)
    a=np.arange(11);b=a+1;before=distances(torch.tensor(z),a,b,k).numpy()
    moved=move_origin(z,z[-1],k)
    after=distances(torch.tensor(moved),a,b,k).numpy();np.testing.assert_allclose(before,after,atol=1e-12)
    p=ball(z,k);p0=p[a];p1=p[b]
    truth=np.arccosh(1+2*((p0-p1)**2).sum(1)/((1-(p0*p0).sum(1))*(1-(p1*p1).sum(1))))/np.sqrt(k)
    np.testing.assert_allclose(before,truth,atol=1e-12)
    assert np.max(np.linalg.norm(p,axis=1))<1

def example():
    sizes=[3,4,2,1];offset=np.cumsum([0]+sizes);layers=[];F=[]
    for a,b in zip(sizes[:-1],sizes[1:]):
        v,u=np.indices((b,a));layers.append((u.ravel(),v.ravel(),np.ones(a*b)));F.append(np.full(a*b,1/(a*b)))
    Y,g,_=geometry(sizes,offset,layers,F,limit=50,landmarks=8)
    return dict(sizes=sizes,nodes=[dict(layer=l) for l,n in enumerate(sizes) for _ in range(n)],edges=[dict(u=int(u+offset[l]),v=int(v+offset[l+1]),s=float(F[l][j]),c=float(F[l][j])) for l,(us,vs,_) in enumerate(layers) for j,(u,v) in enumerate(zip(us,vs))],metric=dict(epsilon=.03,display_reference=dict(unit=g['unit'])),positions=[(Y/g['unit']).tolist()]*2,conditional=dict(defined=True))

def test_hyperbolic_fidelity_and_input_immutability():
    d=example();before=copy.deepcopy(d);h=compute(d,.7,25);assert d==before
    assert h['status']=='available' and h['training_pairs'] and h['validation_pairs']
    for dim in ['3']:
        s,c=h['dimensions'][dim]
        assert s['constraint_residual']<1e-10
        np.testing.assert_allclose(s['ball'],c['ball'],atol=1e-9)
        assert np.linalg.norm(s['ball'][-1])<1e-10
        assert np.max(np.linalg.norm(s['ball_free'],axis=1))<1
        assert 0<=s['neighbor_overlap_at_10']<=1
        assert len(s['heldout']['by_scale'])==4
    d['hyperbolic']=h;p=inspect_pair(d,0,2)
    assert p['s']['intrinsic_raw']>0 and p['s']['hyperbolic_3']>0
    assert inspect_pair(d,1,1)['c']['hyperbolic_3']==0
    d['conditional']['defined']=False
    z=compute(d,1.,25);assert z['dimensions']['3'][1]['status']=='undefined_measure'
