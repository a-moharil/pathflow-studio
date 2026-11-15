import copy,json
import numpy as np
import pytest
import torch
from torch import nn
from lab.core.inspector import signed_diagnostics,channel_catalog,connection_catalog,ChannelModel
from lab.core.evidence import state,attributions,rescaled,tensor,scalar
from lab.core.healthcare import case_analysis
from lab.evidence_jobs import run,cancellation_demo
from test_evidence08 import tiny,R


def enumerate_paths(st):
    paths=[(i,float(v),i) for i,v in enumerate(st['dx'])]
    for l,(u,v,w) in enumerate(st['layers']):
        new=[]
        for node,a,origin in paths:
            for k in np.flatnonzero(u==node):new.append((v[k],a*w[k]*st['sec'][l][v[k]],origin))
        paths=new
    return paths

@pytest.mark.parametrize('activation',['relu','tanh'])
def test_signed_log_propagation_matches_independent_enumeration(activation):
    m,x,r=tiny(activation);st=state(m,x,r,R);d=signed_diagnostics(m,x,r,R,st);paths=enumerate_paths(st)
    P=np.zeros(16);N=np.zeros(16);qp=np.zeros(16);qn=np.zeros(16)
    for _,a,i in paths:
        if a>=0:P[i]+=a;qp[i]+=a*a
        else:N[i]-=a;qn[i]+=a*a
    q=(qp+qn).sum()
    for key,expected in [('P',P),('N',N),('S',P-N),('A_plus',qp/q),('A_minus',qn/q)]:np.testing.assert_allclose(np.array(d[key]).ravel(),expected,atol=1e-12)
    ca=case_analysis(m,x,r,R);np.testing.assert_allclose(np.array(d['A_plus'])+d['A_minus'],ca['input_fields']['conditional'],atol=1e-12)
    assert d['global_values']['reconstruction_residual']<1e-11
    # Regions must aggregate signed totals, not average local cancellation.
    region=np.arange(5);p=P[region].sum();n=N[region].sum();assert 0<=1-abs(p-n)/(p+n)<=1
    json.dumps(d,allow_nan=False)


def test_gradcam_formula_and_compensated_invariance():
    m,x,r=tiny();st=state(m,x,r,R);cat=channel_catalog(m,x,r,R,st)
    y,hs,zs=m(tensor(x).requires_grad_(True),trace=True)
    for l,entry in enumerate(cat['layers']):
        grad=torch.autograd.grad(scalar(y,R).sum(),hs[l+1],retain_graph=True)[0]
        expected=torch.relu((grad.mean((2,3),keepdim=True)*hs[l+1]).sum(1))[0].detach().numpy()
        np.testing.assert_allclose(entry['gradcam'],expected,atol=1e-12)
        assert abs(sum(c['mass'] for c in entry['channels'])-1)<1e-12
    for factor in [.1,10]:
        mm=rescaled(m,1,0,factor);cc=channel_catalog(mm,x,r,R,state(mm,x,r,R))
        for a,b in zip(cat['layers'],cc['layers']):np.testing.assert_allclose(a['gradcam'],b['gradcam'],atol=1e-11)

@pytest.mark.parametrize('activation',['relu','tanh'])
def test_channel_intervention_matches_native_gate(activation):
    m,x,r=tiny(activation)
    for alpha in [0,.4,1]:
        changed=ChannelModel(m,0,1,alpha);gates={'0':{str(i):alpha for i in range(9,18)}}
        for inp in [x,r]:np.testing.assert_allclose(changed(tensor(inp)).detach(),m(tensor(inp),gates).detach(),atol=1e-12)
        ca=case_analysis(changed,x,r,R);d=signed_diagnostics(changed,x,r,R,state(changed,x,r,R))
        assert ca['residual']<1e-11 and d['global_values']['reconstruction_residual']<1e-11
        # Equal endpoints exercises the otherwise ambiguous equal-preactivation secants.
        ds=state(changed,x,x,R);assert not ds['conditional']['defined']
        if alpha==0:assert np.all(ds['sec'][0][9:18]==0)


def test_connection_identity_and_reversal():
    m,x,r=tiny();s=state(m,x,r,R);cn=connection_catalog(m,s,0,1)
    assert len(cn['rows'])==36
    for row in cn['rows']:
        i,pid,sc,sy,sx,dy,dx,ky,kx,w,mass=row
        assert pid>=4 and sy==dy+ky and sx==dx+kx
        assert w==m.layers[0].weight.detach().numpy().ravel()[pid]
    a=signed_diagnostics(m,x,r,R,s);b=signed_diagnostics(m,r,x,R,state(m,r,x,R))
    np.testing.assert_allclose(a['P'],b['N'],atol=1e-12);np.testing.assert_allclose(a['A_plus'],b['A_minus'],atol=1e-12)
    aa=attributions(m,x,r,R,32);bb=attributions(m,r,x,R,32);np.testing.assert_allclose(aa['integrated_gradients'],-np.array(bb['integrated_gradients']),atol=1e-12)


def test_demo_and_undefined():
    demo=cancellation_demo();a,b=demo['rows'];np.testing.assert_allclose(a['field_a'],[[2/3,1/3]],atol=1e-12);np.testing.assert_allclose(a['field_a'],b['field_a'])
    assert b['diagnostics']['C'][0][0]==1 and b['diagnostics']['S'][0][0]==0
    assert [c['frames'][-1]['delta'] for c in b['curves']]==[0,2,1]
    assert a['delta']==3 and b['delta']==1
    m,x,r=tiny();d=signed_diagnostics(m,x,x,R,state(m,x,x,R));assert d['status']=='undefined_measure' and d['global_values']['C'] is None and d['A_plus'] is None
    json.dumps(demo,allow_nan=False)


def test_new_modes_serializable_and_frozen():
    m,x,r=tiny();before=copy.deepcopy(m.state_dict())
    for mode in ['compare','intervene','reverse','demo']:
        p=dict(evidence_mode=mode,evidence_intervention='channel',visual_layer=0,visual_channel=1,ig_steps=8,support_budget=1)
        result=run(m,x,r,R,p,lambda _:None)
        json.dumps(result,allow_nan=False)
        if mode=='intervene':
            assert len(result['curves'])==3 # two available random channel controls
            assert result['curves'][0]['channel_support']['index']==1
            assert all('diagnostics' in f for c in result['curves'] for f in c['frames'])
    for key,v in m.state_dict().items():torch.testing.assert_close(v,before[key])
