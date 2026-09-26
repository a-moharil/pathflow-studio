import copy
import numpy as np
import pytest
import torch
from torch import nn
from lab.core.network import Network
from lab.config import validate
from lab.core.evidence import (state,select_support,field_b,OccurrenceModel,identities,attributions,rescaled,internal_signal,occurrence_gradients,neighborhood)
from lab.core.healthcare import case_analysis

R=dict(scope='selected',a=1,b=0)
def tiny(activation='relu'):
    torch.manual_seed(71)
    m=Network(validate(dict(model='cnn',channels=[2,2],activation=activation,bias=True)))
    m.layers=nn.ModuleList([nn.Conv2d(1,2,2),nn.Conv2d(2,2,2),nn.Linear(8,2)])
    m.shapes=[(1,4,4),(2,3,3),(2,2,2),(2,)];m=m.double()
    with torch.no_grad():
        for l in m.layers[:-1]:l.weight.copy_(l.weight.abs()+.1);l.bias.fill_(.3)
    rng=np.random.default_rng(71);x=rng.random((1,4,4));r=rng.random((1,4,4))*.2
    return m,x,r

def test_field_b_mass_and_parameter_unfolding():
    m,x,r=tiny();s=state(m,x,r,R)
    q=select_support(m,s,x,r,R,dict(support_kind='parameter',support_rule='manual',manual_ids=[0],support_layers='0'))
    b=field_b(m,s,q);assert q['layers'][0]['occurrence_count']==9
    assert q['layers'][0]['parameter_count']==1
    assert abs(np.sum(b['image'])-q['layers'][0]['cut_mass'])<1e-12
    a=np.asarray(case_analysis(m,x,r,R)['input_fields']['conditional']);assert np.all(np.asarray(b['image'])<=a+1e-12)
    assert abs(sum(v['baseline_mass'] for v in b['occurrences'])-b['total_mass'])<1e-12
    assert np.isclose(np.sum(b['normalized_shape']),1)
    for a in b['occurrences']:assert a['kernel_index']==[0,0] and a['layer']==0
    deep=select_support(m,s,x,r,R,dict(support_layers='1'))
    bd=field_b(m,s,deep);assert bd['status']=='no_first_layer_support' and bd['total_mass']==0
    assert np.shape(bd['native'][0]['channels'])==(2,3,3)

@pytest.mark.parametrize('activation',['relu','tanh'])
def test_parameter_attenuation_matches_actual_weight_edit_and_paths(activation):
    m,x,r=tiny(activation);s=state(m,x,r,R)
    q=select_support(m,s,x,r,R,dict(support_kind='parameter',support_rule='manual',manual_ids=[0],support_layers='all'))
    for a in [1.,.4,0.]:
        c=OccurrenceModel(m,s,q,a);actual=copy.deepcopy(m)
        with torch.no_grad():
            for l in [0,1]:actual.layers[l].weight.flatten()[0]*=a
        inp=torch.tensor(x[None]);np.testing.assert_allclose(c(inp).detach(),actual(inp).detach(),atol=1e-12)
        v=case_analysis(c,x,r,R);assert v['residual']<1e-12 and v['signed_field_residual']<1e-12


def test_one_occurrence_changes_one_preactivation():
    m,x,r=tiny();s=state(m,x,r,R)
    q=select_support(m,s,x,r,R,dict(support_rule='manual',manual_ids=[3],support_layers='0'))
    c=OccurrenceModel(m,s,q,0);_,_,z=m(torch.tensor(x[None]),trace=True);_,_,zz=c(torch.tensor(x[None]),trace=True)
    diff=(z[0]-zz[0]).detach().numpy().ravel();u,v,w=s['layers'][0]
    assert np.sum(abs(diff)>1e-12)==1
    assert abs(diff[v[3]]-x.ravel()[u[3]]*w[3])<1e-12


def test_occurrence_gradients_aggregate_to_shared_parameter_gradients():
    m,x,r=tiny();s=state(m,x,r,R);g=occurrence_gradients(m,x,r,R,s)
    y=m(torch.tensor(x[None]))-m(torch.tensor(r[None]));grad=torch.autograd.grad(y[0,1]-y[0,0],[m.layers[0].weight,m.layers[1].weight])
    for l in [0,1]:
        pid=identities(m,l,s['layers'][l]);np.testing.assert_allclose(np.bincount(pid,weights=g[l],minlength=m.layers[l].weight.numel()),grad[l].numpy().ravel(),atol=1e-12)


def test_rescale_preserves_function_attribution_fields_and_metric():
    m,x,r=tiny();s=state(m,x,r,R);a=attributions(m,x,r,R,16);q=select_support(m,s,x,r,R,{})
    b=field_b(m,s,q);signal=internal_signal(m,x,R,0,1)
    for factor in [.1,10.]:
        scaled=rescaled(m,0,1,factor);ss=state(scaled,x,r,R);aa=attributions(scaled,x,r,R,16)
        np.testing.assert_allclose(scaled(torch.tensor(x[None])).detach(),m(torch.tensor(x[None])).detach(),atol=1e-12)
        for key in ['input_gradient','integrated_gradients']:np.testing.assert_allclose(aa[key],a[key],atol=1e-12)
        for key in ['fs','fc']:
            for v,w in zip(s[key],ss[key]):np.testing.assert_allclose(v,w,atol=1e-12)
        np.testing.assert_allclose(field_b(scaled,ss,q)['image'],b['image'],atol=1e-12)
        sig=internal_signal(scaled,x,R,0,1)
        np.testing.assert_allclose(sig['activation'],np.asarray(signal['activation'])*factor,atol=1e-12)
        np.testing.assert_allclose(sig['gradient'],np.asarray(signal['gradient'])/factor,atol=1e-12)
        np.testing.assert_allclose(neighborhood(m,s)['structural'],neighborhood(scaled,ss)['structural'],atol=1e-10)
    assert a['ig_residual']<1e-12


def test_undefined_and_invalid_rescaling():
    m,x,r=tiny();s=state(m,x,x,R)
    with pytest.raises(ValueError):select_support(m,s,x,x,R,{})
    q=select_support(m,s,x,x,R,dict(support_rule='weight'))
    assert field_b(m,s,q)['status']=='undefined_measure'
    m.config['activation']='tanh'
    with pytest.raises(ValueError):rescaled(m,0,0,2)


def test_api_evidence_modes_preserve_saved_model(tmp_path):
    from lab import create_app,storage as st,datasets
    from lab.engine import quantum
    from lab.worker import tick
    app=create_app(tmp_path);c=validate(dict(model='cnn',channels=[2,2],train_size=32,updates=1,checkpoint_every=1));datasets.prepare(tmp_path,c);rid=st.create(tmp_path,c)
    with st.db(tmp_path) as db:db.execute("UPDATE runs SET status='training' WHERE id=?",(rid,))
    quantum(tmp_path,st.run(tmp_path,rid));checkpoint=tmp_path/rid/'checkpoint_00000001.pt';before=checkpoint.read_bytes()
    client=app.test_client();client.get('/healthcare/'+rid)
    with client.session_transaction() as session:token=session['csrf']
    headers={'X-CSRF-Token':token};base=dict(kind='healthcare_evidence',step=1,x=0,reference_type='zero',split='test',support_budget=2,ig_steps=8,sweep_frames=2,case_indices=[0,1])
    for mode,extra in [('compare',{}),('intervene',{}),('intervene',{'evidence_intervention':'pixels','pixel_budget':4}),('rescale',{}),('transform',{}),('neighborhood',{}),('cases',{})]:
        response=client.post(f'/api/run/{rid}/healthcare',json=dict(base,evidence_mode=mode,**extra),headers=headers);assert response.status_code==200
        tick(tmp_path);job=client.get('/api/job/'+response.json['id']).json;assert job['status']=='completed',job['error']
        e=job['data']['evidence'];assert e['mode']==mode
        assert e['baseline']['reconstruction_residual']<1e-10
        if mode=='rescale':
            for f in e['frames']:assert f['audit']['input_gradient_max_change']<1e-10 and abs(f['audit']['delta_change'])<1e-10
        if mode=='intervene':
            for curve in e['curves']:assert abs(curve['frames'][0]['delta']-e['baseline']['delta'])<1e-10
        if mode=='neighborhood':assert e['neighborhood']['conditional'] is not None
        if mode=='transform':assert e['frames'][0]['H_eq']<1e-8
        assert checkpoint.read_bytes()==before


def test_rescale_last_conv_and_mlp_outgoing_compensation():
    m,x,r=tiny();scaled=rescaled(m,1,1,4)
    np.testing.assert_allclose(m(torch.tensor(x[None])).detach(),scaled(torch.tensor(x[None])).detach(),atol=1e-12)
    torch.manual_seed(21);m=Network(validate(dict(widths=[3,2]))).double();x=np.random.default_rng(2).random((1,28,28));r=np.zeros_like(x)
    scaled=rescaled(m,0,1,.2)
    a=attributions(m,x,r,R,8);b=attributions(scaled,x,r,R,8)
    np.testing.assert_allclose(a['input_gradient'],b['input_gradient'],atol=1e-12)
    np.testing.assert_allclose(a['integrated_gradients'],b['integrated_gradients'],atol=1e-12)


def test_reuse_comparison_and_frozen_support_mass():
    from lab.evidence_jobs import run
    m,x,r=tiny();p=dict(evidence_mode='compare',ig_steps=8,support_budget=1)
    e=run(m,x,r,R,p,lambda _:None)
    a=e['baseline'];b=e['reuse_comparison']
    assert b['support']['layers'][0]['parameters']==a['support']['layers'][0]['parameters']
    assert b['support']['layers'][0]['occurrence_count']>=a['support']['layers'][0]['occurrence_count']
    assert b['field_b']['total_mass']>=a['field_b']['total_mass']
    p.update(evidence_mode='intervene',support_rule='manual',manual_ids=[0,1],support_budget=9)
    e=run(m,x,r,R,p,lambda _:None)
    assert all(len(c['support']['layers'][0]['selected_ids'])==2 for c in e['curves'])
    assert e['curves'][0]['field_b']['total_mass']==e['baseline']['field_b']['total_mass']
    assert all(c['frames'][-1]['reconstruction_residual']<1e-12 for c in e['curves'])
