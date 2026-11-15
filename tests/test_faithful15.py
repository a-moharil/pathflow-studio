"""Checks against hand calculations and tiny explicit complete-carrier enumeration."""
import math
import numpy as np
import pytest
import torch
from torch import nn
from lab.joint.tape import Tape,Measure,norm
from lab.joint.audit import measure_audit,graph_hash
from lab.joint.interpret import observables,scalar_geometry
from lab.joint.conventions import settings
from lab.joint.investigate import trace,analyze,embed_subgraph
from lab.joint import engine
from lab.core.metricgraph import undirected_graph
from test_joint10 import enumerate_paths,fixture


def diamond():
    x=torch.tensor([[2.]],dtype=torch.double);t=Tape(x,x*0)
    a=nn.Linear(1,2,bias=False).double();b=nn.Linear(2,1,bias=False).double()
    a.weight.data.copy_(torch.tensor([[2.],[3.]]));b.weight.data.copy_(torch.tensor([[4.,5.]]))
    u=t.add('a','module',[0],module=a);v=t.add('b','module',[u],module=b)
    s=t.add('merge','add',[0,v]);s=t.add('readout','readout',[s],weight=torch.ones_like(x));return t,s

@pytest.mark.parametrize('mode,expected',[('structural',290),('structural_he',145.5),('conditional',1160),('topology',3)])
def test_hand_partition(mode,expected):
    t,s=diamond();m=Measure(t,s,mode);assert math.exp(m.logQ)==pytest.approx(expected)
    paths=enumerate_paths(t,s,mode);q=np.array([a*a for a,_ in paths]);p=q/q.sum()
    assert math.exp(m.logQ)==pytest.approx(q.sum())
    assert observables(m)['H1']==pytest.approx(-(p*np.log(p)).sum(),abs=1e-12)
    a=measure_audit(m);assert a['status']=='passed';assert a['max_error']<1e-12
    # The shortcut and two learned paths have unequal depths. No padding paths.
    merge=next(v for v in a['merges'] if v['name']=='merge');assert merge['visit_mass']==pytest.approx(1)
    assert sum(merge['allocation'])==pytest.approx(1)



def test_sequential_he_cancels_only_after_normalizing():
    t,s=diamond();t.nodes[3].parents=t.nodes[3].parents[1:];t.nodes[3].edges=t.nodes[3].edges[1:]
    a=Measure(t,s,'structural');b=Measure(t,s,'structural_he')
    assert a.logQ-b.logQ==pytest.approx(math.log(2))
    for i in range(s+1):torch.testing.assert_close(a.node_mass(i),b.node_mass(i))


def test_metric_definition_and_parallel_shortest_path(monkeypatch):
    from lab.core import metricfit
    def light(data,steps):data['euclidean']={'dimensions':{}};return None,None
    monkeypatch.setattr(metricfit,'refine',light)
    S=[('x','a',.1),('y','a',.3),('a','z',.4),('x','z',.6)]
    C=[('x','a',0.),('y','a',0.),('a','z',0.),('x','z',1.)]
    # Entire incoming operation class, including both merge branches.
    r=embed_subgraph(S,C,S,classes=['a','a','merge','merge'],metric_settings={'metric_epsilon':.1,'metric_power':2})
    e=r['shared']['edges'];assert [v['enrichment_s'] for v in e]==pytest.approx([.5,1.5,.8,1.2])
    assert e[0]['length_s']==pytest.approx(1/.6**2)
    assert e[0]['enrichment_c'] is None;assert e[0]['length_c']==pytest.approx(100)
    assert e[2]['enrichment_c']==0;assert e[3]['enrichment_c']==2
    from scipy.sparse.csgraph import dijkstra
    A=undirected_graph(3,[0,0,1],[1,1,2],[10,2,3]);D=dijkstra(A,directed=False)
    assert D[0,1]==2;assert D[0,2]==5


def test_full_scalar_enumeration_and_class_means(monkeypatch):
    from lab.core import metricfit
    def light(data,steps):data['euclidean']={'dimensions':{}};return None,None
    monkeypatch.setattr(metricfit,'refine',light)
    t,s=diamond();st=Measure(t,s,'structural_he');cm=Measure(t,s)
    g=scalar_geometry(t,st,cm,metric_settings={'structural_rule':'he_normalized'})
    assert g['status']=='available';assert g['scalar_audit']['selected_readout_edges']==7
    edges=g['shared']['edges'];groups=g['shared']['metric']['classes']
    for cl in groups:
        ids=[e for e in edges if e['comparison_class']==cl['id']]
        for side in ['s','c']:assert np.mean([e['enrichment_'+side] for e in ids])==pytest.approx(1.)
    assert 'profile' in g['responses']['structural']


@pytest.mark.parametrize('architecture',['residual_small','healthcare'])
def test_intervention_fixed_carrier_and_structural_measure(architecture):
    if architecture=='healthcare':
        from lab.joint.model import JointModel
        model=JointModel(width=16,tiny=True).double().eval();size=16
    else:
        model=engine.build(engine.config(dict(architecture='residual_small',task='classification',size=8,classes=2))).double().eval();size=8
    torch.manual_seed(5);x=torch.rand(1,3,size,size,dtype=torch.double);r=torch.rand_like(x)
    t,h,c=trace(model,x,r,{});name=next(iter(c));u,h1,c1=trace(model,x,r,{name:(0,0.)})
    spec=dict(head='classification',a=1,b=0);s=t.readout(h,spec);z=u.readout(h1,spec)
    assert graph_hash(t)==graph_hash(u);assert len(t.nodes)==len(u.nodes)
    for mode in ['structural','structural_he']:
        a=Measure(t,s,mode);b=Measure(u,z,mode)
        assert a.logQ==pytest.approx(b.logQ)
        for i in range(s+1):torch.testing.assert_close(a.node_mass(i),b.node_mass(i))
    m=Measure(u,z);assert m.signed_reconstruction()[1]<1e-9;assert measure_audit(m)['status']=='passed'


def test_conditional_independent_of_structural_rule_and_reversal():
    model=engine.build(engine.config(dict(architecture='residual_small',task='classification',size=8,classes=2))).double().train()
    torch.manual_seed(2);x=torch.rand(1,3,8,8,dtype=torch.double);r=torch.rand_like(x);args=(dict(head='classification',a=1,b=0),'stem.relu',0)
    a=analyze(model,x,r,*args,steps=4);b=analyze(model,x,r,*args,steps=4,analysis_settings={'structural_rule':'he_normalized'})
    c=analyze(model,r,x,*args,steps=4)
    assert model.training;assert a['audit']['conditional']['status']=='passed'
    np.testing.assert_allclose(a['FieldA'],b['FieldA']);np.testing.assert_allclose(a['FieldA'],c['FieldA'],rtol=1e-10,atol=1e-12)
    assert a['delta']==pytest.approx(-c['delta'])
    assert a['structural_logQ']!=b['structural_logQ']


def test_undefined_and_nonfinite_rejected():
    t,s=diamond();t.nodes[0].ref=t.nodes[0].x.clone();m=Measure(t,s)
    assert measure_audit(m)['status']=='undefined_measure'
    for x in [torch.tensor([float('nan')]),torch.tensor([1.,-1.])]:
        with pytest.raises(FloatingPointError):norm(x)
    with pytest.raises(ValueError):settings({'metric_epsilon':float('nan')})


def test_grouped_conv_and_optional_merge():
    torch.manual_seed(6);x=torch.rand(1,2,1,1,dtype=torch.double);t=Tape(x,x*.2)
    conv=nn.Conv2d(2,2,1,groups=2,bias=False).double();u=t.add('depthwise','module',[0],module=conv)
    v=t.add('optional_merge','add',[0,u]);z=t.add('final_merge','add',[0,v]);s=t.add('readout','readout',[z],weight=torch.ones_like(x))
    for mode in ['conditional','structural_he']:
        m=Measure(t,s,mode);expected=sum(a*a for a,_ in enumerate_paths(t,s,mode));assert math.exp(m.logQ)==pytest.approx(expected)
        audit=measure_audit(m);optional=next(v for v in audit['merges'] if v['name']=='optional_merge')
        assert optional['visit_mass']<1;assert sum(optional['allocation'])==pytest.approx(1)


def test_positive_relu_rescaling_preserves_complete_route_masses():
    def make(scale):
        x=torch.tensor([[2.]],dtype=torch.double);t=Tape(x,torch.tensor([[-1.]],dtype=torch.double))
        a=nn.Linear(1,2,bias=True).double();b=nn.Linear(2,1,bias=False).double()
        a.weight.data.copy_(torch.tensor([[2.*scale],[3.]],dtype=torch.double));a.bias.data.copy_(torch.tensor([.4*scale,.2],dtype=torch.double));b.weight.data.copy_(torch.tensor([[4./scale,-5.]],dtype=torch.double))
        u=t.add('a','module',[0],module=a);u=t.add('relu','relu',[u]);u=t.add('b','module',[u],module=b)
        u=t.add('merge','add',[0,u]);return t,t.add('readout','readout',[u],weight=torch.ones_like(x))
    t,s=make(1);r,z=make(9)
    assert float(t.nodes[s].x)==pytest.approx(float(r.nodes[z].x))
    for mode in ['conditional','structural','structural_he']:
        a=Measure(t,s,mode);b=Measure(r,z,mode)
        assert a.logQ==pytest.approx(b.logQ,abs=1e-12)
        for i in range(s+1):torch.testing.assert_close(a.node_mass(i),b.node_mass(i),atol=1e-12,rtol=1e-12)
