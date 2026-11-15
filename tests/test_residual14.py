import math,json
import numpy as np
import pytest
import torch
from lab.joint import engine,data
from lab.joint.residual import ResidualModel
from lab.joint.investigate import trace,analyze
from lab.joint.tape import Measure,Tape
from lab.joint.interpret import observables,region_analysis,source_summary,occurrences
from lab.config import validate
from lab.core.network import Network
from lab.core.pathmath import analyze as legacy_analyze
from test_joint10 import fixture,enumerate_paths

def test_entropy_and_region_against_enumeration():
    t,s=fixture();m=Measure(t,s);paths=enumerate_paths(t,s,'conditional');q=np.array([v*v for v,_ in paths]);p=q/q.sum();p=p[p>0];o=observables(m)
    assert o['H1']==pytest.approx(-(p*np.log(p)).sum(),abs=1e-11)
    assert math.exp(o['log_eta2']+o['log_path_count'])==pytest.approx(1/(p*p).sum(),rel=1e-11)
    r=region_analysis(m,[0,0,1,1]);expected=sum(v*v for v,origin in paths if origin==0)/q.sum();assert r['source_mass']==pytest.approx(expected)
    assert r['operations'][-1]['absolute_mass']==pytest.approx(expected)
    assert r['operations'][-1]['allocation']==pytest.approx(1.)

@pytest.mark.parametrize('activation',['relu','tanh'])
@pytest.mark.parametrize('branches',[2,3])
def test_graph_builder_endpoint_and_measure(activation,branches):
    torch.manual_seed(8);c=engine.config(dict(architecture='residual_custom',task='classification',size=8,classes=2,stage_widths='2,3',stage_depths='1,2',branches=branches,activation=activation,preactivation=True))
    m=engine.build(c).double().eval();x=torch.rand(1,3,8,8,dtype=torch.double);r=torch.rand_like(x);t,heads,cap=trace(m,x,r,{})
    torch.testing.assert_close(t.nodes[heads['classification']].x,m(x)['classification'])
    sink=t.readout(heads,dict(head='classification',a=1,b=0));me=Measure(t,sink);signed,err=me.signed_reconstruction()
    assert err<1e-10;assert signed==pytest.approx(float((m(x)['classification']-m(r)['classification'])[0,1]-(m(x)['classification']-m(r)['classification'])[0,0]),abs=1e-10)
    assert float(me.node_mass(0).sum())==pytest.approx(1.)
    for i,n in enumerate(t.nodes):
        if len(n.parents)>1:assert sum(me.branch_mass(i,k) for k in range(len(n.parents)))==pytest.approx(float(me.node_mass(i).sum()),abs=1e-10)
    # Intervention uses exactly the same traced forward program.
    name=next(iter(cap));ta,ha,ca=trace(m,x,r,{name:(0,0.)});torch.testing.assert_close(ta.nodes[ha['classification']].x,m(x,gates={name:(0,0.)})['classification'])

def test_scalar_budget_preserves_zero_weight_edges():
    c=engine.config(dict(architecture='residual_small',task='classification',size=8,classes=2));m=engine.build(c).double().eval();x=torch.rand(1,3,8,8,dtype=torch.double)
    t,h,_=trace(m,x,x*0,{});t.readout(h,dict(head='classification',a=1,b=0));rows,meta=occurrences(t);assert meta['status']=='available';assert meta['nodes']>1000;assert len(rows)>5000
    empty,status=occurrences(t,limit=10);assert empty is None;assert status['status']=='budget_exceeded'

@pytest.mark.parametrize('head',[False,True])
def test_cnn_head_and_regression(head):
    c=validate(dict(dataset='synthetic-regression',task='regression',model='cnn',channels=[2,3],cnn_head=head,head_widths=[4,3],geometry_every=0));m=Network(c)
    assert len(m.layers)==(5 if head else 3)
    x=np.random.default_rng(7).random((1,28,28));r=x*.2
    a=legacy_analyze(m,x,r,dict(a=0,b=None),geometry_enabled=False);assert a['residual']<1e-10
    assert a['source_diagnostics']['signed_sum']==pytest.approx(a['delta'],abs=1e-10)


def test_regression_dataset_and_training(tmp_path):
    from lab import create_app
    create_app(tmp_path);key,d=data.prepare_classification(tmp_path,'synthetic-regression',42,32)
    assert d['task']=='regression';assert any(r['label']!=int(r['label']) for r in d['rows'])
    rid=engine.create(tmp_path,dict(architecture='residual_small',task='regression',dataset=key,size=8,batch=2,save_every=1,evaluation_count=2))
    from lab import storage as st
    with st.db(tmp_path) as db:db.execute("UPDATE joint_runs SET target=1,request='start' WHERE id=?",(rid,))
    engine.quantum(tmp_path,engine.get(tmp_path,rid));r=engine.get(tmp_path,rid);assert r['step']==1
    history=json.loads((engine.folder(tmp_path,rid)/'history.json').read_text());assert 'mse' in history[0]['val'];assert 'accuracy' not in history[0]['val']


def test_full_attribution_zero_and_region():
    c=engine.config(dict(architecture='residual_small',task='classification',classes=2,size=8));m=engine.build(c).double().eval();x=torch.rand(1,3,8,8,dtype=torch.double);r=torch.zeros_like(x)
    result=analyze(m,x,r,dict(head='classification',a=1,b=0),'stem.relu',0,steps=4,source_region=[0,0,4,4])
    assert result['source_summary']['total_mass']==pytest.approx(1)
    assert np.sum(result['signed_contribution'])==pytest.approx(result['delta'],abs=1e-10)
    json.dumps(result,allow_nan=False)
    z=analyze(m,x,x,dict(head='classification',a=1,b=0),'stem.relu',0,steps=4)
    assert z['FieldA'] is None;assert z['path_observables']['status']=='undefined_measure'

def test_general_resnet50_traced_program():
    c=engine.config(dict(architecture='resnet50',task='classification',classes=2,size=32,hidden_head=False))
    m=engine.build(c).double().eval();x=torch.rand(1,3,32,32,dtype=torch.double);r=x*.8
    t,h,cap=trace(m,x,r,{})
    torch.testing.assert_close(t.nodes[h['classification']].x,m(x)['classification'])
    sink=t.readout(h,dict(head='classification',a=1,b=0));me=Measure(t,sink);signed,error=me.signed_reconstruction()
    assert error<1e-8;assert signed==pytest.approx(float(t.nodes[sink].x-t.nodes[sink].ref),abs=1e-8)
    assert float(me.node_mass(0).sum())==pytest.approx(1,abs=1e-10)
    assert len(cap)>40


def test_shared_module_execution_identity_and_avg_pool():
    from torch import nn,fx
    from lab.joint.residual import Tracer,Program
    class Tied(nn.Module):
        def __init__(self):
            super().__init__();self.conv=nn.Conv2d(1,1,1,bias=False);self.pool=nn.AvgPool2d(2);self.act=nn.ReLU()
        def forward(self,x):return self.pool(self.act(self.conv(x))+self.act(self.conv(x)))
    model=Tied().double().eval();g=fx.GraphModule(model,Tracer().trace(model));x=torch.rand(1,1,4,4,dtype=torch.double);r=torch.rand_like(x);t=Tape(x,r);end=Program(g,t).run(0)
    torch.testing.assert_close(t.nodes[end].x,model(x));assert len({n.name for n in t.nodes})==len(t.nodes)
    identities=[e.kw['parameter_identity'] for n in t.nodes for e in n.edges if 'parameter_identity' in e.kw];assert len(identities)==2 and len(set(identities))==1
    end=t.add('readout','readout',[end],weight=torch.ones_like(t.nodes[end].x));m=Measure(t,end);v,err=m.signed_reconstruction();assert err<1e-12
