import json
import itertools
import numpy as np
import pytest
import torch
from lab.config import validate
from lab.core.network import Network
from lab.core.healthcare import case_analysis,binary_metrics,bootstrap_metrics
from lab import create_app,storage as st,datasets
from lab.engine import quantum,restore
from lab.worker import tick

torch.set_num_threads(2)

@pytest.mark.parametrize('family',['mlp','cnn'])
@pytest.mark.parametrize('activation',['relu','tanh'])
def test_signed_healthcare_field_reconstructs_gated_contrast(family,activation):
    torch.manual_seed(8);m=Network(validate(dict(model=family,widths=[3,2],channels=[2,2],activation=activation)))
    rng=np.random.default_rng(40);x=rng.random((1,28,28));r=rng.random(x.shape)
    for gates in [{},{'0':{'0':0.,'1':.4}}]:
        d=case_analysis(m,x,r,dict(scope='selected',a=1,b=0),gates)
        assert d['signed_field_residual']<1e-12
        assert d['vector_residual']<1e-12
        if d['conditional']['defined']:
            assert 0<=d['path_hellinger']<=1
            assert abs(np.sum(d['input_fields']['conditional'])-1)<1e-10
            assert all(abs(sum(l['conditional_masses'])-1)<1e-10 for l in d['layer_marginals'])
        else:
            assert d['path_hellinger'] is None
            assert abs(d['delta'])<1e-12
    d=case_analysis(m,x,x,dict(a=1,b=0))
    assert d['path_hellinger'] is None and not d['conditional']['defined']
    np.testing.assert_allclose(d['signed_field'],0)


def test_complete_path_hellinger_against_enumeration():
    torch.manual_seed(11);m=Network(validate(dict(widths=[2,2],activation='relu'))).double()
    with torch.no_grad():
        for layer in m.layers:layer.weight.copy_(layer.weight.abs()+.01);layer.bias.fill_(.2)
    rng=np.random.default_rng(70);x=rng.random((1,28,28));r=rng.random(x.shape)
    d=case_analysis(m,x,r,dict(a=1,b=0));weights=[l.weight.detach().numpy() for l in m.layers];dx=(x-r).ravel();ps=[];pc=[];signed=np.zeros(784)
    for i,j,k,o in itertools.product(range(784),range(2),range(2),range(2)):
        w=weights[0][j,i]*weights[1][k,j]*weights[2][o,k]*(1 if o==1 else -1)
        ps.append(w*w);pc.append((dx[i]*w)**2);signed[i]+=dx[i]*w
    ps=np.array(ps)/sum(ps);pc=np.array(pc)/sum(pc)
    truth=np.linalg.norm(np.sqrt(ps)-np.sqrt(pc))/np.sqrt(2)
    assert abs(d['path_hellinger']-truth)<1e-12
    np.testing.assert_allclose(np.asarray(d['signed_field']).ravel(),signed,atol=1e-12)


def test_binary_metrics_ties_single_class_and_intervals():
    d=binary_metrics([0,0,1,1],[.1,.4,.35,.8])
    assert d['roc_auc']==.75
    assert abs(d['average_precision']-5/6)<1e-12
    assert (d['tn'],d['fp'],d['fn'],d['tp'])==(2,0,1,1)
    assert abs(d['brier']-.158125)<1e-12
    ties=binary_metrics([0,1,0,1],[.5]*4);assert ties['roc_auc']==.5 and ties['average_precision']==.5
    single=binary_metrics([0,0],[.1,.2]);assert single['roc_auc'] is None and single['sensitivity'] is None
    ci=bootstrap_metrics(np.array([0,0,1,1]),np.array([.1,.4,.35,.8]),.5)
    assert ci['accuracy']['valid_replicates']==200


def test_healthcare_worker_jobs_preserve_checkpoint(tmp_path):
    app=create_app(tmp_path);c=validate(dict(widths=[3,2],train_size=32,updates=1,checkpoint_every=1));datasets.prepare(tmp_path,c);rid=st.create(tmp_path,c)
    with st.db(tmp_path) as db:db.execute("UPDATE runs SET status='training' WHERE id=?",(rid,))
    quantum(tmp_path,st.run(tmp_path,rid));checkpoint=tmp_path/rid/'checkpoint_00000001.pt';before=checkpoint.read_bytes()
    client=app.test_client();client.get('/healthcare/'+rid)
    with client.session_transaction() as s:token=s['csrf']
    headers={'X-CSRF-Token':token}
    base=dict(step=1,x=0,reference_type='zero',split='test')
    for kind,extra in [('healthcare_case',{}),('healthcare_dose',{'layer':0,'selection':'top','intervention':'internal'}),('healthcare_dose',{'intervention':'region','left':0,'top':0,'right':28,'bottom':28}),('healthcare_cohort',{'cohort_size':4,'cohort_reference':'zero'})]:
        response=client.post(f'/api/run/{rid}/healthcare',json=dict(base,kind=kind,**extra),headers=headers);assert response.status_code==200
        jid=response.json['id'];tick(tmp_path);job=client.get('/api/job/'+jid).json;assert job['status']=='completed',job['error']
        d=job['data'];assert d['context']['checkpoint']==1
        if kind=='healthcare_dose':
            assert abs(d['dose'][0]['contrast_change'])<1e-12
            if extra['intervention']=='region':assert abs(d['dose'][-1]['delta'])<1e-12
            else:assert len(d['controls'])==2
        if kind=='healthcare_cohort':assert d['metrics']['n']>4 and len(d['rows'])==4
        assert checkpoint.read_bytes()==before
    assert client.post(f'/api/run/{rid}/healthcare',json=dict(base,kind='healthcare_case')).status_code==403
