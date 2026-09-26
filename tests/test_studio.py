import json
import numpy as np
import pytest
import torch
from lab import create_app,storage as st,datasets
from lab.config import validate
from lab.core.network import Network
from lab.core.optimizers import build,specification
from lab.core.pathmath import analyze,measure
from lab.engine import quantum,restore
from lab.worker import tick

torch.set_num_threads(2)

@pytest.mark.parametrize('optimizer',['Adam','SGD','RMSprop'])
@pytest.mark.parametrize('model',['mlp','cnn'])
def test_optimizer_training_resume(tmp_path,optimizer,model):
    st.init(tmp_path);c=validate(dict(optimizer=optimizer,model=model,widths=[4,4],channels=[2,2],train_size=32,updates=4,checkpoint_every=2,momentum=.8,centered=True,amsgrad=True))
    datasets.prepare(tmp_path,c);rid=st.create(tmp_path,c)
    with st.db(tmp_path) as db:db.execute("UPDATE runs SET status='training' WHERE id=?",(rid,))
    quantum(tmp_path,st.run(tmp_path,rid));m,ck=restore(tmp_path,st.run(tmp_path,rid));assert ck['step']==2;assert ck['optimizer']['state'];assert ck['optimizer_spec']['class']=='torch.optim.'+optimizer
    with st.db(tmp_path) as db:db.execute("UPDATE runs SET status='pause_requested',request='pause' WHERE id=?",(rid,))
    tick(tmp_path);assert st.run(tmp_path,rid)['status']=='paused'
    with st.db(tmp_path) as db:db.execute("UPDATE runs SET status='training' WHERE id=?",(rid,))
    quantum(tmp_path,st.run(tmp_path,rid));assert st.run(tmp_path,rid)['status']=='completed'
    r2=st.create(tmp_path,c|{'checkpoint_every':4})
    with st.db(tmp_path) as db:db.execute("UPDATE runs SET status='training' WHERE id=?",(r2,))
    quantum(tmp_path,st.run(tmp_path,r2));m1,_=restore(tmp_path,st.run(tmp_path,rid));m2,_=restore(tmp_path,st.run(tmp_path,r2))
    for a,b in zip(m1.parameters(),m2.parameters()):assert torch.equal(a,b)
    with st.db(tmp_path) as db:metrics=[json.loads(r[0]) for r in db.execute('SELECT payload FROM measurements WHERE run=?',(rid,))]
    assert {r['step'] for r in metrics}=={0,2,4}
    assert all(all(k in m for k in ['train_loss','val_loss','test_loss','train_accuracy','val_accuracy','test_accuracy']) for m in metrics)

@pytest.mark.parametrize('model',['mlp','cnn'])
@pytest.mark.parametrize('activation',['relu','tanh'])
def test_path_audit(model,activation):
    c=validate(dict(model=model,activation=activation,widths=[4,3],channels=[2,2]));torch.manual_seed(1);m=Network(c);rng=np.random.default_rng(3);x=rng.random((1,28,28),dtype=np.float32);r=rng.random((1,28,28),dtype=np.float32)
    for gate in [{},{'0':{'0':.2,'1':0}}]:
        a=analyze(m,x,r,dict(a=0,b=1),gate,False);assert a['residual']<1e-12
        assert all(abs(v-1)<1e-10 for v in a['conditional']['cut_sums'])
    z=analyze(m,x,x,dict(a=0),{},False);assert not z['conditional']['defined'];assert z['delta']==0

def test_design_readiness_and_control(tmp_path):
    app=create_app(tmp_path);client=app.test_client();client.get('/')
    with client.session_transaction() as s:token=s['csrf']
    h={'X-CSRF-Token':token};c=dict(widths=[4,4],train_size=32,updates=3,checkpoint_every=1,landmarks=16)
    assert client.post('/api/runs',json=c,headers=h).status_code==400
    checked=client.post('/api/design/check',json=c,headers=h).json;assert checked['dataset']['state']=='prepare'
    j=client.post('/api/datasets/prepare',json=c,headers=h).json['id'];tick(tmp_path)
    assert client.get('/api/job/'+j).json['data']['state']=='ready'
    rid=client.post('/api/runs',json=c,headers=h).json['id'];assert client.post(f'/api/run/{rid}/control',json={'action':'start'}).status_code==403
    client.post(f'/api/run/{rid}/control',json={'action':'start'},headers=h);tick(tmp_path)
    client.post(f'/api/run/{rid}/control',json={'action':'pause'},headers=h)
    assert st.run(tmp_path,rid)['status']=='pause_requested'
    assert client.post(f'/api/run/{rid}/control',json={'action':'start'},headers=h).status_code==400
    tick(tmp_path);r=st.run(tmp_path,rid);assert r['status']=='paused';assert r['step']==r['live_step']
    job=client.post(f'/api/run/{rid}/analysis',json=dict(kind='analysis',step=r['step'],x=0,reference=1),headers=h).json['id'];tick(tmp_path)
    result=client.get('/api/job/'+job).json;assert result['status']=='completed',result.get('error');d=result['data'];assert d['residual']<1e-12
    assert all(v>=1 for v in d['structural_geometry']['coverage_by_layer']);assert d['structural_geometry']['validation_stress'] is not None
    assert st.run(tmp_path,rid)['status']=='paused'
    alljob=client.post(f'/api/run/{rid}/analysis',json=dict(kind='analysis',step=r['step'],x=0,reference=1,readout=dict(scope='all',a=0,b=1)),headers=h).json['id'];tick(tmp_path)
    alld=client.get('/api/job/'+alljob).json
    assert alld['status']=='completed',alld.get('error')
    assert alld['data']['scope']=='all' and alld['data']['vector_residual']<1e-12
    assert len(alld['data']['edges'])==sum(len(layer[0]) for layer in __import__('lab.core.pathmath',fromlist=['graph']).graph(Network(validate(c)),dict(scope='all',a=0,b=1))[2])
    for url in ['/', '/design','/compare','/experiment/'+rid,'/observatory/'+rid]:assert client.get(url).status_code==200

def test_pause_not_published_before_commit(tmp_path,monkeypatch):
    from lab import engine
    st.init(tmp_path);c=validate(dict(widths=[4],train_size=32,updates=4,checkpoint_every=1));datasets.prepare(tmp_path,c);rid=st.create(tmp_path,c)
    with st.db(tmp_path) as db:db.execute("UPDATE runs SET status='training' WHERE id=?",(rid,))
    quantum(tmp_path,st.run(tmp_path,rid))
    with st.db(tmp_path) as db:db.execute("UPDATE runs SET status='pause_requested',request='pause' WHERE id=?",(rid,))
    original=engine.evaluate
    def checked(*a,**kw):
        assert st.run(tmp_path,rid)['status']=='saving';assert st.run(tmp_path,rid)['request']=='pause';return original(*a,**kw)
    monkeypatch.setattr(engine,'evaluate',checked);tick(tmp_path);assert st.run(tmp_path,rid)['status']=='paused'

def test_entropy_enumeration():
    sizes=[2,2,1];layers=[(np.array([0,1,0,1]),np.array([0,0,1,1]),np.array([1.,-2.,3.,.5])),(np.array([0,1]),np.array([0,0]),np.array([2.,-1.]))];start=np.array([.4,.7]);sec=[np.array([.2,.8]),np.ones(1)]
    d,F=measure(sizes,layers,start,sec);q=np.array([start[u]*(w*sec[0][v]*layers[1][2][v])**2 for u,v,w in zip(*layers[0])]);p=q/q.sum();assert d['H1']==pytest.approx(-sum(p*np.log(p)));assert d['eta2']==pytest.approx(1/(4*sum(p*p)))

def test_invalid_nesterov():
    with pytest.raises(ValueError):validate(dict(optimizer='SGD',nesterov=True,momentum=0))

@pytest.mark.parametrize('model',['mlp','cnn'])
def test_all_output_measure_and_vector_audit(model):
    from lab.core.pathmath import graph
    c=validate(dict(dataset='fashion',model=model,widths=[4,3],channels=[2,2]));torch.manual_seed(7);m=Network(c)
    rng=np.random.default_rng(9);x=rng.random((1,28,28),dtype=np.float32);r=rng.random((1,28,28),dtype=np.float32)
    all_readout=dict(scope='all',a=0,b=1)
    a=analyze(m,x,r,all_readout,{},False)
    assert a['vector_residual']<1e-12
    assert np.allclose(a['output_contrasts'],a['output_path_sums'],atol=1e-12)
    assert all(abs(v-1)<1e-10 for v in a['structural']['cut_sums'])
    sizes,offset,layers=graph(m,all_readout)
    assert len(layers[-1][0])==10
    # The all-output partition is the sum of the ten separate endpoint partitions.
    individual=[analyze(m,x,r,dict(a=j),{},False) for j in range(10)]
    for side in ['structural','conditional']:
        expected=np.logaddexp.reduce([v[side]['logZ'] for v in individual if v[side]['defined']])
        assert a[side]['logZ']==pytest.approx(expected)
    assert a['delta']==pytest.approx(individual[0]['delta']-individual[1]['delta'])
    gated=analyze(m,x,r,all_readout,{'0':{'0':0}},False)
    assert gated['vector_residual']<1e-12
    assert gated['structural']==a['structural']

def test_measurement_separate_from_projection():
    from lab.core.pathmath import geometry
    sizes=[3,4,2,1];offset=np.cumsum([0]+sizes);layers=[];F=[]
    for a,b in zip(sizes[:-1],sizes[1:]):
        v,u=np.indices((b,a));layers.append((u.ravel(),v.ravel(),np.ones(a*b)));F.append(np.full(a*b,1/(a*b)))
    _,g,_=geometry(sizes,offset,layers,F,limit=5,landmarks=4,measurement_limit=20)
    assert g['mode']=='landmark';assert len(g['responses']['profile'])==36
    assert g['responses']['status'] in ['available','no_qualified_window']
    _,g,_=geometry(sizes,offset,layers,F,limit=5,landmarks=4,measurement_limit=5)
    assert g['responses']['status']=='budget_exceeded'


def test_backfill_and_delete_lifecycle(tmp_path):
    app=create_app(tmp_path);client=app.test_client();client.get('/')
    with client.session_transaction() as s:h={'X-CSRF-Token':s['csrf']}
    c=validate(dict(widths=[2],train_size=32,updates=2,checkpoint_every=1));datasets.prepare(tmp_path,c);rid=st.create(tmp_path,c)
    with st.db(tmp_path) as db:db.execute("UPDATE runs SET status='training' WHERE id=?",(rid,))
    quantum(tmp_path,st.run(tmp_path,rid))
    payload=dict(measurement_limit=50,readout=dict(scope='all',a=0,b=1))
    a=client.post(f'/api/run/{rid}/measure-missing',json=payload,headers=h).json
    b=client.post(f'/api/run/{rid}/measure-missing',json=payload,headers=h).json
    assert a['jobs']==b['jobs'] and a['checkpoints']==2
    tick(tmp_path)
    assert client.get('/api/job/'+a['jobs'][0]).json['status']=='completed'
    assert client.get(f'/api/run/{rid}/measurement-jobs').status_code==200
    assert client.get(f'/api/run/{rid}/delete-preview').json['shared_datasets_preserved']
    client.post(f'/api/run/{rid}/delete',json={},headers=h)
    assert st.run(tmp_path,rid)['request']=='delete'
    assert client.post(f'/api/run/{rid}/analysis',json={'kind':'analysis'},headers=h).status_code==400
    tick(tmp_path)
    assert not (tmp_path/rid).exists();assert (tmp_path/'datasets').exists()
    assert client.get('/api/run/'+rid).status_code==404
    with st.db(tmp_path) as db:
        for table,key in [('jobs','run'),('measurements','run'),('observations','run')]:assert db.execute(f'SELECT count(*) FROM {table} WHERE {key}=?',(rid,)).fetchone()[0]==0


def test_delete_arriving_during_commit(tmp_path,monkeypatch):
    from lab import engine
    app=create_app(tmp_path);client=app.test_client();client.get('/')
    with client.session_transaction() as s:h={'X-CSRF-Token':s['csrf']}
    c=validate(dict(widths=[2],train_size=32,updates=2));datasets.prepare(tmp_path,c);rid=st.create(tmp_path,c)
    with st.db(tmp_path) as db:db.execute("UPDATE runs SET status='training' WHERE id=?",(rid,))
    original=engine.evaluate
    def wrapped(*a,**kw):client.post(f'/api/run/{rid}/delete',json={},headers=h);return original(*a,**kw)
    monkeypatch.setattr(engine,'evaluate',wrapped);quantum(tmp_path,st.run(tmp_path,rid))
    assert st.run(tmp_path,rid)['request']=='delete';tick(tmp_path);assert client.get('/api/run/'+rid).status_code==404


def test_geometry_schedule(tmp_path):
    st.init(tmp_path);c=validate(dict(widths=[2],train_size=32,updates=2,checkpoint_every=1,geometry_every=1,measurement_limit=50));datasets.prepare(tmp_path,c);rid=st.create(tmp_path,c)
    with st.db(tmp_path) as db:db.execute("UPDATE runs SET status='training' WHERE id=?",(rid,))
    quantum(tmp_path,st.run(tmp_path,rid))
    with st.db(tmp_path) as db:payloads=[json.loads(j[0]) for j in db.execute('SELECT payload FROM jobs WHERE run=?',(rid,))]
    assert {p['step'] for p in payloads}=={0,1};assert all(p['readout']['scope']=='all' for p in payloads)


@pytest.mark.parametrize('model',['mlp','cnn'])
@pytest.mark.parametrize('activation',['relu','tanh'])
def test_input_fields_integration(model,activation):
    from lab.core.pathmath import graph
    c=validate(dict(model=model,activation=activation,widths=[3,2],channels=[1,1]))
    torch.manual_seed(15);m=Network(c)
    rng=np.random.default_rng(8);x=rng.random((1,28,28),dtype=np.float32);r=rng.random((1,28,28),dtype=np.float32)
    readout=dict(scope='all',a=0,b=1)
    a=analyze(m,x,r,readout,{},False)
    assert a['input_fields']['shape']==[1,28,28]
    for side in ['structural','conditional']:
        if a[side]['defined']:
            field=np.array(a['input_fields'][side]);assert field.shape==(28,28)
            assert field.min()>=0 and field.sum()==pytest.approx(1)
        else:assert a['input_fields'][side] is None
    b=analyze(m,r,x,readout,{},False)
    if a['conditional']['defined']:
        np.testing.assert_allclose(a['input_fields']['conditional'],b['input_fields']['conditional'])
    else:assert b['input_fields']['conditional'] is None
    zero=analyze(m,x,x,readout,{},False)
    assert zero['input_fields']['conditional'] is None
    gated=analyze(m,x,r,readout,{'0':{'0':0}},False)
    assert gated['input_fields']['structural']==a['input_fields']['structural']
    assert gated['vector_residual']<1e-12


def test_measurement_settings_existing_run(tmp_path):
    app=create_app(tmp_path);client=app.test_client();client.get('/')
    with client.session_transaction() as s:h={'X-CSRF-Token':s['csrf']}
    c=validate(dict(widths=[2],train_size=32,updates=2));datasets.prepare(tmp_path,c);rid=st.create(tmp_path,c)
    url=f'/api/run/{rid}/measurement-settings'
    assert client.post(url,json={'geometry_every':1,'measurement_limit':3000},headers=h).status_code==200
    assert st.run(tmp_path,rid)['config']['geometry_every']==1
    with st.db(tmp_path) as db:db.execute("UPDATE runs SET status='training' WHERE id=?",(rid,))
    assert client.post(url,json={'geometry_every':0},headers=h).status_code==400
    preview=client.get(f'/api/run/{rid}/delete-preview').json
    assert set(preview['records'])=={'jobs','measurements','observations'}


def test_default_cnn_landmark_has_full_measurements():
    c=validate(dict(model='cnn',dataset='fashion',geometry_limit=500,measurement_limit=2500))
    torch.manual_seed(2);m=Network(c)
    rng=np.random.default_rng(3);x=rng.random((1,28,28),dtype=np.float32);r=rng.random((1,28,28),dtype=np.float32)
    d=analyze(m,x,r,{'scope':'all','a':0,'b':1})
    assert len(d['nodes'])==2001
    assert d['structural_geometry']['mode']=='landmark'
    response=d['structural_geometry']['responses']
    assert len(response['profile'])==36
    assert all(p['window_start']<=p['radius']<=p['window_end'] for p in response['profile'])
    assert response['status'] in ['available','no_qualified_window']


def test_scheduled_default_cnn_result(tmp_path):
    st.init(tmp_path)
    c=validate(dict(model='cnn',dataset='synthetic',train_size=32,updates=1,checkpoint_every=1,geometry_every=1,measurement_limit=2500))
    datasets.prepare(tmp_path,c);rid=st.create(tmp_path,c)
    with st.db(tmp_path) as db:db.execute("UPDATE runs SET status='training' WHERE id=?",(rid,))
    quantum(tmp_path,st.run(tmp_path,rid));tick(tmp_path)
    with st.db(tmp_path) as db:
        rows=db.execute('SELECT payload FROM observations WHERE run=?',(rid,)).fetchall()
    assert rows
    d=json.loads(rows[0][0]);assert d['scope']=='all'
    assert d['structural_geometry']['mode']=='landmark'
    assert len(d['structural_geometry']['responses']['profile'])==36
