"""Independent numerical and integration checks for the residual DAG extension."""
import copy,json,math
import numpy as np
import pytest
import torch
from torch import nn
from lab.joint.tape import Tape,Measure,pool_routes
from lab.joint.model import JointModel
from lab.joint.investigate import analyze
from lab.joint import data,engine

def enumerate_paths(t,sink,mode):
    origin=(t.nodes[0].x-t.nodes[0].ref).flatten() if mode=='conditional' else torch.ones(t.nodes[0].x.numel(),dtype=torch.double)
    paths=[[[(float(v),i)] for i,v in enumerate(origin)]]
    for node in t.nodes[1:sink+1]:
        output=[[] for _ in range(node.x.numel())]
        for j,e in zip(node.parents,node.edges):
            for a in range(t.nodes[j].x.numel()):
                basis=torch.zeros_like(t.nodes[j].x);basis.flatten()[a]=1
                coef=(e.apply(basis,mode,1,'positive')-e.apply(basis,mode,1,'negative')).flatten()
                for b,w in enumerate(coef):
                    if w!=0:output[b]+=[(v*float(w),o) for v,o in paths[j][a]]
        paths.append(output)
    return paths[sink][0]

def fixture():
    x=torch.tensor([[[[1.,-1.],[.5,2.]]]],dtype=torch.double);r=torch.tensor([[[[-.5,2.],[1.,.1]]]],dtype=torch.double)
    t=Tape(x,r);c=nn.Conv2d(1,1,1,bias=True).double();c.weight.data.fill_(-.7);c.bias.data.fill_(.4)
    a=t.add('c','module',[0],module=c);a=t.add('relu','relu',[a]);b=t.add('skip','add',[0,a]);b=t.add('pool','module',[b],module=nn.MaxPool2d(2));b=t.add('up','resize',[b],size=(2,2));z=t.add('cat','cat',[a,b]);z=t.add('out','readout',[z],weight=torch.tensor([[[[1.,-.3],[.2,0.]],[[.7,.5],[-.1,1.]]]],dtype=torch.double));return t,z

@pytest.mark.parametrize('mode',['conditional','structural','topology'])
def test_enumerated_mass(mode):
    t,sink=fixture();m=Measure(t,sink,mode);paths=enumerate_paths(t,sink,mode);q=sum(v*v for v,_ in paths)
    assert math.exp(m.logQ)==pytest.approx(q,rel=1e-12)
    expected=np.zeros(t.nodes[0].x.numel())
    for v,o in paths:expected[o]+=v*v/q
    np.testing.assert_allclose(m.node_mass(0).flatten().numpy(),expected,rtol=1e-12,atol=1e-14)
    for i in range(1,sink+1):assert sum(m.branch_mass(i,k) for k in range(len(t.nodes[i].edges)))==pytest.approx(float(m.node_mass(i).sum()),abs=1e-12)
    if mode=='conditional':
        pred,_=m.signed_reconstruction();assert pred==pytest.approx(sum(v for v,_ in paths),abs=1e-12)
        assert pred==pytest.approx(float(t.nodes[sink].x-t.nodes[sink].ref),abs=1e-12)
        for power in [1,2]:
            p,n,s=m.signed_origin(power);pos=np.zeros(len(expected));neg=pos.copy()
            for v,o in paths:(pos if v>=0 else neg)[o]+=abs(v)**power
            np.testing.assert_allclose((p*math.exp(s)).flatten(),pos,atol=1e-12);np.testing.assert_allclose((n*math.exp(s)).flatten(),neg,atol=1e-12)

def test_pool_switching_padding_ties():
    torch.manual_seed(31);x=torch.randn(1,2,5,5,dtype=torch.double);r=torch.randn_like(x);pool=nn.MaxPool2d(3,2,1)
    t=Tape(x,r);sink=t.add('pool','module',[0],module=pool);e=t.nodes[sink].edges[0]
    torch.testing.assert_close(e.apply(x-r,power=1),pool(x)-pool(r),rtol=1e-11,atol=1e-12)
    torch.testing.assert_close(e.weight.sum(2),torch.ones_like(e.weight.sum(2)))
    a=torch.randn_like(x);b=torch.randn_like(pool(x));torch.testing.assert_close((e.apply(a)*b).sum(),(a*e.apply(b,transpose=True)).sum())
    q,_,_=pool_routes(x,x,pool);assert torch.isfinite(q).all()

def test_zero_measure():
    x=torch.ones(1,1,2,2,dtype=torch.double);t=Tape(x,x);sink=t.add('out','readout',[0],weight=x);m=Measure(t,sink)
    assert m.logQ==-math.inf;assert float(m.node_mass(0).sum())==0

def test_resnet50_forward_and_contrast():
    torch.manual_seed(12);m=JointModel(width=16).double().eval();x=torch.rand(1,3,32,32,dtype=torch.double);r=torch.rand_like(x)
    with torch.no_grad():
        out=m(x);t=Tape(x,r);heads=m(0,recorder=t)
        for head in out:torch.testing.assert_close(t.nodes[heads[head]].x,out[head],atol=1e-12,rtol=1e-12)
        norm=(x-x.new_tensor([.485,.456,.406])[None,:,None,None])/x.new_tensor([.229,.224,.225])[None,:,None,None]
        torch.testing.assert_close(out['classification'],m.classifier(m.encoder(norm)),rtol=1e-12,atol=1e-12)
        for spec in [dict(head='classification',a=2,b=0),dict(head='segmentation',region=[2,3,12,15])]:
            sink=t.readout(heads,spec);measure=Measure(t,sink);pred,err=measure.signed_reconstruction()
            assert abs(pred-float(t.nodes[sink].x-t.nodes[sink].ref))<1e-9
            assert float(measure.node_mass(0).sum())==pytest.approx(1,abs=1e-11)

def test_intervention_and_zero_analysis():
    m=JointModel(width=16,tiny=True).double().eval();x=torch.rand(1,3,32,32,dtype=torch.double);before={k:v.clone() for k,v in m.state_dict().items()}
    d=analyze(m,x,x,dict(head='classification',a=2,b=0),'stem.relu',1,steps=4,gates={'stem.relu':(1,0.)})
    assert d['FieldA'] is None;assert d['reconstruction_residual']==0;assert d['IG_sum']==0;json.dumps(d,allow_nan=False)
    for k,v in m.state_dict().items():torch.testing.assert_close(v,before[k])

def test_dataset_groups_and_content(tmp_path):
    d=data.synthetic(tmp_path);rows=data.split(d,17);groups={s:{r['patient'] for r in rows if r['split']==s} for s in ['train','val','test']}
    assert not groups['train']&groups['test'];assert not groups['train']&groups['val'];assert not groups['test']&groups['val']
    x,m,y,meta=data.sample(rows[0],128);assert x.shape==(3,128,128);assert m.shape==(1,128,128);assert set(m.unique().tolist())<={0.,1.}
    bad=copy.deepcopy(d);bad['rows'][3]['sha256']=bad['rows'][0]['sha256'];detected=False
    for seed in range(10):
        try:data.split(bad,seed)
        except ValueError:detected=True;break
    assert detected

def test_app_legacy_and_joint(tmp_path):
    from lab import create_app
    app=create_app(tmp_path);client=app.test_client();assert client.get('/').status_code==200;assert client.get('/joint',follow_redirects=True).status_code==200
    with client.session_transaction() as sess:token=sess['csrf']
    headers={'X-CSRF-Token':token};data.synthetic(tmp_path)
    r=client.post('/api/joint/create',json=dict(dataset='synthetic',size=64,width=16,seed=17),headers=headers);assert r.status_code==200
    rid=r.json['id'];response=client.get('/api/joint/'+rid);assert response.status_code==200;assert len(response.json['cases'])>0
    assert client.post('/api/joint/'+rid+'/control',json=dict(action='pause'),headers=headers).status_code==200
    assert client.get('/joint/'+rid,follow_redirects=True).status_code==200
    with engine.st.db(tmp_path) as db:assert db.execute('SELECT count(*) FROM runs').fetchone()[0]==0

def test_optimizers_and_controls(tmp_path):
    from lab import storage as st
    st.init(tmp_path);engine.init(tmp_path);data.synthetic(tmp_path)
    for name in ['Adam','SGD','RMSprop']:
        c=engine.config(dict(optimizer=name,dataset='synthetic',size=64,width=16));m=nn.Linear(2,1);o=engine.optimizer(c,m)
        y=m(torch.ones(1,2)).sum();y.backward();o.step();assert type(o).__name__==name
    rid=engine.create(tmp_path,dict(dataset='synthetic',size=64,width=16))
    with st.db(tmp_path) as db:db.execute("UPDATE joint_runs SET request='pause',status='training' WHERE id=?",(rid,))
    assert engine.tick(tmp_path,controls_only=True);assert engine.get(tmp_path,rid)['status']=='paused'
    st.job(tmp_path,dict(kind='joint_analysis'),rid)
    with st.db(tmp_path) as db:db.execute("UPDATE joint_runs SET request='delete' WHERE id=?",(rid,))
    assert engine.tick(tmp_path,controls_only=True);assert not engine.folder(tmp_path,rid).exists();assert (tmp_path/'joint_data'/'synthetic'/'manifest.json').exists()
    with st.db(tmp_path) as db:assert db.execute('SELECT count(*) FROM jobs WHERE run=?',(rid,)).fetchone()[0]==0

def test_pool_reversal_and_bn():
    t,sink=fixture();m=Measure(t,sink);rev=Tape(t.nodes[0].ref,t.nodes[0].x)
    pool=nn.MaxPool2d(2);a=rev.add('pool','module',[0],module=pool)
    fwd=Tape(t.nodes[0].x,t.nodes[0].ref);b=fwd.add('pool','module',[0],module=pool)
    torch.testing.assert_close(rev.nodes[a].edges[0].weight,fwd.nodes[b].edges[0].weight,atol=1e-12,rtol=1e-12)
    bn=nn.BatchNorm2d(1).double().eval();bn.weight.data.fill_(-2);bn.bias.data.fill_(3);bn.running_var.fill_(.2)
    z=fwd.add('bn','module',[0],module=bn);torch.testing.assert_close(fwd.nodes[z].edges[0].apply(fwd.nodes[0].x-fwd.nodes[0].ref,power=1,part='positive')-fwd.nodes[z].edges[0].apply(fwd.nodes[0].x-fwd.nodes[0].ref,power=1,part='negative'),fwd.nodes[z].x-fwd.nodes[z].ref)

def test_uclm_import_contract(tmp_path):
    from PIL import Image
    folder=tmp_path/'downloaded'/'BUS-UCLM';(folder/'images').mkdir(parents=True);(folder/'masks').mkdir()
    img=np.zeros((12,16,3),dtype='uint8');img[:]=80;mask=np.zeros_like(img);mask[3:7,4:9,0]=255
    Image.fromarray(img).save(folder/'images'/'ABCD_001.png');Image.fromarray(mask).save(folder/'masks'/'ABCD_001.png')
    d=data.prepare(tmp_path,source_dir=tmp_path/'downloaded');assert len(d['rows'])==1;assert d['rows'][0]['label']==2;assert d['rows'][0]['patient']=='abcd'
    _,m,_,meta=data.sample(d['rows'][0],32);assert m.sum()>0;assert meta['native_size']==[16,12]

def test_joint_resume_matches_uninterrupted(tmp_path,monkeypatch):
    from lab import storage as st
    st.init(tmp_path);engine.init(tmp_path);data.synthetic(tmp_path)
    def tiny(c,pretrained=False):
        torch.manual_seed(c['seed']);return JointModel(width=16,tiny=True)
    monkeypatch.setattr(engine,'build',tiny)
    c=dict(dataset='synthetic',size=64,width=16,batch=2,save_every=2,evaluation_count=1,seed=17)
    def advance(rid):
        with st.db(tmp_path) as db:db.execute("UPDATE joint_runs SET status='training',request='start',target=2 WHERE id=?",(rid,))
        engine.quantum(tmp_path,engine.get(tmp_path,rid))
    a=engine.create(tmp_path,c);advance(a);engine.CACHE.clear();advance(a);ma,ca=engine.restore(tmp_path,engine.get(tmp_path,a))
    b=engine.create(tmp_path,c);advance(b);advance(b);mb,cb=engine.restore(tmp_path,engine.get(tmp_path,b))
    assert ca['step']==cb['step']==2
    for k,v in ma.state_dict().items():torch.testing.assert_close(v,mb.state_dict()[k],rtol=0,atol=0)
    engine.CACHE.clear()
