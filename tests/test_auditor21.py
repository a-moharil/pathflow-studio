"""Evidence fidelity, inference-unit guards and real native/DAG report integration."""
import copy,json,hashlib
import numpy as np
import pytest
import torch
from scipy import stats
from lab import create_app,storage
from lab.auditor.statistics import holm,population
from lab.auditor.geometry import window_audit,remeasure
from lab.auditor.evidence import report,catalogue
from lab.core.network import Network
from lab.core.pathmath import analyze
from lab.core.healthcare import case_analysis
from lab.config import validate

torch.set_num_threads(2)

def save(root,data,config=None,kind='analysis',request=None):
    config=config or validate(dict(widths=[2],activation='tanh',geometry_every=0))
    rid=storage.create(root,config);jid=storage.job(root,dict(kind=kind,step=0,reference_type='zero',**(request or {})),rid)
    path=root/rid/(jid+'.json');storage.atomic_json(path,data)
    with storage.db(root) as db:db.execute("UPDATE jobs SET status='completed',result=? WHERE id=?",(str(path),jid))
    return rid,jid,path

@pytest.fixture
def real_case():
    torch.manual_seed(12);config=validate(dict(widths=[2],activation='tanh',geometry_every=0));m=Network(config)
    x=np.random.default_rng(3).random((1,28,28));d=case_analysis(m,x,x*0,dict(scope='selected',a=1,b=0));return config,d

def test_real_native_audit_and_undefined(tmp_path,real_case):
    create_app(tmp_path);config,d=real_case;rid,jid,path=save(tmp_path,dict(d,step=0),config)
    before=path.read_bytes();r=report(tmp_path,jid+':analysis:native')
    assert r['structural']['Sdef']==pytest.approx(d['structural']['Sdef'])
    assert r['conditional']['log_eta1']==pytest.approx(d['conditional']['log_eta1'])
    assert r['fidelity']['status']=='passed_recorded_checks'
    assert r['layers'][0]['unit']=='unit'
    assert all('path histogram' not in v['name'] for v in r['distributions'])
    assert r['provenance']['sha256']==hashlib.sha256(before).hexdigest()
    assert path.read_bytes()==before
    z=copy.deepcopy(d);z['conditional'].update(defined=False,eta1=None,eta2=None,H1=None,Sdef=None,log_eta1=None,log_eta2=None)
    z['input_fields']['conditional']=None
    _,j,_=save(tmp_path,z,config);out=report(tmp_path,j+':analysis:native');assert out['conditional']['eta1'] is None
    json.dumps(out,allow_nan=False)

def test_healthcare_dose_adapter(tmp_path,real_case):
    create_app(tmp_path);config,d=real_case
    payload=dict(case=d,context={'checkpoint':0},dose=[dict(alpha=1,contrast_change=0),dict(alpha=0,contrast_change=-d['delta'])],intervention={'mode':'internal'},controls=[])
    _,j,_=save(tmp_path,payload,config,'healthcare_dose');r=report(tmp_path,j+':case:native')
    assert r['intervention']['support']['mode']=='internal'
    assert 'Retained' in r['intervention']['dose_parameter']

def test_real_residual_report_and_null_intervention(tmp_path):
    from lab.joint import engine
    from lab.joint.investigate import analyze as dag_analyze
    create_app(tmp_path);torch.manual_seed(8)
    config=engine.config(dict(architecture='residual_small',task='classification',size=8,classes=2))
    model=engine.build(config).double().eval();x=torch.rand(1,3,8,8,dtype=torch.double)
    d=dag_analyze(model,x,x*0,dict(head='classification',a=1,b=0),'stem.relu',0,steps=4,geometry=False)
    data=dict(heads={'classification':d},checkpoint=0,intervention=None)
    _,j,_=save(tmp_path,data,config,'joint_analysis');r=report(tmp_path,j+':baseline:classification')
    assert r['conditional']['H1']==pytest.approx(d['path_observables']['H1'])
    assert r['fidelity']['status']=='passed_recorded_checks'
    assert not r['geometry_available']
    assert len(catalogue(tmp_path)['records'])==1

def test_reject_pseudoreplication_and_mixed_context(tmp_path,real_case):
    create_app(tmp_path);config,d=real_case;_,j,_=save(tmp_path,d,config);r=report(tmp_path,j+':analysis:native')
    with pytest.raises(ValueError,match='one recorded'):population([r,r],True)
    other=copy.deepcopy(r);other['run_id']='different';other['comparison_signature']='different'
    with pytest.raises(ValueError,match='different checkpoints'):population([r,other],True)
    other['comparison_signature']=r['comparison_signature'];r['parameter_hash']=other['parameter_hash']='same'
    with pytest.raises(ValueError,match='duplicated'):population([r,other],True)

def test_statistical_values_against_scipy(tmp_path,real_case):
    create_app(tmp_path);config,d=real_case;reports=[]
    # Known independent-unit differences exercise the statistical engine separately
    # from the actual-network numerical adapter test above.
    differences=np.array([-1.,-2.,-3.,-4.,-5.,2.,-6.,-7.])
    for i,diff in enumerate(differences):
        _,j,_=save(tmp_path,d,config);r=report(tmp_path,j+':analysis:native');r['conditional']['log_eta1']=r['structural']['log_eta1']+diff;reports.append(r)
    desc=population(reports);assert all(x['p'] is None for x in desc['results'])
    p=population(reports,True);a=p['results'][0]
    assert a['summary']['median']==np.median(differences)
    assert a['p']==pytest.approx(stats.wilcoxon(differences).pvalue)
    assert a['interval'][0]<=a['summary']['median']<=a['interval'][1]
    assert p['results'][2]['p'] is None
    assert holm([.03,.01,.04])==pytest.approx([.06,.03,.06])
    bad=copy.deepcopy(reports);bad[0]['fidelity']['status']='failed';z=population(bad,True)
    assert z['results'][0]['summary']['n']==7 and len(z['results'][0]['excluded'])==1

def test_scaling_accepts_power_law_rejects_flat_and_curved():
    x=np.geomspace(1,100,40);counts=1e5/x
    a=window_audit(x,counts);assert a['status']=='credible_window';assert a['exponent']==pytest.approx(1)
    flat=window_audit(x,np.full(40,10));assert flat['exponent'] is None
    short=window_audit(np.geomspace(1,2,40),counts);assert short['exponent'] is None

def test_geometry_on_exact_diamond_and_scope_guards():
    nodes=[dict(layer=i) for i in [0,1,1,2]]
    edges=[dict(u=a,v=b,s=p,c=q) for a,b,p,q in [(0,1,.2,.8),(0,2,.8,.2),(1,3,.2,.8),(2,3,.8,.2)]]
    graph=dict(nodes=nodes,edges=edges,scope='full_scalar',metric={'epsilon':.03,'power':1},defined={'structural':True,'conditional':True})
    result=remeasure(graph);s=result['results']['structural'];c=result['results']['conditional']
    assert len(s['scales'])==64
    assert [v['boxes'] for v in s['scales']]==[v['boxes'] for v in c['scales']]
    assert all(0<=v['retention']<=1 for v in s['scales'])
    assert s['scaling']['exponent'] is None
    with pytest.raises(ValueError,match='quotients'):remeasure(dict(graph,scope='scoped_quotient'))
    with pytest.raises(ValueError,match='limit'):remeasure(graph,budget=2)
    graph['defined']['conditional']=False;assert remeasure(graph)['results']['conditional']['status']=='undefined_measure'

def test_routes_csrf_hashes_and_export(tmp_path,real_case):
    app=create_app(tmp_path);client=app.test_client();assert client.get('/auditor').status_code==200
    config,d=real_case;_,j,path=save(tmp_path,d,config);rid=j+':analysis:native';before=path.read_bytes()
    assert client.get('/api/auditor/catalogue').json['records'][0]['id']==rid
    assert client.get('/api/auditor/report',query_string={'record':rid,'download':'1'}).status_code==200
    assert client.post('/api/auditor/population',json={'records':[rid]}).status_code==403
    with client.session_transaction() as s:token=s['csrf']
    headers={'X-CSRF-Token':token}
    response=client.post('/api/auditor/population',json={'records':[rid]},headers=headers)
    assert response.status_code==200 and response.json['results'][0]['inference_status']=='descriptive'
    assert client.post('/api/auditor/geometry',json={'record':rid},headers=headers).status_code==400
    assert path.read_bytes()==before
    assert client.get('/api/auditor/report?record=invalid').status_code==400
