import json
import numpy as np
from lab import create_app,storage as st,datasets
from lab.config import validate
from lab.engine import quantum
from lab.worker import tick

def test_region_provenance_and_transform_guard(tmp_path):
    app=create_app(tmp_path);c=validate(dict(model='mlp',widths=[3,2],activation='tanh',train_size=32,updates=1,checkpoint_every=1,geometry_every=0));datasets.prepare(tmp_path,c);rid=st.create(tmp_path,c)
    with st.db(tmp_path) as db:db.execute("UPDATE runs SET status='training' WHERE id=?",(rid,))
    quantum(tmp_path,st.run(tmp_path,rid));cp=tmp_path/rid/'checkpoint_00000001.pt';before=cp.read_bytes();client=app.test_client();html=client.get('/healthcare/'+rid).text
    assert 'hwMount' in html and 'health_workflow.js?v=0.12.0' in html
    with client.session_transaction() as s:token=s['csrf']
    base=dict(kind='healthcare_case',step=1,x=0,reference=1,reference_type='dataset',split='test')
    def submit(p):
        r=client.post('/api/run/'+rid+'/healthcare',json=p,headers={'X-CSRF-Token':token});assert r.status_code==200;tick(tmp_path);return client.get('/api/job/'+r.json['id']).json
    full=submit(base);local=submit(base|dict(source_region=[0,0,14,28]));assert local['status']=='completed',local.get('error');d=local['data'];assert d['request']['source_region']==[0,0,14,28];assert d['workflow_version']==19
    np.testing.assert_allclose(full['data']['case']['input_fields']['conditional'],d['case']['input_fields']['conditional']);assert abs(full['data']['case']['delta']-d['case']['delta'])<1e-12
    assert abs(d['case']['source_region']['source_mass']-np.sum(np.array(d['case']['input_fields']['conditional'])[:,:14]))<1e-12
    bad=submit(base|dict(source_region=[0,0,30,28]));assert bad['status']=='failed' and 'outside' in bad['error']
    bad=submit(base|dict(kind='healthcare_evidence',transform=dict(dx=1,dy=0,angle=0,scale=1)));assert bad['status']=='failed' and 'Reset the case transform' in bad['error']
    assert cp.read_bytes()==before
