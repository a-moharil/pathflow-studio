"""Generate real worker results and rendered pages for offline DOM lifecycle checks."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lab import create_app,storage as st,datasets
from lab.config import validate
from lab.engine import quantum
from lab.worker import tick
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True);root=out/'state';app=create_app(root)
c=validate(dict(model='cnn',channels=[2,2],head_widths=[3],cnn_head=True,activation='tanh',train_size=32,updates=1,checkpoint_every=1,geometry_every=0));datasets.prepare(root,c);rid=st.create(root,c)
with st.db(root) as db:db.execute("UPDATE runs SET status='training' WHERE id=?",(rid,))
quantum(root,st.run(root,rid));client=app.test_client();page=client.get('/healthcare/'+rid).text
with client.session_transaction() as s:csrf=s['csrf']
base=dict(step=1,x=0,reference=1,reference_type='dataset',split='test',source_region=[0,0,14,28],transform=dict(dx=0,dy=0,angle=0,scale=1),ig_steps=8,support_budget=2,support_kind='occurrence',support_layers='0',support_rule='flow',visual_layer=0,visual_channel=0)
reports={};jobs=[]
for name,extras in [('case',dict(kind='healthcare_case')),('compare',dict(kind='healthcare_evidence',evidence_mode='compare')),('intervene',dict(kind='healthcare_evidence',evidence_mode='intervene',evidence_intervention='channel'))]:
 p=base|extras;r=client.post('/api/run/'+rid+'/healthcare',json=p,headers={'X-CSRF-Token':csrf});assert r.status_code==200,r.text;tick(root);job=client.get('/api/job/'+r.json['id']).json;assert job['status']=='completed',job['error'];reports[name]=job['data'];jobs.append(dict(id=r.json['id'],status='completed',payload=json.dumps(p)))
(out/'page.html').write_text(page);(out/'fixtures.json').write_text(json.dumps(dict(run=client.get('/api/run/'+rid).json,dataset=client.get('/api/run/'+rid+'/dataset').json,reports=reports,jobs=jobs)))
print(out)
