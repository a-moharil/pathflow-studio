import sys,os,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from lab import create_app,storage as st
from lab.joint import engine,data,jobs
root=Path(os.environ.get('JOINT_SMOKE_ROOT','/tmp/joint10-smoke'));root.mkdir(exist_ok=True);app=create_app(root);data.synthetic(root)
c=dict(name='Integration smoke',dataset='synthetic',size=64,width=16,seed=17,batch=2,save_every=1,evaluation_count=3)
rid=engine.create(root,c);print('RID',rid,flush=True)
with st.db(root) as db:db.execute("UPDATE joint_runs SET request='start',target=1,status='training' WHERE id=?",(rid,))
torch.set_num_threads(2);start=time.time();engine.tick(root);r=engine.get(root,rid);print('TRAIN',time.time()-start,r['status'],r['error'],flush=True)
if r['status']=='failed':raise RuntimeError(r['error'])
p=dict(kind='joint_analysis',checkpoint='1',x=0,reference=1,reference_type='zero',class_a=2,class_b=0,region=[15,15,48,48],layer='stem.relu',channel=0,alpha=.5,ig_steps=4,memory_gib=4,geometry=True)
jid=st.job(root,p,rid)
with st.db(root) as db:j=dict(db.execute('SELECT * FROM jobs WHERE id=?',(jid,)).fetchone())
start=time.time();jobs.process(root,j);print('ANALYSIS',time.time()-start,flush=True)
(root/'smoke.json').write_text(json.dumps(dict(rid=rid,jid=jid)));d=json.loads((root/'joint'/rid/(jid+'.json')).read_text());print({k:v['reconstruction_residual'] for k,v in d['heads'].items()},flush=True)
print('JSON size', (root/'joint'/rid/(jid+'.json')).stat().st_size,flush=True)
