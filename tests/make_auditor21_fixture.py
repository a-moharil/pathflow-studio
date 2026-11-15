"""Generate real computation payloads for the DOM test, outside the repository."""
import json,sys
from pathlib import Path
import numpy as np
import torch
from lab import create_app,storage
from lab.config import validate
from lab.core.network import Network
from lab.core.pathmath import analyze
from lab.auditor.evidence import catalogue,report
from lab.auditor.statistics import population
from lab.auditor.geometry import remeasure

root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=True);app=create_app(root);torch.set_num_threads(2);reports={}
for seed in range(6):
    torch.manual_seed(seed);c=validate(dict(name='Validation seed '+str(seed),widths=[2],activation='tanh',seed=seed,geometry_every=0,measurement_limit=800))
    m=Network(c);x=np.random.default_rng(32).random((1,28,28));v=analyze(m,x,x*0,dict(scope='selected',a=1,b=0),geometry_enabled=seed==0);v['step']=0
    rid=storage.create(root,c);jid=storage.job(root,dict(kind='analysis',step=0,x=0,reference_type='zero'),rid);p=root/rid/(jid+'.json');storage.atomic_json(p,v)
    with storage.db(root) as db:db.execute("UPDATE jobs SET status='completed',result=? WHERE id=?",(str(p),jid))
    key=jid+':analysis:native';reports[key]=report(root,key)
first=next(iter(reports));r=report(root,first,True);geometry=remeasure(r['_graph']);geometry['record_id']=first
payload=dict(catalogue=catalogue(root),reports=reports,geometry=geometry,descriptive=population(list(reports.values())),inference=population(list(reports.values()),True))
(root/'fixtures.json').write_text(json.dumps(payload,allow_nan=False))
with app.test_client() as client:(root/'page.html').write_text(client.get('/auditor').get_data(as_text=True))
print(root)
