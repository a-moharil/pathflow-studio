"""Additive healthcare joint-task pages; existing routes remain unchanged."""
import json,math
from pathlib import Path
from flask import Blueprint,current_app,jsonify,render_template,request
from .. import storage as st
from . import engine,data
web=Blueprint('joint',__name__)
def root():return current_app.config['ROOT']

@web.get('/joint')
def index():return render_template('joint.html')
@web.get('/joint/<rid>')
def workspace(rid):
    engine.get(root(),rid);return render_template('joint_run.html',rid=rid)
@web.get('/api/joint')
def registry():
    with st.db(root()) as db:ids=[r[0] for r in db.execute('SELECT id FROM joint_runs ORDER BY created DESC')]
    datasets=[]
    for p in (Path(root())/'joint_data').glob('*/manifest.json'):
        d=json.loads(p.read_text());datasets.append(dict(key=p.parent.name,name=d['name'],samples=len(d['rows']),patients=len({r['patient'] for r in d['rows']}),citation=d.get('citation'),label_rule=d.get('label_rule')))
    prepared={d['key']:d for d in datasets}
    catalog=[]
    for key,name in [('bus-uclm-v1','BUS-UCLM v1 · ultrasound, masks and classes'),('synthetic','Synthetic joint-task engineering check')]:
        catalog.append(dict(key=key,name=name,ready=key in prepared,**{k:v for k,v in prepared.get(key,{}).items() if k not in ['key','name']}))
    catalog.extend(dict(d,ready=True) for d in datasets if d['key'] not in ['bus-uclm-v1','synthetic'])
    runs=[engine.get(root(),r) for r in ids]
    with st.db(root()) as db:
        for r in runs:
            r['analysis_count']=db.execute("SELECT count(*) FROM jobs WHERE run=? AND status='completed'",(r['id'],)).fetchone()[0]
            r['pending_measurements']=db.execute("SELECT count(*) FROM jobs WHERE run=? AND status IN ('queued','running')",(r['id'],)).fetchone()[0]
    return jsonify(runs=runs,datasets=datasets,dataset_catalog=catalog)
@web.post('/api/joint/data')
def prepare():
    d=request.get_json();action=d.get('action','download')
    if action not in ['download','import','manifest','synthetic']:raise ValueError('Unknown dataset action')
    return jsonify(job=st.job(root(),dict(kind='joint_data',action=action,path=d.get('path',''))))
@web.post('/api/joint/create')
def create():return jsonify(id=engine.create(root(),request.get_json()))
@web.get('/api/joint/<rid>')
def run(rid):
    r=engine.get(root(),rid);p=engine.folder(root(),rid);r['history']=json.loads((p/'history.json').read_text());d=json.loads((p/'data.json').read_text());r['dataset']={k:v for k,v in d.items() if k!='rows'}
    r['cases']=[{k:v for k,v in a.items() if k not in ['image','mask','sha256']} for a in d['rows'] if a['split']=='test'];r['checkpoints']=sorted([int(f.stem) for f in p.glob('*.pt') if f.stem.isdigit()]);r['checkpoints']=['latest']+r['checkpoints']
    with st.db(root()) as db:r['jobs']=[dict(v) for v in db.execute('SELECT id,status,progress,error FROM jobs WHERE run=? ORDER BY created DESC LIMIT 20',(rid,))]
    return jsonify(r)
@web.post('/api/joint/<rid>/control')
def control(rid):
    r=engine.get(root(),rid);d=request.get_json();action=d.get('action')
    if action not in ['start','pause','stop','delete']:raise ValueError('Invalid run control')
    updates=int(d.get('updates',1))
    if not 1<=updates<=100000:raise ValueError('Updates must be 1..100000')
    with st.db(root()) as db:
        if action=='start':db.execute("UPDATE joint_runs SET target=step+?,request='start',status='training',error='' WHERE id=?",(updates,rid))
        else:db.execute('UPDATE joint_runs SET request=? WHERE id=?',(action,rid))
    return jsonify(ok=True)
@web.post('/api/joint/<rid>/analyze')
def analyze(rid):
    r=engine.get(root(),rid);d=request.get_json();step=str(d.get('checkpoint','latest'))
    if not (step in ['latest','best'] or step.isdigit()) or not (engine.folder(root(),rid)/(step+'.pt')).exists():raise ValueError('Checkpoint unavailable')
    # Freeze latest to a historical checkpoint identity before queueing, avoiding
    # an analysis that silently switches checkpoint while waiting for the worker.
    if step in ['latest','best']:
        import torch
        ck=torch.load(engine.folder(root(),rid)/(step+'.pt'),map_location='cpu',weights_only=False);step=str(ck['step']);path=engine.folder(root(),rid)/(step+'.pt')
        if not path.exists():engine.atomic_save(path,ck)
    d.update(kind='joint_analysis',checkpoint=step)
    steps=int(d.get('ig_steps',16));channel=int(d.get('channel',0));alpha=float(d.get('alpha',1));budget=float(d.get('memory_gib',4))
    if not 4<=steps<=128 or channel<0 or not 0<=alpha<=1 or not 1<=budget<=64:raise ValueError('Invalid analysis settings')
    if d.get('reference_type','dataset') not in ['dataset','zero']:raise ValueError('Invalid reference')
    d.update(ig_steps=steps,channel=channel,alpha=alpha,memory_gib=budget)
    return jsonify(job=st.job(root(),d,rid))
