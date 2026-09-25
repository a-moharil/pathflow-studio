import json,io,time
from pathlib import Path
import numpy as np
from PIL import Image
from flask import Blueprint,current_app,render_template,request,jsonify,abort,send_file
from . import storage as st,datasets
from .config import validate,estimate,DEFAULT
from .core.optimizers import specification
web=Blueprint('web',__name__)
def root():return current_app.config['ROOT']

@web.get('/')
def index():return render_template('index.html')
@web.get('/design')
def design():return render_template('design.html',defaults=DEFAULT)
@web.get('/experiment/<rid>')
def experiment(rid):return render_template('experiment.html',run=st.run(root(),rid))
@web.get('/observatory/<rid>')
def observatory(rid):return render_template('observatory.html',run=st.run(root(),rid))
@web.get('/compare')
def compare():return render_template('compare.html')
@web.get('/api/runs')
def runs():
    with st.db(root()) as db:rows=db.execute('SELECT id FROM runs ORDER BY created DESC').fetchall()
    return jsonify([st.run(root(),r[0]) for r in rows])
@web.post('/api/design/check')
def check():
    c=validate(request.json);return jsonify(config=c,cost=estimate(c),optimizer=specification(c),dataset=datasets.status(root(),c))
@web.post('/api/datasets/prepare')
def prepare():return jsonify(id=st.job(root(),{'kind':'prepare','config':validate(request.json)}))
@web.post('/api/runs')
def create():
    c=validate(request.json)
    if datasets.status(root(),c)['state']!='ready':raise ValueError('Prepare the selected dataset in Design before creating this experiment.')
    return jsonify(id=st.create(root(),c))
@web.get('/api/run/<rid>')
def detail(rid):
    r=st.run(root(),rid)
    with st.db(root()) as db:
        r['metrics']=[json.loads(x[0]) for x in db.execute('SELECT payload FROM measurements WHERE run=? ORDER BY step',(rid,))]
        r['jobs']=[{k:v for k,v in dict(j).items() if k!='result'} for j in db.execute('SELECT * FROM jobs WHERE run=? ORDER BY created DESC LIMIT 30',(rid,))]
    r['checkpoints']=[m['step'] for m in r['metrics']];r['optimizer']=specification(r['config']);return jsonify(r)
@web.post('/api/run/<rid>/control')
def control(rid):
    action=request.json.get('action')
    if action not in ['start','pause','stop']:raise ValueError('Unknown control')
    with st.db(root()) as db:
        db.execute('BEGIN IMMEDIATE');row=db.execute('SELECT * FROM runs WHERE id=?',(rid,)).fetchone()
        if not row:raise KeyError('Run not found')
        r=dict(row);config=json.loads(r['config'])
        if action=='start':
            if r['status'] not in ['ready','paused','stopped','failed']:raise ValueError('Wait until the current operation is committed')
            if r['step']>=config['updates']:raise ValueError('Experiment has completed its configured update count')
            if datasets.status(root(),config)['state']!='ready':raise ValueError('Dataset no longer ready')
            db.execute("UPDATE runs SET status='training',request='',error='' WHERE id=?",(rid,))
        else:
            if r['status'] in ['training','saving','pause_requested','stop_requested']:
                db.execute('UPDATE runs SET status=?,request=? WHERE id=?',(action+'_requested',action,rid))
            elif r['status'] in ['paused','stopped']:pass
            else:raise ValueError('No active training to pause or stop')
    return jsonify(ok=True)
@web.post('/api/run/<rid>/analysis')
def submit(rid):
    st.run(root(),rid);p=request.json
    if p.get('kind','analysis') not in ['analysis','ensemble','dose']:raise ValueError('Unsupported analysis')
    p.setdefault('kind','analysis');return jsonify(id=st.queue_measurement(root(),rid,p))
@web.get('/api/job/<jid>')
def job(jid):
    with st.db(root()) as db:r=db.execute('SELECT * FROM jobs WHERE id=?',(jid,)).fetchone()
    if not r:abort(404)
    r=dict(r);path=r.pop('result')
    if r['status']=='completed':r['data']=json.loads(Path(path).read_text())
    return jsonify(r)
@web.get('/api/run/<rid>/observations')
def observations(rid):
    st.run(root(),rid)
    with st.db(root()) as db:rows=db.execute('SELECT payload FROM observations WHERE run=? ORDER BY step',(rid,)).fetchall()
    return jsonify([json.loads(r[0]) for r in rows])
@web.get('/api/run/<rid>/dataset')
def data_info(rid):
    d=datasets.load(root(),st.run(root(),rid)['config'])
    return jsonify({s:{'count':len(d[s+'_y']),'labels':d[s+'_y'].tolist(),'source_indices':d[s+'_ids'].tolist()} for s in ['train','val','test']})
@web.get('/api/run/<rid>/image/<int:i>')
def image(rid,i):
    d=datasets.load(root(),st.run(root(),rid)['config'])
    if i>=len(d['test_x']):abort(404)
    im=Image.fromarray(np.uint8(d['test_x'][i,0]*255));b=io.BytesIO();im.save(b,format='PNG');b.seek(0);return send_file(b,mimetype='image/png')
@web.get('/api/run/<rid>/export')
def export(rid):
    data=detail(rid).get_json();return send_file(io.BytesIO(json.dumps(data,indent=2).encode()),download_name=rid+'.json',as_attachment=True,mimetype='application/json')
@web.get('/api/run/<rid>/checkpoint/<int:step>')
def checkpoint(rid,step):
    from .engine import checkpoint as path
    st.run(root(),rid)
    with st.db(root()) as db:ready=db.execute('SELECT step FROM measurements WHERE run=? AND step=?',(rid,step)).fetchone()
    if not ready:abort(404)
    return send_file(path(root(),rid,step),as_attachment=True)


@web.get('/api/run/<rid>/delete-preview')
def delete_preview(rid):
    r=st.run(root(),rid);folder=Path(root())/rid
    files=[p for p in folder.rglob('*') if p.is_file()] if folder.exists() else []
    return jsonify(name=r['config']['name'],status=r['status'],files=len(files),bytes=sum(p.stat().st_size for p in files),shared_datasets_preserved=True)

@web.post('/api/run/<rid>/delete')
def delete_experiment(rid):
    with st.db(root()) as db:
        db.execute('BEGIN IMMEDIATE');r=db.execute('SELECT id FROM runs WHERE id=?',(rid,)).fetchone()
        if not r:raise KeyError('Run not found')
        db.execute("UPDATE runs SET status='delete_requested',request='delete',error='' WHERE id=?",(rid,))
        db.execute("UPDATE jobs SET status='cancelled',progress='Experiment deletion requested' WHERE run=? AND status='queued'",(rid,))
    return jsonify(status='delete_requested')

@web.post('/api/run/<rid>/measure-missing')
def measure_missing(rid):
    r=st.run(root(),rid);p=request.json;lo=int(p.get('from',0));hi=int(p.get('to',r['step']));hi=r['step'] if hi<0 else hi
    budget=int(p.get('measurement_limit',r['config'].get('measurement_limit',2500)))
    if not 50<=budget<=4000:raise ValueError('Measurement limit must be 50..4000')
    with st.db(root()) as db:steps=[v[0] for v in db.execute('SELECT step FROM measurements WHERE run=? AND step BETWEEN ? AND ? ORDER BY step',(rid,lo,hi))]
    jobs=[]
    for step in steps:
        payload=dict(kind='analysis',step=step,x=int(p.get('x',0)),reference=int(p.get('reference',1)),readout=p.get('readout',dict(scope='all',a=0,b=1)),gates=p.get('gates',{}),measurement_limit=budget)
        jobs.append(st.queue_measurement(root(),rid,payload))
    return jsonify(jobs=jobs,checkpoints=len(steps))

@web.get('/api/run/<rid>/measurement-jobs')
def measurement_jobs(rid):
    st.run(root(),rid)
    with st.db(root()) as db:rows=db.execute('SELECT id,payload,status,progress,error FROM jobs WHERE run=? ORDER BY created',(rid,)).fetchall()
    return jsonify([dict(r) for r in rows])
