"""Optimizer updates and checkpoint commit protocol, separate from HTTP requests."""
from pathlib import Path
import json
import numpy as np
import torch
from . import storage as st
from .datasets import load,status
from .core.network import Network
from .core.optimizers import build,specification
from .core.pathmath import analyze

def checkpoint(root,rid,step):return Path(root)/rid/f'checkpoint_{step:08d}.pt'

def restore(root,r,step=None):
    torch.manual_seed(r['config']['seed']);m=Network(r['config'])
    with st.db(root) as db:rows=db.execute('SELECT step FROM measurements WHERE run=? ORDER BY step DESC',(r['id'],)).fetchall()
    if step is None:step=rows[0][0] if rows else None
    elif step not in [row[0] for row in rows]:raise ValueError('Checkpoint has not been committed')
    ck=torch.load(checkpoint(root,r['id'],step),map_location='cpu',weights_only=True) if step is not None else None
    if ck:m.load_state_dict(ck['model'])
    return m,ck

def evaluate(m,data,test=True):
    device=next(m.parameters()).device;m.eval();out={}
    with torch.no_grad():
        for split in ['train','val']+(['test'] if test else []):
            n=len(data[split+'_y']);total=0.;correct=0
            for i in range(0,n,256):
                x=torch.from_numpy(data[split+'_x'][i:i+256]).to(device);y=torch.from_numpy(data[split+'_y'][i:i+256]).to(device);z=m(x)
                if m.config.get('task')=='regression':total+=float(torch.nn.functional.mse_loss(z[:,0],y,reduction='sum'))
                else:total+=float(torch.nn.functional.cross_entropy(z,y,reduction='sum'));correct+=int((z.argmax(1)==y).sum())
            out[split+'_loss']=total/n;out[split+'_accuracy']=None if m.config.get('task')=='regression' else correct/n
    return out

def commit(root,r,m,opt,rng,step,data,finalize=True):
    rid=r['id'];c=r['config']
    with st.db(root) as db:db.execute("UPDATE runs SET status='saving',live_step=? WHERE id=?",(step,rid))
    test=step==0 or step==c['updates'] or step%(c['checkpoint_every']*c['test_every'])==0
    scores=evaluate(m,data,test)
    # Analysis uses a copy: optimizer-linked parameter dtype/device are never mutated.
    probe_model=Network(c);probe_model.load_state_dict({k:v.detach().cpu() for k,v in m.state_dict().items()})
    probe=analyze(probe_model,data['test_x'][0],data['test_x'][1],dict(a=0,b=None if c.get('task')=='regression' else 1),geometry_enabled=False)
    p=checkpoint(root,rid,step);tmp=p.with_suffix('.tmp')
    with st.db(root) as db:already_committed=db.execute('SELECT 1 FROM measurements WHERE run=? AND step=?',(rid,step)).fetchone()
    if not already_committed:
        torch.save(dict(model={k:v.detach().cpu() for k,v in m.state_dict().items()},optimizer=opt.state_dict(),generator=rng.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if next(m.parameters()).is_cuda else None,step=step,optimizer_spec=specification(c)),tmp);tmp.replace(p)
    scores.update(step=step,probe=probe,test_evaluated=test)
    with st.db(root) as db:
        db.execute('BEGIN IMMEDIATE')
        request=db.execute('SELECT request FROM runs WHERE id=?',(rid,)).fetchone()[0]
        next_status='delete_requested' if request=='delete' else 'paused' if request=='pause' else 'stopped' if request=='stop' else 'completed' if step>=c['updates'] else 'training'
        db.execute('INSERT OR REPLACE INTO measurements VALUES(?,?,?)',(rid,step,json.dumps(scores,allow_nan=False)))
        db.execute('UPDATE runs SET step=?,live_step=?,status=?,request=? WHERE id=?',(step,step,next_status,'delete' if request=='delete' else '',rid))
    every=c.get('geometry_every',0)
    with st.db(root) as db:previous_step=db.execute('SELECT max(step) FROM measurements WHERE run=? AND step<?',(rid,step)).fetchone()[0]
    due=bool(every and (step==0 or step==c['updates'] or step//every>(previous_step or 0)//every))
    if next_status!='delete_requested' and due:
        st.queue_measurement(root,rid,dict(kind='analysis',step=step,x=0,reference=1,readout=dict(scope='all',a=0,b=None if c.get('task')=='regression' else 1),gates={},measurement_limit=c.get('measurement_limit',2500)))
    return next_status

def quantum(root,r):
    c=r['config'];rid=r['id']
    if status(root,c)['state']!='ready':raise ValueError('Dataset not ready. Prepare it in experiment design.')
    data=load(root,c);device=torch.device('cuda' if c['device']=='auto' and torch.cuda.is_available() else 'cpu' if c['device']=='auto' else c['device'])
    if device.type=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA unavailable; select CPU in a new experiment')
    m,ck=restore(root,r);m.to(device);opt=build(m,c);rng=torch.Generator().manual_seed(c['seed']+1);step=0
    if ck:
        opt.load_state_dict(ck['optimizer']);rng.set_state(ck['generator']);torch.set_rng_state(ck['torch_rng']);step=ck['step']
        if device.type=='cuda' and ck['cuda_rng'] is not None:torch.cuda.set_rng_state_all(ck['cuda_rng'])
    else:
        if commit(root,r,m,opt,rng,0,data)!='training':return
    for _ in range(min(c['checkpoint_every'],c['updates']-step)):
        if st.run(root,rid)['request']:break
        idx=torch.randint(len(data['train_y']),(min(c['batch_size'],len(data['train_y'])),),generator=rng)
        x=torch.from_numpy(data['train_x'][idx.numpy()]).to(device);y=torch.from_numpy(data['train_y'][idx.numpy()]).to(device)
        m.train();opt.zero_grad();z=m(x);loss=torch.nn.functional.mse_loss(z[:,0],y) if c.get('task')=='regression' else torch.nn.functional.cross_entropy(z,y)
        if not torch.isfinite(loss):raise ValueError('Non-finite loss; reduce learning rate')
        loss.backward();opt.step();step+=1
        with st.db(root) as db:db.execute('UPDATE runs SET live_step=? WHERE id=?',(step,rid))
    commit(root,r,m,opt,rng,step,data)
