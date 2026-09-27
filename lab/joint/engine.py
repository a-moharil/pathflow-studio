"""Joint-task worker integration; independent namespace preserves legacy runs."""
import json,math,os,shutil,time,uuid
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .. import storage as st
from . import data
from .model import JointModel
CACHE={}

def init(root):
    with st.db(root) as db:db.execute('CREATE TABLE IF NOT EXISTS joint_runs(id TEXT PRIMARY KEY,config TEXT,status TEXT,step INTEGER DEFAULT 0,target INTEGER DEFAULT 0,request TEXT DEFAULT "",error TEXT DEFAULT "",created REAL)')
    (Path(root)/'joint').mkdir(exist_ok=True)

def config(raw):
    if raw.get('architecture','resnet50_joint')!='resnet50_joint':raise ValueError('Joint workspace supports the ResNet-50 encoder with parallel segmentation/classification heads')
    c=dict(architecture='resnet50_joint',preset=str(raw.get('preset','custom')),name=str(raw.get('name','Joint healthcare experiment'))[:160],dataset=str(raw.get('dataset','bus-uclm-v1')),size=int(raw.get('size',256)),width=int(raw.get('width',32)),seed=int(raw.get('seed',17)),pretrained=bool(raw.get('pretrained',False)),freeze=bool(raw.get('freeze',False)),optimizer=raw.get('optimizer','Adam'),lr=float(raw.get('lr',.0001)),weight_decay=float(raw.get('weight_decay',.0001)),batch=int(raw.get('batch',2)),device=raw.get('device','cpu'),task=raw.get('task','joint'),class_weight=float(raw.get('class_weight',1)),mask_weight=float(raw.get('mask_weight',1)),dice_weight=float(raw.get('dice_weight',1)),save_every=int(raw.get('save_every',25)),evaluation_count=int(raw.get('evaluation_count',32)),checkpoint_metric=raw.get('checkpoint_metric','loss'))
    if c['size'] not in [64,128,256,384,512] or c['width'] not in [16,32,64]:raise ValueError('Unsupported resolution/decoder width')
    if c['optimizer'] not in ['Adam','SGD','RMSprop'] or c['device'] not in ['cpu','cuda'] or c['task'] not in ['joint','classification','segmentation']:raise ValueError('Invalid training configuration')
    if not (1<=c['batch']<=32 and 1<=c['save_every']<=10000 and 0<=c['evaluation_count']<=10000 and 0<c['lr']<=1 and 0<=c['weight_decay']<=1):raise ValueError('Hyperparameters outside supported range')
    if any(not math.isfinite(c[k]) or not 0<=c[k]<=100 for k in ['class_weight','mask_weight','dice_weight']):raise ValueError('Invalid loss weights')
    if c['checkpoint_metric'] not in ['loss','accuracy','dice']:raise ValueError('Invalid checkpoint metric')
    if c['task']=='classification' and c['checkpoint_metric']=='dice' or c['task']=='segmentation' and c['checkpoint_metric']=='accuracy':raise ValueError('Checkpoint criterion must measure a trained task')
    if not 0<=c['seed']<2**32:raise ValueError('Seed must be an unsigned 32-bit integer')
    if (c['task']!='segmentation' and c['class_weight']<=0) or (c['task']!='classification' and c['mask_weight']+c['dice_weight']<=0):raise ValueError('Selected task must have positive loss weight')
    if not all(ch.isalnum() or ch in '-_' for ch in c['dataset']):raise ValueError('Invalid dataset key')
    return c

def get(root,rid):
    with st.db(root) as db:r=db.execute('SELECT * FROM joint_runs WHERE id=?',(rid,)).fetchone()
    if not r:raise KeyError('Joint experiment not found')
    r=dict(r);r['config']=json.loads(r['config']);return r

def folder(root,rid):
    if len(rid)!=16 or any(c not in '0123456789abcdef' for c in rid):raise ValueError('Invalid run ID')
    return Path(root)/'joint'/rid

def create(root,raw):
    c=config(raw);path=Path(root)/'joint_data'/c['dataset']/'manifest.json'
    if not path.exists():raise ValueError('Prepare or import this dataset first')
    d=json.loads(path.read_text());rows=data.split(d,c['seed']);rid=uuid.uuid4().hex[:16];p=folder(root,rid);p.mkdir()
    st.atomic_json(p/'data.json',dict(d,rows=rows));st.atomic_json(p/'history.json',[])
    with st.db(root) as db:db.execute('INSERT INTO joint_runs(id,config,status,created) VALUES(?,?,?,?)',(rid,json.dumps(c),'ready',time.time()))
    return rid

def build(c,pretrained=False):
    torch.manual_seed(c['seed']);m=JointModel(width=c['width'],pretrained=pretrained)
    if c['freeze']:
        for p in m.encoder.parameters():p.requires_grad_(False)
    return m

def optimizer(c,m):
    cls={'Adam':torch.optim.Adam,'SGD':torch.optim.SGD,'RMSprop':torch.optim.RMSprop}[c['optimizer']]
    kwargs=dict(lr=c['lr'],weight_decay=c['weight_decay'])
    if c['optimizer']=='SGD':kwargs['momentum']=.9
    if c['optimizer']=='RMSprop':kwargs.update(alpha=.99,eps=1e-8,momentum=0.)
    return cls([p for p in m.parameters() if p.requires_grad],**kwargs)

def atomic_save(path,obj):
    tmp=path.with_suffix('.tmp');torch.save(obj,tmp);os.replace(tmp,path)

def restore(root,r,checkpoint='latest',analysis=False):
    p=folder(root,r['id']);c=r['config'];path=p/(str(checkpoint)+'.pt')
    if not path.exists():raise ValueError('Checkpoint not available; train at least one update first')
    ck=torch.load(path,map_location='cpu',weights_only=False);m=build(c);m.load_state_dict(ck['model'])
    return m.eval(),ck

def loadstate(root,r):
    rid=r['id'];c=r['config']
    if rid in CACHE:return CACHE[rid]
    if c['device']=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA requested but this PyTorch installation has no available CUDA device')
    path=folder(root,rid)/'latest.pt';m=build(c,pretrained=c['pretrained'] and not path.exists()).to(c['device']);opt=optimizer(c,m);step=0
    if path.exists():
        ck=torch.load(path,map_location=c['device'],weights_only=False);m.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);step=ck['step']
    state=dict(model=m,opt=opt,step=step);CACHE.clear();CACHE[rid]=state;return state

def loss(out,y,mask,c):
    ce=F.cross_entropy(out['classification'],y);bce=F.binary_cross_entropy_with_logits(out['segmentation'],mask)
    prob=out['segmentation'].sigmoid();dice=(2*(prob*mask).sum((1,2,3))+1e-6)/(prob.sum((1,2,3))+mask.sum((1,2,3))+1e-6);dl=1-dice.mean()
    total=(c['class_weight']*ce if c['task']!='segmentation' else ce*0)+(c['mask_weight']*bce+c['dice_weight']*dl if c['task']!='classification' else bce*0)
    return total,dict(loss=float(total.detach()),ce=float(ce.detach()),bce=float(bce.detach()),dice_loss=float(dl.detach()))

def batch(rows,size,device):
    samples=[data.sample(r,size,device) for r in rows]
    return torch.stack([v[0] for v in samples]),torch.stack([v[1] for v in samples]),torch.tensor([v[2] for v in samples],device=device)

@torch.no_grad()
def evaluate(m,rows,c):
    m.eval();limit=c['evaluation_count'];selected=rows if not limit else rows[:limit];result={k:0. for k in ['loss','ce','bce','dice_loss','accuracy','dice','iou']}
    for start in range(0,len(selected),c['batch']):
        rr=selected[start:start+c['batch']];x,mask,y=batch(rr,c['size'],c['device']);out=m(x);_,v=loss(out,y,mask,c);pred=out['segmentation']>0
        target=mask>0;intersection=(pred&target).sum((1,2,3));union=(pred|target).sum((1,2,3));den=pred.sum((1,2,3))+target.sum((1,2,3))
        v.update(accuracy=float((out['classification'].argmax(1)==y).float().mean()),dice=float(torch.where(den>0,2*intersection/den,torch.ones_like(den,dtype=torch.float32)).mean()),iou=float(torch.where(union>0,intersection/union,torch.ones_like(union,dtype=torch.float32)).mean()))
        for k in result:result[k]+=v[k]*len(rr)
    return dict({k:v/len(selected) for k,v in result.items()},samples=len(selected),population=len(rows),scope='full split' if len(selected)==len(rows) else 'fixed leading subset; not a full-split estimate')

def quantum(root,r):
    rid=r['id'];p=folder(root,rid);c=r['config'];request=r['request']
    if request=='delete':
        CACHE.pop(rid,None);
        if p.exists():shutil.rmtree(p)
        with st.db(root) as db:
            db.execute('DELETE FROM jobs WHERE run=?',(rid,));db.execute('DELETE FROM joint_runs WHERE id=?',(rid,))
        return
    if request in ['pause','stop']:
        with st.db(root) as db:db.execute("UPDATE joint_runs SET status='paused',request='' WHERE id=?",(rid,))
        return
    state=loadstate(root,r);m,opt=state['model'],state['opt'];rows=json.loads((p/'data.json').read_text())['rows'];train=[a for a in rows if a['split']=='train']
    rng=np.random.default_rng(c['seed']+state['step']);idx=rng.choice(len(train),c['batch'],replace=len(train)<c['batch']);x,mask,y=batch([train[i] for i in idx],c['size'],c['device'])
    m.train()
    # Fixed running statistics during fine-tuning are an explicit v1 policy,
    # avoiding batch-size-dependent inference semantics.
    for module in m.modules():
        if isinstance(module,nn.BatchNorm2d):module.eval()
    opt.zero_grad(set_to_none=True);out=m(x);total,v=loss(out,y,mask,c)
    if not torch.isfinite(total):raise RuntimeError('Non-finite training loss')
    total.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),10.);opt.step();state['step']+=1;step=state['step']
    ck=dict(model=m.state_dict(),optimizer=opt.state_dict(),step=step,config=c,format='joint-1')
    atomic_save(p/'latest.pt',ck)
    history=json.loads((p/'history.json').read_text());entry=dict(step=step,train_batch=v)
    if step%c['save_every']==0 or step>=r['target']:
        for split in ['train','val','test']:entry[split]=evaluate(m,[a for a in rows if a['split']==split],c)
        atomic_save(p/(str(step)+'.pt'),ck)
        metric=c['checkpoint_metric'];score=entry['val'][metric];best_path=p/'best.json';best=json.loads(best_path.read_text()) if best_path.exists() else None
        if best is None or (score<best['score'] if metric=='loss' else score>best['score']):
            atomic_save(p/'best.pt',ck);st.atomic_json(best_path,dict(step=step,metric=metric,score=score,split='val'))
    history.append(entry);st.atomic_json(p/'history.json',history)
    with st.db(root) as db:db.execute("UPDATE joint_runs SET step=?,status=?,request=CASE WHEN request='start' THEN '' ELSE request END WHERE id=?",(step,'paused' if step>=r['target'] else 'training',rid))

def tick(root,controls_only=False):
    with st.db(root) as db:
        if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='joint_runs'").fetchone():return False
        query="SELECT id FROM joint_runs WHERE "+("request IN ('pause','stop','delete')" if controls_only else "request<>'' OR status='training'")+" ORDER BY CASE WHEN request IN ('pause','stop','delete') THEN 0 ELSE 1 END,created LIMIT 1"
        r=db.execute(query).fetchone()
    if not r:return False
    try:quantum(root,get(root,r[0]))
    except Exception:
        import traceback
        CACHE.pop(r[0],None)
        with st.db(root) as db:db.execute("UPDATE joint_runs SET status='failed',request='',error=? WHERE id=?",(traceback.format_exc(),r[0]))
    return True
