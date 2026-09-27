import json,time,math
from pathlib import Path
import torch
from .. import storage as st
from . import data,engine
from .investigate import analyze,readout_tensor

def process(root,j):
    p=json.loads(j['payload']);progress=lambda msg:st.progress(root,j['id'],msg)
    if p['kind']=='joint_data':
        action=p['action']
        if action=='synthetic':result=data.synthetic(root)
        elif action=='manifest':key,result=data.import_manifest(root,p['path']);result=dict(result,key=key)
        else:result=data.prepare(root,progress,p.get('path') if action=='import' else None)
        result={k:v for k,v in result.items() if k!='rows'};dest=Path(root)/'joint_data'
    else:
        r=engine.get(root,j['run']);c=r['config'];dest=engine.folder(root,r['id']);rows=[v for v in json.loads((dest/'data.json').read_text())['rows'] if v['split']=='test']
        estimated=.8+(c['size']/256)**2*2.5
        if estimated>p['memory_gib']:raise ValueError(f'Conservative analysis estimate {estimated:.1f} GiB exceeds selected {p["memory_gib"]} GiB budget. Choose a larger budget or a lower-resolution experiment; analysis never silently resizes the trained input.')
        idx=int(p.get('x',0));refidx=int(p.get('reference',1));zero=p.get('reference_type')=='zero'
        if not 0<=idx<len(rows) or (not zero and not 0<=refidx<len(rows)):raise ValueError('Invalid case indices')
        model,ck=engine.restore(root,r,p['checkpoint']);model=model.double().eval()
        x,target,label,meta=data.sample(rows[idx],c['size'],dtype=torch.float64);x=x[None]
        ref=torch.zeros_like(x) if zero else data.sample(rows[refidx],c['size'],dtype=torch.float64)[0][None]
        size=c['size'];region=p.get('region',[0,0,size,size]);layer=p.get('layer','stem.relu');channel=p['channel'];alpha=p['alpha']
        specs=[dict(head='classification',a=int(p.get('class_a',2)),b=None if int(p.get('class_b',0))<0 else int(p.get('class_b',0))),dict(head='segmentation',region=region)]
        result=dict(version='joint-investigation-1',checkpoint=ck['step'],config=c,request=p,image=x[0].permute(1,2,0).tolist(),reference_image=ref[0].permute(1,2,0).tolist(),target=target[0].tolist(),label=label,case=rows[idx]['id'],reference_case='zero in raw [0,1] input coordinates' if zero else rows[refidx]['id'],reference_label=None if zero else rows[refidx]['label'],preprocessing=meta,estimated_memory_gib=estimated,heads={})
        for spec in specs:
            result['heads'][spec['head']]=analyze(model,x,ref,spec,layer,channel,p['ig_steps'],geometry=bool(p.get('geometry',False)),progress=lambda msg,h=spec['head']:progress(h+': '+msg))
        result['intervention']=None
        if alpha!=1:
            gates={layer:(channel,alpha)};post={}
            for spec in specs:post[spec['head']]=analyze(model,x,ref,spec,layer,channel,p['ig_steps'],gates=gates,progress=lambda msg,h=spec['head']:progress('Intervened '+h+': '+msg))
            doses=[]
            with torch.no_grad():
                for dose in [1.,.75,.5,.25,0.]:
                    out=model(x,gates={layer:(channel,dose)});out0=model(ref,gates={layer:(channel,dose)})
                    doses.append(dict(alpha=dose,**{s['head']:dict(value=float(readout_tensor(out,s)),reference_value=float(readout_tensor(out0,s)),delta=float(readout_tensor(out,s)-readout_tensor(out0,s))) for s in specs}))
            result['intervention']=dict(layer=layer,channel=channel,alpha=alpha,semantics='Post-ReLU channel attenuation applied at both endpoints; Field B is baseline support, not post-intervention effect',heads=post,dose_response=doses)
    dest.mkdir(exist_ok=True,parents=True);path=dest/(j['id']+'.json');st.atomic_json(path,result)
    with st.db(root) as db:db.execute("UPDATE jobs SET status='completed',result=?,progress='Ready' WHERE id=?",(str(path),j['id']))
