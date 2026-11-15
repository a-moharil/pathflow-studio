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
        if action=='classification':
            key,result=data.prepare_classification(root,p['dataset'],p.get('seed',42),p.get('train_size',512),progress);result=dict(result,key=key)
        elif action=='synthetic':result=data.synthetic(root)
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
        from ..core import transforms
        spec_transform=transforms.specification(p.get('transform'))
        if spec_transform!={'dx':0.,'dy':0.,'angle':0.,'scale':1.}:
            x=torch.from_numpy(transforms.image(x[0].numpy(),spec_transform))[None]
            ref=torch.from_numpy(transforms.image(ref[0].numpy(),spec_transform))[None]
            from scipy.ndimage import affine_transform
            import numpy as np
            M,b=transforms.matrix(target.shape[-2:],spec_transform);inverse=np.linalg.inv(M);swap=np.array([[0,1],[1,0]])
            target=torch.from_numpy(affine_transform(target[0].numpy(),swap@inverse@swap,offset=-swap@inverse@b,order=0,mode='constant',cval=0,prefilter=False))[None]
        size=c['size'];region=p.get('region',[0,0,size,size]);layer=p.get('layer','stem.relu');channel=p['channel'];alpha=p['alpha']
        specs=[dict(head='classification',a=int(p.get('class_a',2)),b=None if int(p.get('class_b',0))<0 else int(p.get('class_b',0))),dict(head='segmentation',region=region)]
        if c.get('classification_only'):specs=specs[:1]
        progress('Checkpoint loaded; computing '+str(len(specs))+' readout(s)')
        result=dict(version='joint-investigation-15',checkpoint=ck['step'],config=c,request=p,image=x[0].permute(1,2,0).tolist(),reference_image=ref[0].permute(1,2,0).tolist(),target=target[0].tolist() if c.get('has_masks',True) else None,label=label,case=rows[idx]['id'],reference_case='zero in raw [0,1] input coordinates' if zero else rows[refidx]['id'],reference_label=None if zero else rows[refidx]['label'],preprocessing=meta,estimated_memory_gib=estimated,heads={})
        for spec in specs:
            result['heads'][spec['head']]=analyze(model,x,ref,spec,layer,channel,p['ig_steps'],geometry=p.get('geometry_scope','operations') if p.get('geometry',True) else False,fit_hyperbolic=bool(p.get('fit_hyperbolic',False)),progress=lambda msg,h=spec['head']:progress(h+': '+msg),source_region=p.get('source_region'),analysis_settings=p.get('analysis_settings'))
        layer=next(iter(result['heads'].values()))['selected_layer'];p['layer']=layer
        result['intervention']=None
        if alpha!=1:
            gates={layer:(channel,alpha)};post={}
            for spec in specs:post[spec['head']]=analyze(model,x,ref,spec,layer,channel,p['ig_steps'],gates=gates,geometry=p.get('geometry_scope','operations') if p.get('geometry',True) else False,fit_hyperbolic=bool(p.get('fit_hyperbolic',False)),progress=lambda msg,h=spec['head']:progress('Intervened '+h+': '+msg),source_region=p.get('source_region'),analysis_settings=p.get('analysis_settings'))
            for name,d in post.items():
                d['post_selected_support_occupation']=d['FieldB'];d['FieldB']=result['heads'][name]['FieldB'];d['FieldB_scope']=result['heads'][name]['FieldB_scope']
            doses=[]
            with torch.no_grad():
                for dose in [1.,.75,.5,.25,0.]:
                    out=model(x,gates={layer:(channel,dose)});out0=model(ref,gates={layer:(channel,dose)})
                    doses.append(dict(alpha=dose,**{s['head']:dict(value=float(readout_tensor(out,s)),reference_value=float(readout_tensor(out0,s)),delta=float(readout_tensor(out,s)-readout_tensor(out0,s))) for s in specs}))
            result['intervention']=dict(layer=layer,channel=channel,alpha=alpha,semantics='Postactivation channel attenuation applied at both endpoints; Field B is baseline support, not post-intervention effect',heads=post,dose_response=doses)
    dest.mkdir(exist_ok=True,parents=True);path=dest/(j['id']+'.json');st.atomic_json(path,result)
    with st.db(root) as db:db.execute("UPDATE jobs SET status='completed',result=?,progress='Ready' WHERE id=?",(str(path),j['id']))
