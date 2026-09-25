import json
from pathlib import Path
import numpy as np
from . import storage as st
from .datasets import load
from .engine import restore
from .core.pathmath import analyze
def validate_gates(m,gates):
    result={}
    for l,units in gates.items():
        layer=int(l)
        if not 0<=layer<len(m.layers)-1:raise ValueError('Only hidden activations may be intervened on')
        n=int(np.prod(m.shapes[layer+1]));result[str(layer)]={}
        for i,alpha in units.items():
            if not 0<=int(i)<n or not 0<=float(alpha)<=1:raise ValueError('Invalid unit or attenuation')
            result[str(layer)][str(int(i))]=float(alpha)
    return result

def process_job(root,j):
    payload=json.loads(j['payload']);r=st.run(root,j['run']);c=r['config'];kind=payload.get('kind','analysis')
    if kind=='download':
        d=load(root,c,True);result={split:int(len(d[split+'_y'])) for split in ['train','val','test']}
    else:
        d=load(root,c,False);step=int(payload.get('step',r['step']));m,ck=restore(root,r,step)
        budget=int(payload.get('measurement_limit',c.get('measurement_limit',2500)))
        if not 50<=budget<=4000:raise ValueError('Measurement limit must be 50..4000')
        m.config=dict(m.config,measurement_limit=budget)
        a=int(payload.get('x',0));reference_type=payload.get('reference_type','dataset');b=int(payload.get('reference',1) or 0);readout=payload.get('readout',dict(a=0,b=1));gates=validate_gates(m,payload.get('gates',{}))
        if reference_type not in ['dataset','zero']:raise ValueError('Unknown reference type')
        if not 0<=a<len(d['test_x']) or (reference_type=='dataset' and not 0<=b<len(d['test_x'])):raise ValueError('Invalid contrast indices')
        if kind=='ensemble':
            if reference_type!='dataset' or any(float(payload.get('transform',{}).get(k,v))!=v for k,v in [('dx',0),('dy',0),('angle',0),('scale',1)]):raise ValueError('Pair ensembles require dataset references and identity transforms; use individual investigation frames for this context')
            relation=payload.get('relation','same');pairs=[];labels=d['test_y']
            for i in range(min(64,len(labels))):
                for k in range(i+1,min(64,len(labels))):
                    if (labels[i]==labels[k])==(relation=='same'):pairs.append((i,k))
            pairs=[pairs[i] for i in np.linspace(0,len(pairs)-1,min(16,len(pairs)),dtype=int)] if pairs else []
            result=dict(step=step,relation=relation,readout=readout,gates=gates,samples=[])
            for i,k in pairs:
                v=analyze(m,d['test_x'][i],d['test_x'][k],readout,gates,False);result['samples'].append(dict(x=i,reference=k,**v))
        else:
            from .core.transforms import specification,image,transport,hellinger
            transform=specification(payload.get('transform'))
            original_x=d['test_x'][a];original_ref=np.zeros_like(original_x) if reference_type=='zero' else d['test_x'][b]
            tx=image(original_x,transform);tr=image(original_ref,transform)
            result=analyze(m,tx,tr,readout,gates,True)
            baseline=analyze(m,original_x,original_ref,readout,gates,False)
            A=baseline['input_fields']['conditional'];B=result['input_fields']['conditional'];transported=transport(A,transform)
            result['investigation']=dict(transform=transform,convention='Pixel-centred affine; positive angle clockwise on image; bilinear zero padding; original images resampled independently; probability uses forward bilinear mass deposition',baseline=A,transported=transported,recomputed=B,residual=(np.asarray(B)-np.asarray(transported['field'])).tolist() if B is not None and transported['field'] is not None else None,H_eq=hellinger(B,transported['field']),H_inv=hellinger(B,A),original_image=np.asarray(original_x).tolist(),transformed_image=tx.tolist(),reference_image=tr.tolist(),baseline_delta=baseline['delta'])
            # Freeze the display distance unit per run/readout at its first analysis.
            unit_path=Path(root)/r['id']/'metric_units.json'
            units=json.loads(unit_path.read_text()) if unit_path.exists() else {}
            unit_key=json.dumps(readout,sort_keys=True)
            if unit_key not in units:
                units[unit_key]=dict(unit=result['structural_geometry']['unit'],checkpoint=step)
                st.atomic_json(unit_path,units)
            reference=units[unit_key];ratio=result['structural_geometry']['unit']/reference['unit']
            result['positions']=(np.asarray(result['positions'])*ratio).tolist()
            result['metric']['display_reference']=reference
            import hashlib
            from scipy.linalg import orthogonal_procrustes
            frame_path=Path(root)/r['id']/('frame_'+hashlib.sha256(unit_key.encode()).hexdigest()[:16]+'.npz')
            pos=np.asarray(result['positions'])
            if frame_path.exists():
                with np.load(frame_path) as frame:base=frame['structural']
                centre=pos[0].mean(0);target=base.mean(0);R,_=orthogonal_procrustes(pos[0]-centre,base-target)
                result['positions']=((pos-centre)@R+target).tolist()
            else:np.savez(frame_path,structural=pos[0])
            if gates:result['native']=analyze(m,tx,tr,readout,{},False)
            result.update(analysis_version=4,reference_type=reference_type,transform=transform,measurement_settings=dict(limit=budget,epsilon=.03,estimator='bcan-1'),step=step,x=a,reference=None if reference_type=='zero' else b,labels=[int(d['test_y'][a]),None if reference_type=='zero' else int(d['test_y'][b])])
            from .core.hyperbolic import euclidean_diagnostics
            result['projection_diagnostics']=euclidean_diagnostics(result)
            result['context']=dict(experiment=r['id'],checkpoint=step,x=a,reference_type=reference_type,reference=result['reference'],readout=readout,gates=gates,transform=transform,metric=result['metric'],representation=payload.get('hyperbolic'),analysis_version=4)
            if payload.get('hyperbolic'):
                from .core.hyperbolic import compute
                options=payload['hyperbolic'];k=float(options.get('curvature',1));steps=int(options.get('steps',150))
                if not .01<=k<=4 or not 25<=steps<=500:raise ValueError('Curvature magnitude must be .01..4 and fit steps 25..500')
                st.progress(root,j['id'],'Fitting 3-D Lorentz representation')
                result['hyperbolic']=compute(result,k,steps)
                if result['hyperbolic']['status']=='available':
                    from .core.hyperbolic import ball,move_origin
                    import math
                    hp=Path(root)/r['id']/('hyper_frame_'+hashlib.sha256((unit_key+str(k)).encode()).hexdigest()[:16]+'.npz')
                    entries=result['hyperbolic']['dimensions']['3'];struct=np.asarray(entries[0]['lorentz'])
                    if hp.exists():
                        with np.load(hp) as frame:target=frame['structural']
                        rotation,_=orthogonal_procrustes(struct[:,1:],target[:,1:])
                        for entry in entries:
                            if entry['status']!='available':continue
                            z=np.asarray(entry['lorentz']);z[:,1:]=z[:,1:]@rotation
                            free_anchor=np.array([math.cosh(.4),-math.sinh(.4),0,0])/math.sqrt(k)
                            entry.update(lorentz=z.tolist(),ball=ball(z,k).tolist(),ball_free=ball(move_origin(z,free_anchor,k),k).tolist())
                    else:np.savez(hp,structural=struct)
                    result['hyperbolic']['temporal_alignment']='Common spatial rotation toward first structural fit for this readout and curvature, after sink boost; no rescaling'
            if kind=='dose':
                result['dose_response']=[]
                for alpha in [0.,.25,.5,.75,1.]:
                    gg={l:{i:alpha for i in units} for l,units in gates.items()}
                    v=analyze(m,tx,tr,readout,gg,False)
                    result['dose_response'].append(dict(alpha=alpha,delta=v['delta'],conditional=v['conditional'],residual=v['residual']))
    path=Path(root)/r['id']/f"analysis_{j['id']}.json";st.atomic_json(path,result)
    with st.db(root) as db:
        db.execute("UPDATE jobs SET status='completed',result=? WHERE id=?",(str(path),j['id']))
        if kind in ['analysis','dose']:
            summary={k:v for k,v in result.items() if k not in ['nodes','edges','positions','input_fields','hyperbolic','investigation']}
            db.execute('INSERT OR REPLACE INTO observations VALUES(?,?,?,?)',(j['id'],r['id'],result['step'],json.dumps(summary)))

