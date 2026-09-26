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
    if kind.startswith('healthcare_'):
        from .healthcare_jobs import process
        return process(root,j)
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
            from .core.metricfit import refine,deformation
            fit_steps=int(payload.get('embedding_steps',120));sensitivity=bool(payload.get('dimension_sensitivity',False))
            if not 25<=fit_steps<=1000:raise ValueError('Embedding iterations must be 25..1000')
            st.progress(root,j['id'],'Fitting independent Euclidean 2-D and 3-D metrics')
            for side in ['structural_geometry','conditional_geometry']:
                result[side]['embedding_diagnostic_role']='Initial spectral preview only; displayed coordinates use euclidean.dimensions and projection_diagnostics'
            design,distances=refine(result,fit_steps,sensitivity)
            for dim,entries in result['euclidean']['dimensions'].items():
                frame_path=Path(root)/r['id']/('metric3_e'+dim+'_'+hashlib.sha256(unit_key.encode()).hexdigest()[:16]+'.npz')
                structural=np.asarray(entries[0]['positions']);centre=structural.mean(0)
                if frame_path.exists():
                    with np.load(frame_path) as frame:base=frame['structural']
                    target=base.mean(0);rotation,_=orthogonal_procrustes(structural-centre,base-target)
                    for entry in entries:
                        if entry['status']=='available':entry['positions']=((np.asarray(entry['positions'])-centre)@rotation+target).tolist()
                else:np.savez(frame_path,structural=structural)
            result['positions']=[entry.get('positions',result['euclidean']['dimensions']['3'][0]['positions']) for entry in result['euclidean']['dimensions']['3']]
            if gates:result['native']=analyze(m,tx,tr,readout,{},False)
            result.update(analysis_version=6,reference_type=reference_type,transform=transform,measurement_settings=dict(limit=budget,epsilon=.03,estimator='bcan-1'),step=step,x=a,reference=None if reference_type=='zero' else b,labels=[int(d['test_y'][a]),None if reference_type=='zero' else int(d['test_y'][b])])
            result['projection_diagnostics']={side:{dim:dict(entry[idx].get('heldout',{}),status=entry[idx].get('quality',entry[idx]['status'])) for dim,entry in result['euclidean']['dimensions'].items()} for idx,side in enumerate(['s','c'])}
            result['context']=dict(experiment=r['id'],checkpoint=step,x=a,reference_type=reference_type,reference=result['reference'],readout=readout,gates=gates,transform=transform,metric=result['metric'],representation=payload.get('hyperbolic'),analysis_version=6)
            if payload.get('hyperbolic'):
                from .core.hyperbolic import compute,numerical_validity
                options=payload['hyperbolic'];k=float(options.get('curvature',1));steps=int(options.get('steps',150))
                if not .01<=k<=4 or not 25<=steps<=500:raise ValueError('Curvature magnitude must be .01..4 and fit steps 25..500')
                st.progress(root,j['id'],'Fitting 3-D Lorentz representation')
                try:
                    result['hyperbolic']=compute(result,k,steps,design,distances,sensitivity)
                except FloatingPointError as error:
                    result['hyperbolic']=dict(status='numerical_invalid',reason=str(error),dimensions={})
                if result['hyperbolic']['status']!='available':
                    for evidence in result['distance_evidence']['sides'].values():evidence.pop('hyperbolic',None)
                if result['hyperbolic']['status']=='available':
                    from .core.hyperbolic import ball,move_origin
                    import math
                    hp=Path(root)/r['id']/('metric3_hyper_frame_'+hashlib.sha256((unit_key+str(k)).encode()).hexdigest()[:16]+'.npz')
                    entries=result['hyperbolic']['dimensions']['3'];struct=np.asarray(entries[0]['lorentz'])
                    if hp.exists():
                        with np.load(hp) as frame:target=frame['structural']
                        rotation,_=orthogonal_procrustes(struct[:,1:],target[:,1:])
                        for entry in entries:
                            if entry['status']!='available':continue
                            z=np.asarray(entry['lorentz']);before=z.copy();z[:,1:]=z[:,1:]@rotation
                            free_anchor=np.array([math.cosh(.4),-math.sinh(.4),0,0])/math.sqrt(k)
                            free=move_origin(z,free_anchor,k)
                            check=numerical_validity(z,k,free=free,before=before,pairs=design['train'])
                            if not check['valid']:
                                result['hyperbolic']=dict(status='numerical_invalid',reason=check['reason'],numerical_validation=check,dimensions={})
                                for evidence in result['distance_evidence']['sides'].values():evidence.pop('hyperbolic',None)
                                break
                            entry.update(lorentz=z.tolist(),ball=ball(z,k).tolist(),ball_free=ball(free,k).tolist(),numerical_validation=check)
                    else:np.savez(hp,structural=struct)
                    result['hyperbolic']['temporal_alignment']='Common spatial rotation toward first structural fit for this readout and curvature, after sink boost; no rescaling'
            temporal_key=json.dumps(dict(x=a,reference_type=reference_type,reference=result['reference'],readout=readout,gates=gates,transform=transform,unit=reference['unit'],hyperbolic=payload.get('hyperbolic'),embedding_steps=fit_steps,dimension_sensitivity=sensitivity),sort_keys=True)
            temporal_path=Path(root)/r['id']/('metric3_probe_'+hashlib.sha256(temporal_key.encode()).hexdigest()[:16]+'.json')
            evidence=result['distance_evidence'];result['temporal_deformation']={}
            if temporal_path.exists():
                previous=json.loads(temporal_path.read_text())
                if previous['evidence']['pairs']==evidence['pairs']:
                    for side,current in evidence['sides'].items():
                        before=previous['evidence']['sides'].get(side)
                        if before is None:continue
                        for space in ['euclidean','hyperbolic']:
                            for dim,pred in current.get(space,{}).items():
                                old=before.get(space,{}).get(dim)
                                if old is not None:result['temporal_deformation'][side+'_'+space+'_'+dim]=dict(reference_checkpoint=previous['step'],current_checkpoint=step,**deformation(before['intrinsic'],current['intrinsic'],old,pred))
            else:st.atomic_json(temporal_path,dict(step=step,evidence=evidence))
            result['embedding_settings']=dict(version='metric-fit-3',steps=fit_steps,dimension_sensitivity=sensitivity)
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
            summary={k:v for k,v in result.items() if k not in ['nodes','edges','positions','input_fields','hyperbolic','investigation','euclidean','distance_evidence']}
            db.execute('INSERT OR REPLACE INTO observations VALUES(?,?,?,?)',(j['id'],r['id'],result['step'],json.dumps(summary)))

