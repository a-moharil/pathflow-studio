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
        a=int(payload.get('x',0));b=int(payload.get('reference',1));readout=payload.get('readout',dict(a=0,b=1));gates=validate_gates(m,payload.get('gates',{}))
        if not 0<=a<len(d['test_x']) or not 0<=b<len(d['test_x']):raise ValueError('Invalid contrast indices')
        if kind=='ensemble':
            relation=payload.get('relation','same');pairs=[];labels=d['test_y']
            for i in range(min(64,len(labels))):
                for k in range(i+1,min(64,len(labels))):
                    if (labels[i]==labels[k])==(relation=='same'):pairs.append((i,k))
            pairs=[pairs[i] for i in np.linspace(0,len(pairs)-1,min(16,len(pairs)),dtype=int)] if pairs else []
            result=dict(step=step,relation=relation,readout=readout,gates=gates,samples=[])
            for i,k in pairs:
                v=analyze(m,d['test_x'][i],d['test_x'][k],readout,gates,False);result['samples'].append(dict(x=i,reference=k,**v))
        else:
            result=analyze(m,d['test_x'][a],d['test_x'][b],readout,gates,True)
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
            if gates:result['native']=analyze(m,d['test_x'][a],d['test_x'][b],readout,{},False)
            result.update(analysis_version=2,measurement_settings=dict(limit=budget,epsilon=.03,estimator='bcan-1'),step=step,x=a,reference=b,labels=[int(d['test_y'][a]),int(d['test_y'][b])])
            if kind=='dose':
                result['dose_response']=[]
                for alpha in [0.,.25,.5,.75,1.]:
                    gg={l:{i:alpha for i in units} for l,units in gates.items()}
                    v=analyze(m,d['test_x'][a],d['test_x'][b],readout,gg,False)
                    result['dose_response'].append(dict(alpha=alpha,delta=v['delta'],conditional=v['conditional'],residual=v['residual']))
    path=Path(root)/r['id']/f"analysis_{j['id']}.json";st.atomic_json(path,result)
    with st.db(root) as db:
        db.execute("UPDATE jobs SET status='completed',result=? WHERE id=?",(str(path),j['id']))
        if kind in ['analysis','dose']:
            summary={k:v for k,v in result.items() if k not in ['nodes','edges','positions','input_fields']}
            db.execute('INSERT OR REPLACE INTO observations VALUES(?,?,?,?)',(j['id'],r['id'],result['step'],json.dumps(summary)))

