"""Serialized healthcare jobs at committed checkpoints; no training mutation."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.special import softmax
from . import storage as st
from .datasets import load
from .engine import restore
from .core.healthcare import case_analysis,binary_metrics,bootstrap_metrics,group_summary
from .core.transforms import specification,image,transport,hellinger


def brief(result):
    return dict(delta=result['delta'],margin=result['margin'],reference_margin=result['reference_margin'],probability=result['probabilities'][1],path_hellinger=result['path_hellinger'],cancellation=result['cancellation'],**{key:result['conditional'][key] for key in ['H1','eta1','eta2','Sdef','logZ']})


def selected_units(model,layer,index):
    if not 0<=layer<len(model.layers)-1:raise ValueError('Select a hidden layer')
    shape=model.shapes[layer+1];count=shape[0]
    if not 0<=index<count:raise ValueError('Channel or unit index is outside the layer')
    size=int(np.prod(shape[1:])) if len(shape)>1 else 1
    return list(range(index*size,(index+1)*size))


def process(root,job):
    p=json.loads(job['payload']);run=st.run(root,job['run']);config=run['config']
    if config['dataset'] not in ['pneumonia','synthetic']:raise ValueError('Healthcare investigation currently supports binary PneumoniaMNIST and synthetic validation experiments')
    data=load(root,config,False);step=int(p.get('step',run['step']));model,checkpoint=restore(root,run,step)
    if checkpoint is None:raise ValueError('Train to a saved checkpoint before investigation')
    model=model.cpu().double().eval();split=p.get('split','test')
    if split not in ['test','val']:raise ValueError('Use validation or test split')
    X=data[split+'_x'];Y=data[split+'_y'];ids=data[split+'_ids'];readout=dict(scope='selected',a=1,b=0)
    x=int(p.get('x',0));r=int(p.get('reference',1));reference_type=p.get('reference_type','dataset')
    if not 0<=x<len(X):raise ValueError('Input index out of bounds')
    if reference_type not in ['zero','dataset']:raise ValueError('Invalid reference type')
    if reference_type=='dataset' and not 0<=r<len(X):raise ValueError('Reference index out of bounds')
    transform=specification(p.get('transform'));threshold=float(p.get('threshold',.5))
    if not np.isfinite(threshold) or not 0<=threshold<=1:raise ValueError('Threshold must be 0..1')
    original=X[x];reference=np.zeros_like(original) if reference_type=='zero' else X[r]
    context=dict(experiment=run['id'],dataset=config['dataset'],architecture=config,checkpoint=step,split=split,x=x,x_source_index=int(ids[x]),reference_type=reference_type,reference=r if reference_type=='dataset' else None,reference_source_index=int(ids[r]) if reference_type=='dataset' else None,readout=readout,transform=transform,healthcare_version=1,analysis_core_version=6,threshold=threshold)
    report=dict(context=context,class_names=['Normal','Pneumonia'] if config['dataset']=='pneumonia' else ['Class 0','Class 1'],kind=p['kind'],research_scope='Model computation on benchmark images. Fields are not lesion segmentations; probabilities are uncalibrated model outputs.')
    if p['kind']=='healthcare_cohort':
        st.progress(root,job['id'],f'Evaluating all {len(X)} {split} images')
        logits=[]
        with torch.no_grad():
            for begin in range(0,len(X),128):logits.append(model(torch.as_tensor(X[begin:begin+128],dtype=torch.float64)).numpy())
        probability=softmax(np.concatenate(logits),axis=1)[:,1]
        metrics=binary_metrics(Y,probability,threshold);metrics['intervals']=bootstrap_metrics(Y,probability,threshold)
        size=int(p.get('cohort_size',32))
        if not 1<=size<=128:raise ValueError('Intrinsic cohort sample must be 1..128 images')
        policy=p.get('cohort_reference','zero')
        if policy not in ['zero','same','cross']:raise ValueError('Invalid cohort reference policy')
        rng=np.random.default_rng(604);sample=np.sort(rng.choice(len(X),min(size,len(X)),replace=False));rows=[]
        identity=transform==specification()
        for k,i in enumerate(sample):
            st.progress(root,job['id'],f'Intrinsic cohort case {k+1}/{len(sample)}')
            ref=np.zeros_like(X[i]);ref_index=None
            if policy!='zero':
                candidates=np.flatnonzero((Y==Y[i]) if policy=='same' else (Y!=Y[i]));candidates=candidates[candidates!=i]
                if not len(candidates):
                    rows.append(dict(x=int(i),status='no_reference',label=int(Y[i]),group=('T' if int(probability[i]>=threshold)==Y[i] else 'F')+('P' if probability[i]>=threshold else 'N')));continue
                ref_index=int(rng.choice(candidates));ref=X[ref_index]
            v=case_analysis(model,X[i],ref,readout);w=v if identity else case_analysis(model,image(X[i],transform),image(ref,transform),readout)
            moved=transport(v['input_fields']['conditional'],transform)
            rows.append(dict(x=int(i),source_index=int(ids[i]),label=int(Y[i]),predicted=int(probability[i]>=threshold),group=('T' if int(probability[i]>=threshold)==Y[i] else 'F')+('P' if probability[i]>=threshold else 'N'),reference=ref_index,status='available',input_difference_l2=float(np.linalg.norm(X[i]-ref)),H_eq=hellinger(w['input_fields']['conditional'],moved['field']),H_inv=hellinger(w['input_fields']['conditional'],v['input_fields']['conditional']),transformed_delta=w['delta'],transform_margin_change=w['margin']-v['margin'],**brief(v)))
        usable=[row for row in rows if row['status']=='available']
        report.update(metrics=metrics,predictions=[dict(x=i,source_index=int(ids[i]),label=int(Y[i]),probability=float(probability[i]),predicted=int(probability[i]>=threshold)) for i in range(len(X))],rows=rows,groups=group_summary(usable),sampling=dict(seed=604,requested=size,selected=len(sample),available=len(usable),policy=policy,population=len(X),interval_unit='Image-level bootstrap; repeated references and unknown patient identities limit independence. Intervals on pair-derived observables are exploratory.'),context=dict({k:v for k,v in context.items() if k not in ['x','x_source_index','reference','reference_source_index','reference_type']},cohort_reference=policy,cohort_size=size))
    else:
        st.progress(root,job['id'],'Computing exact baseline fields')
        base=case_analysis(model,original,reference,readout)
        tx=image(original,transform);tr=image(reference,transform)
        current=base if transform==specification() else case_analysis(model,tx,tr,readout)
        moved=transport(base['input_fields']['conditional'],transform)
        report.update(case=current,baseline=base,images=dict(original=original.tolist(),input=tx.tolist(),reference=tr.tolist()),labels=[int(Y[x]),None if reference_type=='zero' else int(Y[r])],transformation=dict(transported=moved,H_eq=hellinger(current['input_fields']['conditional'],moved['field']),H_inv=hellinger(current['input_fields']['conditional'],base['input_fields']['conditional']),margin_change=current['margin']-base['margin'],residual=(np.asarray(current['input_fields']['conditional'])-np.asarray(moved['field'])).tolist() if current['input_fields']['conditional'] is not None and moved['field'] is not None else None),input_difference_l2=float(np.linalg.norm(tx-tr)))
        if p['kind']=='healthcare_dose':
            mode=p.get('intervention','internal');alphas=[1.,.75,.5,.25,0.];dose=[];controls=[]
            if mode=='internal':
                layer=int(p.get('layer',0));selection=p.get('selection','top');index=int(p.get('index',0))
                if not 0<=layer<len(current['layer_marginals']):raise ValueError('Invalid intervention layer')
                masses=current['layer_marginals'][layer]['conditional_masses']
                if selection=='top':
                    if masses is None:raise ValueError('Conditional measure undefined; choose a manual target')
                    index=int(np.argmax(masses))
                elif selection!='manual':raise ValueError('Invalid target selection')
                units=selected_units(model,layer,index);count=model.shapes[layer+1][0]
                rng=np.random.default_rng(605);others=np.array([i for i in range(count) if i!=index]);chosen=rng.choice(others,min(5,len(others)),replace=False)
                for target in [index]+[int(v) for v in chosen]:
                    curve=[];uu=selected_units(model,layer,target)
                    for alpha in alphas:
                        st.progress(root,job['id'],f'Attenuating layer {layer+1}, target {target}, α={alpha}')
                        v=case_analysis(model,tx,tr,readout,{str(layer):{str(i):alpha for i in uu}})
                        row=dict(alpha=alpha,output_change=v['margin']-current['margin'],contrast_change=v['delta']-current['delta'],**brief(v))
                        if target==index:row.update(fields=v['input_fields'],signed_field=v['signed_field'],signed_field_residual=v['signed_field_residual'])
                        curve.append(row)
                    if target==index:dose=curve
                    else:controls.append(dict(index=target,curve=curve))
                spec=dict(mode=mode,layer=layer,index=index,selection=selection,kind=current['layer_marginals'][layer]['kind'],units=units,policy='Same postactivation attenuation applied to both input and reference. Random controls use other targets in the same layer with the same number of execution units.',random_seed=605)
            elif mode=='region':
                box=[int(p.get(k,v)) for k,v in [('left',8),('top',8),('right',20),('bottom',20)]];left,top,right,bottom=box
                if not 0<=left<right<=28 or not 0<=top<bottom<=28:raise ValueError('Region must satisfy 0 ≤ left < right ≤ 28 and 0 ≤ top < bottom ≤ 28')
                for alpha in alphas:
                    changed=tx.copy();changed[:,top:bottom,left:right]=alpha*tx[:,top:bottom,left:right]+(1-alpha)*tr[:,top:bottom,left:right]
                    v=case_analysis(model,changed,tr,readout)
                    dose.append(dict(alpha=alpha,output_change=v['margin']-current['margin'],contrast_change=v['delta']-current['delta'],fields=v['input_fields'],signed_field=v['signed_field'],signed_field_residual=v['signed_field_residual'],image=changed.tolist(),**brief(v)))
                spec=dict(mode=mode,box=box,policy='Selected region of transformed x interpolated toward transformed reference; reference held fixed. Input perturbation, not an internal gate intervention.',conditional_region_mass=float(np.asarray(current['input_fields']['conditional'])[top:bottom,left:right].sum()) if current['conditional']['defined'] else None,signed_region_contribution=float(np.asarray(current['signed_field'])[top:bottom,left:right].sum()))
            else:raise ValueError('Invalid intervention mode')
            report.update(dose=dose,controls=controls,intervention=spec);report['context']['intervention']=spec
        elif p['kind']!='healthcare_case':raise ValueError('Unknown healthcare analysis')
    path=Path(root)/run['id']/f"healthcare_{job['id']}.json";st.atomic_json(path,report)
    with st.db(root) as db:db.execute("UPDATE jobs SET status='completed',result=?,progress='Healthcare analysis ready' WHERE id=?",(str(path),job['id']))
