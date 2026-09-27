"""Measured visual-evidence reports at frozen checkpoints."""
import numpy as np
import torch
from .core.evidence import (state,attributions,select_support,field_b,OccurrenceModel,rescaled,internal_signal,neighborhood,metric_graph,raw_transport,tensor,scalar,identities,VERSION)
from .core.healthcare import case_analysis
from .core.transforms import specification,image,transport,hellinger
from scipy.sparse.csgraph import dijkstra
from .core.inspector import signed_diagnostics, channel_catalog, connection_catalog, ChannelModel


def summary(case):
    return dict(reference_margin=case['reference_margin'],reference_probability=case['reference_probabilities'][1],delta=case['delta'],margin=case['margin'],probability=case['probabilities'][1],eta1=case['conditional']['eta1'],eta2=case['conditional']['eta2'],H1=case['conditional']['H1'],Sdef=case['conditional']['Sdef'],logZ=case['conditional']['logZ'],cancellation=case['cancellation'],path_hellinger=case['path_hellinger'],reconstruction_residual=case['residual'],signed_residual=case['signed_field_residual'])


def snapshot(model,x,r,readout,p,include_attr=True):
    st=state(model,x,r,readout);case=case_analysis(model,x,r,readout)
    support=(dict(status='undefined_measure',reason='Conditional measure vanishes; flow-ranked support is undefined',kind=p.get('support_kind','occurrence'),layers=[]) if not st['conditional']['defined'] and p.get('support_rule','flow')=='flow' else select_support(model,st,x,r,readout,p));b=field_b(model,st,support)
    result=dict(image=np.asarray(x).tolist(),reference=np.asarray(r).tolist(),field_a=case['input_fields']['conditional'],structural_a=case['input_fields']['structural'],signed_field=case['signed_field'],field_b=b,support=support,layer_marginals=case['layer_marginals'],**summary(case))
    result['diagnostics']=signed_diagnostics(model,x,r,readout,st)
    if include_attr:
        result['attribution']=attributions(model,x,r,readout,p.get('ig_steps',64))
        result['catalog']=channel_catalog(model,x,r,readout,st)
        layer=int(p.get('visual_layer',result['catalog']['default_layer']));index=int(p.get('visual_channel',0))
        result['connections']=connection_catalog(model,st,layer,index) if layer>=0 else dict(status='not_applicable',rows=[])
    return result,st


def measure_intervention(model,x,r,readout,st,support,alpha):
    changed=OccurrenceModel(model,st,support,alpha);case=case_analysis(changed,x,r,readout)
    return dict(diagnostics=signed_diagnostics(changed,x,r,readout,state(changed,x,r,readout)),alpha=alpha,field_a=case['input_fields']['conditional'],signed_field=case['signed_field'],layer_marginals=case['layer_marginals'],**summary(case))


def difference(a,b):
    if a is None or b is None:return None
    return float(np.max(np.abs(np.asarray(a)-np.asarray(b))))


def run(model,x,r,readout,p,progress,dataset=None):
    model=model.cpu().double().eval();mode=p.get('evidence_mode','compare')
    if mode not in ['compare','intervene','rescale','transform','neighborhood','cases','reverse','demo']:raise ValueError('Unknown visual evidence mode')
    progress('Computing attribution and baseline computational fields')
    base,st=snapshot(model,x,r,readout,p)
    result=dict(version=VERSION,mode=mode,settings=p,baseline=base,readout=readout,notes='Field B is baseline mass on selected convolution support, not signed attribution or a post-intervention field. Input attribution uses the same scalar readout; IG uses the same reference.')
    if mode=='compare' and base['support']['status']=='available':
        support=base['support'];alt=dict(support,kind='parameter' if support['kind']=='occurrence' else 'occurrence',rule='linked reuse comparison',layers=[])
        for entry in support['layers']:
            l=entry['layer'];pid=identities(model,l,st['layers'][l]);parameters=entry['parameters']
            if alt['kind']=='parameter':selected=parameters;edges=np.flatnonzero(np.isin(pid,parameters))
            else:
                edges=np.array([int(np.flatnonzero(pid==i)[np.argmax(st['fc'][l][pid==i])]) for i in parameters]);selected=edges.tolist()
            alt['layers'].append(dict(layer=l,selected_ids=selected,occurrences=edges.tolist(),parameters=parameters,occurrence_count=len(edges),parameter_count=len(parameters),cut_mass=float(st['fc'][l][edges].sum())))
        result['reuse_comparison']=dict(support=alt,field_b=field_b(model,st,alt),policy='Same associated parameter identities. Occurrence selection unfolds to all reuses; parameter selection is compared with the largest-mass occurrence per selected identity. This illustrates scope, not equal-size interventions.')
    alphas=[1.,.75,.5,.25,0.]
    if mode=='intervene':
        intervention=p.get('evidence_intervention','primitive');curves=[]
        if intervention=='primitive':
            if base['support']['status']!='available':raise ValueError('Primitive convolution intervention requires a CNN; use pixel comparisons or existing unit gates for an MLP')
            supports=[('Selected support',base['support'])]
            for seed in [809,810,811]:
                matched=dict(base['support'],rule='random',seed=seed,layers=[])
                for entry in base['support']['layers']:
                    control=select_support(model,st,x,r,readout,{**p,'support_layers':str(entry['layer']),'support_budget':len(entry['selected_ids']),'support_rule':'random','seed':seed})
                    matched['layers'].extend(control['layers'])
                supports.append((f'Random support {seed}',matched))
            for name,support in supports:
                frames=[]
                for alpha in alphas:
                    progress(f'{name}: attenuation {alpha}')
                    frames.append(measure_intervention(model,x,r,readout,st,support,alpha))
                curves.append(dict(name=name,support=support,field_b=field_b(model,st,support),frames=frames))
            result['policy']='Same multiplicative edge factor for x and reference, applied before activation. Frozen support per curve. Controls match selected identity count per layer; parameter controls can affect different occurrence counts, which are reported. Intervention weights enter the counterfactual path calculation; baseline Field B stays fixed.'
        elif intervention=='channel':
            layer=int(p.get('visual_layer',base['catalog']['default_layer']));index=int(p.get('visual_channel',0))
            # One channel/unit per support; controls retain the same layer and cardinality.
            count=model.shapes[layer+1][0] if 0<=layer<len(model.layers)-1 else 0
            if not 0<=index<count:raise ValueError('Select a valid hidden channel/unit')
            rng=np.random.default_rng(909);controls=rng.choice(count,min(3,count),replace=False)
            selections=[('Selected channel/unit',index)]+[(f'Random channel/unit {j+1}',int(k)) for j,k in enumerate(controls)]
            for name,k in selections:
                frames=[]
                for alpha in alphas:
                    progress(f'{name}: attenuation {alpha}')
                    changed=ChannelModel(model,layer,k,alpha);case=case_analysis(changed,x,r,readout)
                    ds=signed_diagnostics(changed,x,r,readout,state(changed,x,r,readout))
                    frames.append(dict(alpha=alpha,field_a=case['input_fields']['conditional'],signed_field=case['signed_field'],diagnostics=ds,**summary(case)))
                tile=base['catalog']['layers'][layer]['channels'][k]
                curves.append(dict(name=name,channel_support=dict(layer=layer,index=k,kind='postactivation channel/unit',baseline_mass=tile['mass']),channel_footprint=tile['occupation'],frames=frames))
            result['policy']='One postactivation channel/unit, same layer and channel count for random controls. Gates apply to both endpoints. Channel occupation footprint is native node mass, not primitive Field B. Controls may overlap selected channel; this is an exploratory comparison.'
        elif intervention=='pixels':
            k=int(p.get('pixel_budget',32))
            if not 1<=k<=784:raise ValueError('Pixel budget must be 1..784')
            scores=[('Field A',base['field_a']),('Absolute IG',np.abs(base['attribution']['integrated_gradients'])),('Absolute input gradient',np.abs(base['attribution']['input_gradient']))]
            rng=np.random.default_rng(812)
            for name,a in scores+[(f'Random pixels {j+1}',None) for j in range(3)]:
                if a is None and name=='Field A':continue
                selected=rng.choice(784,k,replace=False) if a is None else np.lexsort((np.arange(784),-np.asarray(a).ravel()))[:k]
                mask=np.zeros(784);mask[selected]=1;mask=mask.reshape(28,28);frames=[]
                for alpha in alphas:
                    progress(f'{name}: pixel attenuation {alpha}')
                    changed=r+(1-mask[None]+alpha*mask[None])*(x-r);case=case_analysis(model,changed,r,readout)
                    frames.append(dict(diagnostics=signed_diagnostics(model,changed,r,readout,state(model,changed,r,readout)),alpha=alpha,image=changed.tolist(),field_a=case['input_fields']['conditional'],signed_field=case['signed_field'],**summary(case)))
                curves.append(dict(name=name,pixel_mask=mask.tolist(),pixel_count=k,frames=frames))
            result['policy']='Equal number of individually ranked pixels, not contiguous regions; x interpolates toward the fixed reference. All rankings and masks are frozen before intervention. Random seed 812. No inference about clinical lesion removal.'
        else:raise ValueError('Invalid evidence intervention')
        result.update(curves=curves,intervention=intervention)
    elif mode=='rescale':
        layer=int(p.get('rescale_layer',0));index=int(p.get('rescale_index',0));frames=[]
        factors=[.1,.31622776601683794,1.,3.1622776601683795,10.]
        original_signal=internal_signal(model,x,readout,layer,index) if 0<=layer<len(model.layers)-1 and 0<=index<model.shapes[layer+1][0] else None
        if original_signal is None:raise ValueError('Choose an existing hidden layer and channel/unit')
        baseline_row=dijkstra(metric_graph(st,'fs'),directed=False,indices=0)
        if not np.isfinite(baseline_row).all():raise ValueError('Rescaling metric audit requires a connected execution graph; this architecture leaves disconnected input nodes')
        baseline_c=dijkstra(metric_graph(st,'fc'),directed=False,indices=0) if st['conditional']['defined'] else None
        reselect=p.get('rescale_support','fixed')=='reselect'
        for factor in factors:
            progress(f'Compensated ReLU rescaling {factor:g}')
            m=rescaled(model,layer,index,factor);snap,t=snapshot(m,x,r,readout,p)
            if not reselect:snap['support']=base['support'];snap['field_b']=field_b(m,t,base['support'])
            cam_layer=int(p.get('visual_layer',base['catalog']['default_layer']))
            cam_change=difference(snap['catalog']['layers'][cam_layer]['gradcam'],base['catalog']['layers'][cam_layer]['gradcam']) if cam_layer>=0 else None
            snap.update(factor=factor,signal=internal_signal(m,x,readout,layer,index),audit=dict(gradcam_max_change=cam_change,margin_change=snap['margin']-base['margin'],delta_change=snap['delta']-base['delta'],input_gradient_max_change=difference(snap['attribution']['input_gradient'],base['attribution']['input_gradient']),ig_max_change=difference(snap['attribution']['integrated_gradients'],base['attribution']['integrated_gradients']),field_a_max_change=difference(snap['field_a'],base['field_a']),field_b_max_change=difference(snap['field_b']['image'],base['field_b']['image']),structural_marginal_max_change=max(difference(a,b) for a,b in zip(st['fs'],t['fs'])),conditional_marginal_max_change=max(difference(a,b) for a,b in zip(st['fc'],t['fc'])) if st['conditional']['defined'] else None,structural_source_distance_max_change=difference(baseline_row,dijkstra(metric_graph(t,'fs'),directed=False,indices=0)),conditional_source_distance_max_change=difference(baseline_c,dijkstra(metric_graph(t,'fc'),directed=False,indices=0)) if baseline_c is not None else None))
            frames.append(snap)
        result.update(frames=frames,original_signal=original_signal,policy='Frozen model copies, positive compensated channel/unit scaling, biases included. Same input coordinates and reference. Numeric distance audit uses exact shortest paths from input node 0; it is not an all-pairs comparison. Fixed support and reselected support are explicitly distinguished.')
    elif mode=='transform':
        count=int(p.get('sweep_frames',5));parameter=p.get('sweep_parameter','dx');start=float(p.get('sweep_from',0));end=float(p.get('sweep_to',3))
        if not 2<=count<=21 or parameter not in ['dx','dy','angle','scale'] or not np.isfinite([start,end]).all():raise ValueError('Invalid transformation sweep')
        frames=[]
        for value in np.linspace(start,end,count):
            t=specification({parameter:value});progress(f'Transformation {parameter}={value:g}')
            tx=image(x,t);tr=image(r,t);snap,ts=snapshot(model,tx,tr,readout,p,False)
            moved=transport(base['field_a'],t);bm=raw_transport(base['field_b']['image'],t)
            fixed=field_b(model,ts,base['support'])
            snap.update(transform=t,transported_a=moved,transported_b=bm,fixed_identity_b=fixed,H_eq=hellinger(snap['field_a'],moved['field']),margin_change=snap['margin']-base['margin'],residual=(np.asarray(snap['field_a'])-moved['field']).tolist() if snap['field_a'] is not None and moved['field'] is not None else None)
            frames.append(snap)
        result.update(frames=frames,policy='Transported baseline Field B is only a spatial footprint transport. Recomputed fixed-identity support and reselected support are separate outputs; neither claims an exact rotated occurrence correspondence. Images use bilinear zero padding; transported mass loss is reported.')
    elif mode=='neighborhood':
        result['neighborhood']=neighborhood(model,st,int(p.get('metric_node',0)))
        if base['support']['status']=='available':
            result['intervened']=measure_intervention(model,x,r,readout,st,base['support'],0.)
            model2=OccurrenceModel(model,st,base['support'],0.)
            result['intervened_neighborhood']=neighborhood(model2,state(model2,x,r,readout),int(p.get('metric_node',0)))
        result['policy']='Exact graph shortest paths; same intrinsic length rule, no embedding. Node union includes nearest structural and conditional neighbors. Native and intervened neighbor unions can differ; inspect IDs. Proximity alone does not establish causal influence.'
    elif mode=='cases':
        if dataset is None:raise ValueError('Case dataset unavailable')
        ids=list(dict.fromkeys(int(i) for i in p.get('case_indices',[0,1])))
        if not 2<=len(ids)<=8 or any(i<0 or i>=len(dataset) for i in ids):raise ValueError('Select 2..8 valid case indices')
        rows=[]
        for i in ids:
            progress(f'Comparative case {i}')
            snap,ss=snapshot(model,dataset[i],r,readout,p);snap['index']=i
            if snap['support']['status']=='available':snap['dose']=[measure_intervention(model,dataset[i],r,readout,ss,snap['support'],a) for a in alphas]
            else:
                # MLP comparison uses equal-size pixel masks selected by Field A.
                k=min(784,max(1,int(p.get('pixel_budget',32))));score=snap['field_a']
                if score is not None:
                    mask=np.zeros(784);mask[np.lexsort((np.arange(784),-np.asarray(score).ravel()))[:k]]=1;mask=mask.reshape(28,28)
                    snap['dose']=[dict(alpha=a,**summary(case_analysis(model,r+(1-mask[None]+a*mask[None])*(dataset[i]-r),r,readout))) for a in alphas]
            rows.append(snap)
        pairs=[]
        for i in range(len(rows)):
            for j in range(i+1,len(rows)):
                a=np.abs(rows[i]['attribution']['integrated_gradients']).ravel();b=np.abs(rows[j]['attribution']['integrated_gradients']).ravel();den=np.linalg.norm(a)*np.linalg.norm(b)
                pairs.append(dict(a=rows[i]['index'],b=rows[j]['index'],absolute_ig_cosine=float(a@b/den) if den else None,margin_gap=abs(rows[i]['margin']-rows[j]['margin']),eta1_gap=abs(rows[i]['eta1']-rows[j]['eta1']) if rows[i]['eta1'] is not None and rows[j]['eta1'] is not None else None))
        pairs.sort(key=lambda q:-(q['absolute_ig_cosine'] if q['absolute_ig_cosine'] is not None else -1))
        result.update(cases=rows,pairs=pairs,policy='User-specified cases, same checkpoint and reference, no hidden search. All candidate pairs shown ranked by cosine similarity of absolute IG. CNN dose uses selected primitive support with the same per-layer identity budget; MLP dose uses equal-count Field A pixels. Similarity and vulnerability differences are observations, not guaranteed outcomes.')
    if mode=='reverse':
        reverse,_=snapshot(model,r,x,readout,p)
        reverse['orientation']='reference → input reversed';base['orientation']='original contrast'
        result.update(frames=[base,reverse],policy='Same network and endpoint images. Reversing the contrast preserves Field A and reverses signed IG and Δg. Absolute IG is also preserved; this is orientation invariance, not attribution failure.',reverse_audit=dict(field_a_max_change=difference(base['field_a'],reverse['field_a']),ig_antisymmetry_error=difference(base['attribution']['integrated_gradients'],-np.asarray(reverse['attribution']['integrated_gradients']))))
    if mode=='demo':
        result['demonstration']=cancellation_demo()
        result['policy']='Constructed two-input, three-hidden-unit ReLU example. Separate from the selected medical checkpoint. Readout signs are changed explicitly; these are different functions.'
    return result


def cancellation_demo():
    from torch import nn
    from .core.network import Network
    from .config import validate
    m=Network(validate(dict(model='mlp',widths=[3],activation='relu',bias=False))).double()
    m.layers=nn.ModuleList([nn.Linear(2,3,bias=False,dtype=torch.float64),nn.Linear(3,2,bias=False,dtype=torch.float64)])
    m.shapes=[(1,1,2),(3,),(2,)];x=np.array([[[1.,1.]]]);r=np.zeros_like(x);readout=dict(scope='selected',a=1,b=0)
    with torch.no_grad():m.layers[0].weight.copy_(torch.tensor([[1.,0.],[1.,0.],[0.,1.]]));m.layers[1].weight.zero_()
    rows=[]
    for sign in [1.,-1.]:
        with torch.no_grad():m.layers[1].weight[1].copy_(torch.tensor([1.,sign,1.]))
        snap,ss=snapshot(m,x,r,readout,dict(ig_steps=64));snap['name']='Constructive' if sign==1 else 'Cancelling';snap['curves']=[]
        for name,indices in [('Positive route',[0]),('Second route',[1]),('Both routes',[0,1])]:
            values=[]
            for alpha in [1.,.75,.5,.25,0.]:
                changed=m
                for i in indices:changed=ChannelModel(changed,0,i,alpha)
                with torch.no_grad():delta=float((scalar(changed(tensor(x)),readout)-scalar(changed(tensor(r)),readout))[0])
                values.append(dict(alpha=alpha,delta=delta))
            snap['curves'].append(dict(name=name,frames=values))
        rows.append(snap)
    return dict(rows=rows,definition='h1=ReLU(x1), h2=ReLU(x1), h3=ReLU(x2); g=h1 ± h2+h3; x=(1,1), reference=(0,0). Nonzero path amplitudes (1,±1,1).')
