"""Healthcare research analyses. All fields derive from exact finite-contrast paths."""
import math
import numpy as np
import torch
from scipy.special import softmax
from scipy.stats import rankdata
from .pathmath import analyze, graph, measure


def case_analysis(model,x,reference,readout,gates=None):
    """Exact input-origin signed contributions and normalized path diagnostics."""
    if readout.get('scope','selected')!='selected':
        raise ValueError('Healthcare fields require one selected scalar readout')
    result=analyze(model,x,reference,readout,gates,False)
    sizes,offset,layers=graph(model,readout)
    with torch.no_grad():
        _,hs,zs=model(torch.as_tensor(x[None],dtype=torch.float64),gates,True)
        _,h0,z0=model(torch.as_tensor(reference[None],dtype=torch.float64),gates,True)
    sec=[]
    for l in range(len(zs)):
        a=zs[l].numpy().ravel();b=z0[l].numpy().ravel()
        derivative=np.ones_like(a) if l==len(zs)-1 else ((a>0).astype(float) if model.config['activation']=='relu' else 1-np.tanh(a)**2)
        if l<len(zs)-1 and gates and str(l) in gates:
            for idx,alpha in gates[str(l)].items():derivative[int(idx)]*=alpha
        sec.append(np.divide(hs[l+1].numpy().ravel()-h0[l+1].numpy().ravel(),a-b,out=derivative,where=a!=b))
    sec.append(np.ones(1));dx=(x.astype(float)-reference.astype(float)).ravel()
    _,Fs=measure(sizes,layers,np.ones(sizes[0]),[np.ones(n) for n in sizes[1:]])
    _,Fc=measure(sizes,layers,dx*dx,sec)
    back=np.ones(1)
    for l in reversed(range(len(layers))):
        u,v,w=layers[l];back=np.bincount(u,weights=w*sec[l][v]*back[v],minlength=sizes[l])
    signed=(dx*back).reshape(x.shape).sum(axis=0)
    cross,_=measure(sizes,layers,np.abs(dx),[np.sqrt(np.abs(s)) for s in sec])
    h=None
    if result['structural']['defined'] and result['conditional']['defined'] and cross['defined']:
        logbc=cross['logZ']-.5*(result['structural']['logZ']+result['conditional']['logZ'])
        h=float(np.sqrt(max(0.,-np.expm1(min(0.,logbc)))))
    layers_out=[]
    for l in range(len(model.layers)-1):
        _,v,_=layers[l];shape=model.shapes[l+1];s=np.bincount(v,weights=Fs[l],minlength=sizes[l+1]);c=np.bincount(v,weights=Fc[l],minlength=sizes[l+1])
        def describe(mass):
            if not np.any(mass>0):return None
            positive=mass[mass>0];entropy=float(-np.sum(positive*np.log(positive)))
            return dict(entropy=entropy,effective_units=math.exp(entropy),participation_units=float(1/np.sum(mass**2)),maximum_mass=float(mass.max()))
        channels_s=s.reshape(shape).sum(axis=(1,2)) if len(shape)==3 else s
        channels_c=c.reshape(shape).sum(axis=(1,2)) if len(shape)==3 else c
        layers_out.append(dict(layer=l,shape=list(shape),kind='channel' if len(shape)==3 else 'unit',structural=describe(s),conditional=describe(c) if result['conditional']['defined'] else None,structural_masses=channels_s.tolist(),conditional_masses=channels_c.tolist() if result['conditional']['defined'] else None))
    a=readout['a'];b=readout.get('b')
    margin=lambda logits:float(logits[a]-(logits[b] if b is not None else 0))
    result.update(signed_field=signed.tolist(),signed_field_sum=float(signed.sum()),signed_field_residual=abs(float(signed.sum())-result['delta']),path_hellinger=h,layer_marginals=layers_out,probabilities=softmax(result['logits']).tolist(),reference_probabilities=softmax(result['reference_logits']).tolist(),margin=margin(result['logits']),reference_margin=margin(result['reference_logits']))
    s=np.asarray(result['input_fields']['structural']);c=result['input_fields']['conditional']
    result['field_redistribution']=(np.asarray(c)-s).tolist() if c is not None else None
    result['field_b']=dict(status='not_implemented',reason='Primitive-occurrence intervention footprint is not implemented. Signed contrast contributions are a separate quantity.')
    return result


def binary_metrics(labels,probability,threshold=.5,curves=True):
    y=np.asarray(labels,int);p=np.asarray(probability,float);n=len(y)
    if not n or set(np.unique(y))-set([0,1]) or not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError('Binary evaluation requires labels 0/1 and finite probabilities')
    pred=p>=threshold;pos=y==1;neg=~pos;tp=int(np.sum(pos&pred));tn=int(np.sum(neg&~pred));fp=int(np.sum(neg&pred));fn=int(np.sum(pos&~pred));P=int(pos.sum());N=int(neg.sum())
    sensitivity=tp/P if P else None;specificity=tn/N if N else None
    auc=float((rankdata(p)[pos].sum()-P*(P+1)/2)/(P*N)) if P and N else None
    order=np.argsort(-p,kind='stable');ys=y[order];ps=p[order];ends=np.r_[np.flatnonzero(np.diff(ps)),n-1]
    tps=np.cumsum(ys)[ends];fps=(ends+1)-tps
    recall=tps/P if P else np.zeros(len(ends));precision=tps/(ends+1)
    ap=float(np.sum(np.diff(np.r_[0.,recall])*precision)) if P else None
    clipped=np.clip(p,1e-15,1-1e-15);bins=[];ece=0.
    for i in range(10):
        mask=(p>=i/10)&((p<(i+1)/10) if i<9 else (p<=1));count=int(mask.sum())
        mean=float(p[mask].mean()) if count else None;frequency=float(y[mask].mean()) if count else None
        if count:ece+=count/n*abs(mean-frequency)
        bins.append(dict(low=i/10,high=(i+1)/10,count=count,mean_probability=mean,positive_frequency=frequency))
    result=dict(n=n,positive=P,negative=N,threshold=threshold,tp=tp,tn=tn,fp=fp,fn=fn,accuracy=(tp+tn)/n,sensitivity=sensitivity,specificity=specificity,precision=tp/(tp+fp) if tp+fp else None,balanced_accuracy=(sensitivity+specificity)/2 if P and N else None,roc_auc=auc,average_precision=ap,brier=float(np.mean((p-y)**2)),nll=float(-np.mean(y*np.log(clipped)+(1-y)*np.log(1-clipped))),ece=ece)
    if curves:result.update(calibration=bins,roc=dict(fpr=np.r_[0.,fps/N].tolist() if N else None,tpr=np.r_[0.,recall].tolist() if P else None),pr=dict(recall=np.r_[0.,recall].tolist(),precision=np.r_[1.,precision].tolist()))
    return result


def bootstrap_metrics(y,p,threshold,replicates=200):
    rng=np.random.default_rng(602);keys=['accuracy','sensitivity','specificity','roc_auc','average_precision','brier'];values={key:[] for key in keys};n=len(y)
    for _ in range(replicates):
        ids=rng.integers(n,size=n);m=binary_metrics(np.asarray(y)[ids],np.asarray(p)[ids],threshold,False)
        for key in keys:
            if m[key] is not None:values[key].append(m[key])
    return {key:dict(low=float(np.quantile(v,.025)),high=float(np.quantile(v,.975)),valid_replicates=len(v),unit='image; patient clustering unavailable') if v else None for key,v in values.items()}


def group_summary(rows):
    rng=np.random.default_rng(603);groups={}
    for group in ['TN','FP','FN','TP']:
        subset=[r for r in rows if r['group']==group];summary={}
        for key in ['delta','path_hellinger','H1','eta1','eta2','Sdef','logZ','cancellation','H_eq','H_inv','transform_margin_change']:
            values=np.array([r[key] for r in subset if r[key] is not None],float);n=len(values)
            interval=None
            if n>1:
                means=np.mean(values[rng.integers(n,size=(200,n))],axis=1);interval=np.quantile(means,[.025,.975]).tolist()
            summary[key]=dict(n=n,mean=float(values.mean()) if n else None,interval=interval)
        groups[group]=dict(n=len(subset),observables=summary)
    return groups
