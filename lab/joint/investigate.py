"""Checkpoint-bound dual-head fields and bounded, explicitly scoped geometry."""
import math
import numpy as np
import torch
from .tape import Tape,Measure
from .conventions import settings,structural_mode,description
from .audit import analysis_state,measure_audit,graph_hash,digest_tensor

CONVENTIONS=dict(decomposition='finite-dag-1',structural='declared coefficient product',metric='comparison-class-enrichment-15',max_pool='Local straight-line upper-envelope occupation; structural uniform valid candidates; ties lowest index',normalization='ImageNet affine preprocessing is inside the analysed graph; raw [0,1] coordinates outside',geometry='Selected Conv1 occurrence subgraph; induced-subgraph distances, not full-network distances')

def arr(t):return t.detach().cpu().tolist()
def scalar(v):return v if math.isfinite(v) else None

def readout_tensor(out,spec):
    if spec['head']=='classification':
        a=int(spec['a']);b=spec.get('b');z=out['classification'][:,a]
        return z if b is None else z-out['classification'][:,int(b)]
    x0,y0,x1,y1=map(int,spec['region']);return out['segmentation'][:,:,y0:y1,x0:x1].mean((1,2,3))

def attributions(model,x,ref,spec,layer,steps,gates):
    from ..progress import stage
    stage("Grad-CAM and input gradients")
    capture={};xx=x.detach().requires_grad_(True);out=model(xx,gates=gates,capture=capture);score=readout_tensor(out,spec).sum()
    if layer not in capture:raise ValueError('Unknown feature channel layer')
    feature=capture[layer];gx,gf=torch.autograd.grad(score,(xx,feature),allow_unused=True);cam=None if gf is None else (feature*gf.mean((-2,-1),keepdim=True)).sum(1).relu()[0]
    grad_sum=torch.zeros_like(x)
    stage("Integrated Gradients",0,steps+1,"gradient evaluations")
    for i in range(steps+1):
        z=(ref+(x-ref)*(i/steps)).detach().requires_grad_(True)
        y=readout_tensor(model(z,gates=gates),spec).sum();g=torch.autograd.grad(y,z)[0]
        grad_sum+=g*(.5 if i in [0,steps] else 1.)
        stage("Integrated Gradients",i+1,steps+1,"gradient evaluations")
    ig=(x-ref)*grad_sum/steps
    with torch.no_grad():delta=float(readout_tensor(model(x,gates=gates),spec)-readout_tensor(model(ref,gates=gates),spec))
    return dict(IG=arr(ig[0].sum(0)),input_gradient=arr(gx[0].sum(0)),input_gradient_channels=arr(gx[0]),GradCAM=arr(cam) if cam is not None else None,GradCAM_status='available' if cam is not None else 'not applicable: selected feature has no influence on this readout',IG_sum=float(ig.sum()),IG_residual=float(ig.sum())-delta,IG_steps=steps,GradCAM_layer=layer,GradCAM_scope='native feature coordinates, endpoint x; positive Grad-CAM',gradient_scope='signed sum over RGB channels; per-channel values included')

@torch.no_grad()
def trace(model,x,ref,gates):
    tape=Tape(x,ref);capture={};heads=model(0,recorder=tape,gates=gates,capture=capture);return tape,heads,capture

def conv1_footprint(measure,channel):
    t=measure.tape;i=next(i for i,n in enumerate(t.nodes) if n.edges and n.edges[0].kind=='conv');n=t.nodes[i];e=n.edges[0];j=n.parents[0]
    if measure.logQ==-math.inf:return None
    a,sa=measure.alpha[j];b,sb=measure.beta[i];selected=torch.zeros_like(b);selected[:,channel]=b[:,channel]
    back=e.apply(selected,measure.mode,2,transpose=True)
    return (a*back*math.exp(sa+sb-measure.logQ))[0].sum(0)

def geometry_edges(measure,channel,window):
    t=measure.tape;i=next(i for i,n in enumerate(t.nodes) if n.edges and n.edges[0].kind=='conv');n=t.nodes[i];e=n.edges[0];j=n.parents[0]
    # At most 3x3 destination positions; all valid kernel occurrences into them.
    h,w=n.x.shape[-2:];cy,cx=window;xlo=max(0,cx-1);xhi=min(w,cx+2);ylo=max(0,cy-1);yhi=min(h,cy+2)
    coef=e.coeff(measure.mode,2,'absolute');a,sa=measure.alpha[j];b,sb=measure.beta[i];factor=math.exp(sa+sb-measure.logQ) if measure.logQ!=-math.inf else 0.
    edges=[]
    for yy in range(ylo,yhi):
        for xx in range(xlo,xhi):
            for local_c in range(coef.shape[1]):
                c=(channel//(coef.shape[0]//e.kw['groups']))*coef.shape[1]+local_c
                for ky in range(coef.shape[2]):
                    for kx in range(coef.shape[3]):
                        sy=yy*e.kw['stride'][0]-e.kw['padding'][0]+ky*e.kw['dilation'][0];sx=xx*e.kw['stride'][1]-e.kw['padding'][1]+kx*e.kw['dilation'][1]
                        if 0<=sy<a.shape[2] and 0<=sx<a.shape[3]:
                            mass=float(a[0,c,sy,sx]*coef[channel,local_c,ky,kx]*b[0,channel,yy,xx])*factor
                            edges.append((f'preprocess:{c}:{sy}:{sx}',f'conv1:{channel}:{yy}:{xx}',mass))
    return edges

def operation_edges(measure):
    t=measure.tape
    return [(t.nodes[j].name,n.name,measure.branch_mass(i,k)) for i,n in enumerate(t.nodes[:measure.sink+1]) for k,j in enumerate(n.parents)]

def embed_subgraph(structural,conditional,topology,scope=None,fit_hyperbolic=False,classes=None,metric_settings=None):
    from ..core.metricfit import refine
    from ..core.hyperbolic import compute
    names=list(dict.fromkeys(v for a,b,_ in structural for v in [a,b]));idx={v:i for i,v in enumerate(names)};count=len(names)
    if count>2500:return dict(status='budget_exceeded',nodes=count,scope=scope or CONVENTIONS['geometry'])
    q=settings(metric_settings)
    classes=classes if classes is not None else [b for a,b,m in structural]
    if len(classes)!=len(structural):raise ValueError('One declared comparison class required per occurrence')
    grouped={}
    for k,key in enumerate(classes):grouped.setdefault(str(key),[]).append(k)
    edges=[None]*len(structural);class_report=[]
    for key,indices in grouped.items():
        sums=[math.fsum(structural[k][2] for k in indices),math.fsum(conditional[k][2] for k in indices) if conditional is not None else 0.]
        class_report.append(dict(id=key,count=len(indices),structural_mass=sums[0],conditional_mass=sums[1] if conditional is not None else None,
            structural_status='available' if sums[0]>0 else 'zero_class_mass',conditional_status='available' if sums[1]>0 else 'zero_class_mass' if conditional is not None else 'undefined_measure'))
        for k in indices:
            a,b,mass=structural[k];cm=conditional[k][2] if conditional is not None else 0.
            if any(not math.isfinite(v) or v<0 for v in [mass,cm]):raise ValueError('Invalid edge probability')
            es=len(indices)*mass/sums[0] if sums[0]>0 else None;ec=len(indices)*cm/sums[1] if sums[1]>0 else None
            edges[k]=dict(u=idx[a],v=idx[b],s=mass,c=cm,comparison_class=key,enrichment_s=es,enrichment_c=ec,
                length_s=((es if es is not None else 0)+q['metric_epsilon'])**(-q['metric_power']),
                length_c=((ec if ec is not None else 0)+q['metric_epsilon'])**(-q['metric_power']))
    shared=dict(nodes=[dict(label=name,layer=0 if i==0 else 1,index=i) for i,name in enumerate(names)],sizes=[1,count-1,1],edges=edges,conditional=dict(defined=conditional is not None),
        metric=dict(epsilon=q['metric_epsilon'],power=q['metric_power'],explicit_edge_lengths=True,version='comparison-class-enrichment-15',
            enrichment='|K| F_e / sum_K F',classes=class_report,zero_class='Undefined enrichment; metric-only zero extension',display_reference=dict(unit=1.)),scope=scope or CONVENTIONS['geometry'])
    # Fixed topology-only display unit, independent of weights and contrast values.
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path
    u=np.array([e['u'] for e in edges]);v=np.array([e['v'] for e in edges])
    hops=shortest_path(csr_matrix((np.ones(2*len(edges)),(np.r_[u,v],np.r_[v,u])),shape=(count,count)),directed=False,unweighted=True)
    shared['metric']['display_reference']=dict(unit=float(np.median(hops[np.isfinite(hops)&(hops>0)])),convention='Median unweighted hop distance of this fixed scoped topology; distances divided by this recorded unit for fitting only')
    from ..progress import stage
    stage('Fitting independently in Euclidean 2-D and 3-D')
    design,targets=refine(shared,steps=120)
    if fit_hyperbolic:
        stage('Fitting Lorentz 3-D representation')
        shared['hyperbolic']=compute(shared,steps=150,design=design,T=targets)
    shared['projection_diagnostics']={side:{dim:dict(status=entry[k].get('quality','undefined'),stress=entry[k].get('heldout',{}).get('stress')) for dim,entry in shared['euclidean']['dimensions'].items()} for k,side in enumerate(['s','c'])}
    return dict(status='available',scope=shared['scope'],nodes=names,edges=[[e['u'],e['v']] for e in edges],shared=shared,channel=None,BCANw_status='Not computed: displayed scope is not a full scalar-network BCANw measurement')

@analysis_state
def analyze(model,x,ref,spec,layer,channel,steps=16,gates=None,geometry=False,fit_hyperbolic=False,progress=lambda s:None,source_region=None,analysis_settings=None):
    q=settings(analysis_settings)
    gates=gates or {};progress('Tracing finite endpoint contrasts');t,heads,capture=trace(model,x,ref,gates);sink=t.readout(heads,spec)
    if layer not in capture:
        if layer=='stem.relu' and capture:layer=next(iter(capture))
        else:raise ValueError('Invalid selected layer')
    selected=capture[layer]
    if not 0<=channel<t.nodes[selected].x.shape[1]:raise ValueError('Invalid selected channel')
    progress('Computing exact tensor path marginals');m=Measure(t,sink);defined=m.logQ!=-math.inf
    reconstructed,local_error=m.signed_reconstruction();delta=float(t.nodes[sink].x-t.nodes[sink].ref)
    result=dict(conventions=dict(CONVENTIONS,geometry='Operation quotient; see geometry.scope and recorded explicit lengths' if geometry=='operations' else CONVENTIONS['geometry']),readout=spec,delta=delta,signed_reconstruction=reconstructed,reconstruction_residual=abs(reconstructed-delta),largest_forward_contrast_error=local_error,logQ=scalar(m.logQ),Q=scalar(math.exp(m.logQ)) if -745<m.logQ<709 else None,conditional_status='available' if defined else 'undefined: zero total squared path amplitude',FieldA=arr(m.node_mass(0)[0].sum(0)) if defined else None,native_channel_occupation=arr(m.node_mass(selected)[0,channel]) if defined else None,layers=[dict(name=name,channels=int(t.nodes[i].x.shape[1]),height=int(t.nodes[i].x.shape[-2]),width=int(t.nodes[i].x.shape[-1])) for name,i in capture.items()],selected_layer=layer,selected_channel=channel)
    from .interpret import source_summary,region_analysis,observables
    scale=max(1.,max(float(n.x.abs().max())+float(n.ref.abs().max()) for n in t.nodes))
    if local_error>1e-7*scale or abs(reconstructed-delta)>1e-7*scale:raise ValueError('Finite-response reconstruction failed; analysis invalid')
    import hashlib
    digest=hashlib.sha256()
    for name,p in model.state_dict().items():digest.update(name.encode());digest.update(p.detach().cpu().contiguous().numpy().tobytes())
    result['provenance']=dict(parameter_hash=digest.hexdigest(),architecture_graph_hash=graph_hash(t),input_hash=digest_tensor(x),reference_hash=digest_tensor(ref),software_version='0.12.0',analysis_conventions=description(q),response_version='finite-dag-3',dtype=str(x.dtype),mode='eval',relu_equal_zero=0,maxpool_tie='lowest index',structural=q['structural_rule'],local_tolerance=1e-7*scale)
    result['parameter_occurrence_groups']=[dict(operation=n.name,parameter_identity=e.kw['parameter_identity'],mass=m.branch_mass(i,k)) for i,n in enumerate(t.nodes[:sink+1]) for k,e in enumerate(n.edges) if 'parameter_identity' in e.kw]
    result['parameter_quotient_note']='Operation totals retain stable shared weight-tensor identity; sums over repeated uses are expected multiplicities, not necessarily probabilities. Scalar coordinates remain primitive.'
    result['source_summary']=source_summary(result['FieldA'])
    result['source_region']=region_analysis(m,source_region)
    result['layer_fields']=[] if defined else None
    if defined:
        for name,i in capture.items():
            values=m.node_mass(i)
            result['layer_fields'].append(dict(name=name,shape=list(values.shape[1:]),native_absolute=arr(values[0].sum(0)) if values.ndim==4 else None,unit_absolute=arr(values.flatten()) if values.ndim!=4 else None))

    result['path_observables']=observables(m)
    result['audit']=dict(conditional=measure_audit(m))
    result['operation_occupation']=[dict(name=n.name,mass=float(m.node_mass(i).sum()),incoming=[m.branch_mass(i,k) for k in range(len(n.edges))]) for i,n in enumerate(t.nodes[:sink+1])]
    selected_mass=m.node_mass(selected)
    result['channel_gallery']=[dict(channel=c,mass=float(selected_mass[0,c].sum()),activation=arr(t.nodes[selected].x[0,c]),occupation=arr(selected_mass[0,c]) if defined else None) for c in range(min(t.nodes[selected].x.shape[1],64))]
    result['gallery_note']='First 64 channels shown; numeric channel selector supports every channel.'
    first_feature=next(iter(capture));result['FieldB']=arr(conv1_footprint(m,channel)) if defined and layer==first_feature else None
    result['FieldB_scope']='Conv1 incoming occurrence support for selected stem channel, baseline conditional mass' if layer==first_feature else 'Unavailable in raw-image coordinates for deeper channel selection; native channel occupation shown instead'
    if defined:
        progress('Computing sign-separated path diagnostics')
        p,n,s=m.signed_origin(1);p=p[0].sum(0);n=n[0].sum(0);total=p+n
        result.update(log_absolute_path_scale=s,P=arr(p*math.exp(s)) if s<700 else None,N=arr(n*math.exp(s)) if s<700 else None,P_scaled=arr(p),N_scaled=arr(n),cancellation=arr(torch.where(total>0,1-(p-n).abs()/total,torch.full_like(total,torch.nan)).nan_to_num(-1)),global_cancellation=float(1-(p.sum()-n.sum()).abs()/total.sum()) if float(total.sum())>0 else None,sign_separated_reconstruction=scalar(float(p.sum()-n.sum())*math.exp(s)) if s<700 else None)
        continuation=[None]*len(t.nodes);continuation[sink]=torch.ones_like(t.nodes[sink].x)
        for i in range(sink,0,-1):
            if continuation[i] is None:continue
            for j,e in zip(t.nodes[i].parents,t.nodes[i].edges):
                v=e.apply(continuation[i],'conditional',1,'positive',True)-e.apply(continuation[i],'conditional',1,'negative',True)
                continuation[j]=v if continuation[j] is None else continuation[j]+v
        signed=((x-ref)*continuation[0])[0].sum(0);result['signed_contribution']=arr(signed);result['signed_source_residual']=abs(float(signed.sum())-delta)
        result['cancellation_precision_note']='P and N can be enormous in deep signed networks. Their subtraction can lose the net response in floating point; use the independently propagated signed reconstruction for the output audit.'
        result['sign_subtraction_residual']=abs(result['sign_separated_reconstruction']-delta) if result['sign_separated_reconstruction'] is not None else None
        pp,nn,ss=m.signed_origin(2);result.update(A_plus=arr(pp[0].sum(0)*math.exp(ss-m.logQ)),A_minus=arr(nn[0].sum(0)*math.exp(ss-m.logQ)))
    progress('Computing structural carrier')
    st=Measure(t,sink,structural_mode(q));result['structural_FieldA']=arr(st.node_mass(0)[0].sum(0)) if st.logQ!=-math.inf else None;result['structural_logQ']=scalar(st.logQ);result['structural_status']='available' if st.logQ!=-math.inf else 'undefined: zero structural path energy'
    result['audit']['structural']=measure_audit(st)
    result['structural_observables']=observables(st)
    result['conventions'].update(description(q))
    if geometry and st.logQ==-math.inf:
        result['geometry']=dict(status='undefined_structural_measure',reason='No structural path mass; a geometry is not fabricated')
    elif geometry:
        progress('Computing declared comparison classes and scoped distances')
        top=Measure(t,sink,'topology')
        if geometry=='scalar':
            from .interpret import scalar_geometry
            result['geometry']=scalar_geometry(t,st,m,fit_hyperbolic,q)
        elif geometry=='operations':
            result['geometry']=embed_subgraph(operation_edges(st),operation_edges(m) if defined else None,operation_edges(top),scope='Operation quotient: each edge sums full-model occurrence marginals. Distances belong to the quotient, not the scalar execution metric.',fit_hyperbolic=fit_hyperbolic,metric_settings=q)
            result['geometry']['channel']=None
        else:
            conv=next(n for n in t.nodes if n.edges and n.edges[0].kind=='conv');ch=channel if layer==first_feature else 0;window=(conv.x.shape[-2]//2,conv.x.shape[-1]//2)
            se=geometry_edges(st,ch,window)
            result['geometry']=embed_subgraph(se,geometry_edges(m,ch,window) if defined else None,geometry_edges(top,ch,window),classes=['displayed Conv1 occurrence family']*len(se),scope='Conv1 induced subgraph; comparison class is the entire displayed occurrence family; not a full-network metric',fit_hyperbolic=fit_hyperbolic,metric_settings=q);result['geometry']['channel']=ch
    else:result['geometry']=dict(status='not_requested')
    result['conventions']['geometry']=result['geometry'].get('scope',result['geometry']['status'])
    del st,m,t
    progress('Computing IG and native Grad-CAM')
    result.update(attributions(model,x,ref,spec,layer,steps,gates))
    with torch.no_grad():
        out=model(x,gates=gates);out0=model(ref,gates=gates)
        result['task']=getattr(model,'config',{}).get('task','classification');reg=result['task']=='regression';result['prediction']=arr(out['classification'][0] if reg else out['classification'].softmax(1)[0]);result['reference_prediction']=arr(out0['classification'][0] if reg else out0['classification'].softmax(1)[0]);result['mask_probability']=arr(out['segmentation'].sigmoid()[0,0]) if 'segmentation' in out else None;result['reference_mask_probability']=arr(out0['segmentation'].sigmoid()[0,0]) if 'segmentation' in out0 else None
    return result
