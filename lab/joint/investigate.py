"""Checkpoint-bound dual-head fields and bounded, explicitly scoped geometry."""
import math
import numpy as np
import torch
from .tape import Tape,Measure

CONVENTIONS=dict(decomposition='finite-dag-1',structural='raw-affine-carrier-1',metric='topology-reference-1 (experimental)',max_pool='Local straight-line upper-envelope occupation; structural uniform valid candidates; ties lowest index',normalization='ImageNet affine preprocessing is inside the analysed graph; raw [0,1] coordinates outside',geometry='Selected Conv1 occurrence subgraph; induced-subgraph distances, not full-network distances')

def arr(t):return t.detach().cpu().tolist()
def scalar(v):return v if math.isfinite(v) else None

def readout_tensor(out,spec):
    if spec['head']=='classification':
        a=int(spec['a']);b=spec.get('b');z=out['classification'][:,a]
        return z if b is None else z-out['classification'][:,int(b)]
    x0,y0,x1,y1=map(int,spec['region']);return out['segmentation'][:,:,y0:y1,x0:x1].mean((1,2,3))

def attributions(model,x,ref,spec,layer,steps,gates):
    capture={};xx=x.detach().requires_grad_(True);out=model(xx,gates=gates,capture=capture);score=readout_tensor(out,spec).sum()
    if layer not in capture:raise ValueError('Unknown feature channel layer')
    feature=capture[layer];gx,gf=torch.autograd.grad(score,(xx,feature));cam=(feature*gf.mean((-2,-1),keepdim=True)).sum(1).relu()[0]
    grad_sum=torch.zeros_like(x)
    for i in range(steps+1):
        z=(ref+(x-ref)*(i/steps)).detach().requires_grad_(True)
        y=readout_tensor(model(z,gates=gates),spec).sum();g=torch.autograd.grad(y,z)[0]
        grad_sum+=g*(.5 if i in [0,steps] else 1.)
    ig=(x-ref)*grad_sum/steps
    with torch.no_grad():delta=float(readout_tensor(model(x,gates=gates),spec)-readout_tensor(model(ref,gates=gates),spec))
    return dict(IG=arr(ig[0].sum(0)),input_gradient=arr(gx[0].sum(0)),input_gradient_channels=arr(gx[0]),GradCAM=arr(cam),IG_sum=float(ig.sum()),IG_residual=float(ig.sum())-delta,IG_steps=steps,GradCAM_layer=layer,GradCAM_scope='native feature coordinates, endpoint x; positive Grad-CAM',gradient_scope='signed sum over RGB channels; per-channel values included')

@torch.no_grad()
def trace(model,x,ref,gates):
    tape=Tape(x,ref);capture={};heads=model(0,recorder=tape,gates=gates,capture=capture);return tape,heads,capture

def conv1_footprint(measure,channel):
    t=measure.tape;i=next(i for i,n in enumerate(t.nodes) if n.name=='stem.conv');n=t.nodes[i];e=n.edges[0];j=n.parents[0]
    if measure.logQ==-math.inf:return None
    a,sa=measure.alpha[j];b,sb=measure.beta[i];selected=torch.zeros_like(b);selected[:,channel]=b[:,channel]
    back=e.apply(selected,measure.mode,2,transpose=True)
    return (a*back*math.exp(sa+sb-measure.logQ))[0].sum(0)

def geometry_edges(measure,channel,window):
    t=measure.tape;i=next(i for i,n in enumerate(t.nodes) if n.name=='stem.conv');n=t.nodes[i];e=n.edges[0];j=n.parents[0]
    # At most 3x3 destination positions; all valid kernel occurrences into them.
    h,w=n.x.shape[-2:];cy,cx=window;xlo=max(0,cx-1);xhi=min(w,cx+2);ylo=max(0,cy-1);yhi=min(h,cy+2)
    coef=e.coeff(measure.mode,2,'absolute');a,sa=measure.alpha[j];b,sb=measure.beta[i];factor=math.exp(sa+sb-measure.logQ) if measure.logQ!=-math.inf else 0.
    edges=[]
    for yy in range(ylo,yhi):
        for xx in range(xlo,xhi):
            for c in range(a.shape[1]):
                for ky in range(coef.shape[2]):
                    for kx in range(coef.shape[3]):
                        sy=yy*e.kw['stride'][0]-e.kw['padding'][0]+ky;sx=xx*e.kw['stride'][1]-e.kw['padding'][1]+kx
                        if 0<=sy<a.shape[2] and 0<=sx<a.shape[3]:
                            mass=float(a[0,c,sy,sx]*coef[channel,c,ky,kx]*b[0,channel,yy,xx])*factor
                            edges.append((f'input:{c}:{sy}:{sx}',f'conv1:{channel}:{yy}:{xx}',mass))
    return edges

def embed_subgraph(structural,conditional,topology):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path,connected_components
    names=sorted({v for a,b,_ in structural for v in [a,b]});idx={v:i for i,v in enumerate(names)};count=len(names)
    if count>800:return dict(status='budget_exceeded',nodes=count,scope=CONVENTIONS['geometry'])
    result=dict(status='available',scope=CONVENTIONS['geometry'],convention=CONVENTIONS['metric'],nodes=names,edges=[[idx[a],idx[b]] for a,b,_ in structural],epsilon=.03,dimensions={})
    for label,rows in [('structural',structural),('conditional',conditional)]:
        if label=='conditional' and conditional is None:result[label]=dict(status='undefined_conditional_measure');continue
        mass=np.array([v[2] for v in rows]);ref=np.array([v[2] for v in topology]);length=1/(np.divide(mass,ref,out=np.zeros_like(mass),where=ref>0)+.03)
        a,b=np.array(result['edges']).T;graph=csr_matrix((np.r_[length,length],(np.r_[a,b],np.r_[b,a])),shape=(count,count));D=shortest_path(graph,directed=False)
        if not np.isfinite(D).all():result[label]=dict(status='disconnected',components=int(connected_components(graph)[0]));continue
        B=-.5*(D**2-(D**2).mean(0)[None,:]-(D**2).mean(1)[:,None]+(D**2).mean());val,vec=np.linalg.eigh(B);order=np.argsort(val)[::-1];coords={}
        for dim in [2,3]:
            keep=order[:dim];X=vec[:,keep]*np.sqrt(np.maximum(val[keep],0))
            # Independent stress refinement via a compact deterministic full-pair solver.
            xt=torch.tensor(X,dtype=torch.float64,requires_grad=True);target=torch.tensor(D,dtype=torch.float64);opt=torch.optim.Adam([xt],lr=max(float(np.median(D[D>0]))*.01,.001))
            with torch.enable_grad():
                for _ in range(100):
                    opt.zero_grad();dist=torch.cdist(xt,xt);objective=((dist-target).square()/(target.square()+float(np.median(D[D>0]))**2*.01)).mean();objective.backward();opt.step()
            X=xt.detach().numpy();dist=np.linalg.norm(X[:,None]-X[None,:],axis=2);stress=float(np.linalg.norm(dist-D)/np.linalg.norm(D));coords[str(dim)]=dict(positions=X.tolist(),relative_stress=stress,iterations=100,fit='full pairs, relative-scale balanced stress; completion does not imply exact metric preservation')
        result[label]=dict(status='available',mass=mass.tolist(),edge_lengths=length.tolist(),dimensions=coords,distances=D.tolist(),BCANw_status='not computed: this is a selected induced subgraph, not a full-network BCANw result')
    if result.get('conditional',{}).get('status')=='available':
        from scipy.linalg import orthogonal_procrustes
        for dim in ['2','3']:
            A=np.array(result['structural']['dimensions'][dim]['positions']);B=np.array(result['conditional']['dimensions'][dim]['positions']);A-=A.mean(0);B-=B.mean(0);R,_=orthogonal_procrustes(B,A);B=B@R
            result['structural']['dimensions'][dim]['positions']=A.tolist();result['conditional']['dimensions'][dim]['positions']=B.tolist()
    return result

def analyze(model,x,ref,spec,layer,channel,steps=16,gates=None,geometry=False,progress=lambda s:None):
    gates=gates or {};progress('Tracing finite endpoint contrasts');t,heads,capture=trace(model,x,ref,gates);sink=t.readout(heads,spec)
    if layer not in capture:raise ValueError('Invalid selected layer')
    selected=capture[layer]
    if not 0<=channel<t.nodes[selected].x.shape[1]:raise ValueError('Invalid selected channel')
    progress('Computing exact tensor path marginals');m=Measure(t,sink);defined=m.logQ!=-math.inf
    reconstructed,local_error=m.signed_reconstruction();delta=float(t.nodes[sink].x-t.nodes[sink].ref)
    result=dict(conventions=CONVENTIONS,readout=spec,delta=delta,signed_reconstruction=reconstructed,reconstruction_residual=abs(reconstructed-delta),largest_forward_contrast_error=local_error,logQ=scalar(m.logQ),Q=scalar(math.exp(m.logQ)) if -745<m.logQ<709 else None,conditional_status='available' if defined else 'undefined: zero total squared path amplitude',FieldA=arr(m.node_mass(0)[0].sum(0)) if defined else None,native_channel_occupation=arr(m.node_mass(selected)[0,channel]) if defined else None,layers=[dict(name=name,channels=int(t.nodes[i].x.shape[1]),height=int(t.nodes[i].x.shape[-2]),width=int(t.nodes[i].x.shape[-1])) for name,i in capture.items()],selected_layer=layer,selected_channel=channel)
    result['operation_occupation']=[dict(name=n.name,mass=float(m.node_mass(i).sum()),incoming=[m.branch_mass(i,k) for k in range(len(n.edges))]) for i,n in enumerate(t.nodes[:sink+1])]
    selected_mass=m.node_mass(selected)
    result['channel_gallery']=[dict(channel=c,mass=float(selected_mass[0,c].sum()),activation=arr(t.nodes[selected].x[0,c]),occupation=arr(selected_mass[0,c]) if defined else None) for c in range(min(t.nodes[selected].x.shape[1],64))]
    result['gallery_note']='First 64 channels shown; numeric channel selector supports every channel.'
    result['FieldB']=arr(conv1_footprint(m,channel)) if defined and layer=='stem.relu' else None
    result['FieldB_scope']='Conv1 incoming occurrence support for selected stem channel, baseline conditional mass' if layer=='stem.relu' else 'Unavailable in raw-image coordinates for deeper channel selection; native channel occupation shown instead'
    if defined:
        progress('Computing sign-separated path diagnostics')
        p,n,s=m.signed_origin(1);p=p[0].sum(0);n=n[0].sum(0);total=p+n
        result.update(log_absolute_path_scale=s,P=arr(p*math.exp(s)) if s<700 else None,N=arr(n*math.exp(s)) if s<700 else None,P_scaled=arr(p),N_scaled=arr(n),cancellation=arr(torch.where(total>0,1-(p-n).abs()/total,torch.full_like(total,torch.nan)).nan_to_num(-1)),global_cancellation=float(1-(p.sum()-n.sum()).abs()/total.sum()) if float(total.sum())>0 else None,sign_separated_reconstruction=scalar(float(p.sum()-n.sum())*math.exp(s)) if s<700 else None)
        result['cancellation_precision_note']='P and N can be enormous in deep signed networks. Their subtraction can lose the net response in floating point; use the independently propagated signed reconstruction for the output audit.'
        result['sign_subtraction_residual']=abs(result['sign_separated_reconstruction']-delta) if result['sign_separated_reconstruction'] is not None else None
        pp,nn,ss=m.signed_origin(2);result.update(A_plus=arr(pp[0].sum(0)*math.exp(ss-m.logQ)),A_minus=arr(nn[0].sum(0)*math.exp(ss-m.logQ)))
    progress('Computing structural carrier')
    st=Measure(t,sink,'structural');result['structural_FieldA']=arr(st.node_mass(0)[0].sum(0));result['structural_logQ']=scalar(st.logQ);result['structural_status']='available' if st.logQ!=-math.inf else 'undefined: zero structural path energy'
    if geometry:
        progress('Computing topology reference and selected-subgraph distances')
        top=Measure(t,sink,'topology');conv=next(n for n in t.nodes if n.name=='stem.conv');ch=channel if layer=='stem.relu' else 0;window=(conv.x.shape[-2]//2,conv.x.shape[-1]//2)
        result['geometry']=embed_subgraph(geometry_edges(st,ch,window),geometry_edges(m,ch,window) if defined else None,geometry_edges(top,ch,window));result['geometry']['channel']=ch
    else:result['geometry']=dict(status='not_requested')
    del st,m,t
    progress('Computing IG and native Grad-CAM')
    result.update(attributions(model,x,ref,spec,layer,steps,gates))
    with torch.no_grad():
        out=model(x,gates=gates);out0=model(ref,gates=gates)
        result['prediction']=arr(out['classification'].softmax(1)[0]);result['reference_prediction']=arr(out0['classification'].softmax(1)[0]);result['mask_probability']=arr(out['segmentation'].sigmoid()[0,0]);result['reference_mask_probability']=arr(out0['segmentation'].sigmoid()[0,0])
    return result
