"""Exact log-domain path measures on spatially unrolled feedforward execution DAGs."""
import math
import numpy as np
import torch
from torch import nn
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.linalg import eigh, orthogonal_procrustes


def group_logsum(values,indices,size):
    top=np.full(size,-np.inf);np.maximum.at(top,indices,values)
    sums=np.zeros(size);valid=np.isfinite(values)
    np.add.at(sums,indices[valid],np.exp(values[valid]-top[indices[valid]]))
    out=np.full(size,-np.inf);ok=sums>0;out[ok]=top[ok]+np.log(sums[ok]);return out

def graph(model,readout):
    sizes=[int(np.prod(s)) for s in model.shapes]+[1];offset=np.cumsum([0]+sizes)
    layers=[]
    for l,layer in enumerate(model.layers):
        w=layer.weight.detach().cpu().double().numpy();u=[];v=[];weights=[]
        if isinstance(layer,nn.Linear):
            vv,uu=np.indices(w.shape);u=uu.ravel();v=vv.ravel();weights=w.ravel()
        else:
            ch,h,width=model.shapes[l];oc,oh,ow=model.shapes[l+1]
            kh,kw=layer.kernel_size;sy,sx=layer.stride;py,px=layer.padding
            for j in range(oc):
                for y in range(oh):
                    for x in range(ow):
                        dest=(j*oh+y)*ow+x
                        for i in range(ch):
                            for dy in range(kh):
                                iy=y*sy+dy-py
                                if not 0<=iy<h:continue
                                for dx in range(kw):
                                    ix=x*sx+dx-px
                                    if 0<=ix<width:u.append((i*h+iy)*width+ix);v.append(dest);weights.append(w[j,i,dy,dx])
        layers.append((np.asarray(u,dtype=int),np.asarray(v,dtype=int),np.asarray(weights,dtype=float)))
    a=int(readout.get('a',0));b=readout.get('b');classes=sizes[-2]
    scope=readout.get('scope','selected')
    if scope not in ['selected','all']:raise ValueError('Invalid path scope')
    if not 0<=a<classes or (b is not None and (not 0<=int(b)<classes or int(b)==a)):raise ValueError('Choose valid, distinct readout classes')
    if scope=='all':layers.append((np.arange(classes),np.zeros(classes,dtype=int),np.ones(classes)))
    else:layers.append((np.array([a] if b is None else [a,int(b)]),np.array([0] if b is None else [0,0]),np.array([1.] if b is None else [1.,-1.])))
    return sizes,offset,layers

def measure(sizes,layers,start,sec):
    with np.errstate(divide='ignore'):
        logs=[2*np.log(np.abs(w))+2*np.log(np.abs(sec[l][v])) for l,(u,v,w) in enumerate(layers)]
        initial=np.log(start)
    def forward(power):
        aa=[initial*power]
        for l,(u,v,w) in enumerate(layers):aa.append(group_logsum(aa[-1][u]+power*logs[l],v,sizes[l+1]))
        return aa
    aa=forward(1);logZ=float(aa[-1][0]);F=[np.zeros(len(w)) for u,v,w in layers]
    count=np.zeros(sizes[0])
    for l,(u,v,w) in enumerate(layers):count=group_logsum(count[u],v,sizes[l+1])
    logN=float(count[0])
    if not np.isfinite(logZ):return dict(defined=False,logZ=None,H1=None,Sdef=None,eta1=None,eta2=None,log_eta1=None,log_eta2=None,log_path_count=logN),F
    back=[None]*len(sizes);back[-1]=np.zeros(sizes[-1])
    for l in range(len(layers)-1,-1,-1):
        u,v,w=layers[l];back[l]=group_logsum(logs[l]+back[l+1][v],u,sizes[l])
        F[l]=np.exp(aa[l][u]+logs[l]+back[l+1][v]-logZ)
    p0=np.exp(initial+back[0]-logZ);valid=np.isfinite(initial);expected=float(np.sum(p0[valid]*initial[valid]))
    for f,log in zip(F,logs):
        valid=np.isfinite(log);expected+=float(np.sum(f[valid]*log[valid]))
    H=float(np.clip(logZ-expected,0,logN));log_eta1=H-logN;log_eta2=2*logZ-float(forward(2)[-1][0])-logN
    return dict(defined=True,logZ=logZ,H1=H,Sdef=logN-H,eta1=math.exp(log_eta1),eta2=math.exp(log_eta2),log_eta1=log_eta1,log_eta2=log_eta2,log_path_count=logN,cut_sums=[float(f.sum()) for f in F]),F

def input_origin_field(sources, first_cut_flow, shape, defined):
    """Sum first-cut marginals at input sources, then sum image channels."""
    if not defined:return None
    mass=np.bincount(sources,weights=first_cut_flow,minlength=int(np.prod(shape)))
    return mass.reshape(shape).sum(axis=0).tolist()

def bcan(D,strength):
    n=len(D);positive=D[D>0];radii=np.geomspace(positive.min()*.95,positive.max()*1.005,36);counts=[]
    order=np.lexsort((np.arange(n),-strength))
    for radius in radii:
        color=np.full(n,-1)
        for u in order:
            forbidden=set(color[(color>=0)&(D[u]>=radius)].tolist());c=0
            while c in forbidden:c+=1
            color[u]=c
        counts.append(int(color.max()+1))
    profile=[];xi=None
    for i in range(1,len(radii)):
        if counts[i]<=n*.5<counts[i-1]:xi=float(np.exp(np.log(radii[i-1])+(n*.5-counts[i-1])/(counts[i]-counts[i-1])*np.log(radii[i]/radii[i-1])));break
    for i,r in enumerate(radii):
        lo=max(0,i-3);hi=min(len(radii),i+4);x=np.log(radii[lo:hi]);y=np.log(counts[lo:hi]);x=x-x.mean();y=y-y.mean();slope=float(x@y/(x@x));r2=float((x@y)**2/((x@x)*(y@y))) if y@y>0 else 0.
        profile.append(dict(radius=float(r),boxes=counts[i],window_start=float(radii[lo]),window_end=float(radii[hi-1]),window_points=hi-lo,box_fraction=counts[i]/n,deff=-slope,r2=r2,qualified=bool(.015<=counts[i]/n<=.85 and r2>=.9)))
    good=[p['deff'] for p in profile if p['qualified']];near=min(profile,key=lambda p:abs(np.log(p['radius']/xi))) if xi else None
    return dict(dstar=max(good) if good else None,deff=near['deff'] if near and near['qualified'] else None,xi50=xi,profile=profile)

def embed_distances(D,anchors):
    # Landmark classical MDS + out-of-sample Nyström coordinates, no force repulsion.
    A=D[:,anchors];sq=A*A;row=sq.mean(1);grand=sq.mean();B=-.5*(sq-row[:,None]-row[None,:]+grand)
    vals,vec=eigh(B,subset_by_index=[max(0,len(B)-3),len(B)-1]);order=np.argsort(vals)[::-1];vals=vals[order];vec=vec[:,order];keep=vals>max(1e-12,float(vals.max())*1e-12)
    vals=vals[keep];vec=vec[:,keep];cross=-.5*(D.T**2-(D.T**2).mean(1)[:,None]-row[None,:]+grand)
    Y=cross@vec/np.sqrt(vals)[None,:] if len(vals) else np.zeros((D.shape[1],0))
    Y=np.pad(Y,((0,0),(0,3-Y.shape[1])));pred=np.linalg.norm(Y[anchors,None,:]-Y[None,:,:],axis=2)
    stress=float(np.sqrt(np.sum((pred-D)**2)/max(1e-30,np.sum(D*D))))
    return Y,stress,vals.tolist()

def geometry(sizes,offset,layers,F,limit=500,landmarks=64,anchors=None,measurement_limit=None):
    n=int(offset[-1]);us=[];vs=[];lengths=[];strength=np.zeros(n)
    for l,((u,v,w),f) in enumerate(zip(layers,F)):
        U=u+offset[l];V=v+offset[l+1];us.extend(U);vs.extend(V);lengths.extend(1/(len(u)*f+.03));np.add.at(strength,U,f)
        if l==len(layers)-1:np.add.at(strength,V,f)
    us=np.array(us);vs=np.array(vs);lengths=np.array(lengths);A=coo_matrix((np.r_[lengths,lengths],(np.r_[us,vs],np.r_[vs,us])),shape=(n,n)).tocsr()
    full=n<=limit
    if anchors is None:
        if full:anchors=np.arange(n)
        else:
            # Every layer receives a seed; remaining landmarks maximize metric coverage.
            chosen=[int(offset[l]+sizes[l]//2) for l in range(len(sizes))]
            rows=dijkstra(A,directed=False,indices=chosen)
            if not np.isfinite(rows).all():raise ValueError('Disconnected execution graph')
            nearest=rows.min(0);nearest[chosen]=-1
            while len(chosen)<min(n,max(landmarks,len(sizes))):
                nxt=int(np.argmax(nearest));chosen.append(nxt)
                nearest=np.minimum(nearest,dijkstra(A,directed=False,indices=nxt));nearest[chosen]=-1
            anchors=np.asarray(chosen)
    D=dijkstra(A,directed=False,indices=anchors)
    if not np.isfinite(D).all():raise ValueError('Disconnected execution graph cannot be embedded by this metric')
    Y,stress,eigenvalues=embed_distances(D,anchors)
    nonanchors=np.setdiff1d(np.arange(n),anchors)
    held=nonanchors[np.linspace(0,len(nonanchors)-1,min(16,len(nonanchors)),dtype=int)] if len(nonanchors) else np.array([],dtype=int)
    validation=None
    if len(held):
        truth=dijkstra(A,directed=False,indices=held)[:,nonanchors]
        pred=np.linalg.norm(Y[held,None,:]-Y[None,nonanchors,:],axis=2)
        validation=float(np.sqrt(np.sum((pred-truth)**2)/max(1e-30,np.sum(truth**2))))
    unit=float(np.median(D[D>0]))
    budget=limit if measurement_limit is None else measurement_limit
    if n<=budget:
        response=bcan(D if full else dijkstra(A,directed=False),strength)
        response['status']='available' if response['deff'] is not None or response['dstar'] is not None else 'no_qualified_window'
        response['reason']='Computed full intrinsic distances; inspect the scale profile and fit qualification.'
    else:response=dict(deff=None,dstar=None,status='budget_exceeded',reason=f'{n} nodes exceed the full-distance measurement limit {budget}. Increase the analysis budget explicitly.')

    g=dict(mode='full' if full else 'landmark',anchor_count=len(anchors),landmark_indices=anchors.tolist(),
           coverage_by_layer=[int(np.sum((anchors>=offset[l])&(anchors<offset[l+1]))) for l in range(len(sizes))],
           stress=stress,validation_stress=validation,leading_positive_eigenvalues=eigenvalues,
           stress_scope='all distances' if full else 'landmark-to-node distances',
           validation_scope='non-landmark-to-non-landmark distances from up to 16 held-out sources',unit=unit,
           responses=response,estimator_version='bcan-1',measurement_nodes=n,measurement_limit=budget)
    return Y,g,anchors

def analyze(model,x,x0,readout,gates=None,geometry_enabled=True):
    model=model.cpu().double().eval();sizes,offset,layers=graph(model,readout)
    with torch.no_grad():
        y,hs,zs=model(torch.as_tensor(x[None],dtype=torch.float64),gates,True);y0,h0,z0=model(torch.as_tensor(x0[None],dtype=torch.float64),gates,True)
    sec=[]
    for l in range(len(zs)):
        a=zs[l].numpy().ravel();b=z0[l].numpy().ravel();dh=hs[l+1].numpy().ravel()-h0[l+1].numpy().ravel();dz=a-b
        if l==len(zs)-1:s=np.ones_like(a)
        else:
            derivative=(a>0).astype(float) if model.config['activation']=='relu' else 1-np.tanh(a)**2
            if gates and str(l) in gates:
                for idx,alpha in gates[str(l)].items():derivative[int(idx)]*=alpha
            s=np.divide(dh,dz,out=derivative,where=dz!=0)
        sec.append(s)
    sec.append(np.ones(1));dx=(x.astype(float)-x0.astype(float)).ravel();s,Fs=measure(sizes,layers,np.ones(sizes[0]),[np.ones(n) for n in sizes[1:]]);c,Fc=measure(sizes,layers,dx*dx,sec)
    signed=dx.copy();absolute=np.abs(dx)
    for l,(u,v,w) in enumerate(layers):
        if l==len(layers)-1:output_path_sums=signed.copy()
        signed=np.bincount(v,weights=signed[u]*w*sec[l][v],minlength=sizes[l+1]);absolute=np.bincount(v,weights=absolute[u]*np.abs(w*sec[l][v]),minlength=sizes[l+1])
    a=int(readout.get('a',0));b=readout.get('b');delta=float(y[0,a]-y0[0,a]-(y[0,int(b)]-y0[0,int(b)] if b is not None else 0))
    vector_delta=(y[0]-y0[0]).numpy();vector_residual=float(np.max(np.abs(output_path_sums-vector_delta)))
    selected_path_sum=float(output_path_sums[a]-(output_path_sums[int(b)] if b is not None else 0))
    out=dict(cancellation_scope='sum_of_output_amplitudes' if readout.get('scope')=='all' else 'selected_scalar_readout',scope=readout.get('scope','selected'),output_contrasts=vector_delta.tolist(),output_path_sums=output_path_sums.tolist(),vector_residual=vector_residual,contrast_l2=float(np.linalg.norm(vector_delta)),aggregate_path_sum=float(signed[0]),structural=s,conditional=c,delta=delta,path_sum=selected_path_sum,residual=abs(delta-selected_path_sum),absolute_path_sum=float(absolute[0]),cancellation=float(1-abs(signed[0])/absolute[0]) if absolute[0] else None,logits=y[0].tolist(),reference_logits=y0[0].tolist(),sizes=sizes,readout=readout,gates=gates or {},metric=dict(epsilon=.03,power=1,enrichment='number of execution edges in cut × normalized edge marginal'))
    out['input_fields']=dict(version=1,shape=list(model.shapes[0]),
        structural=input_origin_field(layers[0][0],Fs[0],model.shapes[0],s['defined']),
        conditional=input_origin_field(layers[0][0],Fc[0],model.shapes[0],c['defined']),
        definition='Input-origin marginal of the normalized complete-path measure; channels summed.',
        normalization='Unit total mass when defined; null when the partition vanishes.')
    if geometry_enabled:
        S,sg,anchors=geometry(sizes,offset,layers,Fs,model.config['geometry_limit'],model.config['landmarks'],measurement_limit=model.config.get('measurement_limit',model.config['geometry_limit']));out['structural_geometry']=sg
        if c['defined']:
            C,cg,_=geometry(sizes,offset,layers,Fc,model.config['geometry_limit'],model.config['landmarks'],anchors,measurement_limit=model.config.get('measurement_limit',model.config['geometry_limit']));C-=C.mean(0);S-=S.mean(0);rotation,_=orthogonal_procrustes(C,S);C=C@rotation;out['conditional_geometry']=cg
        else:C=S.copy();out['conditional_geometry']=dict(mode='undefined',responses=dict(status='undefined_measure',reason='All conditional path amplitudes vanish.',deff=None,dstar=None))
        scale=sg['unit'];out['positions']=[(S/scale).tolist(),(C/scale).tolist()]
        out['nodes']=[dict(layer=l,index=i,label=f'L{l}:{i}') for l,n in enumerate(sizes) for i in range(n)]
        # All execution edges are retained; no display mass is invented.
        out['edges']=[dict(u=int(u+offset[l]),v=int(v+offset[l+1]),s=float(Fs[l][k]),c=float(Fc[l][k])) for l,(uu,vv,w) in enumerate(layers) for k,(u,v) in enumerate(zip(uu,vv))]
    return out
