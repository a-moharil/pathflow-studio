"""Experimental distance fitting in Lorentz coordinates; intrinsic metric stays external."""
import math
import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.linalg import orthogonal_procrustes


def metric_graph(data, side):
    n=len(data['nodes']);counts=np.zeros(len(data['sizes']),int)
    layer=np.array([q['layer'] for q in data['nodes']])
    u=np.array([e['u'] for e in data['edges']]);v=np.array([e['v'] for e in data['edges']])
    np.add.at(counts,layer[u],1)
    mass=np.array([e[side] for e in data['edges']])
    lengths=1/(counts[layer[u]]*mass+data['metric']['epsilon'])
    return coo_matrix((np.r_[lengths,lengths],(np.r_[u,v],np.r_[v,u])),shape=(n,n)).tocsr()


def lorentz(space,k):
    return torch.cat((torch.sqrt(1/k+(space*space).sum(-1,keepdim=True)),space),-1)


def distances(z,i,j,k):
    inner=z[i,0]*z[j,0]-(z[i,1:]*z[j,1:]).sum(-1)
    return torch.acosh(torch.clamp(k*inner,min=1+1e-12))/math.sqrt(k)


def move_origin(z,anchor,k):
    x=z*math.sqrt(k);a=anchor*math.sqrt(k);dot=x[:,1:]@a[1:]
    return np.column_stack((a[0]*x[:,0]-dot,x[:,1:]+(dot/(a[0]+1)-x[:,0])[:,None]*a[1:]))/math.sqrt(k)


def ball(z,k):
    return z[:,1:]/(z[:,0,None]+1/math.sqrt(k))


def diagnostics(pred,truth):
    rel=np.abs(pred-truth)/np.maximum(truth,1e-12)
    bins=np.quantile(truth,[0,.25,.5,.75,1]);scales=[]
    for l in range(4):
        take=(truth>=bins[l]) & (truth<=bins[l+1] if l==3 else truth<bins[l+1])
        scales.append(dict(low=float(bins[l]),high=float(bins[l+1]),count=int(take.sum()),median_relative_error=float(np.median(rel[take])) if take.any() else None))
    return dict(stress=float(np.linalg.norm(pred-truth)/max(np.linalg.norm(truth),1e-12)),median_relative_error=float(np.median(rel)),p95_relative_error=float(np.quantile(rel,.95)),by_scale=scales)


def fit(n,dim,k,pairs,truth,initial,steps):
    initial=np.array(initial[:,:dim],copy=True)
    norms=np.linalg.norm(initial,axis=1,keepdims=True)
    initial*=np.minimum(1,(math.sinh(6)/math.sqrt(k))/np.maximum(norms,1e-12))
    space=torch.tensor(initial,dtype=torch.float64,requires_grad=True)
    opt=torch.optim.Adam([space],lr=.025);ij=torch.as_tensor(pairs,dtype=torch.long);target=torch.tensor(truth,dtype=torch.float64)
    denominator=(target*target).mean().clamp_min(1e-20);best=float('inf');bestz=None;history=[]
    for step in range(steps+1):
        opt.zero_grad();z=lorentz(space,k);pred=distances(z,ij[:,0],ij[:,1],k)
        loss=((pred-target)**2).mean()/denominator
        if not torch.isfinite(loss):raise ValueError('Nonfinite Lorentz objective; try lower curvature')
        if loss.item()<best:best=loss.item();bestz=z.detach().numpy().copy()
        if step%25==0 or step==steps:history.append(dict(step=step,stress=float(math.sqrt(loss.item()))))
        if step==steps:break
        loss.backward();torch.nn.utils.clip_grad_norm_([space],10.);opt.step()
        with torch.no_grad():
            # Explicit numerical safety domain, reported in metadata.
            norms=space.norm(dim=1,keepdim=True);space.mul_(torch.clamp((math.sinh(6)/math.sqrt(k))/norms.clamp_min(1e-12),max=1))
    return bestz,history


def compute(data,k=1.,steps=150):
    n=len(data['nodes'])
    if n>4000:return dict(status='budget_exceeded',reason='Experimental fitting is limited to 4000 nodes; intrinsic/Euclidean results remain available.')
    unit=data['metric']['display_reference']['unit'];rng=np.random.default_rng(1024)
    sources=np.sort(rng.choice(n,min(48,n),replace=False));train_sources=sources[:max(1,len(sources)*3//4)];held=sources[len(train_sources):]
    # Train and validation use disjoint unordered pairs.
    pairs=np.array([(int(i),j) for i in train_sources for j in range(n) if j!=i and j not in held and (j not in train_sources or i<j)],int)
    if len(pairs)>16000:pairs=pairs[rng.choice(len(pairs),16000,replace=False)]
    validation=np.array([(int(i),j) for i in held for j in range(n) if j!=i and (j not in held or i<j)],int)
    if len(validation)>8000:validation=validation[rng.choice(len(validation),8000,replace=False)]
    rows={int(s):r for r,s in enumerate(sources)}
    result=dict(status='available',curvature=-k,unit=unit,steps=steps,seed=1024,training_pairs=len(pairs),validation_pairs=len(validation),alignment='Readout sink at origin; orthogonal Lorentz spatial alignment, no scale fitting',numerical_radius_limit=6.,dimensions={})
    D={side:dijkstra(metric_graph(data,side),directed=False,indices=sources)/unit for side in ['s','c'] if side=='s' or data['conditional']['defined']}
    fit_truth={side:matrix[[rows[int(i)] for i in pairs[:,0]],pairs[:,1]] for side,matrix in D.items()}
    test_truth={side:matrix[[rows[int(i)] for i in validation[:,0]],validation[:,1]] for side,matrix in D.items()}
    for dim in [3]:
        outputs=[];zs=[]
        for idx,side in enumerate(['s','c']):
            if side not in D:outputs.append(dict(status='undefined_measure'));zs.append(None);continue
            initial=np.asarray(data['positions'][0]) if idx==0 else zs[0][:,1:]
            if idx and np.array_equal(D['s'],D['c']):
                z=zs[0].copy();history=outputs[0]['history']
            else:
                z,history=fit(n,dim,k,pairs,fit_truth[side],initial,steps)
                if history[-1]['stress']>.25:
                    candidate,extra=fit(n,dim,k,pairs,fit_truth[side],z[:,1:],min(100,steps))
                    if extra[-1]['stress']<history[-1]['stress']:z=candidate
                    history+= [dict(step=steps+v['step'],stress=v['stress']) for v in extra]
            z=move_origin(z,z[-1],k)
            if idx:
                R,_=orthogonal_procrustes(z[:,1:],zs[0][:,1:]);z[:,1:]=z[:,1:]@R
            zs.append(z)
            ij=torch.tensor(validation);tz=torch.tensor(z);pred=distances(tz,ij[:,0],ij[:,1],k).numpy()
            E=np.asarray(data['positions'][idx])[:,:dim];ep=np.linalg.norm(E[validation[:,0]]-E[validation[:,1]],axis=1)
            free_anchor=np.zeros(dim+1);free_anchor[0]=math.cosh(.4)/math.sqrt(k);free_anchor[1]=-math.sinh(.4)/math.sqrt(k)
            free=move_origin(z,free_anchor,k)
            # Neighborhood overlap against all nodes for held-out source nodes.
            overlap=[]
            for i in held:
                actual=D[side][rows[int(i)]].copy();actual[i]=np.inf
                hp=distances(tz,torch.full((n,),int(i)),torch.arange(n),k).numpy();hp[i]=np.inf
                top=min(10,n-1);overlap.append(len(set(np.argsort(actual)[:top])&set(np.argsort(hp)[:top]))/top)
            outputs.append(dict(status='available',lorentz=z.tolist(),ball=np.pad(ball(z,k),((0,0),(0,3-dim))).tolist(),ball_free=np.pad(ball(free,k),((0,0),(0,3-dim))).tolist(),history=history,termination='Iteration budget reached; best valid training objective retained',training=diagnostics(distances(torch.tensor(z),torch.tensor(pairs[:,0]),torch.tensor(pairs[:,1]),k).numpy(),fit_truth[side]),heldout=diagnostics(pred,test_truth[side]),euclidean_same_pairs=diagnostics(ep,test_truth[side]),neighbor_overlap_at_10=float(np.mean(overlap)),constraint_residual=float(np.max(np.abs(-z[:,0]**2+(z[:,1:]**2).sum(1)+1/k)))))
        result['dimensions'][str(dim)]=outputs
    return result


def inspect_pair(data,u,v):
    n=len(data['nodes'])
    if not 0<=u<n or not 0<=v<n:raise ValueError('Node index out of bounds')
    unit=data['metric']['display_reference']['unit'];out={}
    for idx,side in enumerate(['s','c']):
        if idx and not data['conditional']['defined']:out[side]=None;continue
        raw=float(dijkstra(metric_graph(data,side),directed=False,indices=u)[v]);values=dict(intrinsic_raw=raw,intrinsic_display_units=raw/unit)
        for dim in [2,3]:
            E=np.array(data['positions'][idx]);values['euclidean_'+str(dim)]=float(np.linalg.norm(E[u,:dim]-E[v,:dim]))
            h=data.get('hyperbolic',{});entry=h.get('dimensions',{}).get(str(dim),[])
            if entry:
                z=torch.tensor(entry[idx]['lorentz']);values['hyperbolic_'+str(dim)]=0. if u==v else float(distances(z,torch.tensor([u]),torch.tensor([v]),-h['curvature'])[0])
        out[side]=values
    return out


def euclidean_diagnostics(data):
    """Existing projection evaluated on deterministic held-out source rows."""
    n=len(data['nodes']);sources=np.random.default_rng(912).choice(n,min(12,n),replace=False);unit=data['metric']['display_reference']['unit'];out={}
    for idx,side in enumerate(['s','c']):
        if idx and not data['conditional']['defined']:out[side]={'status':'undefined_measure'};continue
        D=dijkstra(metric_graph(data,side),directed=False,indices=sources)/unit
        values={}
        for dim in [2,3]:
            E=np.asarray(data['positions'][idx])[:,:dim];pred=np.linalg.norm(E[sources,None]-E[None],axis=-1);mask=np.ones(D.shape,bool);mask[np.arange(len(sources)),sources]=False
            diagnostic=diagnostics(pred[mask],D[mask]);overlap=[]
            for row,i in enumerate(sources):
                a=D[row].copy();b=pred[row].copy();a[i]=b[i]=np.inf;top=min(10,n-1);overlap.append(len(set(np.argsort(a)[:top])&set(np.argsort(b)[:top]))/top)
            diagnostic.update(neighbor_overlap_at_10=float(np.mean(overlap)),status='high_distortion' if diagnostic['stress']>.25 else 'available',method='Existing MDS/landmark projection, no extra optimization',constraint_residual=0.)
            values[str(dim)]=diagnostic
        out[side]=values
    return out
