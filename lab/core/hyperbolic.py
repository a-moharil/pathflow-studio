"""Experimental distance fitting in Lorentz coordinates; intrinsic metric stays external."""
from .fitsettings import dimensions, FitHistory, stop_reason
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
    lengths=np.array([e['length_'+side] for e in data['edges']]) if data['metric'].get('explicit_edge_lengths') else 1/(counts[layer[u]]*mass+data['metric']['epsilon'])
    if not np.isfinite(lengths).all() or np.any(lengths<=0):raise ValueError('Metric edge lengths must be finite and positive')
    from .metricgraph import undirected_graph
    return undirected_graph(n,u,v,lengths)


def lorentz(space,k):
    return torch.cat((torch.sqrt(1/k+(space*space).sum(-1,keepdim=True)),space),-1)


def distances(z,i,j,k):
    # Minkowski chord avoids subtracting two nearly equal numbers around 1.
    # Time differences use their rationalized form on the upper hyperboloid.
    A=z[i,1:];B=z[j,1:];diff=A-B
    dt=(diff*(A+B)).sum(-1)/(z[i,0]+z[j,0])
    q=torch.clamp((diff*diff).sum(-1)-dt*dt,min=0)
    return 2*torch.asinh(torch.sqrt(k*q)/2)/math.sqrt(k)


def lorentz_objective(flat,n,dim,k,pairs,truth,weights):
    """Distance stress and its matched analytic gradient, including the floor.

    A chord-squared floor of 1e-24 regularizes only the nondifferentiable
    coincidence. Its flat region has exactly zero derivative. No coordinates
    are displaced to make the picture look separated.
    """
    X=flat.reshape(n,dim);time=np.sqrt(1/k+(X*X).sum(1));i,j=pairs.T
    A=X[i];B=X[j];diff=A-B;ta=time[i];tb=time[j]
    dt=(diff*(A+B)).sum(1)/(ta+tb)
    raw=(diff*diff).sum(1)-dt*dt;q=np.maximum(raw,1e-24)
    pred=2*np.arcsinh(np.sqrt(k*q)/2)/math.sqrt(k);error=pred-truth
    loss=float(np.mean(weights*error**2))
    factor=4/len(pairs)*weights*error/(np.sqrt(q)*np.sqrt(4+k*q))
    factor=np.where(raw>1e-24,factor,0.)
    gradient=np.zeros_like(X)
    np.add.at(gradient,i,factor[:,None]*(diff-dt[:,None]*A/ta[:,None]))
    np.add.at(gradient,j,factor[:,None]*(-diff+dt[:,None]*B/tb[:,None]))
    if not np.isfinite(loss) or not np.isfinite(gradient).all():
        raise FloatingPointError('Nonfinite Lorentz objective or gradient')
    return loss,gradient.ravel()


def numerical_validity(z,k,free=None,before=None,pairs=None):
    """Reject invalid coordinates independently of distance approximation quality."""
    z=np.asarray(z,dtype=float);reason=[]
    if z.ndim!=2 or not np.isfinite(z).all():
        return dict(valid=False,reason='Nonfinite or malformed Lorentz coordinates')
    if np.any(z[:,0]<=0):reason.append('Coordinates are not on the upper sheet')
    # Extended precision audits stored float64 coordinates, not reconstructed ones.
    zz=z.astype(np.longdouble)
    residual=float(np.max(np.abs(k*(-zz[:,0]**2+(zz[:,1:]**2).sum(1))+1)))
    if not np.isfinite(residual) or residual>1e-6:reason.append('Dimensionless Lorentz constraint residual exceeds 1e-6')
    free_residual=None
    if free is not None:
        free=np.asarray(free,dtype=float)
        if not np.isfinite(free).all() or np.any(free[:,0]<=0):
            reason.append('Released-anchor coordinates are nonfinite or not on the upper sheet')
        else:
            ff=free.astype(np.longdouble)
            free_residual=float(np.max(np.abs(k*(-ff[:,0]**2+(ff[:,1:]**2).sum(1))+1)))
            if not np.isfinite(free_residual) or free_residual>1e-6:reason.append('Released-anchor Lorentz constraint residual exceeds 1e-6')
    with np.errstate(divide='ignore',invalid='ignore'):
        radii=np.linalg.norm(ball(z,k),axis=1)
        free_radii=np.linalg.norm(ball(free,k),axis=1) if free is not None else radii
    if not np.isfinite(radii).all() or not np.isfinite(free_radii).all() or max(radii.max(),free_radii.max())>=1-1e-10:
        reason.append('Ball coordinates are nonfinite or too close to the numerical boundary')
    drift=None
    if before is not None and pairs is not None:
        i,j=np.asarray(pairs).T
        d0=distances(torch.tensor(before),i,j,k).numpy();d1=distances(torch.tensor(z),i,j,k).numpy()
        drift=float(np.max(np.abs(d1-d0)/np.maximum(1.,d0)))
        if not np.isfinite(drift) or drift>1e-6:reason.append('Alignment changes represented distances beyond numerical tolerance')
    return dict(valid=not reason,reason='; '.join(reason) or 'Numerical checks passed',dimensionless_constraint_residual=residual if np.isfinite(residual) else None,released_anchor_residual=free_residual if free_residual is not None and np.isfinite(free_residual) else None,
                constraint_tolerance=1e-6,maximum_ball_radius=float(radii.max()) if np.isfinite(radii).all() else None,
                boundary_margin=1e-10,alignment_distance_drift=drift if drift is not None and np.isfinite(drift) else None)


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
    """Analytic float64 Lorentz-distance gradients; best finite iterate retained."""
    from scipy.optimize import minimize
    from .metricfit import objective_weights
    initial=np.array(initial[:,:dim],copy=True);cap=math.sinh(12)/math.sqrt(k);initial=np.clip(initial,-cap,cap)
    i,j=pairs.T;weights,floor=objective_weights(truth);best=[float('inf'),initial.copy()];history=[]
    def objective(flat):
        loss,gradient=lorentz_objective(flat,n,dim,k,pairs,truth,weights)
        X=flat.reshape(n,dim)
        if np.isfinite(loss) and loss<best[0]:best[:]=[loss,X.copy()]
        return loss,gradient
    initial_loss=objective(initial.ravel())[0];history=FitHistory(objective,initial_loss)
    def callback(flat):
        history(flat)
    solution=minimize(objective,initial.ravel(),jac=True,method='L-BFGS-B',bounds=[(-cap,cap)]*(n*dim),callback=callback,options={'maxiter':steps,'maxfun':max(15000,steps*40),'ftol':1e-10,'gtol':1e-7,'maxls':35})
    X=best[1];z=np.column_stack((np.sqrt(1/k+(X*X).sum(1)),X))
    return z,dict(initial_objective=initial_loss,best_objective=best[0],iterations=int(solution.nit),evaluations=int(solution.nfev),converged=bool(solution.success),termination=str(solution.message),relative_error_floor=floor,history=history.finish(solution),max_iterations=steps,max_evaluations=max(15000,steps*40),stop_reason=stop_reason(solution,steps),coordinate_bound=cap,bound_fraction=float(np.mean(np.abs(X)>=cap*.999)))


def compute(data,k=1.,steps=200,design=None,T=None,sensitivity=False):
    from .metricfit import pair_design,targets,deformation,euclidean_distances
    n=len(data['nodes'])
    if n>4000:return dict(status='budget_exceeded',reason='Lorentz fitting is limited to 4000 nodes; intrinsic and Euclidean results remain available.')
    design=pair_design(data) if design is None else design;T=targets(data,design) if T is None else T
    pairs=design['train'];validation=design['validation'];sources=design['sources'];unit=data['metric']['display_reference']['unit']
    result=dict(status='available',version='metric-fit-4',curvature=-k,unit=unit,steps=steps,seed=1024,training_pairs=len(pairs),validation_pairs=len(validation),minimum_training_degree=int(design['degree'].min()),unconstrained_nodes=int(np.sum(design['degree']==0)),alignment='Readout sink at origin; spatial orthogonal isometries, no scale fitting',dimensions={},deformation={})
    for dim in ([3]+dimensions(sensitivity)):
        outputs=[];zs=[]
        for idx,side in enumerate(['s','c']):
            if side not in T:outputs.append(dict(status='undefined_measure'));zs.append(None);continue
            entry=data.get('euclidean',{}).get('dimensions',{}).get(str(dim))
            initial=np.asarray(entry[idx]['positions']) if entry else np.pad(np.asarray(data['positions'][idx]),((0,0),(0,max(0,dim-3))))
            if idx and np.array_equal(T['s']['rows'],T['c']['rows']):z=zs[0].copy();optimization=outputs[0]['optimization'].copy()
            else:
                z,optimization=fit(n,dim,k,pairs,T[side]['train'],initial,steps)
                if optimization['best_objective']>.06:
                    rng=np.random.default_rng(710+dim);restart=rng.normal(size=(n,dim))*np.median(T[side]['train'])/math.sqrt(2*dim)
                    alternate,other=fit(n,dim,k,pairs,T[side]['train'],restart,steps)
                    attempt_reports=[dict(optimization),dict(other)]
                    attempts=[optimization['best_objective'],other['best_objective']]
                    if other['best_objective']<optimization['best_objective']:z,optimization=alternate,other
                    optimization=dict(optimization,attempts=attempt_reports,restart_objectives=attempts,selection='Smallest fitting objective; validation not used for selection')
            before=z.copy()
            check=numerical_validity(z,k)
            if not check['valid']:
                return dict(status='numerical_invalid',reason=check['reason'],numerical_validation=check,dimensions={})
            z=move_origin(z,z[-1],k)
            if idx:
                R,_=orthogonal_procrustes(z[:,1:],zs[0][:,1:]);z[:,1:]=z[:,1:]@R
            zs.append(z);tz=torch.tensor(z,dtype=torch.float64)
            pred=distances(tz,torch.tensor(validation[:,0]),torch.tensor(validation[:,1]),k).numpy()
            train_pred=distances(tz,torch.tensor(pairs[:,0]),torch.tensor(pairs[:,1]),k).numpy()
            E=initial;ep=euclidean_distances(E,validation)
            free_anchor=np.zeros(dim+1);free_anchor[0]=math.cosh(.4)/math.sqrt(k);free_anchor[1]=-math.sinh(.4)/math.sqrt(k)
            free=move_origin(z,free_anchor,k)
            check=numerical_validity(z,k,free=free,before=before,pairs=pairs)
            if not check['valid']:
                return dict(status='numerical_invalid',reason=check['reason'],numerical_validation=check,dimensions={})
            overlap=[]
            for row,i in enumerate(sources):
                actual=T[side]['rows'][row].copy();actual[i]=np.inf;hp=distances(tz,torch.full((n,),int(i)),torch.arange(n),k).numpy();hp[i]=np.inf;top=min(10,n-1);overlap.append(len(set(np.argsort(actual)[:top])&set(np.argsort(hp)[:top]))/top)
            heldout=diagnostics(pred,T[side]['validation']);residual=float(np.max(np.abs(-z[:,0]**2+(z[:,1:]**2).sum(1)+1/k)))
            outputs.append(dict(status='available',quality='high_distortion' if heldout['stress']>.25 or heldout['median_relative_error']>.25 else 'inspect_diagnostics',lorentz=z.tolist(),ball=ball(z,k).tolist(),ball_free=ball(free,k).tolist(),numerical_validation=check,neighbor_sample_count=len(sources),neighbor_source_ids=sources.tolist(),optimization=optimization,history=optimization['history'],termination=optimization['termination'],training=diagnostics(train_pred,T[side]['train']),heldout=heldout,euclidean_same_pairs=diagnostics(ep,T[side]['validation']),neighbor_overlap_at_10=float(np.mean(overlap)),constraint_residual=residual))
            evidence=data.get('distance_evidence',{}).get('sides',{}).get(side)
            if evidence is not None:
                pp=np.array(data['distance_evidence']['pairs']);evidence.setdefault('hyperbolic',{})[str(dim)]=distances(tz,torch.tensor(pp[:,0]),torch.tensor(pp[:,1]),k).numpy().tolist()
        result['dimensions'][str(dim)]=outputs
        if 'c' in T:
            pp=[]
            for z in zs:pp.append(distances(torch.tensor(z),torch.tensor(validation[:,0]),torch.tensor(validation[:,1]),k).numpy())
            result['deformation'][str(dim)]=deformation(T['s']['validation'],T['c']['validation'],*pp)
    return result


def inspect_pair(data,u,v):
    n=len(data['nodes'])
    if not 0<=u<n or not 0<=v<n:raise ValueError('Node index out of bounds')
    unit=data['metric']['display_reference']['unit'];out={}
    for idx,side in enumerate(['s','c']):
        if idx and not data['conditional']['defined']:out[side]=None;continue
        raw=float(dijkstra(metric_graph(data,side),directed=False,indices=u)[v]);values=dict(intrinsic_raw=raw,intrinsic_display_units=raw/unit)
        for dim in [2,3]:
            entry=data.get('euclidean',{}).get('dimensions',{}).get(str(dim));E=np.array(entry[idx]['positions']) if entry else np.array(data['positions'][idx])[:,:dim];values['euclidean_'+str(dim)]=float(np.linalg.norm(E[u]-E[v]))
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
