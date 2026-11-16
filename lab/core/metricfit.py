"""Versioned, pair-covered metric fitting. Targets are never changed for display."""
from .fitsettings import dimensions, FitHistory, stop_reason, DIAGNOSTIC_DIMS, FIT_VERSION
import hashlib
import numpy as np
from scipy.linalg import eigh,orthogonal_procrustes
from scipy.optimize import minimize
from scipy.sparse.csgraph import dijkstra
from .hyperbolic import metric_graph,diagnostics

VERSION=FIT_VERSION
DEFAULT_STEPS=3000      # ceiling; fits normally stop earlier on the plateau rule
PLATEAU_WINDOW=100      # iterations
PLATEAU_TOLERANCE=2e-3  # relative objective decrease over the window
PLATEAU_MINIMUM=300     # never declare a plateau before this many iterations
SETTLED=('converged','plateau')
NEAR_PLATEAU=.02       # at the ceiling, below this 100-iteration gain the shape is reported as near final
RESTART_SEED=710        # one seed for every dimension (nested draw)
RESTART_THRESHOLD=.06   # a converged objective above this also receives a random restart
LIFT_SCALE=.1           # added axes enter at this fraction of the lower picture's spread

def pair_design(data):
    n=len(data['nodes']);rng=np.random.default_rng(1024);layers=np.array([v['layer'] for v in data['nodes']]);sources=[]
    for layer in np.unique(layers):
        ids=np.flatnonzero(layers==layer);sources.extend(rng.choice(ids,min(2,len(ids)),replace=False).tolist())
    remaining=np.setdiff1d(np.arange(n),sources)
    if len(sources)<min(n,48):sources.extend(rng.choice(remaining,min(n,48)-len(sources),replace=False).tolist())
    sources=np.asarray(sources,dtype=int);candidates=set();mandatory=set()
    # Every node participates in at least four independently selected source constraints.
    for node in range(n):
        other=sources[sources!=node]
        for source in rng.choice(other,min(4,len(other)),replace=False):mandatory.add(tuple(sorted((node,int(source)))))
    for source in sources:
        for node in range(n):
            if node!=source:candidates.add(tuple(sorted((int(source),node))))
    free=np.asarray(sorted(candidates-mandatory),dtype=int).reshape(-1,2);rng.shuffle(free)
    count=min(8000,max(1,len(free)//5));validation=free[:count];extra=free[count:]
    budget=max(24000,len(mandatory));train=np.asarray(sorted(mandatory)+extra[:max(0,budget-len(mandatory))].tolist(),dtype=int)
    if not len(validation):raise ValueError('Graph too small for disjoint pair validation')
    degrees=np.bincount(train.ravel(),minlength=n)
    assert degrees.min()>0 and not (set(map(tuple,train))&set(map(tuple,validation)))
    return dict(sources=sources,train=train,validation=validation,degree=degrees,seed=1024)

def targets(data,design):
    unit=data['metric']['display_reference']['unit'];sources=design['sources'];lookup={int(s):i for i,s in enumerate(sources)};out={}
    for side in ['s','c']:
        if side=='c' and not data['conditional']['defined']:continue
        D=dijkstra(metric_graph(data,side),directed=False,indices=sources)/unit
        if not np.isfinite(D).all():raise ValueError('Disconnected metric cannot be fitted')
        values={'rows':D}
        for key in ['train','validation']:
            values[key]=np.array([D[lookup[int(a)],b] if a in lookup else D[lookup[int(b)],a] for a,b in design[key]])
        out[side]=values
    return out

def spectral_initial(D,sources,dim):
    square=D[:,sources]**2;row=square.mean(1);grand=square.mean();B=-.5*(square-row[:,None]-row[None,:]+grand)
    vals,vec=eigh(B);order=np.argsort(vals)[::-1];vals=vals[order];vec=vec[:,order];keep=np.flatnonzero(vals>max(1e-12,float(vals.max())*1e-12))[:dim]
    cross=-.5*(D.T**2-(D.T**2).mean(1)[:,None]-row[None,:]+grand)
    Y=cross@vec[:,keep]/np.sqrt(vals[keep])[None,:] if len(keep) else np.zeros((D.shape[1],0))
    Y=np.pad(Y,((0,0),(0,dim-Y.shape[1])));Y-=Y.mean(0)
    return Y,dict(positive_eigenvalues=int(np.sum(vals>1e-10)),negative_eigenvalues=int(np.sum(vals< -1e-10)),positive_spectral_mass_retained=float(np.maximum(vals[keep],0).sum()/max(np.maximum(vals,0).sum(),1e-30)),scope='Sampled source-distance Gram matrix; diagnostic, not full-graph embedding dimension')

def objective_weights(truth):
    # Relative squared errors with a declared floor prevent tiny pairs dominating.
    floor=max(float(np.median(truth))*.1,1e-12);weights=1/np.maximum(truth,floor)**2
    return weights,floor

def fit_euclidean(initial,pairs,truth,steps=DEFAULT_STEPS):
    """L-BFGS stress fit. Stops at the solver's own tolerance, or when the objective has improved by
    less than PLATEAU_TOLERANCE (relative) over the last PLATEAU_WINDOW iterations, or at ``steps``."""
    n,dim=initial.shape;i,j=pairs.T;weights,floor=objective_weights(truth);best=[float('inf'),initial.copy()]
    def objective(flat):
        X=flat.reshape(n,dim);diff=X[i]-X[j];pred=np.linalg.norm(diff,axis=1);error=pred-truth
        loss=float(np.mean(weights*error**2));gradient=np.zeros_like(X)
        force=(2/len(pairs)*weights*error/np.maximum(pred,1e-15))[:,None]*diff
        np.add.at(gradient,i,force);np.add.at(gradient,j,-force)
        if loss<best[0]:best[:]=[loss,X.copy()]
        return loss,gradient.ravel()
    start=objective(initial.ravel())[0];history=FitHistory(objective,start);plateau=[False]
    def callback(intermediate_result):
        history.record(intermediate_result.fun)
        gain=history.trailing_improvement(PLATEAU_WINDOW)
        if history.iteration>=PLATEAU_MINIMUM and gain is not None and gain<PLATEAU_TOLERANCE:
            plateau[0]=True;raise StopIteration
    solution=minimize(objective,initial.ravel(),jac=True,method='L-BFGS-B',callback=callback,options={'maxiter':steps,'maxfun':max(15000,steps*40),'ftol':1e-10,'gtol':1e-7,'maxls':30})
    Y=best[1];Y-=Y.mean(0);reason=stop_reason(solution,steps,plateau[0])
    return Y,dict(initial_objective=start,best_objective=best[0],iterations=int(solution.nit),evaluations=int(solution.nfev),converged=reason in SETTLED,termination=str(solution.message),relative_error_floor=floor,history=history.finish(solution),max_iterations=steps,max_evaluations=max(15000,steps*40),stop_reason=reason,trailing_improvement=history.trailing_improvement(PLATEAU_WINDOW),plateau_rule=dict(window=PLATEAU_WINDOW,relative_tolerance=PLATEAU_TOLERANCE,minimum_iterations=PLATEAU_MINIMUM))

def euclidean_distances(X,pairs):return np.linalg.norm(X[pairs[:,0]]-X[pairs[:,1]],axis=1)

def neighborhood(X,D,sources):
    values=[]
    for row,node in enumerate(sources):
        a=D[row].copy();b=np.linalg.norm(X-X[node],axis=1);a[node]=b[node]=np.inf;k=min(10,len(X)-1)
        values.append(len(set(np.argsort(a)[:k])&set(np.argsort(b)[:k]))/k)
    return float(np.mean(values))

def deformation(truth_a,truth_b,pred_a,pred_b):
    actual=np.asarray(truth_b)-truth_a;represented=np.asarray(pred_b)-pred_a;den=np.linalg.norm(actual)
    corr=float(np.corrcoef(actual,represented)[0,1]) if np.std(actual)>1e-12 and np.std(represented)>1e-12 else None
    return dict(rms_intrinsic_change=float(np.sqrt(np.mean(actual**2))),rms_represented_change=float(np.sqrt(np.mean(represented**2))),rms_change_error=float(np.sqrt(np.mean((actual-represented)**2))),relative_change_error=float(np.linalg.norm(actual-represented)/den) if den>1e-12 else None,change_correlation=corr,status='available' if den>1e-12 else 'no_intrinsic_change')

def settling(optimization):
    """Display status of a fit: settled, near its plateau at the ceiling, or not settled."""
    if optimization.get('stop_reason') in SETTLED:return 'inspect_diagnostics'
    gain=optimization.get('trailing_improvement')
    return 'near_plateau' if gain is not None and gain<NEAR_PLATEAU else 'not_settled'

def nested_restart(n,dim,scale):
    """One random draw for every dimension: the 2-D restart is the leading columns of the 3-D one."""
    draw=np.random.default_rng(RESTART_SEED).normal(size=(n,max(DIAGNOSTIC_DIMS)))
    return draw[:,:dim]*scale/np.sqrt(2*dim)

def lifted_initial(lower,spectral,dim):
    """Converged lower-dimensional picture plus the next spectral axes, entering at reduced scale.

    The added axes start small, so the fit begins at (almost) the lower-dimensional objective and
    uses the new direction only as far as the intrinsic distances require it.
    """
    extra=np.asarray(spectral[:,lower.shape[1]:dim],dtype=float).copy();spread=float(np.sqrt(np.mean(np.sum(lower**2,1))))
    for col in range(extra.shape[1]):
        if np.std(extra[:,col])<1e-9*max(spread,1e-12):
            extra[:,col]=np.random.default_rng(RESTART_SEED+lower.shape[1]+col).normal(size=len(extra))
        extra[:,col]=extra[:,col]/max(np.std(extra[:,col]),1e-30)*LIFT_SCALE*spread/np.sqrt(lower.shape[1])
    Y=np.hstack([lower,extra]);return Y-Y.mean(0)

def orient_to_lower(high,low):
    """Rotate a fit so its best two-dimensional shadow lies on the plane of the lower fit.

    Orthogonal transformation only: every distance is unchanged. Remaining axes follow a
    fixed sign convention (positive third moment) so orientation is reproducible.
    """
    high=np.asarray(high,dtype=float);low=np.asarray(low,dtype=float);hc=high-high.mean(0);lc=low-low.mean(0)
    target=np.pad(lc,((0,0),(0,high.shape[1]-low.shape[1])))
    R,_=orthogonal_procrustes(hc,target);Y=hc@R
    for col in range(low.shape[1],Y.shape[1]):
        if np.mean(Y[:,col]**3)<0:Y[:,col]*=-1;R[:,col]*=-1
    return Y+high.mean(0)@R,R

def shadow_discrepancy(high,low,robust=False):
    """How far a lower fit is from the best-aligned shadow of a higher fit.

    Default: relative squared difference (sensitive to a few loosely constrained outlying nodes).
    robust=True: median node displacement divided by the median node radius of the lower fit.
    """
    high=np.asarray(high,dtype=float);low=np.asarray(low,dtype=float);lc=low-low.mean(0)
    Y,_=orient_to_lower(high,low);shadow=Y[:,:low.shape[1]]-Y[:,:low.shape[1]].mean(0)
    if robust:return float(np.median(np.linalg.norm(shadow-lc,axis=1))/max(np.median(np.linalg.norm(lc,axis=1)),1e-30))
    return float(np.sum((shadow-lc)**2)/max(np.sum(lc**2),1e-30))

def refine(data,steps=DEFAULT_STEPS,sensitivity=False):
    design=pair_design(data);T=targets(data,design);outputs={};deformations={};lower={}
    for dim in ([2,3]+dimensions(sensitivity)):
        entries=[]
        for idx,side in enumerate(['s','c']):
            if side not in T:entries.append(dict(status='undefined_measure'));continue
            spectral,spectrum=spectral_initial(T[side]['rows'],design['sources'],dim)
            if side in lower:initial,start=lifted_initial(lower[side],spectral,dim),'lifted_from_%d_d'%lower[side].shape[1]
            else:initial,start=spectral,'spectral'
            if idx and np.array_equal(T['s']['rows'],T['c']['rows']):Y=np.asarray(entries[0]['positions']).copy();optimization=entries[0]['optimization'].copy()
            else:
                Y,optimization=fit_euclidean(initial,design['train'],T[side]['train'],steps)
                optimization=dict(optimization,start=start)
                if optimization['best_objective']>RESTART_THRESHOLD:
                    restart=nested_restart(len(initial),dim,float(np.median(T[side]['train'])))
                    alternate,other=fit_euclidean(restart,design['train'],T[side]['train'],steps)
                    other=dict(other,start='random_restart_seed_%d'%RESTART_SEED)
                    attempts=[optimization['best_objective'],other['best_objective']];reports=[dict(optimization),dict(other)]
                    if other['best_objective']<optimization['best_objective']:Y,optimization=alternate,other
                    optimization=dict(optimization,attempts=reports,restart_objectives=attempts,selection='Both attempts receive the full iteration budget; the smaller fitting objective is kept; validation not used for selection')
            if idx:
                R,_=orthogonal_procrustes(Y-Y.mean(0),np.asarray(entries[0]['positions'])-np.asarray(entries[0]['positions']).mean(0));Y=Y@R
            elif side in lower:Y,_=orient_to_lower(Y,lower[side])
            consistency=None
            if side in lower:
                consistency=dict(compared_with='%d-D'%lower[side].shape[1],shadow_discrepancy=shadow_discrepancy(Y,lower[side]),median_shift=shadow_discrepancy(Y,lower[side],robust=True),meaning='shadow_discrepancy: relative squared difference between the lower-dimensional picture and the best-aligned shadow of this one (0 = contained unchanged; outlying nodes weigh heavily). median_shift: median node displacement over median node radius')
            lower[side]=np.asarray(Y)
            train=diagnostics(euclidean_distances(Y,design['train']),T[side]['train']);validation=diagnostics(euclidean_distances(Y,design['validation']),T[side]['validation'])
            entries.append(dict(status='available',positions=np.asarray(Y).tolist(),training=train,heldout=validation,initial_heldout=diagnostics(euclidean_distances(initial,design['validation']),T[side]['validation']),neighbor_overlap_at_10=neighborhood(Y,T[side]['rows'],design['sources']),neighbor_sample_count=len(design['sources']),neighbor_source_ids=design['sources'].tolist(),optimization=optimization,spectrum=spectrum,dimension_consistency=consistency,converged=optimization.get('stop_reason') in SETTLED,quality='high_distortion' if validation['stress']>.25 or validation['median_relative_error']>.25 else settling(optimization)))
        outputs[str(dim)]=entries
        if 'c' in T:deformations[str(dim)]=deformation(T['s']['validation'],T['c']['validation'],euclidean_distances(np.asarray(entries[0]['positions']),design['validation']),euclidean_distances(np.asarray(entries[1]['positions']),design['validation']))
    fingerprint=hashlib.sha256(design['validation'].tobytes()).hexdigest()
    data['euclidean']=dict(version=VERSION,dimensions=outputs,pairs=dict(training=len(design['train']),validation=len(design['validation']),minimum_training_degree=int(design['degree'].min()),unconstrained_nodes=int(np.sum(design['degree']==0)),validation_fingerprint=fingerprint,source_layers=[data['nodes'][i]['layer'] for i in design['sources']],seed=1024),objective='Mean squared relative distance error, denominator floored at 0.1 × median fitting distance; same objective for Euclidean and Lorentz',procedure='2-D from the spectral start; each higher dimension starts from the converged lower picture plus the next spectral axis at reduced scale and is rotated so its best shadow lies on the lower picture; a random restart (same nested draw for every dimension) is tried only when the converged objective exceeds %.2f and kept only if its converged objective is smaller'%RESTART_THRESHOLD,validation_note='Pairs excluded from stress optimization, not from spectral initialization; diagnostic holdout, not strict out-of-sample generalization',deformation=deformations)
    data['positions']=[outputs['3'][0]['positions'],outputs['3'][1].get('positions',outputs['3'][0]['positions'])]
    # An inspectable sample uses the same pair identities in every checkpoint.
    select=np.linspace(0,len(design['validation'])-1,min(512,len(design['validation'])),dtype=int)
    evidence=dict(pairs=design['validation'][select].tolist(),sides={})
    for idx,side in enumerate(['s','c']):
        if side not in T:continue
        evidence['sides'][side]=dict(intrinsic=T[side]['validation'][select].tolist(),euclidean={dim:euclidean_distances(np.array(entries[idx]['positions']),design['validation'][select]).tolist() for dim,entries in outputs.items()})
    data['distance_evidence']=evidence
    return design,T
