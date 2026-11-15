"""Explicit manuscript criteria and bounded, 64-scale intrinsic remeasurement."""
import numpy as np
from scipy.sparse.csgraph import dijkstra
from ..core.metricgraph import undirected_graph


def window_audit(radii, counts):
    x=np.log10(np.asarray(radii,float)); y=np.log(np.asarray(counts,float))
    # Collapse consecutive equal counts, retaining their first observed scale.
    keep=np.r_[True,np.diff(counts)!=0]; x=x[keep]; y=y[keep]
    best=None; nearest=None; candidates=0
    for lo in range(len(x)):
        for hi in range(lo+6,len(x)+1):
            xx=x[lo:hi]; yy=y[lo:hi]; z=xx*np.log(10); n=len(z)
            fit=np.polyfit(z,yy,1); pred=np.polyval(fit,z); rss=float(np.sum((yy-pred)**2)); tss=float(np.sum((yy-yy.mean())**2))
            r2=1-rss/tss if tss>0 else 0.; exponent=float(-fit[0]); local=-np.diff(yy)/np.diff(z)
            cv=float(np.std(local)/abs(np.mean(local))) if abs(np.mean(local))>1e-15 else float('inf')
            rss2=float(np.sum((yy-np.polyval(np.polyfit(z,yy,2),z))**2))
            floor=np.finfo(float).eps*max(1.,float(np.sum(yy**2)))
            dbic=float(n*np.log(max(rss,floor)/max(rss2,floor))-np.log(n))
            values=[n,float(xx[-1]-xx[0]),exponent,float(np.exp(yy[0]-yy[-1])),r2,cv,dbic]
            passed=[n>=6,values[1]>=.5,exponent>=.25,values[3]>=5,r2>=.97,cv<=.25,dbic<=2]
            names=['At least 6 distinct counts','At least 0.5 decade','Exponent at least 0.25','At least fivefold count change','R² at least 0.97','Local slope CV at most 0.25','Linear − quadratic BIC at most 2']
            item=dict(start=10**float(xx[0]),end=10**float(xx[-1]),exponent=exponent if all(passed) else None,
                      fitted_slope=exponent,qualified=all(passed),span=values[1],points=n,
                      gates=[dict(name=k,value=v if np.isfinite(v) else None,passed=bool(p)) for k,v,p in zip(names,values,passed)],
                      residuals=(yy-pred).tolist(),log_radius=z.tolist())
            candidates+=1
            if nearest is None or (sum(passed),item['span'])>(sum(g['passed'] for g in nearest['gates']),nearest['span']): nearest=item
            if all(passed) and (best is None or (item['span'],n)>(best['span'],best['points'])): best=item
    return dict(status='credible_window' if best else 'no_credible_window',window=best or nearest,candidates=candidates,
                selection='Longest qualifying span, then most distinct counts; exploratory window search. Repeated plateaus compressed at first scale.',
                exponent=best['exponent'] if best else None,reason=None if best else 'No candidate passes all seven Appendix C.10 criteria.')


def remeasure(graph, budget=800):
    nodes=graph.get('nodes',[]); edges=graph.get('edges',[]); n=len(nodes)
    if not n or not edges: raise ValueError('This saved report contains no scalar metric graph. Compute a full geometry investigation first.')
    if n>budget: raise ValueError(f'{n} scalar nodes exceed the auditor remeasurement limit of {budget}. Existing saved geometry remains inspectable.')
    if graph.get('scope')!='full_scalar': raise ValueError('Operation quotients and induced subgraphs cannot be substituted for the full scalar carrier.')
    u=np.array([e['u'] for e in edges],int); v=np.array([e['v'] for e in edges],int)
    result={}
    for side,key in [('structural','s'),('conditional','c')]:
        if not graph['defined'].get(side): result[side]=dict(status='undefined_measure'); continue
        mass=np.array([e[key] for e in edges],float)
        if 'length_'+key in edges[0]: lengths=np.array([e['length_'+key] for e in edges],float)
        else:
            layers=np.array([nodes[int(i)]['layer'] for i in u]); counts={l:int(np.sum(layers==l)) for l in set(layers)}
            enrich=mass*np.array([counts[l] for l in layers]); metric=graph['metric']
            lengths=(enrich+float(metric['epsilon']))**(-float(metric.get('power',1)))
        D=dijkstra(undirected_graph(n,u,v,lengths),directed=False)
        if not np.isfinite(D).all(): raise ValueError('Disconnected metric carrier; no fabricated finite distances.')
        positive=D[D>0]
        if not len(positive): raise ValueError('At least two distinct metric nodes are required.')
        incoming=np.bincount(v,weights=mass,minlength=n); outgoing=np.bincount(u,weights=mass,minlength=n)
        strength=np.where(np.bincount(v,minlength=n)>0,incoming,outgoing)
        if strength.sum()<=0: raise ValueError('Undefined node mass.')
        nu=strength/strength.sum(); order=np.lexsort((np.arange(n),-strength))
        radii=np.geomspace(positive.min()*.95,positive.max()*1.005,64); rows=[]
        for r in radii:
            colour=np.full(n,-1)
            for vertex in order:
                forbidden=set(colour[(colour>=0)&(D[vertex]>=r)].tolist()); c=0
                while c in forbidden:c+=1
                colour[vertex]=c
            k=int(colour.max()+1); masses=np.bincount(colour,weights=nu)
            rows.append(dict(radius=float(r),boxes=k,box_fraction=k/n,
                             lacunarity=float(np.mean(masses**2)/np.mean(masses)**2),
                             retention=float(mass[colour[u]!=colour[v]].sum()/mass.sum())))
        counts=np.array([r['boxes'] for r in rows]); crossings=np.flatnonzero((counts[1:]<=n*.5)&(counts[:-1]>n*.5)); xi=None
        if len(crossings):
            i=int(crossings[0]); xi=float(np.exp(np.log(radii[i])+(n*.5-counts[i])/(counts[i+1]-counts[i])*np.log(radii[i+1]/radii[i])))
        keep=np.r_[True,np.diff(counts)!=0]; rr=radii[keep]; cc=counts[keep]
        deff=-np.diff(np.log(cc))/np.diff(np.log(rr)); centre=np.sqrt(rr[:-1]*rr[1:]); supported=(cc[:-1]<n)&(cc[1:]>1)
        profile=[dict(radius=float(r),deff=float(d),supported=bool(s)) for r,d,s in zip(centre,deff,supported)]
        idx=int(np.argmax(np.where(supported,deff,-np.inf))) if np.any(supported) else None
        intermediate=[r['lacunarity'] for r in rows if 1<r['boxes']<n]
        target_rows=[min(rows,key=lambda r:abs(r['box_fraction']-f)) for f in [.5,.25,.125,.0625]]
        result[side]=dict(status='available',scales=rows,xi50=xi,dimension_profile=profile,
                          dstar=float(deff[idx]) if idx is not None else None,
                          dstar_method='Adjacent compressed-transition log slope; explicit auditor estimator, not an asserted reproduction of manuscript smoothing',
                          peak_lacunarity=max(intermediate) if intermediate else None,
                          mean_retention=float(np.mean([r['retention'] for r in target_rows])),
                          retention_resolutions=target_rows,scaling=window_audit(radii,counts))
    return dict(version='ecml-audit-1',scope='Full scalar metric carrier',ruler=graph['metric'],nodes=n,
                scales=64,cover='Deterministic greedy diameter cover ordered by node mass; an approximation to minimum box covering',
                provenance='Remeasured from stored exact edge marginals and declared lengths; no retraining, no embedding distances.',results=result)
