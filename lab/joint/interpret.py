"""Source interpretation and requested scalar occurrences of an implicit carrier."""
import math
import numpy as np
import torch
from .tape import Measure,advance

def source_summary(field):
    if field is None:return dict(status='undefined_measure')
    a=np.asarray(field,float);v=a.ravel();positive=v[v>0];H=float(-(positive*np.log(positive)).sum());order=np.argsort(-v,kind='stable');cum=np.cumsum(v[order]);contours=[]
    for target in [.5,.8,.95]:
        k=min(len(v)-1,int(np.searchsorted(cum,target)));threshold=float(v[order[k]]);mask=a>=threshold
        contours.append(dict(target=target,threshold=threshold,mass=float(a[mask].sum()),pixels=int(mask.sum()),fraction=float(mask.mean()),tie_policy='Include all pixels tied at threshold'))
    return dict(status='available',source_entropy=H,effective_source_pixels=math.exp(H),total_mass=float(v.sum()),contours=contours,normalization='Original complete-carrier probability; source entropy is not path entropy')

def region_analysis(measure,region):
    if region is None:return None
    x=measure.tape.nodes[0].x;h,w=x.shape[-2:];x0,y0,x1,y1=map(int,region)
    if not 0<=x0<x1<=w or not 0<=y0<y1<=h:raise ValueError('Source region outside image')
    mask=torch.zeros_like(x);mask[:,:,y0:y1,x0:x1]=1
    m=Measure(measure.tape,measure.sink,source_mask=mask);defined=m.logQ!=-math.inf and measure.logQ!=-math.inf
    mass=math.exp(m.logQ-measure.logQ) if defined else 0.;rows=[]
    for i,node in enumerate(m.tape.nodes[:m.sink+1]):
        value=m.node_mass(i);native=value[0].sum(0) if value.ndim==4 else None
        rows.append(dict(name=node.name,absolute_mass=float(value.sum())*mass,allocation=float(value.sum()) if defined else None,native_absolute=(native*mass).tolist() if native is not None else None,branches=[dict(absolute_mass=m.branch_mass(i,k)*mass,allocation=m.branch_mass(i,k) if defined else None) for k in range(len(node.parents))]))
    return dict(region=region,source_mass=mass,status='available' if defined else 'zero_region_mass',operations=rows,definition='Routes originating in selected source rectangle; absolute mass retains whole-network normalization; allocation conditions on region')

def observables(m):
    if m.logQ==-math.inf:return dict(status='undefined_measure',H1=None,eta1=None,eta2=None,Sdef=None)
    n=m.tape.nodes;src=(n[0].x-n[0].ref).square() if m.mode=='conditional' else torch.ones_like(n[0].x)
    expected=float((m.node_mass(0)*torch.where(src>0,src.clamp_min(torch.finfo(src.dtype).tiny).log(),torch.zeros_like(src))).sum())
    for i,node in enumerate(n[1:m.sink+1],1):
        if m.beta[i] is None:continue
        b,sb=m.beta[i]
        for j,e in zip(node.parents,node.edges):
            for part,sign in [('logpositive',1),('lognegative',-1)]:
                value,scale=advance(e,m.alpha[j],m.mode,part=part)
                if scale!=-math.inf and sb!=-math.inf:expected+=sign*float((value*b).sum())*math.exp(scale+sb-m.logQ)
    top=Measure(m.tape,m.sink,'topology');m4=Measure(m.tape,m.sink,m.mode,power=4)
    H=m.logQ-expected;logN=top.logQ;e1=H-logN;e2=2*m.logQ-m4.logQ-logN
    return dict(status='available',H1=H,Sdef=logN-H,eta1=math.exp(e1),eta2=math.exp(e2),log_eta1=e1,log_eta2=e2,log_path_count=logN,expected_route_length=sum(m.branch_mass(i,k) for i,node in enumerate(n[:m.sink+1]) for k in range(len(node.parents))),definition='Complete-carrier entropy; unequal route lengths retained')

def occurrences(t,limit=2500,edge_limit=200000,structural_mode='structural'):
    """Requested complete scalar graph. Basis expansion only under explicit budget.
    Every occurrence remains separate; production marginals use tensor operators.
    """
    sizes=[n.x.numel() for n in t.nodes];total=sum(sizes)
    if total>limit:return None,dict(status='budget_exceeded',nodes=total,limit=limit,reason='Full scalar carrier exceeds requested materialization budget; no quotient substituted')
    rows=[];count=0
    for i,node in enumerate(t.nodes):
        for k,(j,e) in enumerate(zip(node.parents,node.edges)):
            ni,nj=sizes[j],sizes[i]
            # For the deliberately small scalar preset, expand each local response.
            # No complete paths are enumerated, and valid topology retains zero weights.
            for u in range(ni):
                basis=torch.zeros_like(t.nodes[j].x);basis.view(-1)[u]=1
                topo=e.apply(basis,'topology').flatten();ids=(topo!=0).nonzero().flatten()
                if len(ids)==0:continue
                structural=e.apply(basis,structural_mode).flatten();conditional=e.apply(basis,'conditional').flatten()
                count+=len(ids)
                if count>edge_limit:return None,dict(status='budget_exceeded',edges=count,limit=edge_limit,reason='Scalar occurrence budget exceeded')
                rows.extend((i,k,j,u,int(v),float(structural[v]),float(conditional[v]),float(topo[v])) for v in ids)
    return rows,dict(status='available',nodes=total,edges=count,scope='Full scalar carrier, including zero-weight admissible dependencies')

def scalar_geometry(t,st,cm,hyper=False,metric_settings=None):
    from .investigate import embed_subgraph
    rows,status=occurrences(t,structural_mode=st.mode)
    if rows is None:return status
    top=Measure(t,cm.sink,'topology');S=[];C=[];T=[];classes=[]
    def label(j,u):
        shape=t.nodes[j].x.shape[1:];coord=np.unravel_index(u,shape) if shape else (u,)
        return t.nodes[j].name+' ['+','.join(map(str,coord))+']'
    def mass(m,j,i,u,v,q):
        a=m.alpha[j];b=m.beta[i]
        if b is None or a[1]==-math.inf or b[1]==-math.inf or m.logQ==-math.inf:return 0.
        return float(a[0].flatten()[u]*b[0].flatten()[v])*q*math.exp(a[1]+b[1]-m.logQ)
    for i,k,j,u,v,qs,qc,qt in rows:
        tm=mass(top,j,i,u,v,qt)
        if tm<=0:continue # Outside the selected-readout complete carrier, not an inactive conditional edge.
        classes.append(t.nodes[i].name)
        a,b=label(j,u),label(i,v);S.append((a,b,mass(st,j,i,u,v,qs)));C.append((a,b,mass(cm,j,i,u,v,qc)));T.append((a,b,mass(top,j,i,u,v,qt)))
    result=embed_subgraph(S,C if cm.logQ!=-math.inf else None,T,scope='Full selected-readout scalar carrier; exact occurrence marginals; incoming-operation comparison classes; complete-carrier shortest paths',fit_hyperbolic=hyper,classes=classes,metric_settings=metric_settings)
    status['selected_readout_edges']=len(S);status['excluded_noncarrier_edges']=len(rows)-len(S)
    result['scalar_audit']=status
    if result['status']=='available':
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import dijkstra
        from ..core.pathmath import bcan
        edges=result['shared']['edges'];n=len(result['nodes']);u=np.array([e['u'] for e in edges]);v=np.array([e['v'] for e in edges]);responses={}
        for key,side in [('s','structural'),('c','conditional')]:
            if key=='c' and cm.logQ==-math.inf:
                responses[side]=dict(status='undefined_measure',deff=None,dstar=None);continue
            from ..core.metricgraph import undirected_graph
            lengths=np.array([e['length_'+key] for e in edges]);graph=undirected_graph(n,u,v,lengths);D=dijkstra(graph,directed=False)
            strength=np.bincount(u,weights=[e[key] for e in edges],minlength=n)
            z=bcan(D,strength);z['status']='available' if z['deff'] is not None or z['dstar'] is not None else 'no_qualified_window';responses[side]=z
        result['responses']=responses;result['BCANw_status']='Computed on full scalar carrier using declared comparison-class lengths; inspect profiles and qualified windows'
    return result
