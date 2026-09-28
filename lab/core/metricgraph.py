"""Undirected multigraph metric: parallel alternatives use minimum length."""
import numpy as np
from scipy.sparse import coo_matrix

def undirected_graph(n,u,v,lengths):
    best={}
    for a,b,d in zip(u,v,lengths):
        if not np.isfinite(d) or d<=0:raise ValueError('Metric lengths must be finite and positive')
        a,b=sorted((int(a),int(b)))
        if a==b:continue
        best[a,b]=min(best.get((a,b),float('inf')),float(d))
    u=[a for a,b in best];v=[b for a,b in best];w=list(best.values())
    return coo_matrix((w+w,(u+v,v+u)),shape=(n,n)).tocsr()
