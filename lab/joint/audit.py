"""Independent tensor-level carrier identities on the selected readout DAG."""
import functools,copy,hashlib,json,math
import torch
from .tape import advance

def analysis_state(fn):
    @functools.wraps(fn)
    def run(model,x,ref,*args,**kwargs):
        # Job runner already supplies doubles. Direct callers get an isolated copy
        # only when promotion is needed; no training weights or modes are mutated.
        params=list(model.parameters())
        if any(p.dtype!=torch.float64 for p in params):model=copy.deepcopy(model).double()
        modes=[(m,m.training) for m in model.modules()]
        try:
            model.eval()
            return fn(model,x.double(),ref.double(),*args,**kwargs)
        finally:
            for m,state in modes:m.training=state
    return run

def digest_tensor(x):return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def graph_hash(t):
    rows=[dict(name=n.name,shape=list(n.x.shape),parents=n.parents,
        edges=[dict(kind=e.kind,shape=list(e.shape),weight_shape=list(e.weight.shape),parameter=e.kw.get('parameter_identity')) for e in n.edges]) for n in t.nodes]
    return hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()

def measure_audit(m,tolerance=1e-9):
    if m.logQ==-math.inf:return dict(status='undefined_measure',normalization_applied=False,reason='Zero partition; no epsilon added')
    nodes=m.tape.nodes;children=[[] for n in nodes]
    for i,n in enumerate(nodes[1:m.sink+1],1):
        for k,j in enumerate(n.parents):children[j].append((i,k))
    ein=eout=0.;merges=[];source_cut=sink_cut=0.
    for i,n in enumerate(nodes[:m.sink+1]):
        mass=m.node_mass(i);incoming=torch.zeros_like(n.x);outgoing=torch.zeros_like(n.x)
        b=m.beta[i];a=m.alpha[i]
        if b is not None and b[1]!=-math.inf:
            for j,e in zip(n.parents,n.edges):
                f,sf=advance(e,m.alpha[j],m.mode,power=m.power)
                if sf!=-math.inf:incoming+=f*b[0]*math.exp(sf+b[1]-m.logQ)
        if a[1]!=-math.inf:
            for child,k in children[i]:
                bc=m.beta[child]
                if bc is None or bc[1]==-math.inf:continue
                e=nodes[child].edges[k];back,sback=advance(e,bc,m.mode,power=m.power,transpose=True)
                if sback!=-math.inf:outgoing+=a[0]*back*math.exp(a[1]+sback-m.logQ)
        if i>0:ein=max(ein,float((incoming-mass).abs().max()))
        if i<m.sink:eout=max(eout,float((outgoing-mass).abs().max()))
        if i==0:source_cut=float(outgoing.sum())
        if i==m.sink:sink_cut=float(incoming.sum())
        if len(n.parents)>1:
            total=float(mass.sum());branches=[m.branch_mass(i,k) for k in range(len(n.parents))]
            merges.append(dict(name=n.name,visit_mass=total,incoming_mass=sum(branches),allocation=[v/total for v in branches] if total>0 else None,
                interpretation='Incoming mass equals visit mass. Only a mandatory merge has visit mass one.'))
    source=float(m.node_mass(0).sum());sink=float(m.node_mass(m.sink).sum());cut_error=max(abs(source_cut-1),abs(sink_cut-1))
    error=max(ein,eout,cut_error,abs(source-1),abs(sink-1))
    result=dict(status='passed' if error<=tolerance else 'failed',tolerance=tolerance,source_mass=source,sink_mass=sink,
        incoming_node_max_error=ein,outgoing_node_max_error=eout,source_sink_cut_error=cut_error,
        forward_backward_partition_relative_error=abs(source-1),max_error=error,merges=merges,
        cut_scope='Actual source and sink cuts; operation positions are not assumed to be cuts',normalization_applied=True)
    if result['status']=='failed':raise ArithmeticError('Carrier conservation audit failed: '+str(result))
    return result
