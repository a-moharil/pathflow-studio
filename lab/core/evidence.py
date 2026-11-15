"""Occurrence footprints, exact interventions and function-matched attribution audits."""
import copy
import numpy as np
import torch
from torch import nn
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from .pathmath import graph, measure
from .network import hidden_count
from .healthcare import case_analysis
from .transforms import image, specification, transport, hellinger

VERSION = 'evidence-2'


def tensor(x): return torch.as_tensor(np.asarray(x)[None], dtype=torch.float64)
def scalar(y, readout):
    return y[:,readout['a']] - (y[:,readout['b']] if readout.get('b') is not None else 0)


def state(model, x, reference, readout):
    sizes, offsets, layers = graph(model, readout)
    with torch.no_grad():
        _, hs, zs = model(tensor(x), trace=True)
        _, h0, z0 = model(tensor(reference), trace=True)
    from .network import hidden_count
    sec=[];hidden=hidden_count(model)
    for l, (z,b) in enumerate(zip(zs,z0)):
        a=z.numpy().ravel();b=b.numpy().ravel()
        derivative=np.ones_like(a) if l>=hidden else ((a>0).astype(float) if model.config['activation']=='relu' else 1-np.tanh(a)**2)
        
        for idx,alpha in (getattr(model,'gates',{}).get(str(l),{}) if l<hidden else {}).items():derivative[int(idx)]*=alpha
        sec.append(np.divide(hs[l+1].numpy().ravel()-h0[l+1].numpy().ravel(),a-b,out=derivative,where=a!=b))
    sec.append(np.ones(1));dx=(np.asarray(x,float)-reference).ravel()
    sm,fs=measure(sizes,layers,np.ones(sizes[0]),[np.ones(n) for n in sizes[1:]])
    cm,fc=measure(sizes,layers,dx*dx,sec)
    return dict(sizes=sizes,offsets=offsets,layers=layers,sec=sec,dx=dx,structural=sm,conditional=cm,fs=fs,fc=fc)


def identities(model, l, layer_graph):
    u,v,w=layer_graph;layer=model.layers[l]
    if isinstance(layer,nn.Linear): return v*layer.in_features+u
    ic,h,width=model.shapes[l];oc,oh,ow=model.shapes[l+1]
    source_c=u//(h*width);source_y=(u//width)%h;source_x=u%width
    dest_c=v//(oh*ow);dest_y=(v//ow)%oh;dest_x=v%ow
    ky=source_y-dest_y*layer.stride[0]+layer.padding[0]
    kx=source_x-dest_x*layer.stride[1]+layer.padding[1]
    return ((dest_c*ic+source_c)*layer.kernel_size[0]+ky)*layer.kernel_size[1]+kx


def occurrence_gradients(model,x,reference,readout,st):
    values=[]
    for inp in [x,reference]:
        out,hs,zs=model(tensor(inp).requires_grad_(True),trace=True)
        dz=torch.autograd.grad(scalar(out,readout).sum(),zs)
        values.append([hs[l].detach().numpy().ravel()[u]*dz[l].detach().numpy().ravel()[v] for l,(u,v,w) in enumerate(st['layers'][:-1])])
    return [a-b for a,b in zip(*values)]


def attributions(model,x,reference,readout,steps=64):
    steps=int(steps)
    if not 8<=steps<=1024: raise ValueError('Integrated Gradients intervals must be 8..1024')
    x=np.asarray(x,float);reference=np.asarray(reference,float);dx=x-reference
    def gradient(a):
        t=torch.as_tensor(a,dtype=torch.float64).requires_grad_(True)
        return torch.autograd.grad(scalar(model(t),readout).sum(),t)[0].detach().numpy()
    grad=gradient(x[None])[0];total=np.zeros_like(x)
    # Trapezoidal integration, batched to bound memory. No per-image normalization.
    for start in range(0,steps+1,16):
        ids=np.arange(start,min(start+16,steps+1));points=reference[None]+(ids/steps).reshape(-1,1,1,1)*dx[None]
        weights=np.where((ids==0)|(ids==steps),.5,1.)
        total+=np.sum(gradient(points)*weights.reshape(-1,1,1,1),axis=0)
    ig=dx*total/steps
    with torch.no_grad(): delta=float(scalar(model(tensor(x)),readout)[0]-scalar(model(tensor(reference)),readout)[0])
    return dict(input_gradient=grad.sum(0).tolist(),integrated_gradients=ig.sum(0).tolist(),ig_sum=float(ig.sum()),delta=delta,ig_residual=abs(float(ig.sum())-delta),ig_absolute_mass=float(np.abs(ig).sum()),ig_intervals=steps,quadrature='trapezoidal',target=readout,gradient_units='readout per input unit',ig_units='readout contribution')


def select_support(model,st,x,reference,readout,options):
    from .network import hidden_count
    conv=[l for l,m in enumerate(model.layers) if isinstance(m,nn.Conv2d) and l<hidden_count(model)]
    if not conv: return dict(status='not_applicable',reason='Field B and primitive convolution interventions require a CNN',layers=[],kind=options.get('support_kind','occurrence'))
    chosen=options.get('support_layers','0')
    layers=conv if chosen=='all' else [int(chosen)]
    if any(l not in conv for l in layers):raise ValueError('Select a convolutional layer')
    kind=options.get('support_kind','occurrence');rule=options.get('support_rule','flow');budget=int(options.get('support_budget',16))
    if kind not in ['occurrence','parameter'] or rule not in ['flow','gradient','weight','random','manual']:raise ValueError('Invalid support kind or ranking')
    if not 1<=budget<=1024:raise ValueError('Support budget per layer must be 1..1024')
    if rule=='flow' and not st['conditional']['defined']:raise ValueError('Conditional measure undefined; choose a non-flow selection rule')
    gradients=occurrence_gradients(model,x,reference,readout,st) if rule=='gradient' else None
    rng=np.random.default_rng(int(options.get('seed',808)));result=[]
    for l in layers:
        u,v,w=st['layers'][l];pid=identities(model,l,st['layers'][l]);unique=np.unique(pid)
        if kind=='occurrence':
            candidates=np.arange(len(u));score=st['fc'][l] if rule=='flow' else np.abs(gradients[l]) if rule=='gradient' else np.abs(w)
        else:
            candidates=unique
            score=np.bincount(pid,weights=st['fc'][l],minlength=model.layers[l].weight.numel()) if rule=='flow' else np.abs(np.bincount(pid,weights=gradients[l],minlength=model.layers[l].weight.numel())) if rule=='gradient' else np.abs(model.layers[l].weight.detach().numpy().ravel())
        count=min(budget,len(candidates))
        if rule=='random':selected=np.sort(rng.choice(candidates,count,replace=False))
        elif rule=='manual':
            selected=np.unique(np.asarray(options.get('manual_ids',[]),int))
            if not len(selected) or not np.isin(selected,candidates).all():raise ValueError('Manual IDs must be valid occurrence or parameter IDs in every selected layer')
        else:selected=candidates[np.lexsort((candidates,-score[candidates]))[:count]]
        edges=selected if kind=='occurrence' else np.flatnonzero(np.isin(pid,selected))
        result.append(dict(layer=l,selected_ids=selected.tolist(),occurrences=edges.tolist(),parameters=np.unique(pid[edges]).tolist(),occurrence_count=len(edges),parameter_count=len(np.unique(pid[edges])),cut_mass=float(st['fc'][l][edges].sum()) if st['conditional']['defined'] else None))
    return dict(status='available',kind=kind,rule=rule,budget_per_layer=budget,seed=int(options.get('seed',808)),layers=result,policy='Top scores ranked within each selected layer. Parameter gradients aggregate signed occurrence gradients before taking absolute value. Biases are not selected.')


def field_b(model,st,support):
    if support['status']!='available':return dict(status=support['status'],reason=support.get('reason'),image=None,total_mass=None,native=[])
    if not st['conditional']['defined']:return dict(status='undefined_measure',image=None,total_mass=None,native=[])
    im=np.zeros(model.shapes[0][-2:]);native=[];occurrences=[]
    for entry in support['layers']:
        l=entry['layer'];ids=np.asarray(entry['occurrences'],int);u,v,w=st['layers'][l];pid=identities(model,l,st['layers'][l]);shape=model.shapes[l]
        mass=np.bincount(u[ids],weights=st['fc'][l][ids],minlength=int(np.prod(shape))).reshape(shape)
        native.append(dict(layer=l,source_shape=list(shape),channels=mass.tolist(),spatial=mass.sum(0).tolist(),total_mass=float(mass.sum())))
        if l==0:im=mass.sum(0)
        for i in ids:
            occurrences.append(dict(layer=l,occurrence=int(i),parameter=int(pid[i]),source=list(map(int,np.unravel_index(u[i],shape))),destination=list(map(int,np.unravel_index(v[i],model.shapes[l+1]))),kernel_index=list(map(int,np.unravel_index(pid[i],tuple(model.layers[l].weight.shape))))[2:],baseline_mass=float(st['fc'][l][i])))
    total=float(im.sum());has_first=any(e['layer']==0 for e in support['layers'])
    return dict(status='available' if has_first else 'no_first_layer_support',image=im.tolist(),total_mass=total,normalized_shape=(im/total).tolist() if total>0 else None,native=native,occurrences=occurrences,definition='Baseline conditional mass on selected convolution occurrences at exact source coordinates; image field uses Conv1 only',normalization='Raw selected mass, not renormalized. Native totals are per cut and cannot be summed as a probability of a union.')


class OccurrenceModel(nn.Module):
    """A frozen network with multiplicative factors on selected convolution uses."""
    def __init__(self,model,st,support,alpha):
        super().__init__();self.layers=model.layers;self.config=model.config;self.shapes=model.shapes;self.occurrence_factors={};self.execution=st['layers']
        for entry in support['layers']:
            l=entry['layer'];f=np.ones(len(st['layers'][l][0]));f[entry['occurrences']]=alpha;self.occurrence_factors[l]=f
    def forward(self,x,gates=None,trace=False):
        hs=[x];zs=[]
        for l,layer in enumerate(self.layers):
            if isinstance(layer,nn.Linear):x=x.flatten(1)
            z=layer(x)
            if l in self.occurrence_factors:
                u,v,w=self.execution[l];f=self.occurrence_factors[l];ids=np.flatnonzero(f!=1)
                if len(ids):
                    uu=torch.as_tensor(u[ids],device=x.device);vv=torch.as_tensor(v[ids],device=x.device);gain=torch.as_tensor(w[ids]*(f[ids]-1),dtype=x.dtype,device=x.device)
                    correction=torch.zeros_like(z).flatten(1).index_add(1,vv,x.flatten(1)[:,uu]*gain)
                    z=z+correction.reshape_as(z)
            hidden=hidden_count(self)
            zs.append(z);x=(torch.relu(z) if self.config['activation']=='relu' else torch.tanh(z)) if l<hidden else z
            if gates and str(l) in gates and l<hidden:
                g=torch.ones(x[0].numel(),dtype=x.dtype,device=x.device)
                for i,a in gates[str(l)].items():g[int(i)]=float(a)
                x=x*g.reshape(x.shape[1:])
            hs.append(x)
        return (x,hs,zs) if trace else x


def rescaled(model,layer,index,factor):
    if model.config['activation']!='relu':raise ValueError('Exact compensated rescaling is available only for ReLU')
    from .network import hidden_count
    if not 0<=layer<hidden_count(model):raise ValueError('Select a hidden layer')
    if not 0<=index<model.shapes[layer+1][0] or not .1<=factor<=10:raise ValueError('Invalid channel/unit or rescale factor; use 0.1..10')
    m=copy.deepcopy(model);a=m.layers[layer];b=m.layers[layer+1]
    with torch.no_grad():
        a.weight[index]*=factor
        if a.bias is not None:a.bias[index]*=factor
        if isinstance(b,nn.Conv2d):b.weight[:,index]/=factor
        else:
            size=int(np.prod(m.shapes[layer+1][1:]));b.weight[:,index*size:(index+1)*size]/=factor
    return m


def internal_signal(model,x,readout,layer,index):
    out,hs,zs=model(tensor(x).requires_grad_(True),trace=True);h=hs[layer+1]
    g=torch.autograd.grad(scalar(out,readout).sum(),h)[0]
    a=h.detach().numpy()[0,index];b=g.detach().numpy()[0,index]
    return dict(activation=np.atleast_2d(a).tolist(),gradient=np.atleast_2d(b).tolist(),product=np.atleast_2d(a*b).tolist())


def metric_graph(st,side):
    us=[];vs=[];lengths=[]
    for l,(u,v,w) in enumerate(st['layers']):
        us.extend(u+st['offsets'][l]);vs.extend(v+st['offsets'][l+1]);lengths.extend(1/(len(u)*st[side][l]+.03))
    return coo_matrix((np.r_[lengths,lengths],(np.r_[us,vs],np.r_[vs,us])),shape=(st['offsets'][-1],)*2).tocsr()


def neighborhood(model,st,node=0,count=12):
    n=int(st['offsets'][-1]);node=int(node)
    if not 0<=node<n:raise ValueError(f'Node must be in 0..{n-1}')
    ds=dijkstra(metric_graph(st,'fs'),directed=False,indices=node)
    if not np.isfinite(ds).all():raise ValueError('Intrinsic graph is disconnected for this architecture; this neighborhood comparison requires connected source coordinates')
    dc=dijkstra(metric_graph(st,'fc'),directed=False,indices=node) if st['conditional']['defined'] else None
    selected=np.unique(np.r_[node,np.argsort(ds)[:count+1],np.argsort(dc)[:count+1] if dc is not None else []]).astype(int)
    def label(i):
        l=int(np.searchsorted(st['offsets'][1:],i,side='right'));idx=int(i-st['offsets'][l])
        return 'Readout sink' if l==len(model.shapes) else f'L{l} '+str(tuple(map(int,np.unravel_index(idx,model.shapes[l]))))
    return dict(node=node,node_count=n,ids=selected.tolist(),labels=[label(i) for i in selected],structural=dijkstra(metric_graph(st,'fs'),directed=False,indices=selected)[:,selected].tolist(),conditional=dijkstra(metric_graph(st,'fc'),directed=False,indices=selected)[:,selected].tolist() if dc is not None else None,neighbors_structural=np.argsort(ds)[1:count+1].tolist(),neighbors_conditional=np.argsort(dc)[1:count+1].tolist() if dc is not None else None,unit='Raw intrinsic lengths 1/(cut edge count × marginal + 0.03); no embedding or rescaling')


def raw_transport(a,t):
    moved=transport(a,t)
    return dict(**moved,raw_field=(np.asarray(moved['field'])*moved['retained_mass']).tolist() if moved['field'] is not None else None)
