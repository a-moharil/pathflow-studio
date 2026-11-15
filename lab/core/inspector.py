"""Signed path diagnostics and visual mechanism inspection for frozen models."""
import numpy as np
import torch
from torch import nn
from scipy.special import logsumexp
from .pathmath import group_logsum
from .evidence import tensor, scalar, identities


def finite_exp(a):
    with np.errstate(over='ignore',under='ignore'):
        y=np.exp(a)
    return float(y) if np.isfinite(y) else None


def signed_diagnostics(model,x,reference,readout,st):
    """Sign-separated log-semiring backward propagation, no path enumeration."""
    sizes=st['sizes'];layers=st['layers'];sec=st['sec'];dx=st['dx']
    def origin_logs(power):
        pos=np.zeros(1);neg=np.full(1,-np.inf)
        for l in reversed(range(len(layers))):
            u,v,w=layers[l];gain=w*sec[l][v]
            with np.errstate(divide='ignore'):lg=power*np.log(np.abs(gain))
            good=gain>=0
            pp=group_logsum(lg+np.where(good,pos[v],neg[v]),u,sizes[l])
            nn=group_logsum(lg+np.where(good,neg[v],pos[v]),u,sizes[l])
            pos,neg=pp,nn
        with np.errstate(divide='ignore'):ld=power*np.log(np.abs(dx))
        pp=ld+np.where(dx>=0,pos,neg);nn=ld+np.where(dx>=0,neg,pos)
        # Sum image channels at exact input spatial coordinates.
        shape=model.shapes[0];spatial=int(np.prod(shape[-2:]));idx=np.arange(len(dx))%spatial
        return group_logsum(pp,idx,spatial),group_logsum(nn,idx,spatial)
    lp,ln=origin_logs(1);lqp,lqn=origin_logs(2);lq=float(logsumexp(np.r_[lqp,lqn]))
    def raw(a):return [finite_exp(v) for v in a]
    p=raw(lp);n=raw(ln);t=raw(np.logaddexp(lp,ln));s=[a-b if a is not None and b is not None else None for a,b in zip(p,n)]
    def cancellation(a,b):
        total=np.logaddexp(a,b)
        if not np.isfinite(total):return None
        return float(np.clip(1-abs(np.exp(a-total)-np.exp(b-total)),0,1))
    gp=float(logsumexp(lp));gn=float(logsumexp(ln));P=finite_exp(gp);N=finite_exp(gn);S=P-N if P is not None and N is not None else None
    with torch.no_grad():delta=float((scalar(model(tensor(x)),readout)-scalar(model(tensor(reference)),readout))[0])
    shape=model.shapes[0][-2:]
    def matrix(a):return np.asarray(a,dtype=object).reshape(shape).tolist()
    defined=np.isfinite(lq)
    ap=np.exp(lqp-lq) if defined else None;an=np.exp(lqn-lq) if defined else None
    global_values=dict(P=P,N=N,S=S,T=finite_exp(np.logaddexp(gp,gn)),Q=finite_exp(lq),logQ=lq if defined else None,C=cancellation(gp,gn),delta=delta,reconstruction_residual=abs(S-delta) if S is not None else None)
    return dict(status='available' if defined else 'undefined_measure',global_values=global_values,P=matrix(p),N=matrix(n),S=matrix(s),T=matrix(t),C=matrix([cancellation(a,b) for a,b in zip(lp,ln)]),A_plus=matrix(ap) if defined else None,A_minus=matrix(an) if defined else None,logP=matrix([float(v) if np.isfinite(v) else None for v in lp]),logN=matrix([float(v) if np.isfinite(v) else None for v in ln]),units='P,N,S,T: readout amplitude; Q: squared amplitude; A+/A−: probability; C: dimensionless',precision='Float64 sign-separated log propagation. Raw overflow is null. Cancellation near one can lose net precision; inspect reconstruction residual.')


def channel_catalog(model,x,reference,readout,st):
    from .network import hidden_count
    hidden=hidden_count(model)
    conv=[l for l,m in enumerate(model.layers) if isinstance(m,nn.Conv2d) and l<hidden]
    layers=[]
    # Two backpropagations supply both endpoints at every hidden layer.
    ends=[]
    for inp in [x,reference]:
        y,hs,zs=model(tensor(inp).requires_grad_(True),trace=True)
        grads=torch.autograd.grad(scalar(y,readout).sum(),hs[1:hidden+1]) if hidden else []
        ends.append([(h.detach().numpy()[0],g.detach().numpy()[0]) for h,g in zip(hs[1:hidden+1],grads)])
    for l in range(hidden):
        shape=model.shapes[l+1];act,grad=ends[0][l];ref,refgrad=ends[1][l]
        _,v,_=st['layers'][l]
        mass=np.bincount(v,weights=st['fc'][l],minlength=st['sizes'][l+1]).reshape(shape) if st['conditional']['defined'] else None
        isconv=l in conv
        if isconv:
            weights=grad.mean(axis=(1,2));weights0=refgrad.mean(axis=(1,2));contribution=weights[:,None,None]*act;contribution0=weights0[:,None,None]*ref
            cam=np.maximum(contribution.sum(0),0);cam0=np.maximum(contribution0.sum(0),0)
        else:contribution=contribution0=cam=cam0=None
        channels=[]
        for k in range(shape[0]):
            def m(a):return np.atleast_2d(a[k]).tolist() if a is not None else None
            channels.append(dict(index=k,activation=m(act),reference=m(ref),difference=m(act-ref),occupation=m(mass),mass=float(mass[k].sum()) if mass is not None else None,contribution=m(contribution),reference_contribution=m(contribution0)))
        layers.append(dict(layer=l,kind='convolution' if isconv else 'dense',shape=list(shape),channels=channels,gradcam=cam.tolist() if cam is not None else None,reference_gradcam=cam0.tolist() if cam0 is not None else None))
    return dict(status='available',layers=layers,default_layer=conv[-1] if conv else (hidden-1),definition='Grad-CAM uses mean spatial gradients of the same readout and postactivation feature maps. Endpoint maps are not a contrast decomposition. Dense layers have unit maps, not Grad-CAM.')


def connection_catalog(model,st,layer,index):
    from .network import hidden_count
    if not 0<=layer<hidden_count(model):raise ValueError('Select a hidden layer')
    shape=model.shapes[layer+1]
    if not 0<=index<shape[0]:raise ValueError('Select a valid channel/unit')
    if not isinstance(model.layers[layer],nn.Conv2d):return dict(status='not_applicable',rows=[],reason='Primitive convolution connections require a convolution layer')
    u,v,w=st['layers'][layer];pid=identities(model,layer,st['layers'][layer]);area=int(np.prod(shape[1:]));ids=np.flatnonzero(v//area==index)
    # Compact exact identity rows: occurrence, parameter, source-channel/y/x, destination-y/x, ky/kx, weight, mass.
    rows=[]
    for i in ids:
        sc,sy,sx=np.unravel_index(u[i],model.shapes[layer]);dc,dy,dx=np.unravel_index(v[i],shape);oc,ic,ky,kx=np.unravel_index(pid[i],tuple(model.layers[layer].weight.shape))
        rows.append([int(i),int(pid[i]),int(sc),int(sy),int(sx),int(dy),int(dx),int(ky),int(kx),float(w[i]),float(st['fc'][layer][i]) if st['conditional']['defined'] else None])
    return dict(status='available',layer=layer,channel=index,columns=['occurrence','parameter','source_channel','source_y','source_x','destination_y','destination_x','kernel_y','kernel_x','weight','conditional_mass'],rows=rows)


class ChannelModel(nn.Module):
    """Postactivation gate wrapper; same multiplicative gate at both endpoints."""
    def __init__(self,model,layer,index,alpha):
        super().__init__();self.base=model;self.layers=model.layers;self.shapes=model.shapes;self.config=model.config
        from .network import hidden_count
        if not 0<=layer<hidden_count(self) or not 0<=index<self.shapes[layer+1][0]:raise ValueError('Invalid hidden channel/unit')
        if not 0<=alpha<=1:raise ValueError('Attenuation must be in [0,1]')
        size=int(np.prod(self.shapes[layer+1][1:]));self.gates={str(layer):{str(i):alpha for i in range(index*size,(index+1)*size)}}
        self.occurrence_factors=getattr(model,'occurrence_factors',{})
    def forward(self,x,gates=None,trace=False):
        combined={k:dict(v) for k,v in self.gates.items()}
        for l,values in (gates or {}).items():
            dest=combined.setdefault(l,{})
            for i,a in values.items():dest[i]=dest.get(i,1)*a
        return self.base(x,combined,trace)
