"""Tensor DAG finite contrasts. Nonnegative mass messages carry logarithmic scales.
No path enumeration and no per-layer probability renormalization.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F

@dataclass
class Node:
    name:str
    x:torch.Tensor
    ref:torch.Tensor
    parents:list
    edges:list

class Edge:
    def __init__(self,kind,shape,weight=None,**kw):
        self.kind,self.shape,self.weight,self.kw=kind,shape,weight,kw
    def coeff(self,mode,power,part):
        w=self.kw.get('structural',self.weight) if mode in ['structural','structural_he'] else self.weight
        if mode=='structural_he' and self.kind in ['conv','linear']:
            fan_in=self.weight[0].numel();w=w*math.sqrt(fan_in/2.)
        if mode=='topology':w=torch.ones_like(w) if self.kind!='dot' else (w!=0).to(w.dtype)
        if part in ['logpositive','lognegative']:
            q=w.abs().pow(power);l=torch.where(q>0,q.clamp_min(torch.finfo(q.dtype).tiny).log(),torch.zeros_like(q));return q*(l.clamp(min=0) if part=='logpositive' else (-l).clamp(min=0))
        if part=='positive':return w.clamp(min=0).pow(power)
        if part=='negative':return (-w).clamp(min=0).pow(power)
        return w.abs().pow(power)
    def apply(self,x,mode='conditional',power=2,part='absolute',transpose=False):
        k=self.kind;w=self.coeff(mode,power,part);p=self.kw
        if k=='conv':
            if transpose:return torch.nn.grad.conv2d_input(self.shape,w,x,stride=p['stride'],padding=p['padding'],dilation=p['dilation'],groups=p['groups'])
            return F.conv2d(x,w,None,p['stride'],p['padding'],p['dilation'],p['groups'])
        if k=='linear':return x@w if transpose else F.linear(x,w)
        if k=='reshape':return (x*w).reshape(self.shape if transpose else p['output'])
        if k=='diag':return x*w
        if k=='dot':return x*w if transpose else (x*w).sum().reshape(1)
        if k=='slice':
            if transpose:return x[:,p['start']:p['stop']]*w
            y=x.new_zeros(p['output']);y[:,p['start']:p['stop']]=x*w;return y
        if k=='resize':
            idx=p['indices'];flat=x.flatten(2)
            if transpose:
                y=x.new_zeros((x.shape[0],x.shape[1],self.shape[-2]*self.shape[-1]))
                y.scatter_add_(2,idx.expand_as(flat),flat*w);return y.reshape(self.shape)
            return (flat[:,:,idx]*w).reshape(p['output'])
        if k=='gap4':
            if transpose:return (x*w).expand(self.shape)
            return x.sum((-2,-1),keepdim=True)*w
        if k=='gap':
            if transpose:return (x[:,:,None,None]*w).expand(self.shape)
            return x.sum((-2,-1))*w
        if k=='pool':
            if transpose:
                vals=x.flatten(2).unsqueeze(2)*w
                return F.fold(vals.reshape(x.shape[0],-1,vals.shape[-1]),self.shape[-2:],p['kernel'],padding=p['padding'],stride=p['stride'])
            vals=F.unfold(x,p['kernel'],padding=p['padding'],stride=p['stride']).reshape(x.shape[0],x.shape[1],-1,w.shape[-1])
            return (vals*w).sum(2).reshape(p['output'])
        raise ValueError(k)

def pool_routes(x,ref,module):
    """Upper-envelope interval integration, fixed lowest-index tie convention.
    Invalid padded coordinates never acquire an occurrence or probability mass.
    """
    k=module.kernel_size;s=module.stride;p=module.padding
    if isinstance(k,tuple):kh,kw=k
    else:kh=kw=k
    if kh*kw>25 or module.ceil_mode or module.dilation!=1:raise ValueError('Unsupported max pool configuration')
    valid=F.unfold(torch.ones_like(x[:,:1]),k,padding=p,stride=s)>0
    a=F.unfold(ref,k,padding=p,stride=s).reshape(x.shape[0],x.shape[1],kh*kw,-1)
    b=F.unfold(x,k,padding=p,stride=s).reshape_as(a)-a
    valid=valid.reshape(1,1,kh*kw,-1).expand_as(a)
    flat_a=a.permute(0,1,3,2).reshape(-1,kh*kw);flat_b=b.permute(0,1,3,2).reshape_as(flat_a)
    flat_v=valid.permute(0,1,3,2).reshape_as(flat_a);q=torch.zeros_like(flat_a)
    for start in range(0,len(q),4096):
        aa=flat_a[start:start+4096];bb=flat_b[start:start+4096];vv=flat_v[start:start+4096]
        cuts=[torch.zeros(len(aa),device=x.device,dtype=x.dtype),torch.ones(len(aa),device=x.device,dtype=x.dtype)]
        for i in range(kh*kw):
            for j in range(i):
                den=bb[:,i]-bb[:,j];v=(aa[:,j]-aa[:,i])/torch.where(den!=0,den,torch.ones_like(den))
                v=torch.where((den!=0)&vv[:,i]&vv[:,j],v,torch.zeros_like(v));cuts.append(v.clamp(0,1))
        cuts=torch.stack(cuts,1).sort(1).values
        out=q[start:start+4096]
        for j in range(cuts.shape[1]-1):
            length=cuts[:,j+1]-cuts[:,j];mid=(cuts[:,j]+cuts[:,j+1])/2
            winner=(aa+bb*mid[:,None]).masked_fill(~vv,-torch.inf).argmax(1)
            out.scatter_add_(1,winner[:,None],length[:,None])
    q=q.reshape(a.shape[0],a.shape[1],a.shape[3],a.shape[2]).permute(0,1,3,2).contiguous()
    structural=valid.to(x.dtype)/valid.sum(2,keepdim=True)
    return q,structural,valid

class Tape:
    def __init__(self,x,ref):
        if x.shape!=ref.shape or x.shape[0]!=1:raise ValueError('Analysis uses one matched pair')
        self.nodes=[Node('input',x.detach().clone(),ref.detach().clone(),[],[])];self.parameter_ids={}
    @torch.no_grad()
    def add(self,name,kind,parents,**kw):
        xs=[self.nodes[i].x for i in parents];rs=[self.nodes[i].ref for i in parents];x,r=xs[0],rs[0]
        edges=[]
        if kind=='module':
            m=kw['module'];y=m(x);y0=m(r)
            if isinstance(m,nn.Conv2d):
                if m.padding_mode!='zeros':raise ValueError('Only zero padding supported')
                edges=[Edge('conv',x.shape,m.weight.detach(),stride=m.stride,padding=m.padding,dilation=m.dilation,groups=m.groups)]
            elif isinstance(m,nn.Linear):edges=[Edge('linear',x.shape,m.weight.detach())]
            elif isinstance(m,nn.BatchNorm2d):
                if m.training or not m.track_running_stats:raise ValueError('BatchNorm requires evaluation mode and frozen running statistics')
                w=m.weight/torch.sqrt(m.running_var+m.eps) if m.affine else torch.rsqrt(m.running_var+m.eps)
                edges=[Edge('diag',x.shape,w[None,:,None,None])]
            elif isinstance(m,nn.AvgPool2d):
                if m.ceil_mode:raise ValueError('Ceil-mode average pooling requires a separate response handler')
                kh,kernel_width=m.kernel_size if isinstance(m.kernel_size,tuple) else (m.kernel_size,m.kernel_size)
                stride=m.stride or m.kernel_size;valid=F.unfold(torch.ones_like(x),m.kernel_size,padding=m.padding,stride=stride).reshape(x.shape[0],x.shape[1],kh*kernel_width,-1)
                den=m.divisor_override or (kh*kernel_width if m.count_include_pad else valid.sum(2,keepdim=True))
                w=valid/den;edges=[Edge('pool',x.shape,w,kernel=m.kernel_size,padding=m.padding,stride=stride,output=y.shape)]
                original=edges[0].coeff;edges[0].coeff=lambda mode,power,part,orig=original,v=valid:orig(mode,power,part)*v if mode=='topology' else orig(mode,power,part)
            elif isinstance(m,nn.MaxPool2d):
                q,struct,valid=pool_routes(x,r,m)
                e=Edge('pool',x.shape,q,structural=struct,kernel=m.kernel_size,stride=m.stride,padding=m.padding,output=y.shape)
                # Topology keeps only valid spatial candidates.
                original=e.coeff
                e.coeff=lambda mode,power,part,orig=original,v=valid:orig(mode,power,part)*v if mode=='topology' else orig(mode,power,part)
                edges=[e]
            else:raise ValueError('Unsupported module '+type(m).__name__)
        elif kind in ['relu','tanh']:
            y=x.relu() if kind=='relu' else x.tanh();y0=r.relu() if kind=='relu' else r.tanh();d=x-r
            w=torch.where(d!=0,(y-y0)/torch.where(d!=0,d,torch.ones_like(d)),((x>0).to(x.dtype) if kind=='relu' else 1-x.tanh().square()))
            edges=[Edge('diag',x.shape,w,structural=torch.ones_like(w))]
        elif kind=='affine':
            w=kw['factor'];y=x*w+kw['offset'];y0=r*w+kw['offset'];edges=[Edge('diag',x.shape,w)]
        elif kind=='gate':
            w=kw['factor'];y=x*w;y0=r*w;edges=[Edge('diag',x.shape,w,structural=torch.ones_like(w))]
        elif kind=='add':
            if any(a.shape!=x.shape for a in xs):raise ValueError('Additive merge requires matching tensor shapes')
            coeff=kw.get('coefficients',[1.]*len(xs))
            y=sum(a*v for a,v in zip(xs,coeff));y0=sum(a*v for a,v in zip(rs,coeff));edges=[Edge('diag',a.shape,a.new_tensor(v)) for a,v in zip(xs,coeff)]
        elif kind=='cat':
            y=torch.cat(xs,1);y0=torch.cat(rs,1);start=0
            for a in xs:
                edges.append(Edge('slice',a.shape,a.new_tensor(1.),start=start,stop=start+a.shape[1],output=y.shape));start+=a.shape[1]
        elif kind=='resize':
            size=kw['size'];y=F.interpolate(x,size=size,mode='nearest');y0=F.interpolate(r,size=size,mode='nearest')
            idx=F.interpolate(torch.arange(x.shape[-2]*x.shape[-1],device=x.device,dtype=x.dtype).reshape(1,1,*x.shape[-2:]),size=size,mode='nearest').long().flatten()
            edges=[Edge('resize',x.shape,x.new_tensor(1.),indices=idx,output=y.shape)]
        elif kind=='gap':
            y=x.mean((-2,-1));y0=r.mean((-2,-1));edges=[Edge('gap',x.shape,x.new_tensor(1/(x.shape[-2]*x.shape[-1])))]
        elif kind=='flatten':
            y=x.flatten(1);y0=r.flatten(1);edges=[Edge('reshape',x.shape,x.new_tensor(1.),output=y.shape)]
        elif kind=='identity':
            y=x;y0=r;edges=[Edge('diag',x.shape,x.new_tensor(1.))]
        elif kind=='readout':
            w=kw['weight'];y=(x*w).sum().reshape(1);y0=(r*w).sum().reshape(1);edges=[Edge('dot',x.shape,w)]
        else:raise ValueError(kind)
        if kind=='module' and isinstance(kw['module'],(nn.Conv2d,nn.Linear)):
            m=kw['module'];pid=self.parameter_ids.setdefault(id(m.weight),name+'.weight')
            for edge in edges:edge.kw['parameter_identity']=pid
        self.nodes.append(Node(name,y.detach().clone(),y0.detach().clone(),parents,edges));return len(self.nodes)-1
    def attenuate(self,index,factor):
        """Postactivation intervention changes a response, not graph topology."""
        node=self.nodes[index]
        if len(node.edges)!=1 or node.edges[0].kind!='diag' or 'structural' not in node.edges[0].kw:
            raise ValueError('Gate attenuation requires a captured activation response')
        if not bool(torch.isfinite(factor).all()) or bool(((factor<0)|(factor>1)).any()):raise ValueError('Invalid attenuation')
        node.x=node.x*factor;node.ref=node.ref*factor
        node.edges[0].weight=node.edges[0].weight*factor
        return index

    def readout(self,heads,spec):
        kind=spec['head'];idx=heads[kind];z=self.nodes[idx].x;w=torch.zeros_like(z)
        if kind=='classification':
            a=int(spec.get('a',1));b=spec.get('b',0)
            if not 0<=a<z.shape[1] or (b is not None and (not 0<=int(b)<z.shape[1] or int(b)==a)):raise ValueError('Choose distinct valid class indices')
            w[:,a]=1
            if b is not None:w[:,int(b)]=-1
        else:
            h,ww=z.shape[-2:];x0,y0,x1,y1=map(int,spec.get('region',[0,0,ww,h]))
            if not (0<=x0<x1<=ww and 0<=y0<y1<=h):raise ValueError('Invalid fixed mask region')
            w[:,:,y0:y1,x0:x1]=1/((y1-y0)*(x1-x0))
        return self.add('readout','readout',[idx],weight=w)

def norm(x,scale=0.):
    if not bool(torch.isfinite(x).all()):raise FloatingPointError('Non-finite path message')
    if bool((x<0).any()):raise FloatingPointError('Negative positive-mass message')
    m=float(x.max())
    return (x/m,scale+math.log(m)) if m>0 else (x, -math.inf)

def plus(a,b):
    if a is None:return b
    if b is None:return a
    x,s=a;y,t=b
    if s==-math.inf:return b
    if t==-math.inf:return a
    u=max(s,t);return norm(x*math.exp(s-u)+y*math.exp(t-u),u)

def advance(edge,msg,mode,power=2,part='absolute',transpose=False):
    x,s=msg
    return norm(edge.apply(x,mode,power,part,transpose),s)

class Measure:
    def __init__(self,tape,sink,mode='conditional',source_mask=None,power=2):
        self.tape,self.sink,self.mode=tape,sink,mode;self.power=power
        nodes=tape.nodes;self.alpha=[None]*len(nodes);self.beta=[None]*len(nodes)
        source=(nodes[0].x-nodes[0].ref).abs().pow(power) if mode=='conditional' else torch.ones_like(nodes[0].x)
        if source_mask is not None:source=source*source_mask
        self.alpha[0]=norm(source)
        for i,n in enumerate(nodes[1:sink+1],1):
            a=None
            for j,e in zip(n.parents,n.edges):a=plus(a,advance(e,self.alpha[j],mode,power=power))
            self.alpha[i]=a
        self.logQ=self.alpha[sink][1]+math.log(float(self.alpha[sink][0].sum())) if self.alpha[sink][1]!=-math.inf else -math.inf
        self.beta[sink]=norm(torch.ones_like(nodes[sink].x))
        for i in range(sink,0,-1):
            if self.beta[i] is None:continue
            for j,e in zip(nodes[i].parents,nodes[i].edges):self.beta[j]=plus(self.beta[j],advance(e,self.beta[i],mode,power=power,transpose=True))
    def node_mass(self,i):
        a,b=self.alpha[i],self.beta[i]
        if b is None or self.logQ==-math.inf:return torch.zeros_like(self.tape.nodes[i].x)
        return a[0]*b[0]*math.exp(a[1]+b[1]-self.logQ)
    def branch_mass(self,i,k):
        if self.beta[i] is None or self.logQ==-math.inf:return 0.
        j=self.tape.nodes[i].parents[k];e=self.tape.nodes[i].edges[k]
        f=advance(e,self.alpha[j],self.mode,power=self.power);b=self.beta[i]
        return float((f[0]*b[0]).sum())*math.exp(f[1]+b[1]-self.logQ) if f[1]!=-math.inf and b[1]!=-math.inf else 0.
    def signed_origin(self,power):
        n=self.tape.nodes;b=[None]*len(n);b[self.sink]=(norm(torch.ones_like(n[self.sink].x)),norm(torch.zeros_like(n[self.sink].x)))
        for i in range(self.sink,0,-1):
            if b[i] is None:continue
            pos,neg=b[i]
            for j,e in zip(n[i].parents,n[i].edges):
                pp=plus(advance(e,pos,self.mode,power,'positive',True),advance(e,neg,self.mode,power,'negative',True))
                nn=plus(advance(e,neg,self.mode,power,'positive',True),advance(e,pos,self.mode,power,'negative',True))
                b[j]=(pp,nn) if b[j] is None else (plus(b[j][0],pp),plus(b[j][1],nn))
        dx=n[0].x-n[0].ref;pos,neg=b[0];s=max(pos[1],neg[1])
        if s==-math.inf:return torch.zeros_like(dx),torch.zeros_like(dx),s
        p=pos[0]*math.exp(pos[1]-s);q=neg[0]*math.exp(neg[1]-s)
        return torch.where(dx>=0,p,q)*dx.abs().pow(power),torch.where(dx>=0,q,p)*dx.abs().pow(power),s
    def signed_reconstruction(self):
        # Signed linear propagation independently checks the decomposition.
        n=self.tape.nodes;v=[n[0].x-n[0].ref]
        residual=0.
        for node in n[1:self.sink+1]:
            y=sum(e.apply(v[j],self.mode,1,'positive')-e.apply(v[j],self.mode,1,'negative') for j,e in zip(node.parents,node.edges))
            v.append(y);residual=max(residual,float((y-(node.x-node.ref)).abs().max()))
        return float(v[self.sink]),residual
