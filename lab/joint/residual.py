"""Graph-driven residual builders. Model names select builders, never response rules."""
from collections import OrderedDict
import operator
import torch
from torch import nn, fx
import torch.nn.functional as F

class AffineInput(nn.Module):
    def __init__(self):
        super().__init__();self.register_buffer('scale',1/torch.tensor([.229,.224,.225]).reshape(1,3,1,1));self.register_buffer('offset',-torch.tensor([.485,.456,.406]).reshape(1,3,1,1)*self.scale)
    def forward(self,x):return x*self.scale+self.offset

class Block(nn.Module):
    def __init__(self,ci,co,stride=1,activation='relu',branches=2,preactivation=False):
        super().__init__();act=nn.ReLU if activation=='relu' else nn.Tanh
        self.pre=act() if preactivation else nn.Identity()
        self.routes=nn.ModuleList([nn.Identity() if ci==co and stride==1 else nn.Conv2d(ci,co,1,stride,bias=False)])
        for _ in range(branches-1):self.routes.append(nn.Sequential(nn.Conv2d(ci,co,3,stride,1),act(),nn.Conv2d(co,co,3,1,1)))
        self.post=nn.Identity() if preactivation else act()
    def forward(self,x):
        z=self.pre(x);y=self.routes[0](z)
        for b in self.routes[1:]:y=y+b(z)
        return self.post(y)

class SmallCore(nn.Module):
    def __init__(self,c):
        super().__init__();act=nn.ReLU if c['activation']=='relu' else nn.Tanh
        widths=c['stage_widths'];self.stem=nn.Sequential(OrderedDict(conv=nn.Conv2d(3,widths[0],3,2,1),relu=act()))
        blocks=[];ci=widths[0];h=(c['size']+1)//2
        for i,(co,depth) in enumerate(zip(widths,c['stage_depths'])):
            for j in range(depth):
                stride=2 if i and j==0 else 1
                blocks.append(Block(ci,co,stride,c['activation'],c['branches'],c['preactivation']));ci=co;h=(h+stride-1)//stride
        self.blocks=nn.Sequential(*blocks)
        self.pool=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(1)) if c['reduction']=='gap' else nn.Flatten(1)
        n=ci if c['reduction']=='gap' else ci*h*h;layers=[]
        for w in c['head_widths'] if c['hidden_head'] else []:layers.extend([nn.Linear(n,w),act()]);n=w
        layers.append(nn.Linear(n,c['classes']));self.head=nn.Sequential(*layers)
    def forward(self,x):return self.head(self.pool(self.blocks(self.stem(x))))

class Tracer(fx.Tracer):
    def is_leaf_module(self,m,qualname):return isinstance(m,AffineInput) or super().is_leaf_module(m,qualname)

class Program(fx.Interpreter):
    def __init__(self,graph,tape=None,gates=None,capture=None):
        super().__init__(graph);self.tape=tape;self.gates=gates or {};self.capture=capture if capture is not None else {};self.used_names=set()
    def run_node(self,n):
        args,kwargs=self.fetch_args_kwargs_from_env(n);t=self.tape
        if n.op in ['placeholder','output','get_attr']:return super().run_node(n)
        name=str(n.target) if n.op=='call_module' else n.name
        if name in self.used_names:name+='@'+n.name
        self.used_names.add(name)
        module=self.fetch_attr(n.target) if n.op=='call_module' else None
        kind=None;extra={};parents=[]
        if module is not None:
            parents=[args[0]]
            if isinstance(module,(nn.Conv2d,nn.Linear,nn.BatchNorm2d,nn.MaxPool2d,nn.AvgPool2d)):kind='module';extra={'module':module}
            elif isinstance(module,nn.ReLU):kind='relu'
            elif isinstance(module,nn.Tanh):kind='tanh'
            elif isinstance(module,nn.Flatten):
                if module.start_dim!=1 or module.end_dim!=-1:raise ValueError('Only batch-preserving flatten supported')
                kind='flatten'
            elif isinstance(module,nn.AdaptiveAvgPool2d):
                if module.output_size not in [1,(1,1)]:raise ValueError('Only global adaptive average pooling supported')
                kind='gap4'
            elif isinstance(module,nn.Identity):kind='identity'
            elif isinstance(module,AffineInput):kind='affine';extra={'factor':module.scale,'offset':module.offset}
        elif n.op=='call_function':
            if n.target in [operator.add,torch.add]:
                if len(args)!=2 or kwargs:raise ValueError('Only tensor additive merge without alpha supported')
                kind='add';parents=list(args)
            elif n.target==torch.flatten:
                if args[1:] not in [(1,),()] or kwargs:raise ValueError('Unsupported flatten')
                kind='flatten';parents=[args[0]]
            elif n.target in [torch.relu,F.relu,torch.tanh]:kind='tanh' if n.target==torch.tanh else 'relu';parents=[args[0]]
        if kind is None:raise ValueError(f'No exact response handler for {n.op}: {n.target}')
        if t is not None:
            if kind=='gap4':
                v=t.add(name,'gap',parents);shape=t.nodes[v].x.shape
                # Preserve actual AdaptiveAvgPool2d shape, then flatten explicitly later.
                from .tape import Node,Edge
                a=t.nodes[v];a.x=a.x[:,:,None,None];a.ref=a.ref[:,:,None,None]
                e=a.edges[0];e.kind='gap4'
                y=v
            else:y=t.add(name,kind,parents,**extra)
        else:y=super().run_node(n)
        if kind in ['relu','tanh']:
            shape=t.nodes[y].x.shape if t is not None else y.shape
            if len(shape)==4:
                if name in self.gates:
                    ch,alpha=self.gates[name]
                    if not 0<=ch<shape[1] or not 0<=alpha<=1:raise ValueError('Invalid gate')
                    template=t.nodes[y].x if t is not None else y;factor=template.new_ones((1,shape[1],1,1));factor[:,ch]=alpha
                    y=t.attenuate(y,factor) if t is not None else y*factor
                self.capture[name]=y
        return y

class ResidualModel(nn.Module):
    def __init__(self,c,pretrained=False):
        super().__init__();self.config=c
        if c['architecture']=='resnet50':
            from torchvision.models import resnet50,ResNet50_Weights
            body=resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
            n=body.fc.in_features;seq=[]
            for w in c['head_widths'] if c['hidden_head'] else []:seq.extend([nn.Linear(n,w),nn.ReLU()]);n=w
            seq.append(nn.Linear(n,c['classes']));body.fc=nn.Sequential(*seq)
            core=nn.Sequential(OrderedDict(preprocess=AffineInput(),encoder=body))
        else:core=SmallCore(c)
        for m in core.modules():
            if isinstance(m,nn.ReLU):m.inplace=False
        tracer=Tracer();self.program=fx.GraphModule(core,tracer.trace(core))
    def forward(self,x,recorder=None,gates=None,capture=None):
        y=Program(self.program,recorder,gates,capture).run(x)
        return {'classification':y}
