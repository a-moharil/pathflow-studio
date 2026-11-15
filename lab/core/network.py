import torch
from torch import nn

class Network(nn.Module):
    def __init__(self,c):
        super().__init__();self.config=c;self.layers=nn.ModuleList();self.shapes=[(1,28,28)]
        n=784;classes=1 if c.get('task')=='regression' else 10 if c['dataset']=='fashion' else 2
        if c['model']=='mlp':
            for m in c['widths']:
                self.layers.append(nn.Linear(n,m,bias=c['bias']));self.shapes.append((m,));n=m
        else:
            ch=1;h=28
            for out in c['channels']:
                self.layers.append(nn.Conv2d(ch,out,c['kernel'],stride=c['stride'],padding=c['padding'],bias=c['bias']))
                h=(h+2*c['padding']-c['kernel'])//c['stride']+1;ch=out;self.shapes.append((ch,h,h))
            n=ch*h*h
            if c.get('cnn_head',False):
                for m in c.get('head_widths',[32,16]):
                    self.layers.append(nn.Linear(n,m,bias=c['bias']));self.shapes.append((m,));n=m
        self.layers.append(nn.Linear(n,classes,bias=c['bias']));self.shapes.append((classes,))
    def activation(self,x):return torch.relu(x) if self.config['activation']=='relu' else torch.tanh(x)
    def forward(self,x,gates=None,trace=False):
        hs=[x];zs=[]
        for l,layer in enumerate(self.layers):
            if isinstance(layer,nn.Linear):x=x.flatten(1)
            z=layer(x);zs.append(z)
            x=self.activation(z) if l<len(self.layers)-1 else z
            if gates and str(l) in gates and l<len(self.layers)-1:
                g=torch.ones(x[0].numel(),device=x.device,dtype=x.dtype)
                for idx,alpha in gates[str(l)].items():g[int(idx)]=float(alpha)
                x=x*g.reshape(x.shape[1:])
            hs.append(x)
        return (x,hs,zs) if trace else x
