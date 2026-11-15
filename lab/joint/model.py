"""Shared ResNet encoder and two parallel heads, with one forward definition.
The optional recorder evaluates this same program at both contrast endpoints.
"""
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

class JointModel(nn.Module):
    def __init__(self, classes=3, width=32, pretrained=False, tiny=False, classification_only=False):
        super().__init__()
        if tiny:  # Test fixture only, never offered as ResNet-50 in the UI.
            from torchvision.models.resnet import ResNet, BasicBlock
            self.encoder=ResNet(BasicBlock,[1,1,1,1],num_classes=classes)
            channels=[64,128,256,512]
        else:
            self.encoder=resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
            channels=[256,512,1024,2048]
        self.encoder.fc=nn.Identity()
        self.classifier=nn.Linear(channels[-1],classes)
        self.classification_only=classification_only
        self.decoder=nn.ModuleList()
        incoming=channels[-1]
        for skip in ([] if classification_only else [channels[2],channels[1],channels[0],64]):
            self.decoder.append(nn.ModuleList([nn.Conv2d(incoming+skip,width,3,padding=1),nn.Conv2d(width,width,3,padding=1)]))
            incoming=width
        self.mask=None if classification_only else nn.Conv2d(width,1,1)
        self.tiny=tiny
        for m in self.modules():
            if isinstance(m,nn.ReLU):m.inplace=False

    def forward(self,x,recorder=None,gates=None,capture=None):
        gates=gates or {};capture={} if capture is None else capture
        def op(name,kind,inputs,**kw):
            if recorder is not None:
                y=recorder.add(name,kind,inputs,**kw)
            else:
                vals=inputs
                if kind=='module':y=kw['module'](vals[0])
                elif kind=='relu':y=F.relu(vals[0])
                elif kind=='add':y=vals[0]+vals[1]
                elif kind=='cat':y=torch.cat(vals,1)
                elif kind=='resize':y=F.interpolate(vals[0],size=kw['size'],mode='nearest')
                elif kind=='gap':y=vals[0].mean((-2,-1))
                elif kind=='gate':y=vals[0]*kw['factor']
                elif kind=='affine':y=vals[0]*kw['factor']+kw['offset']
                else:raise ValueError(kind)
            if kind=='relu':
                if name in gates:
                    channel,alpha=gates[name]
                    shape=recorder.nodes[y].x.shape if recorder is not None else y.shape
                    if not 0<=channel<shape[1] or not 0<=alpha<=1:raise ValueError('Invalid channel gate')
                    template=recorder.nodes[y].x if recorder is not None else y
                    factor=torch.ones((1,shape[1],1,1),dtype=template.dtype,device=template.device)
                    factor[:,channel]=alpha
                    y=recorder.attenuate(y,factor) if recorder is not None else y*factor
                capture[name]=y
            return y
        def mod(name,m,z):return op(name,'module',[z],module=m)
        def shape(z):return recorder.nodes[z].x.shape[-2:] if recorder is not None else z.shape[-2:]
        original=shape(x);e=self.encoder
        template=recorder.nodes[x].x if recorder is not None else x
        std=template.new_tensor([.229,.224,.225]).reshape(1,3,1,1);mean=template.new_tensor([.485,.456,.406]).reshape(1,3,1,1)
        x=op('preprocess','affine',[x],factor=1/std,offset=-mean/std)
        z=mod('stem.conv',e.conv1,x);z=mod('stem.bn',e.bn1,z);stem=op('stem.relu','relu',[z])
        z=mod('stem.pool',e.maxpool,stem);stages=[]
        for si,layer in enumerate([e.layer1,e.layer2,e.layer3,e.layer4]):
            for bi,b in enumerate(layer):
                n=f'encoder.stage{si+1}.block{bi+1}';identity=z
                h=mod(n+'.conv1',b.conv1,z);h=mod(n+'.bn1',b.bn1,h);h=op(n+'.relu1','relu',[h])
                h=mod(n+'.conv2',b.conv2,h);h=mod(n+'.bn2',b.bn2,h)
                if hasattr(b,'conv3'):
                    h=op(n+'.relu2','relu',[h]);h=mod(n+'.conv3',b.conv3,h);h=mod(n+'.bn3',b.bn3,h)
                if b.downsample is not None:
                    identity=mod(n+'.shortcut.conv',b.downsample[0],identity);identity=mod(n+'.shortcut.bn',b.downsample[1],identity)
                z=op(n+'.add','add',[identity,h]);z=op(n+'.out','relu',[z])
            stages.append(z)
        cls=mod('classification.logits',self.classifier,op('classification.pool','gap',[z]))
        if self.classification_only:return {'classification':cls}
        for di,(layers,skip) in enumerate(zip(self.decoder,[stages[2],stages[1],stages[0],stem])):
            n=f'decoder.level{di+1}';z=op(n+'.up','resize',[z],size=shape(skip));z=op(n+'.cat','cat',[z,skip])
            for ci,m in enumerate(layers):z=op(n+f'.relu{ci+1}','relu',[mod(n+f'.conv{ci+1}',m,z)])
        mask=mod('segmentation.logits',self.mask,z);mask=op('segmentation.resize','resize',[mask],size=original)
        return {'classification':cls,'segmentation':mask}
