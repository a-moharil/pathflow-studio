"""Explicit registry. Display names never drive dynamic attribute lookup."""
import torch
REGISTRY={'Adam':torch.optim.Adam,'SGD':torch.optim.SGD,'RMSprop':torch.optim.RMSprop}

def specification(c):
    name=c['optimizer'];kw=dict(lr=c['lr'],weight_decay=c['decay'])
    if name=='Adam':kw.update(betas=(c['beta1'],c['beta2']),eps=c['eps'],amsgrad=c['amsgrad'])
    elif name=='SGD':kw.update(momentum=c['momentum'],dampening=c['dampening'],nesterov=c['nesterov'])
    elif name=='RMSprop':kw.update(alpha=c['alpha'],eps=c['eps'],momentum=c['momentum'],centered=c['centered'])
    else:raise ValueError('Unsupported optimizer')
    return {'class':'torch.optim.'+name,'parameters':kw,'weight_decay_semantics':'coupled L2'}

def build(model,c):
    return REGISTRY[c['optimizer']](model.parameters(),**specification(c)['parameters'])
