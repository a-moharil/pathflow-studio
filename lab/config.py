"""Validated configuration and architecture cost estimates."""
DEFAULT = dict(name='New experiment', dataset='synthetic', model='mlp', widths=[16]*6,
               channels=[4,6,8], kernel=3, stride=2, padding=1, activation='relu', bias=True,
               optimizer='Adam', lr=.001, decay=0., momentum=0., beta1=.9, beta2=.999, eps=1e-8, alpha=.99, centered=False, amsgrad=False, dampening=0., nesterov=False, test_every=1, batch_size=32,
               updates=100, checkpoint_every=10, seed=42, train_size=512,
               device='cpu', geometry_limit=500, landmarks=64, measurement_limit=2500, geometry_every=0)

def validate(raw):
    c=DEFAULT|raw
    if c['optimizer']=='RMSProp': c['optimizer']='RMSprop'
    for key,options in dict(dataset=['synthetic','fashion','pneumonia'], model=['mlp','cnn'],
                            activation=['relu','tanh'],optimizer=['Adam','SGD','RMSprop'],device=['cpu','cuda','auto']).items():
        if c[key] not in options: raise ValueError(f'Invalid {key}')
    for key,lo,hi in [('measurement_limit',50,4000),('geometry_every',0,100000),('test_every',1,100),('batch_size',1,512),('updates',1,100000),('checkpoint_every',1,1000),('seed',0,2147483647),('train_size',32,60000),('geometry_limit',50,1200),('landmarks',8,128),('kernel',1,7),('stride',1,3),('padding',0,3)]:
        c[key]=int(c[key])
        if not lo<=c[key]<=hi:raise ValueError(f'{key} must be {lo}..{hi}')
    for key,lo,hi in [('lr',1e-6,1.),('decay',0.,1.),('momentum',0.,.999),('beta1',0.,.999999),('beta2',0.,.999999),('alpha',0.,.999999),('eps',1e-16,1.),('dampening',0.,1.)]:
        c[key]=float(c[key])
        if not lo<=c[key]<=hi:raise ValueError(f'Invalid {key}')
    for key in ['widths','channels']:
        if isinstance(c[key],str): c[key]=[int(x.strip()) for x in c[key].split(',')]
        if not 1<=len(c[key])<=16 or not all(isinstance(v,int) and 1<=v<=256 for v in c[key]):raise ValueError(f'Invalid {key}')
    for key in ['centered','amsgrad','nesterov','bias']:
        if not isinstance(c[key],bool):raise ValueError(f'{key} must be boolean')
    if c['nesterov'] and c['optimizer']=='SGD' and (c['momentum']<=0 or c['dampening']!=0):raise ValueError('Nesterov requires positive momentum and zero dampening')
    c['name']=str(c['name'])[:100];c['bias']=bool(c['bias'])
    estimate(c)
    return c

def estimate(c):
    classes=10 if c['dataset']=='fashion' else 2
    nodes=784;edges=0;params=0;n=784
    if c['model']=='mlp':
        for m in c['widths']+[classes]:
            edges+=n*m;params+=n*m+(m if c['bias'] else 0);nodes+=m;n=m
    else:
        h=28;ch=1
        for out in c['channels']:
            nh=(h+2*c['padding']-c['kernel'])//c['stride']+1
            if nh<1: raise ValueError('Convolution collapses image; reduce depth/kernel or change stride/padding')
            edges+=nh*nh*out*ch*c['kernel']**2;params+=out*ch*c['kernel']**2+(out if c['bias'] else 0)
            nodes+=nh*nh*out;h=nh;ch=out
        n=h*h*ch;edges+=n*classes;params+=n*classes+(classes if c['bias'] else 0);nodes+=classes
    if nodes>30000 or edges>2000000: raise ValueError('Local execution-graph budget exceeded (30,000 nodes / 2 million edges). Choose a narrower or strided model.')
    return dict(nodes=nodes+1,edges_upper_bound=edges+2,parameters=params,full_distance_mb=round((nodes+1)**2*8/1e6,1),checkpoint_mb=round(params*16/1e6,2))
