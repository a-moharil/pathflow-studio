"""Validated configuration and architecture cost estimates."""
DEFAULT = dict(name='New experiment', dataset='synthetic', model='mlp', widths=[16]*6,
               channels=[4,6,8], kernel=3, stride=2, padding=1, activation='relu', bias=True,
               optimizer='Adam', lr=.001, decay=0., momentum=0., beta1=.9, beta2=.999, eps=1e-8, alpha=.99, centered=False, amsgrad=False, dampening=0., nesterov=False, test_every=1, batch_size=32,
               updates=100, checkpoint_every=10, seed=42, train_size=512,
               device='cpu', geometry_limit=500, landmarks=64, measurement_limit=2500, geometry_every=0)

FIELD_CLASSES = {'synthetic-interference': 3, 'synthetic-orbits': 4}


def class_count(c):
    """Number of outputs: 1 for regression, 10 for Fashion-MNIST, the field datasets' own
    counts, otherwise 2."""
    if c.get('task')=='regression':return 1
    if c['dataset']=='fashion':return 10
    return FIELD_CLASSES.get(c['dataset'],2)


def validate(raw):
    c=DEFAULT|raw
    c['task']=raw.get('task','classification')
    if c['task'] not in ['classification','regression']:raise ValueError('Invalid task')
    if (c['task']=='regression') != (c['dataset']=='synthetic-regression'):raise ValueError('Use synthetic-regression for regression, or a classification dataset for classification')
    # CNN readout: 'dense' flattens into a linear output (optionally after a dense hidden
    # head); 'gap' is fully convolutional: 1x1 convolution to the outputs, then global
    # spatial averaging. Configurations without the key keep the original dense readout.
    c['cnn_readout']=raw.get('cnn_readout','dense')
    if c['cnn_readout'] not in ['dense','gap']:raise ValueError('CNN readout must be dense or gap')
    c['cnn_head']=raw.get('cnn_head',False)
    if not isinstance(c['cnn_head'],bool):raise ValueError('CNN hidden MLP must be boolean')
    if c['model']=='cnn' and c['cnn_readout']=='gap' and c['cnn_head']:
        raise ValueError('A dense hidden head needs the flatten + dense readout; turn the head off for the 1×1 convolution + global average readout.')
    head_required=c['model']=='cnn' and c['cnn_head']
    v=raw.get('head_widths',[32,16])
    try:c['head_widths']=[int(i.strip()) for i in v.split(',') if i.strip()] if isinstance(v,str) else list(v)
    except (TypeError,ValueError):
        if head_required:raise ValueError('Head widths must be comma-separated integers, for example 32,16')
        c['head_widths']=[32,16]
    if not 1<=len(c['head_widths'])<=8 or any(type(i)!=int or not 1<=i<=256 for i in c['head_widths']):
        if head_required:raise ValueError('Head widths must contain 1..8 widths in 1..256')
        c['head_widths']=[32,16]  # unused while the head is off
    if c['optimizer']=='RMSProp': c['optimizer']='RMSprop'
    for key,options in dict(dataset=['synthetic','fashion','pneumonia','synthetic-regression','synthetic-interference','synthetic-orbits'], model=['mlp','cnn'],
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
    # Optional per-layer kernel sizes (one per convolution layer); blank uses `kernel` for all.
    k=raw.get('kernels')
    if isinstance(k,str):k=[x.strip() for x in k.split(',') if x.strip()]
    if k:
        try:k=[int(x) for x in k]
        except (TypeError,ValueError):raise ValueError('Kernel sizes must be comma-separated integers, for example 3,3,1')
        if c['model']=='cnn' and len(k)!=len(c['channels']):raise ValueError(f'Give one kernel size per convolution layer ({len(c["channels"])} channels listed, {len(k)} kernel sizes).')
        if any(not 1<=x<=7 for x in k):raise ValueError('Kernel sizes must be 1..7')
        c['kernels']=k
    else:c['kernels']=None
    for key in ['centered','amsgrad','nesterov','bias']:
        if not isinstance(c[key],bool):raise ValueError(f'{key} must be boolean')
    if c['nesterov'] and c['optimizer']=='SGD' and (c['momentum']<=0 or c['dampening']!=0):raise ValueError('Nesterov requires positive momentum and zero dampening')
    # Ambiguity (class overlap) of the synthetic field datasets; recorded only for them.
    if c['dataset'] in FIELD_CLASSES:
        try:c['ambiguity']=round(float(raw.get('ambiguity',.35)),2)
        except (TypeError,ValueError):raise ValueError('Ambiguity must be a number in 0..0.8')
        if not 0<=c['ambiguity']<=.8:raise ValueError('Ambiguity must be in 0..0.8')
        c['field_version']=int(raw.get('field_version',2))
        if c['field_version'] not in (1,2):raise ValueError('Unknown field dataset version')
    else:
        c.pop('ambiguity',None);c.pop('field_version',None)
    c['name']=str(c['name'])[:100];c['bias']=bool(c['bias'])
    estimate(c)
    return c

def estimate(c):
    classes=class_count(c)
    nodes=784;edges=0;params=0;n=784
    if c['model']=='mlp':
        for m in c['widths']+[classes]:
            edges+=n*m;params+=n*m+(m if c['bias'] else 0);nodes+=m;n=m
    else:
        h=28;ch=1
        kernels=c.get('kernels') or [c['kernel']]*len(c['channels'])
        for out,kernel in zip(c['channels'],kernels):
            nh=(h+2*c['padding']-kernel)//c['stride']+1
            if nh<1: raise ValueError('Convolution collapses image; reduce depth/kernel or change stride/padding')
            # Upper bound: border occurrences that fall in padding are not executed.
            edges+=nh*nh*out*ch*kernel**2;params+=out*ch*kernel**2+(out if c['bias'] else 0)
            nodes+=nh*nh*out;h=nh;ch=out
        n=h*h*ch
        if c.get('cnn_readout','dense')=='gap':
            # 1x1 readout convolution to one map per output, then global averaging edges.
            edges+=h*h*classes*ch+h*h*classes;params+=ch*classes+(classes if c['bias'] else 0);nodes+=h*h*classes+classes
        elif c.get('cnn_head',False):
            for m in c.get('head_widths',[32,16]):
                edges+=n*m;params+=n*m+(m if c['bias'] else 0);nodes+=m;n=m
        if c.get('cnn_readout','dense')!='gap':
            edges+=n*classes;params+=n*classes+(classes if c['bias'] else 0);nodes+=classes
    if nodes>30000 or edges>2000000: raise ValueError('Local execution-graph budget exceeded (30,000 nodes / 2 million edges). Choose a narrower or strided model.')
    return dict(nodes=nodes+1,edges_upper_bound=edges+2,parameters=params,full_distance_mb=round((nodes+1)**2*8/1e6,1),checkpoint_mb=round(params*16/1e6,2),measurement_within_budget=nodes+1<=c.get('measurement_limit',2500),distance_source_runs=nodes+1,bcan_radius_count=36,working_memory_planning_mb=round((nodes+1)**2*8*4/1e6,1))
