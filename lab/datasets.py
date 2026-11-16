"""Fixed split identities and deterministic subsets; downloads only when requested."""
from pathlib import Path
import numpy as np
from functools import lru_cache
from .fields_data import DEFAULT_AMBIGUITY,generate,is_field

@lru_cache(maxsize=2)
def _read_cache(path):
    with np.load(path) as z:return {k:z[k] for k in z.files}

def _stem(c):
    stem=f"{c['dataset']}_{c['seed']}_{c['train_size']}"
    # Field datasets also depend on their ambiguity; existing cache names are unchanged.
    if is_field(c['dataset']):
        stem+=f"_a{int(round(c.get('ambiguity',DEFAULT_AMBIGUITY)*100)):02d}"
        if c.get('field_version',1)>1:stem+=f"_v{c['field_version']}"
    return stem

def load(root,c,download=False):
    folder=Path(root)/'datasets';folder.mkdir(exist_ok=True)
    cache=folder/f"{_stem(c)}.npz"
    if cache.exists():
        return _read_cache(str(cache))
    rng=np.random.default_rng(c['seed']);out={}
    if is_field(c['dataset']):
        # Runs recorded before 0.16.3 have no field_version and keep the version-1 rendering.
        out=generate(c['dataset'],c['seed'],c['train_size'],c.get('ambiguity',DEFAULT_AMBIGUITY),c.get('field_version',1))
    elif c['dataset'] in ['synthetic','synthetic-regression']:
        for split,n in [('train',c['train_size']),('val',64),('test',80)]:
            y=np.arange(n)%2;x=rng.normal(.1,.07,(n,1,28,28)).astype('float32')
            for i in range(n):
                off=int(rng.integers(8,18))
                if y[i]:x[i,0,off:off+3,:]+=.7
                else:x[i,0,:,off:off+3]+=.7
            x=np.clip(x,0,1)
            # A deterministic image regression target with known regional mechanism.
            if c['dataset']=='synthetic-regression':
                y=(x[:,0,:,:14].mean((1,2))-x[:,0,:,14:].mean((1,2))).astype('float32')
            out.update({split+'_x':x,split+'_y':y if c['dataset']=='synthetic-regression' else y.astype('int64'),split+'_ids':np.arange(n)})
    else:
        if c['dataset']=='fashion':
            from torchvision.datasets import FashionMNIST
            train=FashionMNIST(folder,train=True,download=download);test=FashionMNIST(folder,train=False,download=download)
            ids=rng.permutation(len(train));parts={'train':(train,ids[6000:]),'val':(train,ids[:6000]),'test':(test,np.arange(len(test)))}
        else:
            from medmnist import PneumoniaMNIST
            parts={split:(PneumoniaMNIST(root=str(folder),split=split,download=download,size=28),None) for split in ['train','val','test']}
        for split,(ds,ids) in parts.items():
            if ids is None:ids=np.arange(len(ds))
            if split=='train':
                # Seeded uniform subset; preserve selected source indices.
                ids=rng.permutation(ids)[:min(c['train_size'],len(ids))]
            images=[];labels=[]
            for i in ids:
                image,label=ds[int(i)];images.append(np.asarray(image,dtype='float32')[None]/255.);labels.append(int(np.asarray(label).reshape(-1)[0]))
            out.update({split+'_x':np.stack(images),split+'_y':np.array(labels,dtype='int64'),split+'_ids':np.asarray(ids)})
    # A private temporary name per writer: the web process and the worker may build the
    # same cache at once, and a shared temporary file let one writer replace it mid-write.
    import os,uuid
    tmp=cache.with_name(f'{cache.stem}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp.npz')
    try:
        np.savez_compressed(tmp,**out);tmp.replace(cache)
    finally:
        tmp.unlink(missing_ok=True)
    return out


def cache_path(root,c):
    return Path(root)/'datasets'/f"{_stem(c)}.npz"


def status(root,c):
    """Read-only readiness check; no preparation or download side effects."""
    p=cache_path(root,c)
    if p.exists():
        try:
            data=_read_cache(str(p));splits={}
            for split in ['train','val','test']:
                x=data[split+'_x'];y=data[split+'_y']
                if len(x)!=len(y) or x.shape[1:]!=(1,28,28) or not np.isfinite(x).all():raise ValueError('Prepared data failed shape/finite checks')
                splits[split]={'count':len(y),'target_range':[float(y.min()),float(y.max())]} if c['dataset']=='synthetic-regression' else {'count':len(y),'classes':np.bincount(y).tolist()}
            return {'state':'ready','splits':splits,'key':p.stem}
        except Exception as e:return {'state':'invalid','message':str(e),'key':p.stem}
    if c['dataset'] in ['synthetic','synthetic-regression'] or is_field(c['dataset']):return {'state':'prepare','message':'Generate deterministic synthetic data','key':p.stem}
    folder=Path(root)/'datasets'
    if c['dataset']=='fashion':
        files=['train-images-idx3-ubyte','train-labels-idx1-ubyte','t10k-images-idx3-ubyte','t10k-labels-idx1-ubyte']
        downloaded=all((folder/'FashionMNIST'/'raw'/name).exists() for name in files)
    else:downloaded=(folder/'pneumoniamnist.npz').exists()
    return {'state':'prepare' if downloaded else 'missing','message':'Raw files found; verify and prepare split' if downloaded else 'Download required','key':p.stem}


def prepare(root,c,report=lambda msg:None):
    from filelock import FileLock
    folder=Path(root)/'datasets';folder.mkdir(parents=True,exist_ok=True)
    with FileLock(str(folder/'prepare.lock')):
        report('Checking cached files and preparing the selected split…')
        p=cache_path(root,c)
        if status(root,c)['state']=='invalid':
            _read_cache.cache_clear();p.rename(p.with_suffix('.invalid.npz'))
        if not p.exists() and c['dataset']!='synthetic' and not is_field(c['dataset']):
            report('Downloading or verifying source dataset files; byte progress is shown in the Python console…')
        load(root,c,download=True)
        report('Verifying prepared images and labels…')
        ready=status(root,c)
        if ready['state']!='ready':raise ValueError(ready.get('message','Dataset verification failed'))
        return ready
