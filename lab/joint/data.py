"""BUS-UCLM v1 source adapter and explicit manifest importer. Patient splits precede resize.
Downloaded dataset and weights are shared and never removed with an experiment.
"""
import csv,hashlib,json,random,re,shutil,urllib.request,zipfile
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from ..storage import atomic_json

SOURCE='https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/7fvgj4jsp7-1.zip'
CITATION='Vallez et al., BUS-UCLM, Mendeley Data v1, doi:10.17632/7fvgj4jsp7.1; Scientific Data 12, 242 (2025). CC BY-NC 3.0 according to the v1 repository record.'
CLASSES=['normal','benign','malignant']

def default_path(root):
    settings=Path(root)/'joint_data'/'settings.json'
    if settings.exists():return json.loads(settings.read_text()).get('bus_uclm_path','')
    manifest=Path(root)/'joint_data'/'bus-uclm-v1'/'manifest.json'
    if manifest.exists():
        d=json.loads(manifest.read_text())
        if d.get('source_directory'):return d['source_directory']
        if d.get('rows'):
            import os
            return os.path.commonpath([str(Path(r[k]).parent) for r in d['rows'] for k in ['image','mask']])
    return str(Path.home()/'datasets'/'bus-uclm-v1')

def save_default(root,path):
    base=Path(path).expanduser().resolve()
    if not base.is_dir():raise ValueError('The default BUS-UCLM folder must exist on the server: '+str(base))
    dest=Path(root)/'joint_data';dest.mkdir(parents=True,exist_ok=True)
    atomic_json(dest/'settings.json',dict(bus_uclm_path=str(base)))
    return str(base)

def prepare(root,progress=lambda s:None,source_dir=None):
    dest=Path(root)/'joint_data'/'bus-uclm-v1';manifest=dest/'manifest.json'
    if manifest.exists():return json.loads(manifest.read_text())
    dest.mkdir(parents=True,exist_ok=True)
    if not source_dir and Path(default_path(root)).is_dir():source_dir=default_path(root)
    base=Path(source_dir).expanduser().resolve() if source_dir else dest/'raw'
    if not source_dir and not base.exists():
        archive=dest/'download.zip.part';progress('Downloading BUS-UCLM v1 from the publisher-linked archive')
        try:
            with urllib.request.urlopen(SOURCE,timeout=60) as src,archive.open('wb') as out:
                count=0
                while True:
                    chunk=src.read(1024*1024)
                    if not chunk:break
                    count+=len(chunk)
                    if count>4*1024**3:raise ValueError('Dataset archive exceeded 4 GiB budget')
                    out.write(chunk)
                    if count%(20*1024**2)<1024**2:progress(f'Downloaded {count/1024**2:.0f} MiB')
            extract=dest/'extracting';shutil.rmtree(extract,ignore_errors=True);extract.mkdir()
            with zipfile.ZipFile(archive) as z:
                if sum(v.file_size for v in z.infolist())>8*1024**3:raise ValueError('Expanded archive exceeds 8 GiB budget')
                for v in z.infolist():
                    p=(extract/v.filename).resolve()
                    if not p.is_relative_to(extract.resolve()) or ((v.external_attr>>16)&0o170000)==0o120000:raise ValueError('Unsafe archive member')
                z.extractall(extract)
            extract.rename(base);archive.unlink()
        except Exception as e:
            raise RuntimeError('BUS-UCLM download failed. Use the official dataset page to download v1, extract it, then use Import extracted BUS-UCLM. '+str(e)) from e
    files=list(base.rglob('*.png'))+list(base.rglob('*.PNG'))
    byname={}
    for p in files:byname.setdefault(p.stem.lower(),[]).append(p)
    rows=[]
    from ..progress import stage
    progress('Indexing BUS-UCLM image/mask pairs')
    for done,(key,paths) in enumerate(sorted(byname.items()),1):
        stage('Indexing BUS-UCLM image/mask pairs',done,len(byname),'filenames')
        if not re.fullmatch(r'[a-z]{4}_\d+',key):continue
        masks=[p for p in paths if any(s in p.parent.name.lower() for s in ['mask','label','annot','ground'])]
        images=[p for p in paths if p not in masks]
        if len(images)!=1 or len(masks)!=1:continue
        img,mask=images[0],masks[0]
        ar=np.asarray(Image.open(mask).convert('RGB'));red=(ar[:,:,0]>128)&(ar[:,:,1]<128);green=(ar[:,:,1]>128)&(ar[:,:,0]<128)
        label=2 if red.any() else 1 if green.any() else 0
        # Explicit image-level rule: any malignant region takes precedence.
        rows.append(dict(id=key,patient=key.split('_')[0],image=str(img.resolve()),mask=str(mask.resolve()),label=label,mask_encoding='rgb-lesion',sha256=hashlib.sha256(img.read_bytes()).hexdigest()))
    if not rows:raise ValueError('No unambiguous BUS-UCLM image/mask pairs found. Use a CSV manifest for a different layout.')
    result=dict(name='BUS-UCLM v1',citation=CITATION,source=SOURCE,classes=CLASSES,rows=rows,label_rule='Any red lesion -> malignant; otherwise green -> benign; empty expert mask -> normal. Labels are annotation-derived, not independent supervision.')
    result['source_directory']=str(base)
    atomic_json(manifest,result)
    save_default(root,base)
    return result

def import_manifest(root,path):
    p=Path(path).expanduser().resolve();rows=[]
    with p.open(newline='',encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            if not all(row.get(k) for k in ['id','patient','image','mask','label']):raise ValueError('CSV requires id, patient, image, mask, label')
            label=int(row['label'])
            if label not in range(3):raise ValueError('Labels must be 0 normal, 1 benign, 2 malignant')
            img=(p.parent/row['image']).resolve();mask=(p.parent/row['mask']).resolve()
            with Image.open(img) as a,Image.open(mask) as b:
                if a.size!=b.size:raise ValueError('Image/mask dimensions differ for '+row['id'])
            rows.append(dict(id=row['id'],patient=row['patient'],image=str(img),mask=str(mask),label=label,mask_encoding=row.get('mask_encoding') or 'binary',sha256=hashlib.sha256(img.read_bytes()).hexdigest()))
    if not rows or len({r['id'] for r in rows})!=len(rows):raise ValueError('Empty manifest or duplicate sample IDs')
    for row in rows:
        if row['mask_encoding'] not in ['binary','rgb-lesion']:raise ValueError('Unknown mask encoding')
    key=hashlib.sha256(p.read_bytes()).hexdigest()[:16];dest=Path(root)/'joint_data'/key;dest.mkdir(parents=True,exist_ok=True)
    result=dict(name=p.stem,classes=CLASSES,rows=rows,source=str(p),citation='User-supplied manifest; verify data rights and source labels',label_rule='Explicit CSV labels')
    atomic_json(dest/'manifest.json',result);return key,result

def split(data,seed):
    if data.get('fixed_splits'):return data['rows']
    groups=sorted({r['patient'] for r in data['rows']})
    if len(groups)<6:raise ValueError('At least six independent patient/source groups required')
    random.Random(seed).shuffle(groups);n=len(groups);a=max(1,int(.7*n));b=max(a+1,int(.85*n));b=min(b,n-1)
    mapping={g:('train' if i<a else 'val' if i<b else 'test') for i,g in enumerate(groups)}
    seen={};out=[]
    for row in data['rows']:
        s=mapping[row['patient']];digest=row['sha256']
        if digest in seen and seen[digest]!=s:raise ValueError('Duplicate image crosses patient partitions; correct the manifest')
        seen[digest]=s;out.append(dict(row,split=s))
    if any(len({r['label'] for r in out if r['split']==s})<3 for s in ['train','val','test']):
        raise ValueError('This patient split omits a class. Choose a different split seed; no image-level fallback is made.')
    return out

def sample(row,size,device='cpu',dtype=torch.float32):
    # Letterbox image and mask together; content bounds are exported.
    a=Image.open(row['image']).convert('RGB');m=Image.open(row['mask']).convert('RGB') if row.get('mask') else Image.new('RGB',a.size);w,h=a.size
    if m.size!=a.size:raise ValueError('Image/mask mismatch')
    scale=min(size/w,size/h);nw,nh=max(1,round(w*scale)),max(1,round(h*scale));left,top=(size-nw)//2,(size-nh)//2
    image=Image.new('RGB',(size,size));image.paste(a.resize((nw,nh),Image.Resampling.BILINEAR),(left,top))
    mask=np.asarray(m);mask=(mask[:,:,:2].max(2)>128) if row['mask_encoding']=='rgb-lesion' else (mask.max(2)>0)
    target=Image.new('L',(size,size));target.paste(Image.fromarray(mask.astype('uint8')*255).resize((nw,nh),Image.Resampling.NEAREST),(left,top))
    # Unnormalised [0,1] input is the analysis coordinate system. Normalization is
    # an explicit affine operation in the model wrapper (see engine).
    x=torch.from_numpy(np.array(image).transpose(2,0,1).copy()).to(device=device,dtype=dtype)/255
    y=torch.from_numpy(np.array(target).copy()).to(device=device,dtype=dtype).unsqueeze(0)/255
    return x,y,row['label'],dict(native_size=[w,h],content=[left,top,left+nw,top+nh])

def synthetic(root):
    """Deterministic integration-test dataset, visibly labelled nonmedical."""
    dest=Path(root)/'joint_data'/'synthetic';dest.mkdir(parents=True,exist_ok=True);rows=[]
    for patient in range(12):
        for label in range(3):
            yy,xx=np.mgrid[:64,:64];rng=np.random.default_rng(patient*3+label)
            mask=((xx-32-patient%5)**2+(yy-32)**2<(9+label*3)**2)&(label>0)
            image=(rng.uniform(.1,.3,(64,64))+.5*mask).clip(0,1);key=f'p{patient}_{label}'
            ip=dest/(key+'.png');mp=dest/(key+'_mask.png');Image.fromarray((image*255).astype('uint8')).save(ip);Image.fromarray(mask.astype('uint8')*255).save(mp)
            rows.append(dict(id=key,patient=str(patient),image=str(ip.resolve()),mask=str(mp.resolve()),label=label,mask_encoding='binary',sha256=hashlib.sha256(ip.read_bytes()).hexdigest()))
    result=dict(name='Synthetic engineering check (not medical data)',classes=CLASSES,rows=rows,citation='Generated locally',label_rule='Generated shape label; no medical meaning')
    atomic_json(dest/'manifest.json',result);return result


def prepare_classification(root,dataset,seed=42,train_size=512,progress=lambda s:None):
    from .. import datasets
    if dataset not in ['fashion','pneumonia','synthetic']:raise ValueError('Unsupported classification dataset')
    seed=int(seed);train_size=int(train_size)
    if not 0<=seed<2**32 or not 32<=train_size<=60000:raise ValueError('Invalid dataset seed or training subset')
    c=dict(dataset=dataset,seed=seed,train_size=train_size)
    datasets.prepare(root,c,progress);values=datasets.load(root,c)
    classes=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot'] if dataset=='fashion' else ['Normal','Pneumonia'] if dataset=='pneumonia' else ['Vertical bar','Horizontal bar']
    key=f'{dataset}-{seed}-{train_size}';dest=Path(root)/'joint_data'/key;dest.mkdir(parents=True,exist_ok=True);rows=[]
    from ..progress import stage
    count=sum(len(values[s+'_y']) for s in ['train','val','test']);done=0
    for split in ['train','val','test']:
        folder=dest/split;folder.mkdir(exist_ok=True)
        for x,y,idx in zip(values[split+'_x'],values[split+'_y'],values[split+'_ids']):
            path=folder/(str(int(idx))+'.png')
            if not path.exists():Image.fromarray(np.round(x[0]*255).astype('uint8')).save(path)
            rows.append(dict(id=split+'-'+str(int(idx)),patient=split+'-'+str(int(idx)),image=str(path.resolve()),mask=None,label=int(y),split=split,mask_encoding=None,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            done+=1;stage('Preparing classification adapter',done,count,'images')
    result=dict(name=dataset+' classification',classes=classes,rows=rows,has_masks=False,fixed_splits=True,source_dataset=dataset,source_seed=seed,train_size=train_size,label_rule='Dataset class labels; no mask annotations',preprocessing='Grayscale replicated into RGB, letterboxed to chosen model input size; ImageNet affine normalization inside model',split_policy='Existing dataset adapter splits and source indices retained; identifiers are image IDs, not patient claims')
    atomic_json(dest/'manifest.json',result);return key,result
