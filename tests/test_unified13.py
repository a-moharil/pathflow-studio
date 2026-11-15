import json
import torch
import pytest
from lab import create_app
from lab.joint import data,engine
from lab.joint.model import JointModel
from lab.joint.investigate import analyze,attributions,embed_subgraph

def test_shared_routes_and_design(tmp_path):
    c=create_app(tmp_path).test_client();page=c.get('/design').text
    assert 'id="resnetDesign"' in page and 'id="jointDesign"' in page
    assert c.get('/joint').location=='/design?model=resnet50_joint'
    data.synthetic(tmp_path);rid=engine.create(tmp_path,dict(dataset='synthetic',size=64,width=16,seed=17))
    assert c.get('/joint/'+rid).location=='/experiment/'+rid
    page=c.get('/experiment/'+rid).text
    assert 'id="geometry0"' in page and 'geometry.js' in page

def test_standard_classification_adapter(tmp_path):
    key,d=data.prepare_classification(tmp_path,'synthetic',42,32)
    assert not d['has_masks'] and len(d['classes'])==2
    assert all(r['mask'] is None for r in d['rows'])
    create_app(tmp_path);rid=engine.create(tmp_path,dict(dataset=key,task='classification',size=64,width=16))
    r=engine.get(tmp_path,rid);assert r['config']['classes']==2 and r['config']['classification_only']
    m=engine.build(r['config']);assert len(m.decoder)==0 and m.mask is None
    out=m.eval()(torch.rand(1,3,64,64));assert list(out)==['classification'] and out['classification'].shape==(1,2)
    with pytest.raises(ValueError):engine.create(tmp_path,dict(dataset=key,task='joint'))

def test_disconnected_cam_and_classification_only():
    torch.manual_seed(9);x=torch.rand(1,3,32,32,dtype=torch.double);r=torch.zeros_like(x)
    m=JointModel(width=16,tiny=True).double().eval()
    result=attributions(m,x,r,dict(head='classification',a=2,b=0),'decoder.level1.relu1',4,{})
    assert result['GradCAM'] is None and 'not applicable' in result['GradCAM_status']
    m=JointModel(classes=10,width=16,tiny=True,classification_only=True).double().eval()
    result=analyze(m,x,r,dict(head='classification',a=9,b=0),'stem.relu',0,4,geometry='operations')
    assert result['mask_probability'] is None
    assert result['geometry']['shared']['euclidean']['pairs']['unconstrained_nodes']==0
    assert result['geometry']['shared']['metric']['explicit_edge_lengths']
    assert abs(sum(map(sum,result['FieldA']))-1)<1e-8


@pytest.mark.parametrize('dataset,count',[('fashion',10),('pneumonia',2)])
def test_other_dataset_class_counts(tmp_path,dataset,count):
    import numpy as np
    cache=tmp_path/'datasets';cache.mkdir();arrays={}
    for split in ['train','val','test']:
        arrays[split+'_x']=np.zeros((count,1,28,28),dtype='float32');arrays[split+'_y']=np.arange(count);arrays[split+'_ids']=np.arange(count)
    np.savez(cache/f'{dataset}_42_32.npz',**arrays)
    key,d=data.prepare_classification(tmp_path,dataset,42,32)
    create_app(tmp_path);rid=engine.create(tmp_path,dict(dataset=key,task='classification',size=64,width=16))
    c=engine.get(tmp_path,rid)['config'];assert c['classes']==count
    assert c['classification_only'] and len(c['class_names'])==count
