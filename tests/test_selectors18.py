"""The dataset selector publishes true labels without changing split identities."""
import pytest
from lab import create_app,storage,datasets
from lab.config import validate

@pytest.mark.parametrize('dataset',['synthetic','synthetic-regression'])
def test_selector_catalog(tmp_path,dataset):
    app=create_app(tmp_path);config=validate(dict(dataset=dataset,train_size=32,task="regression" if dataset=="synthetic-regression" else "classification"));rid=storage.create(tmp_path,config)
    if isinstance(rid,dict):rid=rid['id']
    expected=datasets.load(tmp_path,config);client=app.test_client()
    r=client.get('/api/run/'+rid+'/dataset');assert r.status_code==200
    d=r.get_json();assert d['test']['labels']==expected['test_y'].tolist();assert d['test']['source_indices']==expected['test_ids'].tolist()
    assert d['class_names']==(['Vertical stripe','Horizontal stripe'] if dataset=='synthetic' else [])
    page=client.get('/experiment/'+rid).get_data(as_text=True)
    assert '<select id="x"' in page and '<select id="reference"' in page
    assert 'pair_selector.js?v=0.11.4' in page
    assert client.get('/design').status_code==200
