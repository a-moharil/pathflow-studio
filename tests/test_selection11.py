from lab import create_app
from lab.joint import engine

def test_catalog_visible_before_download(tmp_path):
    app=create_app(tmp_path);c=app.test_client();d=c.get('/api/joint').json
    assert d['datasets']==[]
    assert [v['key'] for v in d['dataset_catalog']]==['bus-uclm-v1','synthetic']
    assert all(not v['ready'] for v in d['dataset_catalog'])
    main=c.get('/design').get_data(as_text=True)
    assert 'value="resnet50_joint"' in main;assert 'value="bus-uclm-v1"' in main
    joint=c.get('/joint',follow_redirects=True).get_data(as_text=True)
    assert 'id="jointPreset"' in joint;assert 'id="jointArchitecture"' in joint
    assert 'id="jointCreate"' in joint

def test_joint_architecture_explicit_and_legacy_default():
    assert engine.config({})['architecture']=='resnet50_joint'
    assert engine.config({'architecture':'resnet50_joint','preset':'engineering64'})['preset']=='engineering64'
    import pytest
    with pytest.raises(ValueError):engine.config({'architecture':'not_a_resnet'})
