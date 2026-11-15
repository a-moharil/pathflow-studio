import json,time
from pathlib import Path
import pytest
from lab import create_app,progress
from lab.joint import data

def test_default_persists_and_is_dataset_specific(tmp_path):
    root=tmp_path/'workspace';folder=tmp_path/'images and masks';folder.mkdir()
    app=create_app(root);client=app.test_client();client.get('/joint')
    with client.session_transaction() as s:token=s['csrf']
    headers={'X-CSRF-Token':token}
    assert client.post('/api/joint/data/default',json={'path':str(folder)},headers=headers).status_code==200
    again=create_app(root).test_client();assert again.get('/api/joint').json['bus_uclm_path']==str(folder)
    client.post('/api/joint/data',json={'action':'manifest','path':'/some.csv'},headers=headers)
    assert data.default_path(root)==str(folder)
    assert client.post('/api/joint/data/default',json={'path':str(folder/'missing')},headers=headers).status_code==400

def test_migrate_existing_manifest(tmp_path):
    dest=tmp_path/'joint_data'/'bus-uclm-v1';dest.mkdir(parents=True)
    (dest/'manifest.json').write_text(json.dumps({'rows':[{'image':'/data/bus/images/a.png','mask':'/data/bus/masks/a.png'}]}))
    assert data.default_path(tmp_path)=='/data/bus'

def test_reporter_eta_failure_and_stale(tmp_path,capsys):
    create_app(tmp_path)
    with pytest.raises(RuntimeError):
        with progress.task(tmp_path,'Test',job='abc') as r:
            progress.stage('Measured',2,10,'items');r.stage_started=time.monotonic()-4
            r.publish(True);v=progress.snapshot(tmp_path)
            assert 15.9<v['stage_eta_seconds']<16.2
            progress.stage('Uncounted');assert r.snapshot()['stage_eta_seconds'] is None
            raise RuntimeError('test failure')
    v=progress.snapshot(tmp_path);assert v['status']=='failed' and v['error']=='test failure'
    assert progress.ACTIVE is None
    assert 'stage ETA' in capsys.readouterr().out
    v.update(status='running',updated_at=time.time()-20)
    (tmp_path/'compute_progress.json').write_text(json.dumps(v));assert progress.snapshot(tmp_path)['stale']
