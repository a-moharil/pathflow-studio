"""Ruler sensitivity reproduces the recorded convention and varies only ε (and p)."""
import pytest

from lab import create_app, datasets
from lab.config import validate
from lab.stats.ruler import validate_request
from lab.worker import tick


@pytest.fixture(scope='module')
def analysed(tmp_path_factory):
    root = tmp_path_factory.mktemp('ruler')
    app = create_app(root)
    client = app.test_client()
    client.get('/')
    with client.session_transaction() as s:
        headers = {'X-CSRF-Token': s['csrf']}
    config = dict(name='ruler', model='cnn', channels='3,4', stride=2, padding=1, cnn_readout='gap', train_size=32, updates=2, checkpoint_every=1, landmarks=16)
    datasets.prepare(root, validate(config))
    rid = client.post('/api/runs', json=config, headers=headers).json['id']
    client.post(f'/api/run/{rid}/control', json={'action': 'start'}, headers=headers)
    while tick(root):
        pass
    job = client.post(f'/api/run/{rid}/analysis', json=dict(kind='analysis', step=2, x=0, reference=1, readout=dict(scope='all', a=0, b=1)), headers=headers).json['id']
    while tick(root):
        pass
    return dict(root=root, client=client, headers=headers, rid=rid, job=job)


def test_request_validation():
    assert validate_request({'epsilons': '0.3, 0.003,0.03'})['epsilons'] == [0.003, 0.03, 0.3]
    for bad in [{'epsilons': 'x'}, {'epsilons': []}, {'epsilons': [0]}, {'power': 0}, {'power': 9}]:
        with pytest.raises(ValueError):
            validate_request(bad)


def test_ruler_job_reproduces_saved_geometry(analysed):
    client, rid, h = analysed['client'], analysed['rid'], analysed['headers']
    records = client.get(f'/api/stats/run/{rid}/geometry-records').json['records']
    assert [r['job'] for r in records] == [analysed['job']] and records[0]['measured']
    submitted = client.post('/api/stats/ruler', json=dict(run=rid, source_job=analysed['job'], epsilons='0.003, 0.3'), headers=h)
    assert submitted.status_code == 200
    again = client.post('/api/stats/ruler', json=dict(run=rid, source_job=analysed['job'], epsilons=[0.3, 0.003]), headers=h)
    assert again.json['id'] == submitted.json['id']  # identical request reuses the job
    while tick(analysed['root']):
        pass
    data = client.get('/api/job/' + submitted.json['id']).json['data']
    assert data['version'] == 'ruler-1' and data['recorded'] == dict(epsilon=0.03, power=1.0)
    for side in data['sides']:
        assert side['status'] == 'available'
        assert [r['epsilon'] for r in side['rows']] == [0.003, 0.03, 0.3]  # recorded ε added for the check
        assert side['reproduction']['reproduced'], side['reproduction']
        fractions = [r['floor_fraction'] for r in side['rows']]
        assert fractions == sorted(fractions) and all(0 <= f <= 1 for f in fractions)
    previous = client.get(f'/api/stats/run/{rid}/geometry-records').json['previous']
    assert previous[0]['status'] == 'completed'


def test_ruler_is_blocked_in_presentation_and_for_other_runs(analysed):
    client, rid, h = analysed['client'], analysed['rid'], analysed['headers']
    client.post('/api/presentation', json=dict(on=True), headers=h)
    try:
        assert client.post('/api/stats/ruler', json=dict(run=rid, source_job=analysed['job']), headers=h).status_code == 409
    finally:
        client.post('/api/presentation', json=dict(on=False), headers=h)
    job = client.post('/api/stats/ruler', json=dict(run=rid, source_job='not-a-job', epsilons='0.1'), headers=h).json['id']
    while tick(analysed['root']):
        pass
    failed = client.get('/api/job/' + job).json
    assert failed['status'] == 'failed' and 'completed analysis of this experiment' in failed['error']
