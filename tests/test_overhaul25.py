"""Structural overhaul (0.16.0): admission gate, cases, queue, stage aliases, cache race,
residual geometry indexing and the residual estimate."""
import json
import multiprocessing as mp
import threading

import pytest

from lab import create_app, datasets, storage as st
from lab.cases import coordinates
from lab.config import validate
from lab.stats import index as ix
from lab.worker import tick


def _train(root, **overrides):
    c = validate(dict(dict(widths=[4, 4], train_size=32, updates=4, checkpoint_every=2, landmarks=16, geometry_every=0), **overrides))
    datasets.prepare(root, c)
    rid = st.create(root, c)
    with st.db(root) as db:
        db.execute("UPDATE runs SET status='training' WHERE id=?", (rid,))
    while tick(root):
        pass
    return rid


@pytest.fixture(scope='module')
def ws(tmp_path_factory):
    root = tmp_path_factory.mktemp('overhaul')
    app = create_app(root)
    client = app.test_client()
    client.get('/')
    with client.session_transaction() as s:
        headers = {'X-CSRF-Token': s['csrf']}
    rid = _train(root, name='overhaul')
    return dict(root=root, client=client, headers=headers, rid=rid)


def _analysis(**kw):
    return dict(dict(kind='analysis', step=4, x=0, reference=1, readout=dict(scope='all', a=0, b=1)), **kw)


def test_admission_gate_reuses_exact_results_and_rejects_uncommitted(ws):
    client, rid, h, root = ws['client'], ws['rid'], ws['headers'], ws['root']
    first = client.post(f'/api/run/{rid}/analysis', json=_analysis(), headers=h).json
    assert first['state'] == 'queued'
    again = client.post(f'/api/run/{rid}/analysis', json=_analysis(), headers=h).json
    assert again == dict(id=first['id'], state='pending')
    while tick(root):
        pass
    found = client.post(f'/api/run/{rid}/analysis', json=_analysis(), headers=h).json
    assert found == dict(id=first['id'], state='found')
    # Any difference in the request is new work: reuse is by exact match only.
    other = client.post(f'/api/run/{rid}/analysis', json=_analysis(readout=dict(scope='selected', a=0, b=1)), headers=h).json
    assert other['state'] == 'queued' and other['id'] != first['id']
    client.post(f'/api/job/{other["id"]}/cancel', json={}, headers=h)
    # A checkpoint that was never committed is refused before it reaches the worker.
    bad = client.post(f'/api/run/{rid}/analysis', json=_analysis(step=3), headers=h)
    assert bad.status_code == 400 and 'never committed' in bad.json['error']
    bad = client.post(f'/api/run/{rid}/healthcare', json=dict(kind='healthcare_case', step=99, x=0, reference=1), headers=h)
    assert bad.status_code == 400 and 'never committed' in bad.json['error']
    ws['analysis'] = first['id']


def test_case_coordinates_and_exact_matching(ws):
    client, rid, h, root = ws['client'], ws['rid'], ws['headers'], ws['root']
    assert coordinates(dict(kind='joint_analysis', checkpoint='8', x=3, reference=5)) == dict(
        step=8, split='test', x=3, reference_type='dataset', reference=5)
    assert coordinates(dict(kind='analysis', step=2, x=1, reference=4, reference_type='zero'))['reference'] is None
    assert coordinates(dict(kind='healthcare_cohort', step=2)) is None
    job = ws['analysis']
    case = client.get(f'/api/case/{job}').json
    assert case['coords'] == dict(step=4, split='test', x=0, reference_type='dataset', reference=1)
    assert case['label'] == 'Update 4 · test · input 0 vs reference 1' and case['stages'] == {'analysis': job}
    # No clinical reading for this case yet: an exact null, with the coordinates to pre-fill.
    opened = client.get(f'/api/run/{rid}/case/open?kind=healthcare_case&case={job}').json
    assert opened['match'] is None and opened['coords']['x'] == 0 and opened['exact']
    clinical = client.post(f'/api/run/{rid}/healthcare', json=dict(kind='healthcare_case', step=4, x=0, reference=1, split='test',
                                                                  reference_type='dataset'), headers=h).json['id']
    unrelated = client.post(f'/api/run/{rid}/healthcare', json=dict(kind='healthcare_case', step=2, x=5, reference=6, split='test',
                                                                    reference_type='dataset'), headers=h).json['id']
    while tick(root):
        pass
    opened = client.get(f'/api/run/{rid}/case/open?kind=healthcare_case&case={job}').json
    assert opened['match'] == clinical and opened['exact']
    # Without a case, a stage opens its latest recorded result.
    latest = client.get(f'/api/run/{rid}/case/open?kind=healthcare_case').json
    assert latest['match'] == unrelated and not latest['exact']
    assert client.get(f'/api/case/{clinical}').json['stages'] == {'analysis': job, 'healthcare_case': clinical}
    cases = client.get(f'/api/run/{rid}/cases').json['cases']
    assert {c['job'] for c in cases} >= {job, clinical, unrelated}
    assert next(c for c in cases if c['job'] == clinical)['stage'] == 'clinical'


def test_queue_lists_work_with_open_links(ws):
    client, rid = ws['client'], ws['rid']
    assert client.get('/queue').status_code == 200
    d = client.get('/api/queue').json
    done = next(j for j in d['jobs'] if j['job'] == ws['analysis'])
    assert done['status'] == 'completed' and done['open'] == f'/run/{rid}/geometry?case={ws["analysis"]}'
    assert done['run_name'] == 'overhaul' and done['label'].startswith('Update 4')
    assert d['counts']['completed'] >= 3 and 'heartbeat' in d


def test_stage_aliases_and_rail(ws):
    client, rid = ws['client'], ws['rid']
    expect = {'realisation': f'/experiment/{rid}?section=Fields', 'observables': f'/observatory/{rid}',
              'evidence': f'/run/{rid}/statistics', 'audit': f'/auditor?run={rid}', 'clinical': f'/healthcare/{rid}',
              'geometry': f'/experiment/{rid}?section=Geometry'}
    for view, target in expect.items():
        assert client.get(f'/run/{rid}/{view}').location == target, view
    assert client.get(f'/run/{rid}/geometry?case=abc').location == f'/experiment/{rid}?section=Geometry&case=abc'
    page = client.get(f'/run/{rid}').text
    for label in ('Overview', 'Training', 'Geometry', 'Realisation', 'Observables', 'Evidence', 'Audit', 'Clinical reading'):
        assert f'>{label}</a>' in page, label
    assert 'id="caseChip"' in page and '/static/case.js' in page and 'auditor_link.js' not in page
    shell = client.get('/').text
    for label in ('Library', 'Build', 'Queue', 'Evidence', 'Present'):
        assert label in shell


def test_case_reads_are_allowed_in_presentation(ws):
    client, rid, h = ws['client'], ws['rid'], ws['headers']
    client.post('/api/presentation', json=dict(on=True), headers=h)
    try:
        assert client.get(f'/api/run/{rid}/case/open?kind=analysis').status_code == 200
        assert client.get('/api/queue').status_code == 200
        assert client.post(f'/api/run/{rid}/analysis', json=_analysis(x=2), headers=h).status_code == 409
    finally:
        client.post('/api/presentation', json=dict(on=False), headers=h)


def _build_cache(root, barrier):
    c = validate(dict(dataset='synthetic', train_size=48, seed=7))
    barrier.wait()
    datasets.load(root, c)


def test_concurrent_cache_writers_do_not_collide(tmp_path):
    ctx = mp.get_context('fork')
    barrier = ctx.Barrier(4)
    procs = [ctx.Process(target=_build_cache, args=(str(tmp_path), barrier)) for _ in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
    assert all(p.exitcode == 0 for p in procs)
    files = sorted(f.name for f in (tmp_path / 'datasets').iterdir())
    assert files == ['synthetic_7_48.npz']


def test_atomic_json_concurrent_writers(tmp_path):
    target = tmp_path / 'state.json'
    errors = []

    def write(i):
        try:
            for _ in range(40):
                st.atomic_json(target, dict(writer=i))
        except Exception as e:  # pragma: no cover - the failure being tested
            errors.append(e)

    threads = [threading.Thread(target=write, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and json.loads(target.read_text())['writer'] in range(6)
    assert [f.name for f in tmp_path.iterdir()] == ['state.json']


def test_residual_geometry_is_indexed():
    data = dict(heads=dict(classification=dict(
        structural_observables=dict(log_eta1=-1.0), path_observables=dict(log_eta1=-2.0),
        geometry=dict(responses=dict(structural=dict(dstar=1.7, deff=1.2), conditional=dict(dstar=None, deff=0.9))))))
    rows = {(side, name): value for _, side, name, value in ix.joint_rows(data)}
    assert rows[('geometry_structural', 'dstar')] == 1.7 and rows[('geometry_conditional', 'deff')] == 0.9
    assert ('geometry_conditional', 'dstar') not in rows
    assert ix.INDEX_VERSION >= 2


def test_residual_estimate(ws):
    client, h = ws['client'], ws['headers']
    d = client.post('/api/joint/estimate', json=dict(architecture='residual_small', task='classification', dataset='synthetic',
                                                       size=16, name='e'), headers=h).json
    assert d['parameters'] > 0 and d['activation_units'] > 0 and d['operations'] > 0 and d['blocks'] == 3
    again = client.post('/api/joint/estimate', json=dict(architecture='residual_small', task='classification', dataset='synthetic',
                                                           size=16, name='other name'), headers=h).json
    assert again == d
