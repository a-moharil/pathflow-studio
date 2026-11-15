"""Efficient realisation release: index, lean feeds, statistics, routing, presentation."""
import json

import numpy as np
import pytest
from scipy import stats as sps

from lab import create_app, datasets, storage as st
from lab.config import validate
from lab.stats import engine as E, index as ix
from lab.worker import tick


def _train(root, **overrides):
    c = validate(dict(dict(widths=[4, 4], train_size=32, updates=6, checkpoint_every=2, landmarks=16, geometry_every=0), **overrides))
    datasets.prepare(root, c)
    rid = st.create(root, c)
    with st.db(root) as db:
        db.execute("UPDATE runs SET status='training' WHERE id=?", (rid,))
    while tick(root):
        pass
    return rid


@pytest.fixture(scope='module')
def workspace(tmp_path_factory):
    root = tmp_path_factory.mktemp('efficient')
    app = create_app(root)
    client = app.test_client()
    client.get('/')
    with client.session_transaction() as s:
        token = s['csrf']
    headers = {'X-CSRF-Token': token}
    rid = _train(root, name='primary')
    job = client.post(f'/api/run/{rid}/analysis', json=dict(kind='analysis', step=6, x=0, reference=1,
                                                             readout=dict(scope='all', a=0, b=1)), headers=headers).json['id']
    cohort = client.post(f'/api/run/{rid}/healthcare', json=dict(kind='healthcare_cohort', step=2, split='test', threshold=.5,
                                                                 cohort_size=24, cohort_reference='zero'), headers=headers).json['id']
    while tick(root):
        pass
    return dict(root=root, app=app, client=client, headers=headers, rid=rid, job=job, cohort=cohort)


# ------------------------------------------------------------------------- index

def test_index_sync_is_incremental_and_complete(workspace):
    root, rid = workspace['root'], workspace['rid']
    first = ix.sync(root)
    second = ix.sync(root)
    assert sum(second.values()) == 0
    with st.db(root) as db:
        checkpoints = db.execute('SELECT count(*) FROM measurements WHERE run=?', (rid,)).fetchone()[0]
        probes = db.execute("SELECT count(DISTINCT source_id) FROM observables WHERE run=? AND source='probe'", (rid,)).fetchone()[0]
    assert probes == checkpoints == 4
    contexts = ix.contexts(root, rid)
    assert {c['source'] for c in contexts} >= {'probe', 'analysis'}
    probe = next(c for c in contexts if c['source'] == 'probe')
    series = ix.series(root, rid, probe['key'])
    stored = json.loads(st.db(root).execute('SELECT payload FROM measurements WHERE run=? AND step=6', (rid,)).fetchone()[0])
    value = next(v for s, v, _ in series[('conditional', 'log_eta1')] if s == 6)
    assert value == pytest.approx(stored['probe']['conditional']['log_eta1'])
    assert ('training', 'val_loss') in series
    del first


def test_lean_feeds_are_smaller_and_mergeable(workspace):
    client, rid = workspace['client'], workspace['rid']
    full = client.get(f'/api/run/{rid}').get_data()
    lean = client.get(f'/api/run/{rid}?lean=1').get_data()
    assert len(lean) < len(full) / 3
    newer = client.get(f'/api/run/{rid}?lean=1&since=2').json
    assert [m['step'] for m in newer['metrics']] == [4, 6] and newer['metric_count'] == 4
    assert 'input_fields' not in newer['metrics'][0]['probe'] and 'conditional' in newer['metrics'][0]['probe']
    legacy = client.get(f'/api/run/{rid}/observations').json
    assert isinstance(legacy, list) and legacy
    records = client.get(f'/api/run/{rid}/observations?lean=1&after=0').json
    assert records['total'] == len(legacy) and records['records'][0]['_job']
    after = records['records'][-1]['_rowid']
    assert client.get(f'/api/run/{rid}/observations?lean=1&after={after}').json['records'] == []
    assert len(json.dumps(records['records'][0])) < len(json.dumps(legacy[0]))


# -------------------------------------------------------------------- statistics

def test_run_statistics_are_descriptive_and_consistent(workspace):
    root, rid = workspace['root'], workspace['rid']
    probe = next(c['key'] for c in ix.contexts(root, rid) if c['source'] == 'probe')
    s = E.run_statistics(root, rid, probe)
    eta1 = next(r for r in s['realisation'] if r['name'] == 'log_eta1')
    assert eta1['n'] == 4 and eta1['status'] == 'descriptive'
    assert np.allclose(eta1['difference'], np.asarray(eta1['conditional']) - np.asarray(eta1['structural']))
    assert eta1['retention']['median'] == pytest.approx(float(np.median(np.exp(eta1['difference']))))
    assert all('p' not in t for group in s['trajectories'].values() for t in group)
    assert s['coupling']['status'] == 'descriptive'


def test_turning_points_ignore_plateau_noise():
    points = [(0, 1.0, 'a'), (10, 0.2, 'b'), (20, 0.8, 'c'), (30, 0.9, 'd')]
    assert E.trajectory_summary(points)['turning_points'][0]['step'] == 10
    plateau = [(0, 1.0, 'a'), (10, 0.5, 'b'), (20, 0.4999, 'c'), (30, 0.5, 'd')]
    assert E.trajectory_summary(plateau)['turning_points'] == []


def test_cohort_statistics_match_scipy_and_flag_design():
    rng = np.random.default_rng(1)
    rows = []
    for i in range(40):
        label = i % 2
        group = ['TN', 'TP'][label] if i % 5 else ['FP', 'FN'][label]
        rows.append(dict(status='available', label=label, group=group, eta1=float(rng.normal(1 + (group in ('FP', 'FN')), .5)),
                         H1=float(rng.normal()), H_eq=1e-16, H_inv=0.0))
    result = dict(rows=rows, request=dict(transform=dict(dx=0, dy=0, angle=0, scale=1)), sampling=dict(policy='zero', population=40))
    s = E.cohort_statistics(result)
    errors = next(e for e in s['errors'] if e['observable'] == 'eta1')
    a = [r['eta1'] for r in rows if r['group'] in ('FP', 'FN')]
    b = [r['eta1'] for r in rows if r['group'] in ('TN', 'TP')]
    u = sps.mannwhitneyu(a, b, alternative='two-sided')
    assert errors['auroc'] == pytest.approx(u.statistic / (len(a) * len(b)))
    assert errors['p'] == pytest.approx(u.pvalue)
    assert errors['auroc_interval'][0] < errors['auroc'] < errors['auroc_interval'][1]
    assert {e['name'] for e in s['excluded_observables']} == {'H_eq', 'H_inv'}
    assert all(e['observable'] not in ('H_eq', 'H_inv') for e in s['errors'])
    tested = [e for e in s['errors'] if e.get('p') is not None]
    assert [e['p_holm'] for e in tested] == pytest.approx(E.holm([e['p'] for e in tested]))
    assert s['confounding'] is None  # errors span both classes here
    one_class = [dict(r, group='FP' if r['group'] in ('FP', 'FN') and r['label'] == 0 else ('TN' if r['label'] == 0 else 'TP')) for r in rows]
    assert 'confounded' in E.cohort_statistics(dict(rows=one_class))['confounding']


def test_holm_matches_definition():
    p = [0.01, 0.04, 0.03, 0.2]
    assert E.holm(p) == pytest.approx([0.04, 0.09, 0.09, 0.2])


def test_cross_run_friedman_and_guards(tmp_path):
    create_app(tmp_path)
    runs = []
    for optimizer in ['Adam', 'RMSprop', 'SGD']:
        for seed in [1, 2, 3]:
            runs.append(_train(tmp_path, optimizer=optimizer, seed=seed, lr=.01, name=f'{optimizer}{seed}'))
    for rid in runs:
        ix.sync(tmp_path, rid)
    configs = [st.run(tmp_path, r) for r in runs]
    descriptive = E.cross_run(tmp_path, configs, 'conditional', 'log_eta1', 'final', 'optimizer')
    assert descriptive['inference']['status'] == 'descriptive' and len(descriptive['groups']) == 3
    d = E.cross_run(tmp_path, configs, 'conditional', 'log_eta1', 'final', 'optimizer', independent=True)
    inf = d['inference']
    assert inf['status'] == 'exploratory' and inf['blocks'] == 3
    by = {(e['level_label'], e['seed']): e['value'] for e in d['runs']}
    matrix = np.array([[by[(o, s)] for o in sorted({k[0] for k in by}, key=lambda o: (len(json.dumps(o)), json.dumps(o)))] for s in [1, 2, 3]])
    test = sps.friedmanchisquare(*matrix.T)
    assert inf['statistic'] == pytest.approx(test.statistic) and inf['p'] == pytest.approx(test.pvalue)
    assert inf['kendall_w'] == pytest.approx(test.statistic / (3 * 2))
    other = _train(tmp_path, widths=[3, 3], name='narrow')
    ix.sync(tmp_path, other)
    with pytest.raises(ValueError, match='architecture'):
        E.cross_run(tmp_path, configs + [st.run(tmp_path, other)], 'conditional', 'log_eta1', 'final', 'optimizer')
    rows = E.ledger(cross=d)
    assert rows and '\\toprule' in E.ledger_latex(rows) and E.ledger_csv(rows).startswith('section,')


# ----------------------------------------------------------- routing and pages

def test_run_workspace_pages_and_aliases(workspace):
    client, rid = workspace['client'], workspace['rid']
    for url in ['/', f'/run/{rid}', f'/run/{rid}/statistics', '/statistics', '/compare', f'/experiment/{rid}', f'/observatory/{rid}', f'/auditor?run={rid}']:
        assert client.get(url).status_code == 200, url
    page = client.get(f'/observatory/{rid}').text
    assert 'id="runBar"' in page and f'href="/run/{rid}/statistics"' in page and 'aria-current="page">Observatory' in page
    assert 'plotly.min.js' not in client.get('/').text and 'region_explorer.js' not in client.get('/').text
    assert 'plotly.min.js' in client.get(f'/run/{rid}/statistics').text
    assert client.get(f'/run/{rid}/geometry').location == f'/experiment/{rid}?section=Geometry'
    assert client.get(f'/run/{rid}/observatory?context=x').location == f'/observatory/{rid}?context=x'
    assert client.get(f'/run/{rid}/nonsense').status_code == 404
    overview = client.get(f'/api/stats/run/{rid}/overview').json
    assert overview['tiles']['conditional_log_eta1']['step'] == 6 and overview['tiles']['retention']['value'] > 0
    summary = client.get('/api/runs/summary').json
    assert summary['runs'][0]['engine'] == 'native' and summary['runs'][0]['analysis_count'] >= 1
    stats = client.get(f'/api/stats/cohort/{workspace["cohort"]}').json
    assert stats['unit'].startswith('Images') and stats['sampled'] == 24


def test_ledger_exports(workspace):
    client, rid, h = workspace['client'], workspace['rid'], workspace['headers']
    csv = client.post('/api/stats/ledger', json=dict(run=rid, cohort=workspace['cohort'], format='csv'), headers=h)
    assert csv.status_code == 200 and csv.text.startswith('section,quantity')
    tex = client.post('/api/stats/ledger', json=dict(run=rid, format='tex'), headers=h)
    assert '\\begin{tabular}' in tex.text


def test_presentation_mode_blocks_computation_only(workspace):
    client, rid, h = workspace['client'], workspace['rid'], workspace['headers']
    assert client.post('/api/presentation', json=dict(on=True), headers=h).json['on'] is True
    try:
        assert 'data-presentation="on"' in client.get(f'/run/{rid}').text
        blocked = client.post(f'/api/run/{rid}/analysis', json=dict(kind='analysis', step=6), headers=h)
        assert blocked.status_code == 409 and blocked.json['presentation']
        assert client.post(f'/api/run/{rid}/control', json=dict(action='start'), headers=h).status_code == 409
        assert client.post(f'/api/run/{rid}/healthcare', json=dict(kind='healthcare_case'), headers=h).status_code == 409
        assert client.post(f'/api/run/{rid}/delete', json={}, headers=h).status_code == 409
        # Read-only statistics and stopping remain available.
        assert client.post('/api/stats/ledger', json=dict(run=rid), headers=h).status_code == 200
        assert client.post(f'/api/run/{rid}/control', json=dict(action='stop'), headers=h).status_code in (200, 400)
    finally:
        client.post('/api/presentation', json=dict(on=False), headers=h)
    assert client.post('/api/presentation', json=dict(on=True), headers={}).status_code == 403


def test_auditor_catalogue_cache_matches_uncached(workspace):
    from lab.auditor.evidence import catalogue
    root = workspace['root']
    first = catalogue(root)
    with st.db(root) as db:
        cached = db.execute('SELECT count(*) FROM artifact_index').fetchone()[0]
    second = catalogue(root)
    assert cached >= 1 and first['records'] == second['records'] and first['records']
