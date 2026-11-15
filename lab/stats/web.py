"""Run workspace pages, statistics APIs and lean feeds.

All endpoints read recorded results. None of them queues work or touches checkpoints.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from flask import Blueprint, abort, current_app, jsonify, redirect, render_template, request, Response

from .. import storage as st
from . import engine as E
from . import index as ix
from .navigation import describe_run

web = Blueprint('stats', __name__)
_cohort_cache: dict = {}


def root():
    return current_app.config['ROOT']


def _info(rid):
    info = describe_run(root(), rid)
    if not info:
        raise KeyError('Experiment not found')
    return info


# ------------------------------------------------------------------------- pages

@web.get('/run/<rid>')
def overview(rid):
    info = _info(rid)
    return render_template('run_overview.html', rid=rid, engine=info['engine'], config=info['run']['config'], assets='charts')


@web.get('/run/<rid>/statistics')
def run_statistics(rid):
    info = _info(rid)
    return render_template('run_statistics.html', rid=rid, engine=info['engine'], config=info['run']['config'], assets='charts')


@web.get('/run/<rid>/<view>')
def run_view(rid, view):
    """Stable aliases for every run view; the canonical pages keep their URLs."""
    info = _info(rid)
    joint = info['engine'] == 'joint'
    observables = f'/experiment/{rid}?section=resnetObservatory' if joint else f'/observatory/{rid}'
    targets = {
        'training': f'/experiment/{rid}?section=' + ('training' if joint else 'Training'),
        'geometry': f'/experiment/{rid}?section=' + ('geometry' if joint else 'Geometry'),
        'realisation': f'/experiment/{rid}?section=' + ('fieldsPanel' if joint else 'Fields'),
        'observables': observables, 'observatory': observables,
        'evidence': f'/run/{rid}/statistics',
        'clinical': f'/healthcare/{rid}', 'healthcare': f'/healthcare/{rid}',
        'audit': f'/auditor?run={rid}', 'auditor': f'/auditor?run={rid}',
    }
    if view in ('clinical', 'healthcare') and joint:
        abort(404)
    if view not in targets:
        abort(404)
    query = request.query_string.decode()
    target = targets[view]
    if query:
        target += ('&' if '?' in target else '?') + query
    return redirect(target)


@web.get('/statistics')
def cohort_page():
    return render_template('statistics.html', assets='charts')


# ------------------------------------------------------------------- registry feed

@web.get('/api/runs/summary')
def runs_summary():
    """Both engines in one pass, with grouped counts instead of per-run queries."""
    with st.db(root()) as db:
        native = [dict(r) for r in db.execute('SELECT id, config, status, step, live_step, error, created FROM runs ORDER BY created DESC')]
        observations = dict(db.execute('SELECT run, count(*) FROM observations GROUP BY run').fetchall())
        completed = dict(db.execute("SELECT run, count(*) FROM jobs WHERE status='completed' GROUP BY run").fetchall())
        pending = dict(db.execute("SELECT run, count(*) FROM jobs WHERE status IN ('queued','running') GROUP BY run").fetchall())
        try:
            joint = [dict(r) for r in db.execute('SELECT id, config, status, step, error, created FROM joint_runs ORDER BY created DESC')]
        except Exception:
            joint = []
    out = []
    for r in native:
        r['config'] = json.loads(r['config'])
        r.update(engine='native', analysis_count=observations.get(r['id'], 0), pending_measurements=pending.get(r['id'], 0))
        out.append(r)
    for r in joint:
        r['config'] = json.loads(r['config'])
        r.update(engine='joint', joint=True, analysis_count=completed.get(r['id'], 0), pending_measurements=pending.get(r['id'], 0))
        out.append(r)
    out.sort(key=lambda r: r.get('created') or 0, reverse=True)
    active = any(r['status'] in ('training', 'saving', 'pause_requested', 'stop_requested', 'delete_requested') or r['pending_measurements'] for r in out)
    return jsonify(runs=out, active=active)


@web.get('/api/stats/run/<rid>/status')
def run_status(rid):
    info = _info(rid)
    r = info['run']
    with st.db(root()) as db:
        pending = db.execute("SELECT count(*) FROM jobs WHERE run=? AND status IN ('queued','running')", (rid,)).fetchone()[0]
    return jsonify(status=r['status'], step=r['step'], live_step=r.get('live_step'), pending=pending, engine=info['engine'],
                   active=r['status'] in ('training', 'saving', 'pause_requested', 'stop_requested') or bool(pending))


# ------------------------------------------------------------------ statistics API

def _default_context(rid, prefer=('probe', 'analysis', 'joint_analysis', 'dose')):
    contexts = ix.contexts(root(), rid)
    for source in prefer:
        candidates = [c for c in contexts if c['source'] == source]
        if candidates:
            return max(candidates, key=lambda c: c['records'])['key']
    return ''


@web.get('/api/stats/run/<rid>/contexts')
def run_contexts(rid):
    _info(rid)
    ix.sync(root(), rid)
    return jsonify(contexts=ix.contexts(root(), rid), default=_default_context(rid))


@web.get('/api/stats/run/<rid>')
def run_stats(rid):
    _info(rid)
    ix.sync(root(), rid)
    context = request.args.get('context')
    if context is None:
        context = _default_context(rid)
    return jsonify(E.run_statistics(root(), rid, context))


@web.get('/api/stats/run/<rid>/overview')
def run_overview(rid):
    info = _info(rid)
    ix.sync(root(), rid)
    r, engine = info['run'], info['engine']
    config = r['config']
    joint = engine == 'joint'
    probe_key = _default_context(rid, prefer=('joint_analysis',) if joint else ('probe',))
    # Residual-DAG runs have no fixed probe; their geometry comes from the recorded analyses.
    analysis_key = _default_context(rid, prefer=('joint_analysis',) if joint else ('analysis',))
    probe = ix.series(root(), rid, probe_key)
    analysis = ix.series(root(), rid, analysis_key) if analysis_key else {}

    def latest(series, side, name):
        points = [p for p in series.get((side, name), []) if p[0] is not None]
        return dict(step=points[-1][0], value=points[-1][1]) if points else None

    def trajectory(series, side, name):
        steps, values, _ = E._per_step(series.get((side, name), []))
        return dict(steps=steps, values=values)

    with st.db(root()) as db:
        jobs = db.execute('SELECT payload, status FROM jobs WHERE run=?', (rid,)).fetchall()
    kinds: dict = {}
    for payload, status in jobs:
        kind = json.loads(payload).get('kind', 'analysis')
        entry = kinds.setdefault(kind, dict(completed=0, pending=0, failed=0))
        if status == 'completed':
            entry['completed'] += 1
        elif status in ('queued', 'running'):
            entry['pending'] += 1
        elif status == 'failed':
            entry['failed'] += 1
    tiles = dict(
        structural_log_eta1=latest(probe, 'structural', 'log_eta1'),
        conditional_log_eta1=latest(probe, 'conditional', 'log_eta1'),
        structural_deficit=latest(probe, 'structural', 'normalised_deficit'),
        conditional_deficit=latest(probe, 'conditional', 'normalised_deficit'),
        structural_dstar=latest(analysis, 'geometry_structural', 'dstar'),
        conditional_dstar=latest(analysis, 'geometry_conditional', 'dstar'),
        val_loss=latest(probe, 'training', 'val_loss') or latest(probe, 'training', 'val_loss_loss'),
        val_accuracy=latest(probe, 'training', 'val_accuracy') or latest(probe, 'training', 'val_accuracy_accuracy'),
    )
    s, c = tiles['structural_log_eta1'], tiles['conditional_log_eta1']
    if s and c and s['step'] == c['step']:
        import math
        tiles['retention'] = dict(step=s['step'], value=math.exp(c['value'] - s['value']))
    training = {k[1]: trajectory(probe, *k) for k in probe if k[0] == 'training'}
    return jsonify(E._clean(dict(
        rid=rid, engine=engine, name=config.get('name'), status=r['status'], step=r['step'],
        configuration={k: config.get(k) for k in ('dataset', 'model', 'architecture', 'widths', 'channels', 'activation', 'optimizer',
                                                  'lr', 'seed', 'updates', 'checkpoint_every', 'geometry_every', 'train_size', 'task') if k in config},
        jobs=kinds, tiles=tiles, probe_context=json.loads(probe_key) if probe_key else None,
        analysis_context=json.loads(analysis_key) if analysis_key else None,
        series=dict(training=training,
                    structural_log_eta1=trajectory(probe, 'structural', 'log_eta1'),
                    conditional_log_eta1=trajectory(probe, 'conditional', 'log_eta1'),
                    structural_dstar=trajectory(analysis, 'geometry_structural', 'dstar'),
                    conditional_dstar=trajectory(analysis, 'geometry_conditional', 'dstar')))))


@web.get('/api/stats/run/<rid>/cohorts')
def run_cohorts(rid):
    _info(rid)
    with st.db(root()) as db:
        rows = db.execute("SELECT id, payload, created FROM jobs WHERE run=? AND status='completed' ORDER BY created DESC", (rid,)).fetchall()
    out = []
    for jid, payload, created in rows:
        p = json.loads(payload)
        if p.get('kind') == 'healthcare_cohort':
            out.append(dict(job=jid, created=created, step=p.get('step'), split=p.get('split', 'test'),
                            policy=p.get('cohort_reference', 'zero'), size=p.get('cohort_size'), threshold=p.get('threshold', .5)))
    return jsonify(cohorts=out)


def _cohort_result(jid):
    with st.db(root()) as db:
        row = db.execute("SELECT run, payload, result, status FROM jobs WHERE id=?", (jid,)).fetchone()
    if not row or row['status'] != 'completed' or json.loads(row['payload']).get('kind') != 'healthcare_cohort':
        raise ValueError('Select a completed healthcare cohort job.')
    path = Path(row['result']).resolve()
    if not path.is_relative_to(Path(root()).resolve()) or not path.is_file():
        raise ValueError('Cohort artefact is missing.')
    key = (jid, path.stat().st_mtime_ns)
    if key not in _cohort_cache:
        result = json.loads(path.read_text())
        stats = E.cohort_statistics(result)
        stats.update(job=jid, run=row['run'], request=json.loads(row['payload']))
        _cohort_cache.clear() if len(_cohort_cache) > 32 else None
        _cohort_cache[key] = stats
    return _cohort_cache[key]


@web.get('/api/stats/cohort/<jid>')
def cohort_stats(jid):
    return jsonify(_cohort_result(jid))


def _native_runs(ids):
    runs = []
    for rid in ids:
        try:
            runs.append(st.run(root(), rid))
        except KeyError:
            raise ValueError('Cross-run statistics currently compare native MLP/CNN runs. '
                             'Use the ECML auditor population view for residual-DAG records.')
    return runs


@web.post('/api/stats/cross')
def cross():
    p = request.get_json() or {}
    ids = p.get('runs') or []
    if not isinstance(ids, list) or len(ids) > 200:
        raise ValueError('Select up to 200 runs.')
    for rid in ids:
        ix.sync(root(), rid)
    started = time.time()
    result = E.cross_run(root(), _native_runs(ids), p.get('side', 'conditional'), p.get('name', 'log_eta1'),
                         str(p.get('state', 'final')), p.get('factor', 'optimizer'), p.get('context_kind', 'probe'),
                         bool(p.get('independent')))
    result['seconds'] = round(time.time() - started, 3)
    return jsonify(result)


@web.post('/api/stats/ledger')
def ledger():
    p = request.get_json() or {}
    run_stats_result = cohort = cross_result = None
    if p.get('run'):
        ix.sync(root(), p['run'])
        context = p.get('context')
        run_stats_result = E.run_statistics(root(), p['run'], context if context is not None else _default_context(p['run']))
    if p.get('cohort'):
        cohort = _cohort_result(p['cohort'])
    if p.get('cross'):
        c = p['cross']
        cross_result = E.cross_run(root(), _native_runs(c.get('runs', [])), c.get('side', 'conditional'), c.get('name', 'log_eta1'),
                                   str(c.get('state', 'final')), c.get('factor', 'optimizer'), c.get('context_kind', 'probe'),
                                   bool(c.get('independent')))
    rows = E.ledger(run_stats_result, cohort, cross_result)
    fmt = p.get('format', 'json')
    if fmt == 'csv':
        return Response(E.ledger_csv(rows), mimetype='text/csv')
    if fmt == 'tex':
        return Response(E.ledger_latex(rows), mimetype='text/x-tex')
    return jsonify(rows=rows, columns=E.LEDGER_COLUMNS)


# ------------------------------------------------------------- ruler sensitivity

@web.get('/api/stats/run/<rid>/geometry-records')
def geometry_records(rid):
    """Completed native analyses that saved execution edges (eligible for ruler sensitivity)."""
    info = _info(rid)
    if info['engine'] != 'native':
        return jsonify(records=[], note='Ruler sensitivity reads native MLP/CNN analyses; residual-DAG geometry uses its own recorded lengths.')
    with st.db(root()) as db:
        rows = db.execute("SELECT j.id, j.payload, o.payload FROM jobs j JOIN observations o ON o.job=j.id "
                          "WHERE j.run=? AND j.status='completed' ORDER BY o.step", (rid,)).fetchall()
        rulers = db.execute("SELECT id, payload, status FROM jobs WHERE run=? AND payload LIKE '%ruler_sensitivity%' ORDER BY created DESC", (rid,)).fetchall()
    records = []
    for jid, request, summary in rows:
        summary = json.loads(summary)
        if not summary.get('structural_geometry'):
            continue
        responses = summary['structural_geometry'].get('responses') or {}
        readout = summary.get('readout') or {}
        records.append(dict(job=jid, step=summary.get('step'), scope=readout.get('scope', 'selected'), x=summary.get('x'),
                            reference=summary.get('reference'), reference_type=summary.get('reference_type', 'dataset'),
                            gated=bool(summary.get('gates')), nodes=summary['structural_geometry'].get('measurement_nodes'),
                            measured=responses.get('status') not in (None, 'budget_exceeded')))
    previous = [dict(job=r[0], request=json.loads(r[1]), status=r[2]) for r in rulers]
    return jsonify(records=records, previous=previous)


@web.post('/api/stats/ruler')
def ruler_submit():
    from .ruler import validate_request
    p = request.get_json() or {}
    rid = p.get('run')
    info = _info(rid)
    if info['engine'] != 'native':
        raise ValueError('Ruler sensitivity reads native MLP/CNN analyses.')
    params = validate_request(p)
    payload = dict(kind='ruler_sensitivity', source_job=str(p.get('source_job')), epsilons=params['epsilons'], power=params['power'])
    return jsonify(id=st.queue_measurement(root(), rid, payload))


# ------------------------------------------------------------- presentation mode
#
# A workspace-wide switch enforced on the server. While it is on, requests that would
# queue computation, start training, create experiments or delete data are refused;
# reading saved results, pausing, stopping and cancelling remain available.

BLOCKED_IN_PRESENTATION = (
    ('POST', '/api/runs'), ('POST', '/api/datasets/prepare'), ('POST', '/api/joint/create'),
    ('POST', '/api/joint/data'), ('POST', '/api/auditor/geometry'), ('POST', '/api/stats/ruler'),
)
BLOCKED_SUFFIXES = ('/analysis', '/healthcare', '/measure-missing', '/delete', '/analyze', '/measurement-settings')


def presentation_path():
    return Path(root()) / 'presentation.json'


def presentation_on() -> bool:
    try:
        return bool(json.loads(presentation_path().read_text()).get('on'))
    except (OSError, ValueError):
        return False


@web.app_context_processor
def presentation_context():
    try:
        return dict(presentation=presentation_on())
    except RuntimeError:
        return {}


@web.before_app_request
def presentation_guard():
    if request.method != 'POST' or not presentation_on():
        return None
    path = request.path
    blocked = (request.method, path) in BLOCKED_IN_PRESENTATION or path.endswith(BLOCKED_SUFFIXES)
    if path.endswith('/control'):
        action = (request.get_json(silent=True) or {}).get('action')
        blocked = action in ('start', 'delete')
    if blocked:
        return jsonify(error='Presentation mode is on: saved results are shown and no new computation is submitted. '
                             'Turn presentation mode off in the header to compute.', presentation=True), 409
    return None


@web.get('/api/presentation')
def presentation_state():
    return jsonify(on=presentation_on())


@web.post('/api/presentation')
def presentation_set():
    on = bool((request.get_json() or {}).get('on'))
    st.atomic_json(presentation_path(), dict(on=on, changed=time.time()))
    return jsonify(on=on)
