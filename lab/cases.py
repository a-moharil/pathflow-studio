"""Cases: the unit an investigation follows from page to page.

A case is one input pair at one committed checkpoint: (checkpoint, split, input x,
reference type, reference). Every stage of the studio computes its own result for a case
(geometry and fields, the clinical reading, the residual-DAG investigation), and each of
those results is an ordinary completed job. This module only reads recorded jobs:

* it names a case by the job that first recorded it, so a link can carry ``?case=<job>``;
* it finds, for any stage, the latest completed job of that stage's kind whose case
  coordinates match exactly; nothing approximate is ever substituted;
* it lists recent work across the workspace for the Queue.

No endpoint here queues work or touches checkpoints, so all of them stay available in
presentation mode.
"""
from __future__ import annotations

import json
from pathlib import Path

from flask import Blueprint, current_app, jsonify, render_template, request

from . import storage as st
from .stats.navigation import describe_run

web = Blueprint('cases', __name__)

# Job kinds that record a single case (an input pair). Cohorts, dataset preparation and
# ruler sensitivity checks are not cases.
CASE_KINDS = ('analysis', 'dose', 'ensemble', 'healthcare_case', 'healthcare_dose', 'healthcare_evidence', 'joint_analysis')

# Stage that opens a job of each kind.
STAGE_OF_KIND = {
    'analysis': 'geometry', 'dose': 'geometry', 'ensemble': 'geometry', 'joint_analysis': 'geometry',
    'healthcare_case': 'clinical', 'healthcare_dose': 'clinical', 'healthcare_evidence': 'clinical',
    'healthcare_cohort': 'clinical', 'ruler_sensitivity': 'evidence',
}


def root():
    return current_app.config['ROOT']


def coordinates(payload: dict) -> dict | None:
    """Case coordinates of a job request, or None when the job is not a single case."""
    kind = payload.get('kind', 'analysis')
    if kind not in CASE_KINDS:
        return None
    step = payload.get('checkpoint') if kind == 'joint_analysis' else payload.get('step')
    x = payload.get('x', payload.get('index'))
    if step is None or x is None:
        return None
    try:
        step = int(step)
    except (TypeError, ValueError):
        return None
    reference_type = payload.get('reference_type', 'dataset')
    reference = None if reference_type == 'zero' else payload.get('reference')
    return dict(step=step, split=payload.get('split', 'test'), x=int(x), reference_type=reference_type,
                reference=None if reference is None else int(reference))


def label(c: dict) -> str:
    ref = 'zero reference' if c['reference_type'] == 'zero' else f"reference {c['reference']}"
    return f"Update {c['step']} · {c['split']} · input {c['x']} vs {ref}"


def _jobs(rid: str) -> list[dict]:
    with st.db(root()) as db:
        rows = db.execute("SELECT id, payload, created FROM jobs WHERE run=? AND status='completed' ORDER BY created DESC",
                          (rid,)).fetchall()
    out = []
    for jid, payload, created in rows:
        p = json.loads(payload)
        out.append(dict(job=jid, kind=p.get('kind', 'analysis'), created=created, coords=coordinates(p), payload=p))
    return out


def _summary(rid: str, j: dict) -> dict:
    c = j['coords']
    return dict(job=j['job'], run=rid, kind=j['kind'], created=j['created'], stage=STAGE_OF_KIND.get(j['kind'], 'geometry'),
                coords=c, label=label(c) if c else j['kind'])


@web.get('/api/run/<rid>/cases')
def cases(rid):
    if not describe_run(root(), rid):
        raise KeyError('Experiment not found')
    seen, out = set(), []
    for j in _jobs(rid):
        if not j['coords']:
            continue
        key = json.dumps(j['coords'], sort_keys=True)
        entry = _summary(rid, j)
        entry['first'] = key not in seen
        seen.add(key)
        out.append(entry)
    return jsonify(cases=out)


@web.get('/api/case/<jid>')
def case(jid):
    with st.db(root()) as db:
        row = db.execute('SELECT id, run, payload, status FROM jobs WHERE id=?', (jid,)).fetchone()
    if not row:
        raise KeyError('Case not found')
    p = json.loads(row['payload'])
    c = coordinates(p)
    if not c:
        raise ValueError('This record is not a single case')
    info = describe_run(root(), row['run'])
    stages = {}
    for j in _jobs(row['run']):
        if j['coords'] == c:
            stages.setdefault(j['kind'], j['job'])
    return jsonify(job=jid, run=row['run'], engine=info['engine'] if info else None, kind=p.get('kind', 'analysis'),
                   status=row['status'], coords=c, label=label(c), stages=stages)


@web.get('/api/run/<rid>/case/open')
def open_case(rid):
    """The result a stage should open: the case's own result of ``kind``, else the latest.

    ``kind`` may list several kinds separated by commas (for example the clinical reading
    accepts ``healthcare_case,healthcare_dose``). With ``case`` given, only an exact match
    of its coordinates is returned; ``match`` is null when the stage has not been
    computed for that case, and ``coords`` lets the page pre-fill its controls.
    """
    if not describe_run(root(), rid):
        raise KeyError('Experiment not found')
    kinds = [k for k in (request.args.get('kind') or 'analysis').split(',') if k]
    wanted = request.args.get('case') or ''
    jobs = _jobs(rid)
    target = None
    if wanted:
        source = next((j for j in jobs if j['job'] == wanted), None)
        target = source['coords'] if source else None
    for j in jobs:
        if j['kind'] not in kinds:
            continue
        if target is None and not wanted:
            return jsonify(match=j['job'], coords=j['coords'], case=_summary(rid, j) if j['coords'] else None, exact=False)
        if target is not None and j['coords'] == target:
            return jsonify(match=j['job'], coords=target, case=_summary(rid, j), exact=True)
    return jsonify(match=None, coords=target, case=None, exact=bool(target))


# ----------------------------------------------------------------------------- Queue

def _run_names() -> dict:
    with st.db(root()) as db:
        names = {r[0]: (json.loads(r[1]).get('name', r[0]), 'native') for r in db.execute('SELECT id, config FROM runs')}
        try:
            names.update({r[0]: (json.loads(r[1]).get('name', r[0]), 'joint') for r in db.execute('SELECT id, config FROM joint_runs')})
        except Exception:
            pass
    return names


@web.get('/queue')
def queue_page():
    return render_template('queue.html', assets='minimal')


@web.get('/api/queue')
def queue():
    limit = max(1, min(500, request.args.get('limit', 200, type=int)))
    names = _run_names()
    with st.db(root()) as db:
        rows = db.execute('SELECT id, run, payload, status, progress, error, created FROM jobs ORDER BY created DESC LIMIT ?',
                          (limit,)).fetchall()
    out = []
    for r in rows:
        p = json.loads(r['payload'])
        kind = p.get('kind', 'analysis')
        rid = r['run']
        name, engine = names.get(rid, (None, None)) if rid else (None, None)
        c = coordinates(p)
        stage = STAGE_OF_KIND.get(kind)
        href = None
        if rid and name and r['status'] == 'completed' and stage:
            href = f'/run/{rid}/{stage}' + (f'?case={r["id"]}' if c else '')
        out.append(dict(job=r['id'], run=rid, run_name=name, engine=engine, kind=kind, status=r['status'],
                        progress=r['progress'], error=((r['error'] or '').strip().splitlines() or [''])[-1],
                        created=r['created'], label=label(c) if c else None, open=href))
    with st.db(root()) as db:
        counts = dict(db.execute('SELECT status, count(*) FROM jobs GROUP BY status').fetchall())
    worker = Path(root()) / 'worker.json'
    try:
        heartbeat = json.loads(worker.read_text()).get('heartbeat')
    except (OSError, ValueError):
        heartbeat = None
    return jsonify(jobs=out, counts=counts, heartbeat=heartbeat)
