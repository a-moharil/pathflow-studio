"""Run workspace navigation shared by every run-scoped page of both engines.

A context processor derives the run from the current request (route ``rid`` or the
auditor's ``?run=`` filter) and renders one persistent bar: the run identity, its
committed state and the views that apply to it. Pages that do not apply to a run
(for example Healthcare for a ten-class dataset) are omitted rather than disabled.
"""
from __future__ import annotations

import json

from flask import current_app, request

from .. import storage as st

HEALTHCARE_DATASETS = ('pneumonia', 'synthetic')


def _native(root, rid):
    try:
        return st.run(root, rid)
    except KeyError:
        return None


def _joint(root, rid):
    with st.db(root) as db:
        try:
            row = db.execute('SELECT * FROM joint_runs WHERE id=?', (rid,)).fetchone()
        except Exception:
            return None
    if not row:
        return None
    r = dict(row)
    r['config'] = json.loads(r['config'])
    return r


def describe_run(root, rid):
    """Identity and engine of a run, or None."""
    run = _native(root, rid)
    if run:
        return dict(run=run, engine='native')
    run = _joint(root, rid)
    if run:
        return dict(run=run, engine='joint')
    return None


def tabs_for(rid: str, engine: str, config: dict) -> list[dict]:
    """The stage rail of a run, grouped as Run · Case · Accountability.

    Canonical addresses are ``/run/<rid>/<stage>``; they resolve to the pages that hold
    each view, so older page addresses keep working. Healthcare is not a stage: for binary
    native runs it is offered as the Clinical reading of the same case.
    """
    tabs = [dict(key='overview', label='Overview', group='Run', href=f'/run/{rid}'),
            dict(key='training', label='Training', group='Run', href=f'/run/{rid}/training'),
            dict(key='geometry', label='Geometry', group='Case', href=f'/run/{rid}/geometry',
                 hint='Structural disposition and conditional realisation'),
            dict(key='realisation', label='Realisation', group='Case', href=f'/run/{rid}/realisation',
                 hint='Field A over the contrast, routes and transformations'),
            dict(key='observables', label='Observables', group='Case', href=f'/run/{rid}/observables',
                 hint='Path observables across checkpoints'),
            dict(key='evidence', label='Evidence', group='Accountability', href=f'/run/{rid}/evidence'),
            dict(key='audit', label='Audit', group='Accountability', href=f'/run/{rid}/audit')]
    if engine == 'native' and config.get('dataset') in HEALTHCARE_DATASETS:
        tabs.append(dict(key='clinical', label='Clinical reading', group='Lens', href=f'/run/{rid}/clinical'))
    return tabs


def active_tab(endpoint: str | None) -> str | None:
    section = (request.args.get('section') or '').lower()
    return {
        'stats.overview': 'overview',
        'stats.run_statistics': 'evidence',
        'web.observatory': 'observables',
        'web.healthcare_page': 'clinical',
        'auditor.page': 'audit',
        'web.experiment': {'geometry': 'geometry', 'fields': 'realisation', 'fieldspanel': 'realisation',
                           'resnetobservatory': 'observables'}.get(section, 'training'),
    }.get(endpoint or '')


def context_processor():
    """Inject ``run_nav`` for run-scoped pages; harmless elsewhere."""
    rid = (request.view_args or {}).get('rid') or (request.args.get('run') if request.endpoint == 'auditor.page' else None)
    if not rid or request.path.startswith('/api/'):
        return {}
    info = describe_run(current_app.config['ROOT'], rid)
    if not info:
        return {}
    run, engine = info['run'], info['engine']
    config = run['config']
    family = (config.get('architecture') or 'ResNet') if engine == 'joint' else config.get('model', '').upper()
    return dict(run_nav=dict(
        rid=rid, engine=engine, name=config.get('name', rid), status=run.get('status'), step=run.get('step'),
        family=family, dataset=config.get('dataset'), optimizer=config.get('optimizer'),
        tabs=tabs_for(rid, engine, config), active=active_tab(request.endpoint),
        healthcare=engine == 'native' and config.get('dataset') in HEALTHCARE_DATASETS))
