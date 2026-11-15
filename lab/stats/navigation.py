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
    tabs = [dict(key='overview', label='Overview', href=f'/run/{rid}')]
    tabs.append(dict(key='training', label='Training', href=f'/experiment/{rid}?section=Training'))
    if engine == 'native':
        tabs.append(dict(key='geometry', label='Geometry', href=f'/experiment/{rid}?section=Geometry'))
        tabs.append(dict(key='observatory', label='Observatory', href=f'/observatory/{rid}'))
        if config.get('dataset') in HEALTHCARE_DATASETS:
            tabs.append(dict(key='healthcare', label='Healthcare', href=f'/healthcare/{rid}'))
    else:
        tabs.append(dict(key='geometry', label='Geometry', href=f'/experiment/{rid}?section=geometry'))
        tabs.append(dict(key='investigation', label='Investigation', href=f'/experiment/{rid}?section=fieldsPanel'))
        tabs.append(dict(key='observatory', label='Observatory', href=f'/experiment/{rid}?section=resnetObservatory'))
    tabs.append(dict(key='statistics', label='Statistics', href=f'/run/{rid}/statistics'))
    tabs.append(dict(key='auditor', label='ECML auditor', href=f'/auditor?run={rid}'))
    return tabs


def active_tab(endpoint: str | None) -> str | None:
    section = (request.args.get('section') or '').lower()
    return {
        'stats.overview': 'overview',
        'stats.run_statistics': 'statistics',
        'web.observatory': 'observatory',
        'web.healthcare_page': 'healthcare',
        'auditor.page': 'auditor',
        'web.experiment': {'geometry': 'geometry', 'fields': 'geometry', 'fieldspanel': 'investigation',
                           'resnetobservatory': 'observatory'}.get(section, 'training'),
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
        tabs=tabs_for(rid, engine, config), active=active_tab(request.endpoint)))
