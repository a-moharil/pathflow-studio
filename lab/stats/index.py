"""Flattened, indexed observables for both engines.

The native engine stores each checkpoint measurement and each analysis summary as a
JSON blob of roughly 100 KB (input fields, source diagnostics, projection audits).
Statistics only need a few dozen scalars per record. This module extracts those
scalars once into an indexed table, so that statistical views and lean polling feeds
never parse the heavy blobs again.

Synchronisation is incremental and lazy: ``sync`` is called by read endpoints, and it
only touches sources whose change marker differs from the indexed marker. The worker,
engines and numerical kernels are not modified.

Row semantics
-------------
``source``    probe | analysis | dose | joint_analysis | training
``context``   canonical JSON of the declared measurement condition
``side``      structural | conditional | response | geometry_structural |
              geometry_conditional | training
``name``      observable name (H1, Sdef, eta1, log_eta1, dstar, val_loss, ...)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from .. import storage as st

INDEX_VERSION = 2  # 2: residual-DAG geometry responses indexed

PATH_OBSERVABLES = ['H1', 'Sdef', 'eta1', 'eta2', 'log_eta1', 'log_eta2', 'logZ', 'log_path_count']
RESPONSE_OBSERVABLES = ['delta', 'cancellation', 'contrast_l2', 'path_hellinger', 'residual']
GEOMETRY_OBSERVABLES = ['dstar', 'deff', 'xi50']

# Heavy keys that no polling consumer reads. They remain in the stored blobs and in
# full exports; lean feeds omit them.
HEAVY_PROBE_KEYS = {'input_fields', 'source_summary', 'layer_fields', 'source_diagnostics',
                    'output_path_sums', 'output_contrasts'}
HEAVY_OBSERVATION_KEYS = {'source_summary', 'layer_fields', 'source_diagnostics',
                          'projection_diagnostics', 'temporal_deformation', 'fit_settings',
                          'output_path_sums', 'output_contrasts', 'logits', 'reference_logits'}
HEAVY_GEOMETRY_KEYS = {'landmark_indices', 'coverage_by_layer', 'leading_positive_eigenvalues'}


def finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(',', ':'))


# --------------------------------------------------------------------------- schema

def init(root) -> None:
    with st.db(root) as db:
        db.executescript('''
        CREATE TABLE IF NOT EXISTS observables(
            source_id TEXT, run TEXT, engine TEXT, source TEXT, step INTEGER,
            context TEXT, side TEXT, name TEXT, value REAL,
            PRIMARY KEY(source_id, side, name));
        CREATE INDEX IF NOT EXISTS observables_run_name ON observables(run, name, side);
        CREATE INDEX IF NOT EXISTS observables_run_context ON observables(run, context);
        CREATE TABLE IF NOT EXISTS observable_sources(
            source_id TEXT PRIMARY KEY, run TEXT, kind TEXT, marker TEXT, version INTEGER);
        CREATE TABLE IF NOT EXISTS artifact_index(
            job TEXT PRIMARY KEY, path TEXT, mtime REAL, size INTEGER,
            sha256 TEXT, summary TEXT);
        ''')


# ------------------------------------------------------------------------ extraction

def _measures(raw: dict | None) -> dict:
    raw = raw or {}
    out = {k: raw.get(k) for k in PATH_OBSERVABLES if finite(raw.get(k))}
    for q in (1, 2):
        if f'log_eta{q}' not in out and finite(raw.get(f'eta{q}')) and raw[f'eta{q}'] > 0:
            out[f'log_eta{q}'] = math.log(raw[f'eta{q}'])
    if 'Sdef' in out and finite(out.get('log_path_count')) and out['log_path_count'] > 0:
        out['normalised_deficit'] = out['Sdef'] / out['log_path_count']
    return out


def native_context(payload: dict, kind: str) -> dict:
    """The declared measurement condition of a native record, as in the Observatory."""
    readout = payload.get('readout') or {'a': 0, 'b': 1}
    return dict(
        kind='probe' if kind == 'probe' else 'analysis',
        scope=payload.get('scope', readout.get('scope', 'selected')),
        readout={k: readout.get(k) for k in ('scope', 'a', 'b') if k in readout},
        x=payload.get('x', 0),
        reference=payload.get('reference', 1),
        reference_type=payload.get('reference_type', 'dataset'),
        transform=payload.get('transform') or {},
        gates=payload.get('gates') or {},
    )


def native_rows(payload: dict, kind: str) -> list[tuple[str, str, float]]:
    rows = []
    for side in ('structural', 'conditional'):
        for name, value in _measures(payload.get(side)).items():
            rows.append((side, name, value))
    for name in RESPONSE_OBSERVABLES:
        if finite(payload.get(name)):
            rows.append(('response', name, payload[name]))
    for side in ('structural', 'conditional'):
        responses = (payload.get(side + '_geometry') or {}).get('responses') or {}
        for name in GEOMETRY_OBSERVABLES:
            if finite(responses.get(name)):
                rows.append(('geometry_' + side, name, responses[name]))
        profile = responses.get('profile') or []
        if profile:
            qualified = sum(1 for p in profile if p.get('qualified'))
            rows.append(('geometry_' + side, 'qualified_fraction', qualified / len(profile)))
    return rows


def training_rows(metric: dict) -> list[tuple[str, str, float]]:
    rows = []
    for key, value in metric.items():
        if key in ('step', 'probe', 'test_evaluated'):
            continue
        if finite(value):
            rows.append(('training', key, value))
        elif isinstance(value, dict):  # joint engine: {'val': {'loss': .., 'dice': ..}}
            for sub, inner in value.items():
                if finite(inner):
                    rows.append(('training', f'{key}_{sub}', inner))
    return rows


def joint_rows(data: dict) -> list[tuple[str, str, str, float]]:
    """Rows for every head of a joint residual-DAG analysis: (head, side, name, value)."""
    rows = []
    for stage, heads in (('baseline', data.get('heads') or {}),
                         ('intervention', (data.get('intervention') or {}).get('heads') or {})):
        for head, payload in heads.items():
            label = head if stage == 'baseline' else f'{head} · intervention'
            for side, key in (('structural', 'structural_observables'), ('conditional', 'path_observables')):
                for name, value in _measures(payload.get(key)).items():
                    rows.append((label, side, name, value))
            for name in ('delta', 'path_hellinger'):
                if finite(payload.get(name)):
                    rows.append((label, 'response', name, payload[name]))
            cancellation = payload.get('global_cancellation')
            if finite(cancellation):
                rows.append((label, 'response', 'cancellation', cancellation))
            # Residual-DAG geometry responses (BCANw on the recorded carrier), when computed.
            responses = (payload.get('geometry') or {}).get('responses') or {}
            for side in ('structural', 'conditional'):
                values = responses.get(side) or {}
                for name in GEOMETRY_OBSERVABLES:
                    if finite(values.get(name)):
                        rows.append((label, 'geometry_' + side, name, values[name]))
    return rows


# ---------------------------------------------------------------------------- sync

def _replace(db, source_id, run, engine, source, step, context, rows, kind, marker):
    db.execute('DELETE FROM observables WHERE source_id=?', (source_id,))
    db.executemany('INSERT OR REPLACE INTO observables VALUES(?,?,?,?,?,?,?,?,?)',
                   [(source_id, run, engine, source, step, context, side, name, float(value))
                    for side, name, value in rows])
    db.execute('INSERT OR REPLACE INTO observable_sources VALUES(?,?,?,?,?)',
               (source_id, run, kind, marker, INDEX_VERSION))


def sync(root, run: str | None = None) -> dict:
    """Index new or changed sources. Returns counts of newly indexed sources."""
    init(root)
    added = dict(probe=0, analysis=0, joint=0, training=0)
    run_filter = ' AND m.run=?' if run else ''
    args = (run,) if run else ()
    with st.db(root) as db:
        indexed = {r[0]: (r[1], r[2]) for r in db.execute(
            'SELECT source_id, marker, version FROM observable_sources' + (' WHERE run=?' if run else ''), args)}

        # Native checkpoint measurements: training metrics plus the fixed probe.
        for row in db.execute('SELECT m.run, m.step, length(m.payload) FROM measurements m WHERE 1=1' + run_filter, args):
            rid, step, size = row
            sid = f'probe:{rid}:{step}'
            if indexed.get(sid) == (str(size), INDEX_VERSION):
                continue
            payload = json.loads(db.execute('SELECT payload FROM measurements WHERE run=? AND step=?', (rid, step)).fetchone()[0])
            probe = payload.get('probe') or {}
            context = canonical(native_context(probe, 'probe'))
            rows = native_rows(probe, 'probe') + training_rows(payload)
            _replace(db, sid, rid, 'native', 'probe', step, context, rows, 'probe', str(size))
            added['probe'] += 1

        # Native analysis and dose summaries.
        obs_filter = ' WHERE run=?' if run else ''
        for job, rid, step, size in db.execute('SELECT job, run, step, length(payload) FROM observations' + obs_filter, args).fetchall():
            if indexed.get(job) == (str(size), INDEX_VERSION):
                continue
            payload = json.loads(db.execute('SELECT payload FROM observations WHERE job=?', (job,)).fetchone()[0])
            kind_row = db.execute('SELECT payload FROM jobs WHERE id=?', (job,)).fetchone()
            kind = json.loads(kind_row[0]).get('kind', 'analysis') if kind_row else 'analysis'
            context = native_context(payload, 'analysis')
            context['measurement_limit'] = (payload.get('measurement_settings') or {}).get('limit')
            _replace(db, job, rid, 'native', kind, step, canonical(context), native_rows(payload, kind), kind, str(size))
            added['analysis'] += 1

        # Joint (residual-DAG) runs: history and completed analyses.
        try:
            joint_ids = [r[0] for r in db.execute('SELECT id FROM joint_runs' + (' WHERE id=?' if run else ''), args)]
        except Exception:  # joint engine table absent in very old workspaces
            joint_ids = []
        for rid in joint_ids:
            history = Path(root) / rid / 'history.json'
            if not history.exists():
                history = Path(root) / 'joint' / rid / 'history.json'
            if history.exists():
                marker = f'{history.stat().st_mtime_ns}:{history.stat().st_size}'
                sid = f'history:{rid}'
                if indexed.get(sid) != (marker, INDEX_VERSION):
                    db.execute('DELETE FROM observables WHERE source_id LIKE ?', (f'history:{rid}:%',))
                    for entry in json.loads(history.read_text()):
                        rows = training_rows({k: v for k, v in entry.items() if k != 'train_batch'})
                        if rows:
                            _replace(db, f'history:{rid}:{entry["step"]}', rid, 'joint', 'training', entry['step'],
                                     canonical(dict(kind='training')), rows, 'training', marker)
                    db.execute('INSERT OR REPLACE INTO observable_sources VALUES(?,?,?,?,?)', (sid, rid, 'history', marker, INDEX_VERSION))
                    added['training'] += 1
            for job, payload_text, result in db.execute(
                    "SELECT id, payload, result FROM jobs WHERE run=? AND status='completed'", (rid,)).fetchall():
                request = json.loads(payload_text)
                if request.get('kind') != 'joint_analysis' or not result:
                    continue
                path = Path(result)
                if not path.is_file():
                    continue
                marker = f'{path.stat().st_mtime_ns}:{path.stat().st_size}'
                if indexed.get(job) == (marker, INDEX_VERSION):
                    continue
                data = json.loads(path.read_text())
                db.execute('DELETE FROM observables WHERE source_id=? OR source_id LIKE ?', (job, job + ':%'))
                by_head: dict[str, list] = {}
                for head, side, name, value in joint_rows(data):
                    by_head.setdefault(head, []).append((side, name, value))
                for head, rows in by_head.items():
                    context = dict(kind='joint_analysis', head=head, x=request.get('index', request.get('x')),
                                   reference=request.get('reference'), reference_type=request.get('reference_type', 'dataset'),
                                   transform=request.get('transform') or {}, channel=request.get('channel'), alpha=request.get('alpha'))
                    _replace(db, f'{job}:{head}', rid, 'joint', 'joint_analysis', data.get('checkpoint'), canonical(context), rows, 'joint', marker)
                db.execute('INSERT OR REPLACE INTO observable_sources VALUES(?,?,?,?,?)', (job, rid, 'joint', marker, INDEX_VERSION))
                added['joint'] += 1

        # Prune rows of deleted experiments.
        if not run:
            db.execute('''DELETE FROM observables WHERE run NOT IN (SELECT id FROM runs)
                          AND run NOT IN (SELECT id FROM joint_runs)''')
            db.execute('''DELETE FROM observable_sources WHERE run NOT IN (SELECT id FROM runs)
                          AND run NOT IN (SELECT id FROM joint_runs)''')
    return added


# ------------------------------------------------------------------------- queries

def contexts(root, run: str) -> list[dict]:
    with st.db(root) as db:
        rows = db.execute('''SELECT context, source, count(DISTINCT source_id), min(step), max(step)
                             FROM observables WHERE run=? AND side<>'training'
                             GROUP BY context, source ORDER BY source, context''', (run,)).fetchall()
    out = []
    for context, source, n, lo, hi in rows:
        out.append(dict(key=context, source=source, records=n, first=lo, last=hi, context=json.loads(context)))
    return out


def series(root, run: str, context: str | None = None) -> dict:
    """{(side, name): [(step, value, source_id), ...]} for one declared context.

    Training metrics are attached regardless of context. Multiple records at the same
    checkpoint and context (for instance repeated measurements with different budgets)
    are kept separately; callers decide how to aggregate them and must say so.
    """
    with st.db(root) as db:
        rows = db.execute('''SELECT side, name, step, value, source_id FROM observables
                             WHERE run=? AND (side='training' OR context=?) ORDER BY step''',
                          (run, context or '')).fetchall()
    out: dict = {}
    for side, name, step, value, sid in rows:
        out.setdefault((side, name), []).append((step, value, sid))
    return out


def lean_probe(metric: dict) -> dict:
    probe = metric.get('probe')
    if not isinstance(probe, dict):
        return metric
    return dict(metric, probe={k: v for k, v in probe.items() if k not in HEAVY_PROBE_KEYS})


def lean_observation(obs: dict) -> dict:
    out = {k: v for k, v in obs.items() if k not in HEAVY_OBSERVATION_KEYS}
    for side in ('structural_geometry', 'conditional_geometry'):
        if isinstance(out.get(side), dict):
            out[side] = {k: v for k, v in out[side].items() if k not in HEAVY_GEOMETRY_KEYS}
    return out
