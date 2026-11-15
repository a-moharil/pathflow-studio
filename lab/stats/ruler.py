"""Ruler sensitivity: recompute BCANw from a saved analysis at declared ε (and p).

The theory's flow-induced edge length is the declared ruler

    λ_e = (F̃_e + ε)^(-p),   F̃_e = |E_ℓ| F_e  (architecture-normalised enrichment),

with p = 1 and ε = 0.03 as the studio's recorded convention. This module changes
neither. It reads the saved edge marginals of one completed analysis, rebuilds the
metric for each requested (ε, p), and reruns the same legacy BCANw estimator
(``bcan-1``, 36 radii, strength-ordered greedy colouring) that produced the saved
d*. A recomputation at the recorded convention is included as a reproduction check.

Nothing is written back to the analysis; the result is a separate job artefact.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from .. import storage as st
from ..core.pathmath import bcan

RECORDED = dict(epsilon=0.03, power=1.0)
MAX_EPSILONS = 8


def validate_request(payload: dict) -> dict:
    eps = payload.get('epsilons', [0.003, 0.01, 0.03, 0.1, 0.3])
    if isinstance(eps, str):
        eps = [x.strip() for x in eps.split(',') if x.strip()]
    try:
        eps = sorted({float(x) for x in eps})
    except (TypeError, ValueError):
        raise ValueError('ε values must be numbers, for example 0.003, 0.03, 0.3')
    if not 1 <= len(eps) <= MAX_EPSILONS or any(not 1e-6 <= e <= 10 for e in eps):
        raise ValueError(f'Give 1..{MAX_EPSILONS} ε values in 1e-6..10')
    power = float(payload.get('power', 1.0))
    if power == 0 or not -4 <= power <= 4:
        raise ValueError('p must be non-zero and within −4..4 (negative p is the sign-reversal robustness test)')
    return dict(epsilons=eps, power=power)


def _metric_inputs(data: dict):
    nodes, edges = data.get('nodes'), data.get('edges')
    if not nodes or not edges:
        raise ValueError('Select a completed geometry analysis (one that saved its execution edges).')
    layer = np.array([q['layer'] for q in nodes])
    u = np.array([e['u'] for e in edges])
    v = np.array([e['v'] for e in edges])
    cut = np.bincount(layer[u])[layer[u]]  # |E_ℓ|: edges in the source cut of each edge
    last = layer[u] == layer[u].max()
    return len(nodes), u, v, cut, last


def covering(n, u, v, cut, last, mass, epsilon, power):
    """BCANw exactly as pathmath.geometry: same lengths, strengths and estimator."""
    enrichment = cut * mass
    lengths = (enrichment + epsilon) ** (-power)
    A = coo_matrix((np.r_[lengths, lengths], (np.r_[u, v], np.r_[v, u])), shape=(n, n)).tocsr()
    D = dijkstra(A, directed=False)
    if not np.isfinite(D).all():
        raise ValueError('Disconnected execution graph under this metric')
    strength = np.zeros(n)
    np.add.at(strength, u, mass)
    np.add.at(strength, v[last], mass[last])
    r = bcan(D, strength)
    diameter = float(D.max())
    qualified = sum(1 for p in r['profile'] if p['qualified'])
    return dict(epsilon=epsilon, power=power, dstar=r['dstar'], deff=r['deff'], xi50=r['xi50'],
                xi50_over_diameter=r['xi50'] / diameter if r['xi50'] else None, diameter=diameter,
                qualified_windows=qualified, scales=len(r['profile']),
                floor_fraction=float(np.mean(enrichment < epsilon)),
                profile=[dict(radius=p['radius'], boxes=p['boxes'], deff=p['deff'], qualified=p['qualified']) for p in r['profile']])


def process(root, job) -> None:
    payload = json.loads(job['payload'])
    request = validate_request(payload)
    with st.db(root) as db:
        row = db.execute("SELECT run, payload, result, status FROM jobs WHERE id=?", (payload['source_job'],)).fetchone()
    if not row or row['status'] != 'completed' or row['run'] != job['run']:
        raise ValueError('The source analysis must be a completed analysis of this experiment.')
    path = Path(row['result']).resolve()
    if not path.is_relative_to(Path(root).resolve()) or not path.is_file():
        raise ValueError('The saved analysis file is missing.')
    data = json.loads(path.read_text())
    n, u, v, cut, last = _metric_inputs(data)
    limit = int((data.get('measurement_settings') or {}).get('limit') or data.get('structural_geometry', {}).get('measurement_limit') or 2500)
    if n > limit:
        raise ValueError(f'{n} nodes exceed this analysis measurement budget ({limit}); BCANw was not measured for it.')
    started = time.time()
    sides = []
    for side, key in (('structural', 's'), ('conditional', 'c')):
        if not (data.get(side) or {}).get('defined'):
            sides.append(dict(side=side, status='undefined_measure', rows=[]))
            continue
        mass = np.array([e[key] for e in data['edges']], float)
        saved = (data.get(side + '_geometry') or {}).get('responses') or {}
        rows = []
        grid = sorted({*request['epsilons']} | {RECORDED['epsilon']}) if request['power'] == RECORDED['power'] else request['epsilons']
        for epsilon in grid:
            st.progress(root, job['id'], f'Ruler sensitivity · {side} · ε = {epsilon:g}, p = {request["power"]:g}')
            row_result = covering(n, u, v, cut, last, mass, epsilon, request['power'])
            row_result['recorded_convention'] = epsilon == RECORDED['epsilon'] and request['power'] == RECORDED['power']
            row_result['requested'] = epsilon in request['epsilons']
            rows.append(row_result)
        reference = next((r for r in rows if r['recorded_convention']), None)
        check = None
        if reference is not None:
            same = lambda a, b: (a is None and b is None) or (a is not None and b is not None and abs(a - b) <= 1e-9 * max(1, abs(b)))
            check = dict(saved_dstar=saved.get('dstar'), recomputed_dstar=reference['dstar'], saved_xi50=saved.get('xi50'),
                         recomputed_xi50=reference['xi50'], reproduced=same(reference['dstar'], saved.get('dstar')) and same(reference['xi50'], saved.get('xi50')))
        sides.append(dict(side=side, status='available', rows=rows, reproduction=check))
    result = dict(kind='ruler_sensitivity', version='ruler-1', source_job=payload['source_job'], step=data.get('step'),
                  context=dict(readout=data.get('readout'), x=data.get('x'), reference=data.get('reference'),
                               reference_type=data.get('reference_type'), gates=data.get('gates')),
                  nodes=n, edges=len(u), request=request, recorded=RECORDED, sides=sides, seconds=round(time.time() - started, 2),
                  definition='λ_e = (|E_ℓ|F_e + ε)^(−p); undirected support metric; legacy bcan-1 estimator (36 radii, strength-ordered greedy colouring). '
                             'floor_fraction is the share of edges whose enrichment |E_ℓ|F_e is below ε, where ε dominates the length.',
                  status='Diagnostic of the declared ruler. The recorded convention (ε = 0.03, p = 1) is unchanged.')
    out = Path(root) / job['run'] / f"ruler_{job['id']}.json"
    st.atomic_json(out, result)
    with st.db(root) as db:
        db.execute("UPDATE jobs SET status='completed', result=?, progress='Ruler sensitivity ready' WHERE id=?", (str(out), job['id']))
