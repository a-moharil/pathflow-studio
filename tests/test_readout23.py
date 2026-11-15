"""Fully convolutional readout (1x1 convolution + global spatial averaging): exactness."""
import itertools
import json
import math

import numpy as np
import pytest
import torch

from lab import create_app, datasets, storage as st
from lab.config import validate, estimate
from lab.core.network import Network, SpatialAverage, hidden_count
from lab.core.pathmath import analyze, graph, measure
from lab.worker import tick

TINY = dict(model='cnn', channels=[2, 2], kernels=[3, 3], stride=3, padding=0, cnn_readout='gap', bias=True)


def tiny(seed=0, **overrides):
    torch.manual_seed(seed)
    m = Network(validate(TINY | overrides)).double()
    with torch.no_grad():  # nonzero biases make the finite secants genuinely nonlinear
        for layer in m.layers:
            if getattr(layer, 'bias', None) is not None:
                layer.bias.uniform_(-.2, .2)
    return m


def test_readout_structure_and_forward():
    m = tiny()
    assert [type(l).__name__ for l in m.layers] == ['Conv2d', 'Conv2d', 'Conv2d', 'SpatialAverage']
    assert m.layers[2].kernel_size == (1, 1) and m.shapes[-2:] == [(2, 3, 3), (2,)]
    assert hidden_count(m) == 2
    x = torch.rand(1, 1, 28, 28, dtype=torch.float64)
    h = x
    for layer in m.layers[:2]:
        h = torch.relu(layer(h))
    expected = m.layers[2](h).mean(dim=(2, 3))  # no activation after the 1x1 map
    assert torch.allclose(m(x), expected)
    sizes, offset, layers = graph(m, dict(scope='all', a=0, b=1))
    u, v, w = layers[3]
    assert np.allclose(w, 1 / 9) and len(u) == 18 and set(v.tolist()) == {0, 1}


def test_signed_reconstruction_is_exact():
    rng = np.random.default_rng(3)
    for activation in ['relu', 'tanh']:
        m = tiny(activation=activation)
        x, x0 = rng.random((1, 28, 28)), rng.random((1, 28, 28))
        out = analyze(m, x, x0, dict(scope='all', a=0, b=1), geometry_enabled=False)
        assert out['vector_residual'] < 1e-10 and out['residual'] < 1e-10
        with torch.no_grad():
            y = m(torch.as_tensor(x[None])) - m(torch.as_tensor(x0[None]))
        assert np.allclose(out['output_contrasts'], y[0].numpy())
        names = [f['name'] for f in out['layer_fields']]
        assert names[2] == 'Readout map · 1×1 convolution' and names[3] == 'Output · global average'


def enumerate_paths(sizes, layers, start, sec):
    """Every complete input-to-sink path with its squared amplitude and edge indices."""
    edges_from = []
    for u, v, w in layers:
        table = {}
        for k, (a, b) in enumerate(zip(u, v)):
            table.setdefault(int(a), []).append((int(b), k))
        edges_from.append(table)
    paths = []

    def walk(node, l, weight, chosen):
        if l == len(layers):
            paths.append((weight, chosen))
            return
        for nxt, k in edges_from[l].get(node, []):
            gain = (layers[l][2][k] * sec[l][nxt]) ** 2
            if gain > 0:
                walk(nxt, l + 1, weight * gain, chosen + [k])

    for i in range(sizes[0]):
        if start[i] > 0:
            walk(i, 0, start[i], [])
    return paths


@pytest.mark.parametrize('activation', ['relu', 'tanh'])
def test_measures_match_complete_path_enumeration(activation):
    rng = np.random.default_rng(11)
    m = tiny(seed=5, activation=activation)
    x, x0 = rng.random((1, 28, 28)), rng.random((1, 28, 28))
    readout = dict(scope='all', a=0, b=1)
    sizes, offset, layers = graph(m, readout)
    with torch.no_grad():
        _, hs, zs = m(torch.as_tensor(x[None]), None, True)
        _, h0, z0 = m(torch.as_tensor(x0[None]), None, True)
    hidden = hidden_count(m)
    sec = []
    for l in range(len(zs)):
        a, b = zs[l].numpy().ravel(), z0[l].numpy().ravel()
        d = hs[l + 1].numpy().ravel() - h0[l + 1].numpy().ravel()
        default = np.ones_like(a) if l >= hidden else ((a > 0).astype(float) if activation == 'relu' else 1 - np.tanh(a) ** 2)
        sec.append(np.divide(d, a - b, out=default, where=a != b) if l < hidden else np.ones_like(a))
    sec.append(np.ones(1))
    dx = (x - x0).ravel()
    for start, secants in [(np.ones(sizes[0]), [np.ones(n) for n in sizes[1:]]), (dx * dx, sec)]:
        result, F = measure(sizes, layers, start, secants)
        paths = enumerate_paths(sizes, layers, start, secants)
        Q = np.array([p[0] for p in paths])
        Z = Q.sum()
        p = Q / Z
        assert result['defined'] and math.isclose(result['logZ'], math.log(Z), rel_tol=1e-10)
        assert math.isclose(result['H1'], float(-(p * np.log(p)).sum()), rel_tol=1e-9)
        n_topo = len(enumerate_paths(sizes, layers, np.ones(sizes[0]), [np.ones(n) for n in sizes[1:]]))
        assert math.isclose(result['log_path_count'], math.log(n_topo), rel_tol=1e-12)
        assert math.isclose(result['eta1'], math.exp(result['H1']) / n_topo, rel_tol=1e-9)
        for l in range(len(layers)):  # exact edge marginals, including the averaging edges
            marginal = np.zeros(len(layers[l][0]))
            for weight, chosen in paths:
                marginal[chosen[l]] += weight / Z
            assert np.allclose(F[l], marginal, rtol=1e-9, atol=1e-14)
            assert math.isclose(F[l].sum(), 1.0, rel_tol=1e-10)


def test_config_options_and_estimate():
    with pytest.raises(ValueError, match='dense hidden head'):
        validate(TINY | dict(cnn_head=True))
    with pytest.raises(ValueError, match='one kernel size per convolution layer'):
        validate(TINY | dict(kernels='3'))
    c = validate(dict(model='cnn', cnn_head=False, head_widths=''))  # blank widths are ignored when the head is off
    assert c['cnn_readout'] == 'dense' and c['head_widths'] == [32, 16]
    with pytest.raises(ValueError, match='Head widths'):
        validate(dict(model='cnn', cnn_head=True, head_widths=''))
    for config in [TINY, TINY | dict(kernels='3,1', stride=1, padding=0, channels=[2, 2]), dict(model='cnn'), dict(model='cnn', cnn_head=True)]:
        c = validate(config)
        sizes, _, layers = graph(Network(c), dict(scope='all', a=0, b=1))
        e = estimate(c)
        assert e['nodes'] == sum(sizes)  # input to outputs, plus the readout sink
        assert e['parameters'] == sum(p.numel() for p in Network(c).parameters())
    paper = validate(dict(model='cnn', channels=[4, 4], kernels=[3, 3], stride=1, padding=0, bias=False, cnn_readout='gap', task='regression', dataset='synthetic-regression'))
    assert estimate(paper)['parameters'] == 184  # C1_shallow3 of the CNN panel


def test_dense_configurations_are_unchanged():
    old = dict(model='cnn', channels=[4, 6, 8])  # a stored configuration without the new keys
    keys = list(Network(validate(old)).state_dict())
    assert keys == ['layers.0.weight', 'layers.0.bias', 'layers.1.weight', 'layers.1.bias', 'layers.2.weight', 'layers.2.bias', 'layers.3.weight', 'layers.3.bias']
    assert hidden_count(Network(validate(old))) == 3 and hidden_count(Network(validate(dict(model='mlp')))) == 6


def test_all_jobs_run_on_the_convolutional_readout(tmp_path):
    app = create_app(tmp_path)
    client = app.test_client()
    client.get('/')
    with client.session_transaction() as s:
        headers = {'X-CSRF-Token': s['csrf']}
    config = dict(name='gap', model='cnn', channels='3,4', kernels='3,3', stride=2, padding=1, cnn_readout='gap', train_size=32, updates=2, checkpoint_every=1, landmarks=16)
    datasets.prepare(tmp_path, validate(config))
    rid = client.post('/api/runs', json=config, headers=headers).json['id']
    client.post(f'/api/run/{rid}/control', json={'action': 'start'}, headers=headers)
    while tick(tmp_path):
        pass
    base = dict(step=2, x=0, reference=1, split='test')
    jobs = [client.post(f'/api/run/{rid}/analysis', json=dict(kind='analysis', step=2, x=0, reference=1, readout=dict(scope='all', a=0, b=1), gates={'0': {'0': .5}}), headers=headers),
            client.post(f'/api/run/{rid}/healthcare', json=base | dict(kind='healthcare_case'), headers=headers),
            client.post(f'/api/run/{rid}/healthcare', json=base | dict(kind='healthcare_dose', intervention='internal', layer=1, selection='top'), headers=headers),
            client.post(f'/api/run/{rid}/healthcare', json=base | dict(kind='healthcare_evidence', evidence_mode='compare', source_region=[0, 0, 14, 28], ig_steps=8, support_budget=2, support_kind='occurrence', support_layers='all', support_rule='flow', visual_layer=1, visual_channel=0), headers=headers),
            client.post(f'/api/run/{rid}/healthcare', json=base | dict(kind='healthcare_cohort', cohort_size=6, cohort_reference='zero'), headers=headers)]
    assert all(r.status_code == 200 for r in jobs), [r.json for r in jobs]
    # The readout map is not a hidden layer: gates and interventions on it are refused.
    assert client.post(f'/api/run/{rid}/analysis', json=dict(kind='analysis', step=2, gates={'2': {'0': .5}}), headers=headers).status_code == 200
    while tick(tmp_path):
        pass
    with st.db(tmp_path) as db:
        rows = db.execute('SELECT payload, status, error FROM jobs WHERE run=?', (rid,)).fetchall()
    status = {json.loads(p).get('kind') + str(json.loads(p).get('gates', '')): (s, e[-200:]) for p, s, e in rows}
    refused = [v for k, v in status.items() if "'2'" in k]
    assert refused and refused[0][0] == 'failed' and 'hidden' in refused[0][1]
    assert all(s == 'completed' for k, (s, e) in status.items() if "'2'" not in k), status
    data = client.get('/api/job/' + jobs[0].json['id']).json['data']
    assert data['vector_residual'] < 1e-10 and data['structural_geometry']['responses']['status'] in ('available', 'no_qualified_window')
    case = client.get('/api/job/' + jobs[1].json['id']).json['data']
    assert [m['layer'] for m in case['case']['layer_marginals']] == [0, 1]  # hidden layers only
