"""0.17.1: pictures come from settled fits, and the 3-D picture contains the 2-D one.

Before 0.17.1 the 2-D and 3-D pictures were independent 120-iteration fits, and a random restart
with a different seed per dimension could replace either. Unfinished random-start fits look like
round clouds ("balls") that the intrinsic distances do not support.
"""
import json
import numpy as np
from lab.core import metricfit as mf
from lab.core.fitsettings import FIT_VERSION
from test_hyperbolic import example


def planar_points(n=60, seed=3):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 3*np.pi, n)
    X = np.column_stack((t*np.cos(t), t*np.sin(t), np.zeros(n))) + rng.normal(size=(n, 3))*.02
    pairs = np.array([(i, j) for i in range(n) for j in range(i+1, n)])
    return X, pairs, mf.euclidean_distances(X, pairs)


def test_plateau_rule_stops_a_settled_fit_before_the_ceiling():
    X, pairs, truth = planar_points()
    D = np.linalg.norm(X[:, None]-X[None, :], axis=-1)
    start, _ = mf.spectral_initial(D, np.arange(len(X)), 2)
    Y, report = mf.fit_euclidean(start+np.random.default_rng(1).normal(size=start.shape)*.3, pairs, truth, 5000)
    assert report['stop_reason'] in ('plateau', 'converged') and report['converged']
    assert report['iterations'] < 5000
    assert report['plateau_rule'] == dict(window=mf.PLATEAU_WINDOW, relative_tolerance=mf.PLATEAU_TOLERANCE, minimum_iterations=mf.PLATEAU_MINIMUM)
    # The history records the solver's own objective values; nothing is re-evaluated.
    assert report['evaluations'] < 3*report['iterations']+50


def test_unsettled_fit_is_labelled_as_such():
    X, pairs, truth = planar_points()
    start = np.random.default_rng(4).normal(size=(len(X), 2))
    _, report = mf.fit_euclidean(start, pairs, truth, 30)
    assert report['stop_reason'] == 'iteration_budget' and not report['converged']


def test_restart_draw_is_nested_across_dimensions():
    two = mf.nested_restart(40, 2, 1.)*np.sqrt(4)
    three = mf.nested_restart(40, 3, 1.)*np.sqrt(6)
    np.testing.assert_allclose(two, three[:, :2])


def test_orientation_preserves_distances_and_puts_the_shadow_on_the_lower_plane():
    rng = np.random.default_rng(8)
    low = rng.normal(size=(30, 2))
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    high = np.column_stack((low, rng.normal(size=30)*.05))@Q + 4
    Y, R = mf.orient_to_lower(high, low)
    pairs = np.array([(i, j) for i in range(30) for j in range(i+1, 30)])
    np.testing.assert_allclose(mf.euclidean_distances(Y, pairs), mf.euclidean_distances(high, pairs), atol=1e-10)
    np.testing.assert_allclose(R.T@R, np.eye(3), atol=1e-10)
    assert mf.shadow_discrepancy(high, low) < 1e-2
    assert mf.shadow_discrepancy(high, low, robust=True) < .1
    assert np.mean(Y[:, 2]**3) >= 0


def test_three_d_is_lifted_from_two_d_and_contains_it(monkeypatch):
    # The 10-node example does not fit well in 3-D (settled objective about 0.065, just above
    # RESTART_THRESHOLD), so a random restart is legitimately tried there, and which attempt
    # wins depends on the platform's floating point. Switch restarts off to test the lift
    # itself; restart selection is tested separately below.
    monkeypatch.setattr(mf, 'RESTART_THRESHOLD', float('inf'))
    data = example()
    mf.refine(data, 400)
    e2, e3 = data['euclidean']['dimensions']['2'], data['euclidean']['dimensions']['3']
    assert e2[0]['optimization']['start'] == 'spectral'
    assert e3[0]['optimization']['start'] == 'lifted_from_2_d'
    assert e2[0]['dimension_consistency'] is None
    c = e3[0]['dimension_consistency']
    assert c['compared_with'] == '2-D' and c['median_shift'] < .25
    assert 'lifted' in data['euclidean']['procedure'] or 'lower picture' in data['euclidean']['procedure']
    assert data['euclidean']['version'] == FIT_VERSION
    # Displayed 3-D positions are the 3-D fit (data['positions'] mirrors it).
    np.testing.assert_allclose(data['positions'][0], e3[0]['positions'])


def test_restart_kept_only_when_its_settled_objective_is_smaller(monkeypatch):
    data = example()
    monkeypatch.setattr(mf, 'RESTART_THRESHOLD', -1.)   # force a restart attempt everywhere
    mf.refine(data, 300)
    for dim in ['2', '3']:
        o = data['euclidean']['dimensions'][dim][0]['optimization']
        assert len(o['attempts']) == 2
        assert o['best_objective'] == min(o['restart_objectives'])
        assert o['attempts'][1]['start'] == 'random_restart_seed_%d' % mf.RESTART_SEED
        assert all(a['max_iterations'] == 300 for a in o['attempts'])


def test_geometry_requests_carry_the_fit_version(tmp_path):
    from lab import create_app, storage as st
    from lab.config import validate
    create_app(tmp_path)
    rid = st.create(tmp_path, validate(dict(name='fit31', dataset='synthetic', updates=2, train_size=64, batch_size=8)))
    with st.db(tmp_path) as c:
        c.execute('INSERT INTO measurements VALUES(?,?,?)', (rid, 0, json.dumps({})))
        # An analysis computed under the previous procedure: same request, no fit_version.
        old = dict(kind='analysis', step=0, x=0, reference=1, analysis_version=6)
        c.execute("INSERT INTO jobs(id,run,payload,status,created) VALUES('old',?,?, 'completed', 0)", (rid, json.dumps(old, sort_keys=True)))
    jid, state = st.admit(tmp_path, rid, dict(kind='analysis', step=0, x=0, reference=1))
    assert state == 'queued' and jid != 'old'
    with st.db(tmp_path) as c:
        payload = json.loads(c.execute('SELECT payload FROM jobs WHERE id=?', (jid,)).fetchone()[0])
    assert payload['fit_version'] == FIT_VERSION
    # Non-geometry requests are keyed as before.
    jid2, _ = st.admit(tmp_path, rid, dict(kind='ruler_sensitivity', step=0))
    with st.db(tmp_path) as c:
        assert 'fit_version' not in json.loads(c.execute('SELECT payload FROM jobs WHERE id=?', (jid2,)).fetchone()[0])


def test_interface_defaults_and_viewer_note():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    html = (root/'lab/templates/experiment.html').read_text()
    assert 'id="embeddingSteps" type="number" min="25" max="5000" value="3000"' in html
    js = (root/'lab/static/geometry.js').read_text()
    assert 'fitNote(d)' in js and 'earlier fitting procedure (before 0.17.1)' in js and 'did not settle' in js


def test_display_status_distinguishes_settled_near_plateau_and_unsettled():
    assert mf.settling(dict(stop_reason='plateau')) == 'inspect_diagnostics'
    assert mf.settling(dict(stop_reason='converged')) == 'inspect_diagnostics'
    assert mf.settling(dict(stop_reason='iteration_budget', trailing_improvement=.005)) == 'near_plateau'
    assert mf.settling(dict(stop_reason='iteration_budget', trailing_improvement=.08)) == 'not_settled'
