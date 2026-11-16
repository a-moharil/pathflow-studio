"""Synthetic field datasets (0.16.2): determinism, readout sizes, ambiguity, cache names,
and an end-to-end training and analysis run."""
import numpy as np
import pytest

from lab import create_app, datasets, storage as st
from lab.config import class_count, validate
from lab.core.network import Network
from lab.fields_data import FIELD_DATASETS, generate
from lab.worker import tick


@pytest.mark.parametrize('name,count', [('synthetic-interference', 3), ('synthetic-orbits', 4)])
def test_generation_is_deterministic_balanced_and_bounded(name, count):
    a, b = generate(name, 5, 64, .35), generate(name, 5, 64, .35)
    for split in ('train', 'val', 'test'):
        assert np.array_equal(a[split + '_x'], b[split + '_x'])
        assert a[split + '_x'].shape[1:] == (1, 28, 28) and a[split + '_x'].dtype == np.float32
        assert 0 <= a[split + '_x'].min() and a[split + '_x'].max() <= 1
        assert set(np.bincount(a[split + '_y'])) <= {len(a[split + '_y']) // count, len(a[split + '_y']) // count + 1}
    assert not np.array_equal(a['train_x'], generate(name, 5, 64, .6)['train_x'])
    assert len(FIELD_DATASETS[name]['classes']) == count


def test_config_readout_and_cache_names(tmp_path):
    for name, count in [('synthetic-interference', 3), ('synthetic-orbits', 4), ('synthetic', 2), ('fashion', 10)]:
        c = validate(dict(dataset=name))
        assert class_count(c) == count and Network(c).shapes[-1] == (count,)
        assert Network(validate(dict(dataset=name, model='cnn', cnn_readout='gap'))).shapes[-1][0] == count
    c = validate(dict(dataset='synthetic-orbits', ambiguity='0.5'))
    assert c['ambiguity'] == .5 and c['field_version'] == 2
    assert datasets.cache_path(tmp_path, c).name == 'synthetic-orbits_42_512_a50_v2.npz'
    # A run recorded before 0.16.3 (no field_version) keeps its cache name and rendering.
    legacy = {k: v for k, v in c.items() if k != 'field_version'}
    assert datasets.cache_path(tmp_path, legacy).name == 'synthetic-orbits_42_512_a50.npz'
    assert 'ambiguity' not in validate(dict(dataset='synthetic', ambiguity=.5))
    # Existing cache names are unchanged.
    assert datasets.cache_path(tmp_path, validate(dict(dataset='synthetic'))).name == 'synthetic_42_512.npz'
    with pytest.raises(ValueError, match='Ambiguity'):
        validate(dict(dataset='synthetic-interference', ambiguity=.95))


def test_train_and_analyse_field_run(tmp_path):
    app = create_app(tmp_path)
    c = validate(dict(dataset='synthetic-orbits', widths=[6, 6], train_size=64, updates=4, checkpoint_every=2,
                      landmarks=16, geometry_every=0, ambiguity=.3))
    datasets.prepare(tmp_path, c)
    rid = st.create(tmp_path, c)
    with st.db(tmp_path) as db:
        db.execute("UPDATE runs SET status='training' WHERE id=?", (rid,))
    while tick(tmp_path):
        pass
    client = app.test_client()
    info = client.get(f'/api/run/{rid}/dataset').json
    assert info['class_names'] == ['Rings', 'Spokes', 'Right spiral', 'Left spiral']
    jid, state = st.admit(tmp_path, rid, dict(kind='analysis', step=4, x=0, reference=1, readout=dict(scope='all', a=0, b=1)))
    while tick(tmp_path):
        pass
    job = client.get(f'/api/job/{jid}').json
    assert job['status'] == 'completed' and len(job['data']['logits']) == 4


def test_rendering_versions():
    """Version 2 renders a bright positive phase on a dark ground; version 1 (0.16.2) is
    reproduced exactly for runs created with it."""
    v1, v2 = generate('synthetic-orbits', 5, 64, .35, 1), generate('synthetic-orbits', 5, 64, .35)
    assert abs(v1['train_x'].mean() - .5) < .05 and v2['train_x'].mean() < .25
    assert np.array_equal(np.bincount(v1['train_y']), np.bincount(v2['train_y']))
    assert np.array_equal(v1['train_x'], generate('synthetic-orbits', 5, 64, .35, 1)['train_x'])
