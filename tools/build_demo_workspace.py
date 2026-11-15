"""Build a presentation workspace from real computations.

Every value in the resulting workspace is produced by the studio's own training,
analysis and healthcare jobs. Nothing is fabricated or interpolated. The workspace
is intended for demonstrations and for exercising the statistics views:

* a matched cohort of synthetic MLP runs (optimizer x seed, identical otherwise),
  with scheduled geometry measurements;
* one fully convolutional CNN (1x1 convolution + global average readout);
* a narrow healthcare demonstration run with cohort evaluations at an early,
  an intermediate and the final checkpoint, and one saved case;
* optionally, a PneumoniaMNIST CNN run when that dataset is already prepared.

Usage (from the project directory, with the studio's environment):

    python tools/build_demo_workspace.py --out demo_workspace --scale demo
    PATHFLOW_HOME=demo_workspace python run.py

Scales (wall time measured on one CPU core of a Linux container; laptops vary):
    quick  3 optimizers x 2 seeds,  60 updates  about 1.5 minutes
    demo   3 optimizers x 3 seeds, 150 updates  about 3 minutes
    full   3 optimizers x 5 seeds, 300 updates  about 10 minutes (estimated from demo)
Each scale also trains the narrow healthcare demonstration run (60 updates) and
records three cohort evaluations and one saved case on it.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab import create_app, datasets, storage as st  # noqa: E402
from lab.config import validate  # noqa: E402
from lab.worker import tick  # noqa: E402

SCALES = {
    'quick': dict(seeds=[11, 12], updates=60, checkpoint_every=10, geometry_every=30, train_size=256, cohort=24),
    'demo': dict(seeds=[11, 12, 13], updates=150, checkpoint_every=10, geometry_every=50, train_size=512, cohort=48),
    'full': dict(seeds=[11, 12, 13, 14, 15], updates=300, checkpoint_every=10, geometry_every=50, train_size=1024, cohort=96),
}
OPTIMIZERS = ['Adam', 'RMSprop', 'SGD']


def drain(root: Path, label: str) -> None:
    started = time.time()
    while tick(root):
        pass
    print(f'  {label}: {time.time() - started:.1f} s', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', default='demo_workspace')
    parser.add_argument('--scale', choices=sorted(SCALES), default='demo')
    parser.add_argument('--pneumonia', action='store_true', help='Also train a PneumoniaMNIST CNN if the dataset is prepared')
    args = parser.parse_args()
    root = Path(args.out).resolve()
    scale = SCALES[args.scale]
    app = create_app(root)
    print(f'Building {args.scale} workspace in {root}')
    (root / 'presentation.json').unlink(missing_ok=True)

    runs = []
    for optimizer in OPTIMIZERS:
        for seed in scale['seeds']:
            config = validate(dict(
                name=f'Synthetic MLP · {optimizer} · seed {seed}', dataset='synthetic', model='mlp',
                widths=[16, 16, 16, 16], activation='relu', optimizer=optimizer,
                lr=0.01 if optimizer == 'SGD' else 0.003, momentum=0.9 if optimizer == 'SGD' else 0.0,
                seed=seed, train_size=scale['train_size'], updates=scale['updates'],
                checkpoint_every=scale['checkpoint_every'], geometry_every=scale['geometry_every']))
            datasets.prepare(root, config)
            rid = st.create(root, config)
            with st.db(root) as db:
                db.execute("UPDATE runs SET status='training' WHERE id=?", (rid,))
            drain(root, config['name'])
            runs.append(rid)

    # One fully convolutional CNN (1x1 convolution + global spatial averaging readout),
    # the CNN form used in the research programme's panels.
    config = validate(dict(name='Synthetic CNN · fully convolutional readout', dataset='synthetic', model='cnn',
                           channels=[4, 6, 8], kernel=3, stride=2, padding=1, cnn_readout='gap', activation='relu',
                           optimizer='Adam', lr=0.003, seed=11, train_size=scale['train_size'], updates=scale['updates'],
                           checkpoint_every=scale['checkpoint_every'], geometry_every=scale['geometry_every']))
    rid = st.create(root, config)
    with st.db(root) as db:
        db.execute("UPDATE runs SET status='training' WHERE id=?", (rid,))
    drain(root, config['name'])
    runs.append(rid)

    if args.pneumonia:
        config = validate(dict(name='PneumoniaMNIST CNN · demonstration', dataset='pneumonia', model='cnn',
                               channels=[4, 6, 8], cnn_readout='gap', activation='relu', optimizer='Adam', lr=0.001,
                               train_size=2048, updates=scale['updates'] * 2, checkpoint_every=20,
                               geometry_every=100, seed=7))
        if datasets.status(root, config)['state'] == 'ready':
            rid = st.create(root, config)
            with st.db(root) as db:
                db.execute("UPDATE runs SET status='training' WHERE id=?", (rid,))
            drain(root, config['name'])
            runs.append(rid)
        else:
            print('  PneumoniaMNIST is not prepared here; skipped. Prepare it in Design, then rerun.')

    # Healthcare demonstration: a narrow SGD network that learns gradually, so that
    # early checkpoints still misclassify images and error-versus-correct evidence is
    # defined. Cohorts are recorded at an early, an intermediate and the final checkpoint.
    config = validate(dict(name='Synthetic MLP · healthcare demonstration', dataset='synthetic', model='mlp',
                           widths=[8, 8], activation='relu', optimizer='SGD', lr=0.02, seed=3, train_size=256,
                           updates=60, checkpoint_every=4, geometry_every=20))
    datasets.prepare(root, config)
    target = st.create(root, config)
    with st.db(root) as db:
        db.execute("UPDATE runs SET status='training' WHERE id=?", (target,))
    drain(root, config['name'])
    runs.append(target)
    client = app.test_client()
    client.get('/')
    with client.session_transaction() as session:
        token = session['csrf']
    for step, policy in [(8, 'zero'), (12, 'zero'), (12, 'cross'), (60, 'zero')]:
        response = client.post(f'/api/run/{target}/healthcare', headers={'X-CSRF-Token': token},
                               json=dict(kind='healthcare_cohort', step=step, split='test', threshold=0.5,
                                         cohort_size=min(128, scale['cohort'] * 2), cohort_reference=policy,
                                         transform=dict(dx=0, dy=0, angle=0, scale=1)))
        assert response.status_code == 200, response.text
    # One saved case at the intermediate checkpoint, so the five-stage healthcare
    # workflow and its presenter view open on a measured result.
    case = dict(kind='healthcare_case', step=12, x=0, reference=1, reference_type='dataset', split='test',
                threshold=0.5, transform=dict(dx=0, dy=0, angle=0, scale=1))
    response = client.post(f'/api/run/{target}/healthcare', headers={'X-CSRF-Token': token}, json=case)
    assert response.status_code == 200, response.text
    drain(root, 'Healthcare cohorts and case')

    with st.db(root) as db:
        failed = db.execute("SELECT count(*) FROM jobs WHERE status='failed'").fetchone()[0]
        completed = db.execute("SELECT count(*) FROM jobs WHERE status='completed'").fetchone()[0]
    print(json.dumps(dict(workspace=str(root), runs=len(runs), completed_jobs=completed, failed_jobs=failed), indent=2))


if __name__ == '__main__':
    main()
