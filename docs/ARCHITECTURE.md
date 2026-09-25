# Project architecture

`run.py` creates the Flask application, starts one background worker with the same interpreter, waits for worker recovery, then serves through Waitress. A parent-pipe watcher shuts the worker down when its launcher exits. A file lock prevents duplicate workers sharing one workspace.

## Pages

| Route | Role |
|---|---|
| `/` | List runs and their committed state |
| `/design` | Architecture, optimizer, schedule and explicit dataset preparation |
| `/experiment/<id>` | Training controls and curves, checkpoint inspection and both geometries |
| `/observatory/<id>` | Observable trajectories, time histograms and contrast ensembles |
| `/compare` | Validation loss histories labelled with configuration identities |

## Modules

- `lab/config.py`: validation and execution-graph cost limits.
- `lab/datasets.py`: readiness, deterministic split preparation and source cache reuse.
- `lab/core/network.py`: PyTorch sequential MLP/CNN and activation gates.
- `lab/core/optimizers.py`: explicit optimizer registry and resolved keyword arguments.
- `lab/core/pathmath.py`: execution graph, exact path measures, metric projection and BCANw.
- `lab/engine.py`: update quanta, full-split evaluation and atomic checkpoint publication.
- `lab/analysis_jobs.py`: frozen-checkpoint analyses, ensembles, interventions and stable display frames.
- `lab/worker.py`: serialized dataset, training and analysis work.
- `lab/storage.py`: SQLite WAL records and atomic JSON files.
- `lab/web.py`: validated routes, CSRF checks and browser API.
- `lab/static/geometry.js`: projection and drawing, separate from scientific computation.
- `lab/static/charts.js`: locally bundled Plotly charts with responsive SVG rendering.

## State and recovery

A training quantum advances at most `checkpoint_every` steps. A pause request is checked between minibatches; checkpoint evaluation may take longer. Saving writes weights, optimizer state and RNG state, then publishes the measurement and committed step in one SQLite transaction. Only committed checkpoints can be inspected. A requested pause becomes `paused` only after that publication. `live_step` may exceed committed `step` while a quantum runs.

Restart recovery puts active runs into paused state and requeues interrupted jobs. Incomplete, uncommitted checkpoint files are not offered for analysis. The recovered model and generator come from the last committed checkpoint. Dataset cache creation uses a file lock and atomic replacement. Numerical analyses run on float64 copies and cannot mutate the training optimizer.

Priority is pending run controls, then queued jobs, then training. One serial worker prevents simultaneous GPU/CPU model mutations and makes local resource consumption predictable. Analysis requests can delay training; this is disclosed in the interface and README. Scaling to multiple users or distributed workers requires a separate job scheduler and resource policy.
