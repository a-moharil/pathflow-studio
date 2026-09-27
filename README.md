# Update 0.10.0

Use Design experiment for MLP, CNN and ResNet. Architecture-specific controls appear inline. See docs/UNIFIED_RESNET_13.md for supported tasks, shared geometry and explicit scientific limits.

# Update 0.9.2

Persistent BUS-UCLM folder and live computation reporting are described in docs/PROGRESS_12.md. Existing datasets and experiments are preserved.

# Pathflow Studio 0.9.1

This update fixes dataset/architecture discovery and adds explicit ResNet presets. See `docs/SELECTION_FIX_11.md`. Select BUS-UCLM or ResNet-50 from **Design experiment**, or open **Joint healthcare** directly. Datasets remain visible before download.

# Pathflow Studio 0.9.0

New: the **Joint healthcare** page adds ResNet-50 segmentation + classification and an exact finite-contrast tensor DAG. See [docs/JOINT_TASK_10.md](docs/JOINT_TASK_10.md) for conventions, dataset import and scope limits, and [docs/VALIDATION_10.md](docs/VALIDATION_10.md) for executed validation. Existing MLP/CNN workflows remain available.

# Pathflow Studio 0.8.0

Visual evidence now has visual channel/connection selection, Grad-CAM, signed path and region diagnostics, and contrast/cancellation demonstrations. Start at Healthcare investigation → Visual evidence. Read `docs/VISUAL_INVESTIGATION_09.md` for the complete scientific and interaction guide. Existing training, geometry, healthcare and Evidence 08 workflows are retained.

# Visual evidence update 0.7.0

Healthcare investigation now includes Visual evidence. See `docs/VISUAL_EVIDENCE_08.md` for definitions, controls, comparison protocols and limitations.

# Fitting update 0.6.1

See `docs/FITTING_07.md` for the new iteration budgets and dimension diagnostics.

# Pathflow Studio 0.4.0

A fresh local Flask project for training MLPs and CNNs and inspecting exact path measures, their intrinsic metrics, and low-dimensional projections. This directory is independent of the earlier Pathflow Lab prototype. It does not migrate previous databases.

## Start on Ubuntu / WSL

Use Python 3.10, 3.11 or 3.12. The complete validation was run on Python 3.12, Linux CPU. From this extracted directory:

```bash
python3 -m venv .venv
.venv/bin/python bootstrap.py
.venv/bin/python run.py
```

Open **http://127.0.0.1:5000**. Keep the terminal running. Ctrl+C stops the application and worker. If Ubuntu reports that `venv` is unavailable, install its `python3-venv` package first.

The bootstrap installs pinned CPU PyTorch 2.6.0, torchvision 0.21.0 and the dependencies in requirements.txt. Internet is needed for installation and the first real-dataset download. Plotly is bundled locally, so the interface has no CDN requirement.

## PyCharm

Open this directory as a project. Select `.venv/bin/python` as its interpreter after the setup above. Run `run.py`, or select the included **Pathflow Studio** run configuration. The launcher starts both the web server and the serialized background worker. Do not launch a second worker yourself.

On native Windows, create the environment with `py -3.12 -m venv .venv`, then run `.venv\Scripts\python.exe bootstrap.py` and `.venv\Scripts\python.exe run.py`. Native Windows and CUDA were not validated in this build. The default installation is CPU-only; selecting CUDA requires a separately installed compatible CUDA PyTorch build and hardware.

## First experiment

1. Open **Design experiment**. Choose Synthetic bars, Fashion-MNIST or PneumoniaMNIST. Click **Download / prepare dataset**. Run creation stays disabled until the selected split is verified.
2. Define the architecture, activation, optimizer and schedule. The resolved PyTorch configuration shows exactly which optimizer parameters will be used. One update means one minibatch optimizer step, not an epoch.
3. Create the experiment and click **Start / resume**. Full selected training and validation splits are evaluated at every committed checkpoint. Test evaluation follows its separate checkpoint interval, and always occurs at the initial and final checkpoints.
4. Click **Pause & inspect**. The interface confirms pause only after the worker has saved the checkpoint and measurements. It then computes geometry for that checkpoint. A pause requested during evaluation waits for that evaluation to complete.
5. Change the test pair or scalar logit readout, analyze a checkpoint, and apply gates from either geometry panel. Interventions are frozen-checkpoint probes; they do not modify native training. Resume continues the native optimizer state.
6. Open the **Observatory** for trajectories, temporal histograms and fixed-checkpoint contrast ensembles. Use logarithmic axes for small positive concentration measures. Plot controls export SVG; numerical histories export CSV or JSON.

## What is preserved

- Fixed execution topology, including separate spatial occurrences of shared convolution weights.
- Exact finite-contrast secants, log-domain path normalization and exact edge marginals without enumerating every complete path.
- Signed output reconstruction and zero-measure handling.
- Structural and conditional metrics with an explicit regularization convention.
- Immutable committed checkpoints, optimizer state and minibatch RNG state.
- Independent panel zoom, node size, ghosts, readout anchor, interventions, inset visibility and expanded plots.

## Reading the geometry

The 2-D and 3-D views are projections of the same computed metric, not force layouts. Low-dimensional distortion can remain large. The audit reports projection stress and, for landmark mode, held-out distance stress and layer coverage. BCANw measurements have an independent full-distance node budget (default 2,500; maximum 4,000). They can run while the display uses a landmark projection. If the measurement budget is exceeded, the result reports that reason. Unqualified scaling windows remain explicitly unqualified.

A common readout anchor fixes translation. Orthogonal alignment handles frame orientation without fitting away scale changes. The camera uses one uniform scale and true orthographic rotation. Independent camera fitting can give different magnifications, which are shown in pixels per metric unit. **Match camera scales** supports scale comparisons. **Focus central 95%** is only a camera magnification; outliers may leave the viewport, and the visible node count is shown. **Fit all nodes** restores the full extent.

The default analysis path space includes all output endpoints. Selected-readout mode retains the earlier scalar-logit/margin convention. Both geometries always share the chosen path space. Thin carriers retain zero-mass edges. Flow width is proportional to the square root of normalized edge mass, with a minimum display hairline and an explicit prominence multiplier. Conditional colors show enrichment relative to structural mass. Pulse speed is decorative direction, not training rate, a derivative, or physical transport. Small normalized occupation is distinct from small absolute contrast response: consult Δf, log QΔ and cancellation in the export.

## Limits and costs

- Sequential affine/conv networks with ReLU or tanh only. No pooling, residual connections, normalization, dropout, pretrained models or data augmentation.
- Maximum 30,000 execution nodes and 2 million edges. Full MDS projection is limited to 1,200 nodes; independent full-distance BCANw measurement supports up to 4,000 nodes under an explicit budget. Dense metric analysis can still be expensive. Every execution edge is now retained in rendering, including zero-mass carriers. Static carrier frames are cached; at most 3,000 moving markers are sampled for performance. Large graphs may take time to redraw while changing the camera.
- Full validation/test evaluation and geometry run in a single worker and can temporarily delay training updates. The UI displays committed snapshots, not invented between-checkpoint measurements. **Follow new checkpoints** queues geometry snapshots as training proceeds; some intermediate checkpoints may not be rendered.
- Same-class contrasts need not be weak, and different-class contrasts need not be strong. Class relation is a selection label, not a forced flow rule.
- Temporal histograms are empirical distributions over recorded checkpoints, weighted equally. They are not distributions over random seeds or an inferred population.
- Local, single-user research application bound to 127.0.0.1. It is not a public multi-user service or a medical diagnostic application.

## Data and files

`workspace/` is created on first launch. It contains the SQLite database, dataset caches, checkpoints, analysis JSON and geometry frames. Do not delete it if you want to retain experiments. Override it with `PATHFLOW_HOME`; use `PORT` to change the listening port. Configuration paths are independent of the terminal working directory.

Fashion-MNIST uses 6,000 seeded validation examples from its official training set, a selected training subset from the remaining 54,000, and its official 10,000-example test set. PneumoniaMNIST uses official train/validation/test splits. Both use 28×28 grayscale pixels scaled to [0,1]. Selected source indices are retained. Real datasets are downloaded by torchvision/medmnist, not included in this archive. Synthetic bars require no download.

## Verification

```bash
.venv/bin/python -m pytest -q
```

Optional browser smoke test:

```bash
.venv/bin/python -m pip install playwright==1.51.0
.venv/bin/python -m playwright install chromium
.venv/bin/python tests/browser_smoke.py
```

See `docs/METHODS.md`, `docs/ARCHITECTURE.md` and `docs/VALIDATION.md`. The optional browser test uses localhost port 5089 and temporary data; stop any service already using that port.

## Updating an existing installation

Stop the server. Replace the project source files with this version, preserving your existing `workspace/` folder and virtual environment. Run `python run.py` with the same interpreter used for installation, then reload the browser. Previously completed analysis files retain their previous scope; click **Analyze checkpoint** to produce a new all-output result. Existing automatic observatory probes remain explicitly tied to selected margin f0−f1 and are not mixed with all-output analyses.

Each geometry panel reports retained edges, route bounds intersecting the viewport, zero-mass carriers and animated-marker counts. **Fit new geometry** is enabled by default. Turn it off to keep a fixed camera while watching deformation across checkpoints. It controls magnification only, never stored coordinates.

## Version 0.4 workflow

The registry is now a searchable experiment table. The application uses neutral light surfaces, scientific plot styling and dedicated dark geometry viewports. Training, Geometry and Configuration have separate tabs. The Observatory has a measurement-context selector and explicit per-checkpoint status table.

New experiments created through the form default to geometry measurement every 50 updates. The first committed checkpoint at or after each interval is scheduled, plus initialization and the final checkpoint. Setting the interval to 0 makes it manual. Existing experiment configurations remain manual unless they already contain a schedule. Geometry jobs use the fixed all-output contrast test[0] versus test[1]; custom pairs are measured through the Observatory or geometry panel. They share the serial worker with training and can delay subsequent updates.

Open the Observatory from a computed geometry to carry its scope, pair and gate settings. Select an observable, then **Compute missing measurements** for the chosen update window. Matching queued/running/completed requests are reused. Failed jobs can be resubmitted. A higher budget creates a new measurement request. **Scale profile** displays box counts and local slopes at the selected checkpoint, alongside fit criteria and qualification markers.

Deletion is available in the registry and experiment page. The confirmation lists run file count and size. Queued jobs are cancelled; in-flight work completes its current operation before the single worker removes the run files and its database records. Shared datasets remain. Failed cleanup is visible and can be retried. Deletion is permanent.

Bootstrap now attempts Python's ensurepip if the selected virtual environment lacks pip. On Ubuntu an incomplete Python installation may still require its matching python3-venv package. Use the same interpreter for bootstrap and run.py.


## Version 0.5.0 representation update

See [docs/REPRESENTATION_03.md](docs/REPRESENTATION_03.md) for Lorentz H3, zero-reference and affine field investigations, state/provenance, limits, and playback. Existing setup and working interpreter are retained.


## Version 0.5.1 metric fidelity correction

Read [docs/METRIC_FIDELITY_04.md](docs/METRIC_FIDELITY_04.md) for independent Euclidean fits, fully constrained Lorentz fitting, comparable diagnostics, intrinsic matrix inspection and automatic checkpoint selection. Previous embeddings are recomputed; saved training checkpoints remain usable.

## Version 0.5.2 integrity augmentation

See [docs/METRIC_INTEGRITY_05.md](docs/METRIC_INTEGRITY_05.md). This release fixes
the near-coincident Lorentz gradient, rejects numerically invalid hyperbolic
coordinates, preserves Euclidean results on hyperbolic failure, evaluates
neighborhoods across all sampled layers, and labels spectral-preview diagnostics
separately. Analysis cache version 6 requires reanalysis, not retraining.

## Version 0.6.0 healthcare investigation

Existing PneumoniaMNIST and synthetic binary experiments now link to a Healthcare
Investigation page for case/reference analysis, exact signed input contributions,
complete-path divergence, hidden-layer concentration, intervention controls,
transformation playback and cohort predictive/intrinsic evidence. Supports both
MLP and CNN saved checkpoints, with JSON/CSV/PNG export and persistent analysis
history. See [docs/HEALTHCARE_06.md](docs/HEALTHCARE_06.md) for definitions and limits.
