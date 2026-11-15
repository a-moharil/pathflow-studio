# Efficient realisation · Pathflow Studio 0.14.0

Branch: `efficient-realisation`. Baseline: `integrating-ecml-manuscript` (`f77887f`, version 0.13.0).

This release reorganises how the studio is navigated, how data reaches the browser
and how recorded observables are summarised statistically. Numerical kernels,
training, the worker, job semantics, checkpoint contents and the stored analysis
artefacts are unchanged. Everything added reads recorded results; nothing new
queues computation.

## 1. Navigation

### Run workspace

Every run-scoped page now carries one persistent workspace bar: run identity,
engine, dataset, family, optimizer, live status and the views that apply to the run.
Views that do not apply are omitted. The Healthcare tab, for example, appears only
for binary healthcare datasets.

| Tab | MLP / CNN run | Residual-DAG run |
|---|---|---|
| Overview | `/run/<id>` (new) | `/run/<id>` (new) |
| Training | `/experiment/<id>?section=Training` | `/experiment/<id>?section=training` |
| Geometry | `/experiment/<id>?section=Geometry` | `/experiment/<id>?section=geometry` |
| Investigation | — | `/experiment/<id>?section=fieldsPanel` |
| Observatory | `/observatory/<id>` | `/experiment/<id>?section=resnetObservatory` |
| Healthcare | `/healthcare/<id>` (binary datasets) | — |
| Statistics | `/run/<id>/statistics` (new) | `/run/<id>/statistics` (new) |
| ECML auditor | `/auditor?run=<id>` | `/auditor?run=<id>` |

The canonical URLs are unchanged. `/run/<id>/<view>` provides stable aliases
(`training`, `geometry`, `observatory`, `healthcare`, `auditor`) that redirect and
preserve their query string. In-page section tabs on the experiment and residual
pages now read `?section=` and write it back, so a link opens the intended view and
the workspace bar always marks the active one.

The Observatory's measurement context (`?context=`) is carried to the Statistics
tab, where it preselects the matching recorded context.

### Top navigation

Experiments · Design · Healthcare · Statistics, and a Presentation switch. The ECML
auditor is reached from each run's workspace bar and from the Statistics page
(population audit); `/auditor` itself is unchanged. `/compare` now serves the new
cross-run Statistics page.

## 2. Efficiency

Measured on the demonstration workspace (synthetic MLP, 16 checkpoints, 4 analyses):

| Path | Before | After |
|---|---|---|
| Experiment page poll, every 1.5 s | full run detail (≈ 110 KB per checkpoint, ≈ 1.6 MB here) **plus** all observation summaries (≈ 100 KB each) | ≈ 2.8 KB + 25 B per poll once loaded; 8 s interval when idle; paused in hidden tabs |
| Observatory poll, every 4 s | full run detail + all observations | incremental lean feed; 15 s when nothing is queued |
| Experiment registry, every 3 s | one request per engine with per-run count queries | one grouped summary request; 15 s when idle |
| Compute monitor (every page), every 2 s | constant | 2 s while busy, 10 s idle, paused in hidden tabs |
| ECML auditor catalogue | parses and hashes every saved artefact on each open (748 ms for 36 artefacts) | cached per file mtime and size (6–10 ms) |
| Registry and Healthcare hub page weight | Plotly (4.6 MB) and four widget scripts | neither is loaded |

Mechanisms:

- `GET /api/run/<id>?lean=1&since=N` returns only checkpoints after update N, without
  input fields, source summaries, layer fields or source diagnostics. Without these
  parameters the endpoint is unchanged, so exports remain complete.
- `GET /api/run/<id>/observations?lean=1&after=ROWID` returns only newer summaries,
  tagged with `_job` and `_rowid` for merging. Without parameters the response is unchanged.
- `RunFeed` (`lab/static/poller.js`) keeps the client copy and resynchronises in full if a
  checkpoint was replaced or removed.
- `Poller.every(task, {active, idle})` adapts its interval to whether anything is changing
  and pauses while the tab is hidden.
- `base.html` loads assets by tier: `minimal` (registry, healthcare hub), `charts`
  (Plotly views), `full` (geometry and field workbenches; the default).

## 3. Indexed observables

`lab/stats/index.py` adds two tables, `observables` and `observable_sources`, plus
`artifact_index` for the auditor cache. Each record is a scalar with its run, engine,
source, checkpoint, canonical measurement context, measure side and name. Sources:

- every committed checkpoint's fixed probe (`test[0]` vs `test[1]`, readout 0−1) and its training metrics;
- every native analysis and dose summary, including the saved BCANw responses (d*, d_eff, ξ₅₀ and the qualified-window fraction);
- residual-DAG training history and every head of each completed residual analysis.

Synchronisation is lazy and incremental: read endpoints call `sync`, which indexes only
sources whose change marker (payload length or file mtime and size) differs. Rows of
deleted runs are pruned. On the demonstration workspace the first sync of 180 sources
takes 0.23 s, and a repeated sync takes about 1 ms.

## 4. Statistics

`lab/stats/engine.py` attaches an evidential status to every result:

| Status | Meaning |
|---|---|
| identity | exact numerical relation of recorded quantities |
| descriptive | summary of recorded values; no sampling model |
| exploratory | test or interval under a declared design; not preregistered; not corrected across interactive selections |
| insufficient | the design does not support the quantity |
| unavailable | not recorded for this selection |

### Within one run (`/run/<id>/statistics`)

The unit is the committed checkpoints of one fitted trajectory. They are serially
dependent repeated observations, so these summaries are descriptive and carry no
p-values.

- **Disposition → realisation.** Conditional minus structural log η₁ and log η₂,
  matched by checkpoint, with the retained occupancy η_cond/η_struct. The normalised
  entropy deficit is shown as a descriptive companion only, because S_def = −log η₁
  exactly (and the H₁ gap equals the log η₁ gap).
- **Temporal regime.** First → last change, Spearman trend with the update index, and
  turning points. An interior extremum counts as a turning point only when it lies
  beyond both endpoints by at least 10% of the trajectory's range; plateau noise is
  ignored. This identifies, for example, relaxation before selection.
- **Coupling.** Spearman ρ between trajectories over shared checkpoints, with n per
  cell. Each identity class appears once (log η₁ stands for S_def and H₁).
- **Scale geometry.** d*, ξ₅₀, d_eff and the qualified-window fraction from saved
  legacy `bcan-1` profiles. A missing d* is never imputed. The manuscript's seven
  gates and the 64-scale remeasurement remain in the ECML auditor.
- **Response.** Δf, cancellation and ‖Δf‖₂.

### Healthcare cohort (same page)

The unit is images of one split, conditional on one fitted model and checkpoint.

- Outcome groups (TN, FP, FN, TP): Kruskal–Wallis with ε² = (H − k + 1)/(n − k).
- Misclassified vs correctly classified, and class 1 vs class 0: two-sided Mann–Whitney U;
  AUROC = U/(n₁n₂) with a 1000-resample image-level bootstrap interval (seed 701);
  rank-biserial r = 2·AUROC − 1. Each comparison requires at least five defined values per group.
- Holm correction within each family of the current selection only.
- Under the identity transform, H_eq and H_inv are floating-point zeros. They are
  excluded from testing and listed as such.
- When every misclassified image belongs to one true class, the error comparison is
  flagged as confounded with class, with a pointer to the within-class TN/FP or TP/FN view.
- With cross or same-class references, references may repeat. Pair-derived evidence is
  labelled exploratory.
- η₁, H₁ and S_def carry one ordering within an architecture. Their tests are one piece
  of evidence, and the page says so.

### Across runs (`/statistics`)

The unit is independently trained runs; each run contributes one value at the
selected state (initialisation, final checkpoint or a given update) in either the
fixed-probe or the scheduled all-output analysis context.

- Strata share dataset, model family, task and architecture. Grouping by `widths` or
  `channels` spans architectures deliberately.
- Blocks share every configuration value except the factor, the factor's coupled
  hyperparameters (for optimizer: lr, momentum, β, ε, α and related) and the run name.
  This makes seeds the matching variable.
- Inference is disabled until the user declares independent training units. Then the
  design decides the test: Friedman with Kendall W = χ²/(n(k − 1)) for ≥ 3 complete
  blocks and k ≥ 3; a matched-pair Wilcoxon with rank-biserial and Hodges–Lehmann
  shift for ≥ 6 pairs and k = 2; Kruskal–Wallis with ε² for ≥ 3 runs per group;
  otherwise the result is marked insufficient.
- Rank retention: Spearman ρ between run ranks at initialisation and at the selected state (descriptive).
- If final checkpoints differ in update count, a warning states that training lengths are not matched.
- The paired structural-to-conditional population audit remains in the ECML auditor,
  which also covers residual-DAG records.

### Ledger

Every statistics view exports one ledger as CSV, LaTeX (`booktabs`) or JSON:
section, quantity, unit, n, estimate, interval, statistic, p, Holm-adjusted p, effect and status.

## 5. Presentation mode

The header switch writes `presentation.json` in the workspace. While it is on, the
server refuses, with HTTP 409 and an explanation:

- analysis, dose and healthcare submissions;
- measurement backfill and measurement-schedule changes;
- run start;
- experiment, dataset and joint-run creation;
- deletion;
- auditor 64-scale remeasurement.

Pause, stop, cancel, every read, and the read-only statistics endpoints stay available.
The guard is enforced on the server, so it holds on every page. The existing healthcare
presenter follows the switch, and leaving the presenter through its own control also
turns the switch off, because that presenter hides the header.

## 6. Demonstration workspace

`tools/build_demo_workspace.py --out demo_workspace --scale {quick,demo,full}` builds
a workspace from real computations:

- a matched synthetic MLP cohort (3 optimizers × 2, 3 or 5 seeds) with scheduled geometry;
- a narrow SGD healthcare demonstration run with cohort evaluations at updates 8, 12 and 60, and one saved case.

Update 12 still misclassifies images, so error evidence is defined there. `--pneumonia`
also trains a PneumoniaMNIST CNN when that dataset is already prepared. Start the
studio on it with `PATHFLOW_HOME=demo_workspace python run.py`.

## Compatibility

- Existing workspaces need no migration. The new tables are created on start and filled on first read.
- Existing URLs and API responses without the new query parameters are unchanged.
- The 0.13.0 auditor, the healthcare workflow and the joint engine are untouched, apart
  from the auditor's catalogue cache and the section deep links.
