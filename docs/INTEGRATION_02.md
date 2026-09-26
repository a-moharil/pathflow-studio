# Integration 02 (0.4.2)

This release includes Update 01 and the measurement/deletion/interface checklist.

## Checklist and implementation

- All-output and selected-readout histories: explicit contexts, arbitrary contrast
  pair and saved activation gates, preserved when opening Observatory from Geometry.
- Scheduled geometry: independent of checkpoint cadence; existing experiments can
  now change schedule and BCANw budget in Configuration while paused. Initial and
  final checkpoints are included; intermediate scheduled work uses committed steps.
- Backfill: Compute missing measurements queues/reuses jobs for saved checkpoints
  without retraining. Revisioned analysis requests prevent reuse of pre-update
  cached results when a fresh analysis is requested.
- Independent projection and measurement budgets: landmark projection can coexist
  with full-distance BCANw. The default CNN has a full-distance measurement test.
- Missingness: Not computed, Queued, Computing, Available, Budget exceeded,
  Undefined conditional measure, No qualified scaling window, Failed with error.
  Undefined entropies no longer appear as uncomputed requests.
- Provenance: checkpoint, scope, readout, pair, gates, metric, analysis revision,
  measurement budget and estimator version are recorded. Observatory lets users
  select recorded budgets. Explicit combined-budget mode remains available.
- Estimator inspection: N_B(radius), local slope, R-squared, qualification, window
  endpoints, window sample count and box fractions. d_eff is the qualified slope
  nearest xi50, dstar is the maximum qualified slope. These are conventions, not
  assertions of a universal fractal exponent.
- Cost: matrix storage, four-matrix planning allowance plus unestimated graph/solver
  overhead, shortest-path source count and 36 radii displayed before backfill.
  No uncalibrated wall-time estimate is claimed.
- Deletion: registry, experiment toolbar and Configuration actions. Preview includes
  files, bytes, measurement/analysis/job counts. Active work reaches its safe
  boundary; queued jobs cancel; serial worker removes files and database records.
  Shared dataset caches remain. Failed cleanup is visible and retryable.
- Interface: white/neutral application surfaces, blue actions, white vector-export
  plots and dark geometry viewports; searchable registry includes measurement counts.
  Version 0.4.2 is visible in the header; changed asset versions invalidate caches.
- Update 01 retained: MLP/CNN input fields, pixel inspection, undefined-field
  handling, camera preservation across tabs, anchor release and existing controls.

No new application dependencies or database migration. Existing recorded analyses
are retained; recompute/backfill to obtain the new window metadata and fields.
View settings persist within the open page, not across browser reloads.
Field B/primitive interventions and transformation sweeps remain later milestones.

## Validation

27 pytest tests include optimizer resume, path reconstruction, field marginals,
existing-run schedule updates, all-output API results, active deletion and deletion
arriving during commit, full profiles under landmark projection, and a default-CNN
scheduled analysis reaching the observation store. JavaScript tests verify contexts,
budget filtering, undefined status, camera retention and complete carrier rendering.

Both Chromium browser tests passed with no JavaScript errors. The integration
test changes an existing-run schedule, backfills all-output measurements, then
deletes a training run and confirms shared dataset preservation. The existing
smoke test checks fields, cameras, activation gates, insets and responsive layout.
