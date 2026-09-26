# Validation · Healthcare release 06 / 0.6.0

Checks executed locally on 26 September 2026:

- Full Python suite: **50 passed**. Includes the 43 existing integrity/training/
  geometry/management checks plus seven healthcare tests (including parametrized
  MLP/CNN and ReLU/tanh cases).
- Independent small-MLP complete-path enumeration matches the new full-path
  Hellinger coefficient and signed input-origin contributions to 1e-12 tolerance.
- Signed fields reconstruct output contrasts under native and attenuated gates
  for MLP and CNN, ReLU and tanh. Undefined zero-contrast cases remain undefined.
- Binary metrics checked against known confusion, rank-AUC, average precision
  and Brier values; ties and single-class undefined metrics covered.
- Worker/API integration exercises case, internal dose, region dose and cohort
  jobs. Saved checkpoint bytes remain identical. CSRF enforcement retained.
- Healthcare Chromium workflow passed with no page errors: navigation from the
  experiment, exact fields, layer plots, internal and region sweeps, transformed
  fields, measured transform playback, cohort plots, JSON/CSV/PNG downloads,
  stale-request protection and persistent history reload.
- Existing Euclidean/H3 browser workflow passed with no page errors, including
  checkpoint changes, anchors, labels, orbit, matrix inspection, transformation
  playback, recording and stale-request protection.
- Existing JavaScript checks passed for camera/anchor state, independent 2-D
  coordinates, all 79,490 execution edges (including 39,745 zero-mass channels),
  all-node labels, unavailable-fit fallback, and Observatory scope handling.
- 100 independently enumerated path-field checks passed.
- Healthcare fields and cohort screenshots were visually inspected at 1440 px.

The installer is validated against each bundled supported baseline in isolated
Git repositories, requiring exact equality of installed source with the bundled
project, preservation of a workspace sentinel, expected branch creation and
refusal of dirty or incompatible trees. source-integrity.json records no removed
files and byte-identical scientific, training, embedding and dataset modules
relative to 0.5.2.

Validation scope: synthetic binary data and CPU execution. The browser healthcare
workflow uses a small trained CNN with tanh. This release has not been benchmarked
for classification quality on a downloaded PneumoniaMNIST dataset or on CUDA.
The cohort tests validate calculations, not clinical usefulness. Existing analysis
budgets and limitations, including potentially high low-dimensional embedding
distortion and the pending primitive-term Field B, remain explicitly documented.
