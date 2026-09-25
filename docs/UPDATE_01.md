# Update 01: view consistency and input-origin fields

Baseline: 31b7587765f83e46ccedbd3dd4954f44f53b39c9.

Delivered: Field A for image MLPs and CNNs, structural input marginal for comparison,
shared or fixed color scale, pixel inspection, exact fields in JSON exports,
and camera preservation across tab navigation. Hidden canvas resizes retain the
last valid dimensions. Fit new geometry is now opt-in; the first analysis fits.
Tab navigation retains existing in-memory control state. Reload persistence is
not part of this update.

The input marginal sums first-cut edge marginals by source input index and then
sums channels at each pixel. It is not activation magnitude, a gradient, or a
segmentation. The same contrast, checkpoint, gates and readout used for geometry
produce the fields. Conditional zero partition gives null, not a uniform field.
The color maximum is labelled and may change under the shared-maximum option;
use fixed 0–1 for a stable scale across analyses. Arrays are exported without
rounding or additional normalization. Cached older analysis files remain valid;
recompute a checkpoint to obtain the new fields.

Existing training, optimizers, data preparation, graph topology, metric,
activation gates, anchor release, ghosts, recording and observatory remain.
No dependencies or database migrations are introduced. Histories omit the large
pixel arrays; full analysis exports retain them.

## Remaining agreed milestones

- Persistent investigation records and reload restoration.
- Selectable shared anchor identities and independently linked cameras.
- Primitive convolution/dense-term interventions and Field B.
- Transformation sweeps, probability transport, playback and residuals.
- Saved sweep observatory integration and matched architecture comparisons.

## Verification

- 24 pytest tests, including MLP/CNN ReLU/tanh fields, contrast reversal,
  undefined measures, structural invariance under activation gates, and existing
  training/resume, checkpoint, analysis, scheduling and deletion tests.
- 100 independent complete-path enumeration checks with channel reduction.
- Node checks for camera retention, hidden resize, explicit fitting and anchor.
- Existing carrier test: 79,490 edges including 39,745 zero-mass carriers.
- Existing observatory context isolation test.

## Local acceptance

Analyze a checkpoint. Open Fields and inspect pixel probabilities. Return to
Geometry, pan and zoom both views, release one anchor, adjust node size and ghosts.
Visit Training and Fields, then return to Geometry: the camera and controls must
remain unchanged. Change contrast with Fit new geometry unchecked: magnification
must remain unchanged. Fit all nodes must still reframe explicitly. For identical
input and reference indices, Field A must report undefined. Export analysis must
contain input_fields. Existing recording remains available in Geometry.

Browser verification was interrupted by an execution-environment reset. The updated
browser smoke test is included but a completed browser pass is not claimed.
The 24-test pytest pass completed before the reset.
