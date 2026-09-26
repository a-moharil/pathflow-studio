# Metric integrity augmentation 05 · 0.5.2

This release augments Metric Fix 04 following its pre-integration audit. It is
cumulative from Representation 03 (0.5.0) and also provides an incremental patch
from Metric Fix 04 (0.5.1). There are no new runtime dependencies.

## Numerical Lorentz fitting

For spatial coordinates a,b on the upper hyperboloid with curvature -k, time
coordinates are t(a)=sqrt(1/k+||a||²). The implementation evaluates

    dt = ((a-b) · (a+b)) / (t(a)+t(b))
    q = ||a-b||² - dt²
    distance = 2 asinh(sqrt(k q)/2) / sqrt(k)

This is the Minkowski-chord expression for the same Lorentz distance, with a
rationalized time difference. It avoids cancellation of an acosh argument near
one. The optimization objective uses max(q,1e-24); its analytic derivative is
zero in that floor region. Diagnostic distances have exact zero self-distance.
The previous mismatch between a clipped objective and an unclipped gradient is
removed. The floor is recorded here as a numerical convention, not a change to
the graph metric. Exactly coincident initial points remain a nondifferentiable
case; no rendering jitter or repulsion is introduced. The existing deterministic
optimizer restart remains available for poor fitting objectives.

Numerical validity is checked independently of approximation quality. Before
rendering, the fit must have finite upper-sheet coordinates, dimensionless
Lorentz-constraint residual <=1e-6, ball radii <1-1e-10, and alignment distance
drift <=1e-6 relative to max(1,d). Checks run before and after readout-origin
alignment, and after temporal rotation. The boundary margin is a numerical
acceptance limit, not a proof of a maximum intrinsic radius.

A failed check returns `numerical_invalid` with its reason and no renderable
hyperbolic coordinates. The interface uses the Euclidean view and retains the
intrinsic analysis. A budget-exceeded result also preserves the Euclidean view
without crashing the fidelity panel. If any requested hyperbolic dimension fails,
the hyperbolic result is conservatively rejected as a whole for that analysis.

## Diagnostics and provenance

Neighborhood overlap now evaluates every sampled source, including samples from
every execution layer. The audit exports source IDs and count. This is still a
sampled-source statistic, not an all-node or uniformly node-weighted estimate.

The general audit explicitly labels the old spectral-preview diagnostics. The
fidelity table and displayed-embedding diagnostics refer to the refined 2-D/3-D
coordinates. BCANw responses continue to use intrinsic graph distances, not the
rendered coordinates. Numerical validity does not imply low distance distortion.

Pair selection retains the version-04 deterministic coverage construction. Each
node participates in fitting; pair identities are shared across representations.
Distance-quartile errors are diagnostic, not a guarantee of stratified distance
sampling. Diagnostic pairs can inform spectral initialization, so their errors
must not be described as strictly unseen statistical validation.

## Integrity and migration

The exact path-measure, finite-secant, CNN-unrolling and BCANw implementation in
`lab/core/pathmath.py` is byte-identical to version 0.5.1. Training, optimizers,
dataset preparation, interventions, deletion and recording are preserved. The
release includes checks of complete edge rendering, labels, camera state,
anchors, observatory scope, and independent path-field enumeration.

Saved checkpoints, datasets and experiment workspaces are not modified by the
installer. Analysis cache version is now 6, and fit/frame version is metric-fit-3.
Reanalyzing a saved checkpoint computes corrected diagnostics without retraining.
Old analysis records remain available for historical inspection. New temporal
references start with the first analysis of a context under this version and
identify the reference checkpoint explicitly.

No claim of isometry, global solver convergence, or elimination of all overlap
is made. Inspect stress, local-neighbor preservation and deformation diagnostics
before interpreting visual contraction. Primitive-term Field B remains pending;
this release does not relabel a different quantity as that field.

See VALIDATION_05.md for checks executed for this package, and the bundle-level
INTEGRATION.md for the Git workflow. Source integrity hashes are included in
source-integrity.json at the bundle root.
