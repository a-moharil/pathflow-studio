# Representation and investigation update · 0.5.0

## Scientific state

Built on Integration 02 (0.4.2). The full graph, path algebra, BCANw convention,
training, schedules, deletion, activation gates, Observatory and local controls
remain. Analysis version 4 keys cached work independently of older results.
Every result exports its checkpoint, input, explicit reference type, readout,
affine transform, gates, metric unit/settings and requested Lorentz settings.
View state is separate. Outdated responses cannot replace a newer request.
Both Fields and Geometry receive the same completed result; expensive analysis
is atomic rather than displaying new fields alongside stale geometry.

## Geometry

Euclidean 2-D/3-D remain available. The additional option fits H3 in four Lorentz
coordinates, then projects all three spatial coordinates into a unit Poincare
ball. A sphere silhouette and three rotating reference circles reveal volume.
Shift-drag and yaw/pitch rotate; ordinary drag pans. Camera orbit has its own
speed and pause. Edge and node depth sorting is a painter approximation, not a
GPU depth buffer; crossing translucent edges can remain visually ambiguous.
Hyperbolic geodesics use adaptive subdivision with a 0.75 CSS-pixel midpoint
criterion and depth cap 7 (128 segments). At extreme zoom the cap may limit
accuracy. Screen lengths are never used as intrinsic metric measurements.

Lorentz fitting uses float64, fixed curvature, and the unchanged frozen intrinsic
distance unit. Deterministic seed 1024 samples up to 48 distance-source rows,
16,000 fitting pairs and 8,000 disjoint validation pairs. The normalized squared
stress objective is fitted with Adam, learning rate .025, gradient clipping 10,
and a declared initial-frame radial safety cap 6. Fit budget is 25–500 steps;
training stress > .25 triggers at most min(100, requested steps) additional steps.
The best valid objective is retained. This is not a guarantee of convergence,
conformality, exact isometry, or absence of overlaps. Fits are limited to 4,000
nodes; an unavailable fit leaves the Euclidean option and explains the status.

Conditional fitting starts from structural coordinates. A Lorentz boost places
the readout sink at the origin, followed by a spatial orthogonal alignment.
Subsequent analyses share the first structural frame for the same readout and
curvature using an additional common spatial rotation, without fitted scaling.
Symmetric or degenerate configurations can still leave orientation ambiguity.
Released anchors apply a common fixed Lorentz isometry to the panel and ghost.
Anchor identity remains the readout sink. Arbitrary anchor selection is not added.
No repulsion, jitter or probability-dependent coordinate spreading is used.

Each Lorentz fit exports objective history, termination reason, training stress,
held-out distortion by distance scale, neighbor overlap and constraint residual.
Existing Euclidean coordinates are compared on exactly the same held-out pairs;
these are not independently optimal Euclidean stress fits. Euclidean-only jobs
also evaluate deterministic distance-source rows. The pair inspector reports
intrinsic raw/unit distances, Euclidean/hyperbolic distances, and current screen
pixels. A descriptive high-distortion threshold is .25, not a scientific theorem.

All-node labels draw every retained node identity. Dense text can overlap. CNN
labels include channel and spatial coordinates; MLP labels identify layer/unit.
View settings persist per experiment/panel in browser local storage. Camera pan
is retained while navigating but is not persisted across a browser restart.

## Frozen transformation investigations

Field A is still the normalized complete-path input-origin marginal. The UI
compares baseline, transported baseline, recomputed transformed realization, and
signed residual. A transformation residual is explicitly not Field B.
Both members of a pair are transformed at the same frozen checkpoint. Zero
reference is a real zero tensor, evaluated through the biased/intervened model.
Current preprocessing is [0,1], so zero is black. If preprocessing changes, zero
continues to mean zero in model coordinates, not necessarily black in raw data.

Transforms use the pixel centre ((W-1)/2,(H-1)/2), clockwise positive rotation in
image coordinates, pixel translations, and isotropic scale. Every frame samples
the original pair independently using bilinear interpolation and zero padding.
Probability transport instead forward-deposits each source pixel's mass into
four neighboring target cells, records mass leaving support, and normalizes only
when retained mass is positive. H_eq compares recomputed against transported
normalized mass; H_inv compares recomputed against the unchanged baseline.
Output contrast and log unnormalized path energy accompany these shape metrics.
A transformation need not preserve task labels, especially for clinical images.
Field B / primitive-term targeting remains pending, explicitly marked in the UI.
Existing postactivation interventions are preserved and apply to both inputs.

## Playback, jobs and exports

Checkpoint playback visits saved checkpoints in update order, reusing cached
analyses or computing missing states under a frozen probe/readout/gate context.
Transformation playback computes 2–21 measured frames from fixed originals.
Each completed frame is labelled; waiting time is computation, not simulated time.
The structural-to-conditional slider and short Euclidean transition are display
interpolations, not measured intermediate states. Decorative pulses encode
travel direction only. Camera orbit changes no scientific quantity.
Stop cancels queued work where possible. A running numerical operation finishes
and saves its result, but cannot replace an invalidated display. Completed work
remains exportable in the job log. Recording saves the two geometry canvases with
checkpoint/contrast/transform/readout/gate provenance. It does not record the
whole application or field canvases. Browser MediaRecorder availability applies.

## Validation scope and operational limits

See VALIDATION.md for actual test results. Synthetic image data exercise both MLP
and CNN path calculations, zero input with nonzero biases, and transforms.
Browser workflows use synthetic MLP checkpoints, including a deep eight-hidden-
layer model in the existing smoke test. FashionMNIST and PneumoniaMNIST download
and training code is unchanged; those datasets were not downloaded for this release.
There is no promise of clinical validity or medical transformation invariance.
No new application dependencies are required. Keep the same working virtualenv.
Large dense graphs and all-node labels can reduce frame rate. Geometric fitting
is serial CPU work; this release does not introduce background GPU rendering.
