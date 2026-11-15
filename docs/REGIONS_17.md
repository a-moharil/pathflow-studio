# Update 17 · input contrasts and local layer occupation

Version 0.11.3 extends the delivered 0.11.2 application. Existing training,
interventions, intrinsic metrics, embeddings, anchors, geometry controls,
Field A, signed diagnostics and formal Field B remain in place.

## Interpreting the new views

The original right-hand input map is Field A. Its contours are enclosed-mass
sets of an input-origin marginal; they are not different layers.

The new input panel shows the two actual analyzed images and their pixel
contrast. For C input channels, the displayed contrast is
`D(s) = sum_c |x(c,s) - x0(c,s)| / C`.
Red overlays encode D, gold overlays encode Field A. The selected overlay can
be turned off or attenuated. Square-root intensity scaling is disclosed;
raw contrast and Field A have separate numerical maxima. A region can have
large path occupation without large raw contrast, or vice versa.

Drag a rectangle on x, x0 or D. Coordinates are in the analyzed input grid,
after the experiment's preprocessing and requested transform. The same box
is outlined in all views. A zero reference is represented by the actual zero
array. Click Compute routes from selected input box. This is source tracing,
not masking the input, retraining, or an activation intervention.

For input source set R, the downstream quantity is
`M_R(v) = sum_pi p(pi|c) 1[origin(pi) in R] 1[v in pi]`.
The implementation restricts only the starting squared-contrast mass while
retaining all frozen weights and finite secants. It rescales the resulting
normalized marginal by p(origin in R), preserving whole-realization units.
Dividing by source mass is an explicit alternative, undefined at zero mass.

Select a layer/operation to inspect its native map. CNN feature maps sum
channels at each spatial site. Dense layers show unit-index bars. Residual
whole-input maps cover captured activation layers; selected-source maps cover
recorded operations, including branches. An individual residual operation's
mass need not equal the full source mass.

The 50%, 80%, and 95% mask options rank positive spatial masses in the selected
map, include threshold ties, and report the actual enclosed mass. They describe
that map's occupation, not predicted segmentation or anatomical ground truth.
Drag a box within a spatial layer map to inspect its mass. This local feature
box is distinct from the input source box; it does not silently back-project
onto input pixels or activate an intervention. Dense units have no invented
image-space footprint.

Saved results from earlier releases may not contain layer shapes. The app asks
for recomputation rather than guessing spatial layouts. The selected input box
and previously computed box are labelled separately when they differ.

## Compatibility correction

The Update 16 selector was initialized on joint-design pages without a selector
host. Update 17 scopes initialization to experiment pages, preserving residual
and healthcare experiment creation. In the native Fields tab, the training
toolbar scrolls normally so it does not cover image-selection targets; other
tabs retain their existing toolbar behavior.

## Validation

See VALIDATION_17.md for checks completed for this release.
