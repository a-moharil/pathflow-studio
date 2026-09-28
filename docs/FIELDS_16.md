# Update 16 — source-region selection and honest image coordinates

Version 0.11.2. Additive interface update; no path weights, secants, marginals,
metric estimators, fitting procedures, interventions or training rules changed.

Native MLP/CNN signed source heatmaps now have explicit image-pixel axes,
fixed half-pixel bounds and square pixels with constrained domains. The chart
has a compact responsive container. All-output mode explicitly labels the
signed sum of output contrasts, rather than implying a selected margin.

The native Fields panel explicitly states that formal Field B is unavailable.
Source-region routes and signed source contribution remain separately named.
Existing residual/healthcare primitive-support Field B remains intact.

Both native and residual panels now have a Field A rectangle selector: drag in
either direction, click for one pixel, select the full image, or clear. Numeric
bounds remain available for keyboard entry. Right/bottom bounds are exclusive.
Invalid or empty coordinates cannot trigger the region compute button. The
preview uses square-root colours and reports the exact untransformed source
mass. Selection does not recompute automatically: status distinguishes the
selected region from the region used by the displayed downstream results.

Field A and edge marginals group the same complete-path measure differently.
Spatial source concentration alone does not imply a large signed output
contrast or faithful geometric contraction.

Validation: Node regression checks cover forward/reverse rectangle dragging,
one-pixel selection, full/clear controls, exact source mass, invalid/blank
bounds, stale region messaging, pixel-axis labels and ranges, readout labels,
and undefined-measure state. Guarded installers are tested on delivered source
snapshots at repository root and nested layouts. Full model/browser execution
was not rerun in this environment; browser download was unavailable.
