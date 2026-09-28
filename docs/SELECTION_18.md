# Update 18 · 0.11.4

The MLP/CNN Fields and Geometry workspace now shares explicit selectors for x and x°.
Each option identifies the split-local index, ground-truth label (or numerical regression
target), human-readable class name when known, and original source identity. Residual and
joint-task selectors use the same controller with their stored case catalog.

Selecting a relation immediately cycles to a matching reference and computes the new
analysis. Buttons hold x fixed. They exclude the identical sample, cycle through eligible
references, and report when no alternative exists. A failed search never submits the old
pair under a newly selected relation. Manual dropdown selection is unrestricted and its
actual relation is reported. Zero references have no dataset label. Swapping two dataset
samples updates controls and requires analysis. Existing readouts and transformation
settings remain unchanged; true labels are not predicted labels.

For regression, nearer/farther choices are pools ranked by absolute target difference,
with deterministic index tie-breaking. Each pool contains ceil((n-1)/2) references; the
middle example belongs to both when n-1 is odd. Successive clicks cycle within the pool.

## Colour semantics

Field A contour colours are probability thresholds, not neural-network layers.
White is the 50% enclosed-mass set, gold the 80% set, and pink the 95% set. Each contour
has its own visibility toggle. Ties can make actual enclosed mass exceed its target.
These boundaries are rank-based mass sets, not necessarily nonlinear activation cuts.

The single-layer spatial view uses dark-to-gold occupation intensity, or a gold binary
mask of the selected enclosed-mass set. Cyan rectangles represent selections. Dense
unit bars are blue, with gold indicating mask membership. Labels disclose the display
mapping and mass. Numerical arrays are not transformed by these display choices.

Compare selected layers provides independent layer/operation checkboxes. Spatial maps
remain channel sums at native feature coordinates; dense maps remain native unit bars.
Intensity comparisons use a shared maximum across currently visible cards. Mask
thresholds are computed separately for each layer's mass. Selecting/deselecting a layer
changes only display state. No false pixel alignment or superposition is introduced
between feature grids of different resolutions. Source-route profiles also have explicit
layer checkboxes, in addition to Plotly's trace legend.

## Scope and integrity

No weights, path amplitudes, marginal calculations, normalization, intervention semantics,
metric lengths, or embedding calculations changed. Existing data, checkpoints, geometry
controls, source boxes, and analysis histories remain. Existing Update 17 results supply
all needed layer arrays. This update does not turn source tracing into Field B.
