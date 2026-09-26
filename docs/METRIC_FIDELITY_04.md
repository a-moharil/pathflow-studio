> Historical 0.5.1 specification. Numerical evaluation and acceptance are augmented by METRIC_INTEGRITY_05.md in 0.5.2.

# Metric fidelity correction · 0.5.1

This update fixes representation defects in 0.5.0. It preserves the graph,
weights, exact path measures, intrinsic edge lengths, BCANw calculations, fields,
activation interventions, training, scheduling, deletion and recording.

## Every node is fitted

Source selection is deterministic, seeded, and includes nodes from every layer.
Up to 48 source rows are used (more only if layer coverage requires them).
Every node has at least four distinct fitting neighbors for ordinary lab graphs.
Small graphs use the available neighbors and require enough remaining pairs for
a disjoint diagnostic set. Pair IDs, rather than complete nodes, are withheld.
There is no sorted-index train/test split. The UI reports minimum training degree
and the count of unconstrained nodes. This must be zero.

The common fitting set contains mandatory coverage pairs plus random source-to-node
pairs, at most max(24,000, mandatory pair count). Up to 8,000 remaining pairs form
a disjoint diagnostic set. This is a sampled objective, not an all-pairs theorem.
Quartiles of intrinsic distance expose small/intermediate/large-scale errors.
Source-to-node sampling is not uniform sampling of every unordered graph pair.

The diagnostic pairs are excluded from stress optimization. However, the spectral
initializer uses the sampled distance rows, which can include diagnostic entries.
These diagnostics are therefore not claimed to be strictly independent statistical
generalization estimates. Their role is to inspect approximation away from the
optimized pair set. The same IDs are used for both measures and both ambient spaces.

## Independent Euclidean 2-D and 3-D representations

Each dimension receives its own classical landmark MDS initialization followed
by explicit distance-stress refinement. Alignment happens within that dimension
using orthogonal transformations and translations only. The 2-D renderer reads
its own coordinate table; it no longer drops one coordinate from a rotated 3-D fit.
The three-dimensional compatibility `positions` field is retained for old clients.
New clients use the versioned `euclidean.dimensions` tables.

The optional 8-D diagnostic fits help assess sensitivity to display dimension.
They are separate numerical fits, not alternate screen projections. A poor 3-D
fit does not prove a high intrinsic dimension, and a better 8-D fit does not establish
the graph's dimension. Sampled Gram spectra report positive/negative eigenvalue
counts and retained positive spectral mass with their restricted scope explicitly
identified. No dimension estimator is silently replaced by these diagnostics.

## Comparable objectives and convergence

Both Euclidean and Lorentz fitting minimize the mean squared relative distance
error, using denominator max(D_ij, 0.1 × median fitting distance). The floor prevents
extremely short pairs dominating. It is recorded and affects the fitting objective,
not the intrinsic metric. Ordinary normalized absolute stress, median and 95th
percentile relative error, distance-quartile errors, and neighborhood overlap are
reported separately. This prevents one objective number concealing a tradeoff.

Float64 L-BFGS-B uses analytic gradients, a declared iteration budget, and reports
termination, evaluations, iterations and best objective. For a fitting objective
above .06, a second deterministic random initialization is optimized under exactly
the same metric objective; the better fitting result is retained. This is an
optimizer restart, not display jitter or repulsion. Diagnostic pairs are not used
to select the restart. There is no claim of global optimality or guaranteed low
stress. Meeting a solver stopping criterion is distinct from metric fidelity.

Lorentz time coordinates are reconstructed from spatial coordinates. Curvature
and metric units remain fixed through each comparison. Spatial optimization has
an explicit numerical box bound sinh(12)/sqrt(kappa) per coordinate; bound occupancy
is reported. This bound can restrict extreme configurations. A readout-origin
Lorentz boost and spatial orthogonal transformations align the results. Large
coordinates can still lose precision; the constraint residual is exported.

High distortion is marked when diagnostic stress or median relative distance
error exceeds .25. This is a descriptive warning threshold, not a scientific
acceptance theorem. Other results remain labelled 'inspect diagnostics', never
'faithful' merely because fitting completed. Low-dimensional overlap may persist.

## Deformation and intrinsic evidence

The audit compares changes in intrinsic distances with changes in represented
distances, reporting RMS intrinsic change, RMS represented change, RMS change
error, relative change error and correlation. A zero intrinsic-change denominator
is reported as undefined rather than divided by an epsilon to manufacture a score.
Structural-to-conditional comparisons use the full common diagnostic pair set.

Checkpoint comparisons use up to 512 deterministic diagnostic pairs and the first
saved reference for the same input, reference type, transformation, readout, gates,
curvature settings, fitting budget and distance unit. The reference checkpoint is
explicit and is not necessarily update zero or the immediately previous frame.
Isometric alignment cannot remove these distance changes. The scatter plot and
JSON export expose actual intrinsic and represented distances, rather than only
aggregate errors. Changing a setting starts a separate comparison context.

A 64-node submatrix inspector computes exact shortest paths on the entire graph
then selects the labelled rows/columns. It is explicitly a sampled submatrix.
Full intrinsic matrix export is allowed up to the saved measurement node budget,
with an absolute 4,000-node cap. Export can be expensive and temporarily occupy an
HTTP worker; it is manually requested and never triggered by playback.
Neighborhood-overlap values remain sampled and can be affected by exact ties.

## Workflow and compatibility

Selecting a checkpoint now automatically analyzes or retrieves its saved result.
It disables Follow so that inspecting an earlier checkpoint is not immediately
overridden by the latest one. The old result stays visible with a pending status
until the matching result arrives; stale responses are discarded. Independent
cameras, anchors, ghosts, labels, node/flow controls and recording remain available.

Analysis cache version is 5; embedding version is metric-fit-2. Old checkpoints
are reusable without retraining. Old embedding results are preserved but not
silently reused as corrected results. New dimension-specific frame files avoid
mixing with the old 3-D alignment convention. No runtime dependency is added.

The default Euclidean fit budget is 120 iterations per initialization, adjustable
25–1,000. Hyperbolic fitting remains optional with its separate 25–500 budget.
The optional 8-D fits increase runtime. Large graphs can remain slow on CPU;
metric refinement is not a frame-rate operation. Flow animation and camera orbit
remain independent of numerical fitting and never imply additional measured states.
