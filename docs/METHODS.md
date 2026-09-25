# Numerical construction

Analysis evaluates the frozen stored weights in float64, while training uses PyTorch float32. Reconstruction is audited against that same float64 forward evaluation, not a mixture of precisions.

For each fixed scalar readout, the execution graph includes each affine/convolutional occurrence and a final linear readout selection. A class margin retains separate paths through both selected class logits, with final coefficients +1 and −1. It does not first subtract output weight rows, which would merge distinct paths.

Let a path have weight product Wπ, and let Δxi be its source contrast. For each hidden unit, compute the finite activation secant

`s = (α σ(z) − α σ(z°)) / (z − z°)`.

At equal preactivations, use the activation derivative multiplied by α. The choice at equality does not affect exact reconstruction because the incoming preactivation contrast is zero. ReLU uses derivative zero at zero. The final affine/logit and readout operators have secant one.

Structural mass is `Qstrπ = Wπ²`. Conditional amplitude is `Aπ = Δxi Wπ ∏s`, and conditional mass is `QΔπ = Aπ²`. Normalization gives `pπ = Qπ / Z`. No complete-path enumeration is required in production. Forward/backward log-sum-exp messages compute Z and all edge marginals, with a second forward pass for squared masses.

Metrics:

- `H1 = log Z − E_p[log Q]` in nats.
- `Sdef = log Npaths − H1`.
- `eta1 = exp(H1) / Npaths`.
- `eta2 = Z² / (Npaths * sum Qπ²)`.
- `log_eta1` and `log_eta2` are exported to retain very small values that may underflow on conversion to ordinary probabilities.
- `sum Aπ` is reconstructed independently through signed forward messages and compared with the real output difference.

Topological path count includes all architectural paths to the selected readout, even if a learned weight currently vanishes. CNN parameter sharing changes which numerical coefficients repeat; it does not identify spatial execution edges. All paths in these sequential architectures traverse the same operator cuts, so a layerwise common gain factor would cancel from normalized p. Absolute `logZ` in this implementation uses raw weights, not a He-renormalized gain convention.

## Metric and projection

The carrier is the undirected version of the execution graph for shortest-path distances. Every edge has length `1/(ne * F_e + 0.03)`, where ne is the number of execution edges in its layer cut. The positive regularizer retains finite lengths for zero-mass edges. It is an explicit demo/laboratory convention and is not a claim that epsilon has been taken to zero.

Full mode computes the complete shortest-path distance matrix. Landmark mode uses a seed in every execution layer followed by deterministic farthest-point metric coverage and computes landmark-to-all distances. Classical MDS is computed on landmark distances, then extended to all nodes with a Nyström-style out-of-sample inner-product construction. Full mode is the special case with all nodes as landmarks. Only positive eigenvalues are used for up to three coordinates; non-Euclidean distance structure can leave substantial residual stress.

Stress compares projected distances with intrinsic distances on exactly the measured pairs. Full mode reports all-pairs stress. Landmark mode reports only landmark-to-node stress; a separate validation stress evaluates distances from up to 16 held-out non-landmark sources to non-landmark nodes. Neither number guarantees local neighborhood fidelity. The same structural landmark indices are used for conditional geometry. Orthogonal Procrustes aligns conditional coordinates, with no fitted scale factor. The worker fixes the display distance unit at the first analyzed checkpoint for each run/readout and reuses it across later analyses. Later structural embeddings are also orthogonally aligned to the first analyzed structural frame, and the same rigid transform is applied to their conditional embedding. Camera pan/zoom and readout translation do not change stored distances or observables. Initial per-panel camera fitting is labelled and is separate from the common intrinsic display unit.

## BCANw response

Full-distance mode evaluates 36 logarithmically spaced radii spanning 0.95 times the smallest positive distance to 1.005 times the largest. Greedy diameter-box coloring orders nodes by descending exact node path mass, breaking ties by node index. A previously used color is forbidden if it contains a node at distance >= radius.

Local effective dimensions are negative OLS slopes of log box count against log radius using a seven-point centered window, shortened at endpoints. A fit is qualified if R² >= .90 and .015 <= NB/N <= .85. `dstar` is the maximum qualified local slope. `deff` is the qualified slope at the nearest sampled radius to xi50, where NB crosses N/2 with log-radius interpolation. Missing qualified fits remain null.

These quantities are finite-scale responses. They do not establish a universal fractal dimension. In version 0.4 the projection and measurement budgets are independent. A landmark display may accompany full-distance BCANw measurements. If the separate measurement budget is exceeded, no approximate substitute BCANw value is reported.

## Intervention provenance

Interventions are post-training activation multipliers on a frozen checkpoint. The multiplier is applied to both x and x°. It is not incorporated into structural weight products. Dose scans jointly change the explicitly selected gate set. The native optimizer is not updated, and training checkpoints are immutable.

## Display and histogram semantics

No collision avoidance, force repulsion or coordinate jitter is applied. The 2-D display uses the first two axes of the 3-D embedding; reported stress uses all three axes and must not be interpreted as a 2-D fidelity guarantee. Both cameras scale x and y equally. A focus control can exclude outliers from the viewport, with visible counts, without changing positions or metrics. Curved carriers are decorative paths between the same endpoints.

Empirical temporal histograms divide bin counts by the total number of recorded observations in the requested window. Undefined, nonpositive-on-log-axis and outside-range observations are reported and are not silently renormalized away. Fixed limits fix the bin edges. Log scale uses geometric bin edges. This is probability per bin, not a probability density per unit value.

Automatic path histories use test indices 0 and 1 and readout f0−f1. Geometry histories require a corresponding full analysis. Local conditional histories additionally match input indices and interventions. Changing a pair does not splice its values into another pair's time series.

## Version 0.3 path-space scopes

All-output scope is the disjoint union of complete paths terminating at each output logit. A unit-weight terminal edge from each output to a bookkeeping sink allows the same dynamic-programming normalization. Squared amplitudes are summed over paths before normalization; output amplitudes are never combined before squaring. Thus Z_all = sum_k Z_k for both structural and conditional measures. A sum of signed logit contrasts is not substituted for the vector response.

The export includes every output contrast, its independently reconstructed signed path sum, maximum componentwise residual, and Euclidean norm of the output contrast vector. The selected scalar contrast remains an additional readout diagnostic. In all-output mode, the legacy cancellation statistic describes cancellation in the sum over signed output amplitudes; its scope is explicitly labelled in the export and it must not be interpreted as vector response attenuation.

All-output and selected-readout frames and metric reference units are separately keyed. Automatic checkpoint probes retain the selected readout. Panels no longer discard edges above 50,000. Every carrier is rasterized, subject only to viewport clipping. Positive-mass animated markers are sampled independently of carrier completeness, with a maximum of 3,000 per panel. No individual complete-path trajectories are sampled or claimed: each edge visualizes its exact marginal.

## Measurement provenance and schedules (0.4)

Analysis records carry the path scope, scalar readout, input pair, gates, metric epsilon, full-distance node budget and estimator version bcan-1. Observable histories match the selected context; automatic legacy selected-readout probes are used only in that same context. Backfill operates on immutable saved checkpoints.

An available scale profile can still have no qualified slope at the half-node crossing. The interface treats absence of a qualified value separately from absence of computation, budget exhaustion, zero conditional mass, queued work and failed work. Projection stress remains an embedding diagnostic and is not a substitute for a BCANw response.
