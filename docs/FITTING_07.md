# Pathflow Studio 0.6.1 — fitting budgets and dimension diagnostics

Euclidean and Lorentz maximum iterations are independently configurable from 25 to 5,000 per attempt. Defaults remain 120 and 150. L-BFGS-B may stop earlier. The evaluation limit is max(15,000, 40 × maximum iterations). A difficult fit may run a second initialization; selection uses the fitting objective, never diagnostic-pair error.

Enable diagnostic fits and choose 8, 16 and/or 32 dimensions (Ctrl/Cmd for multiple selection). Euclidean always fits 2-D and 3-D. When enabled, Lorentz fits 3-D and the same additional dimensions. Higher dimensions are diagnostic only; the displayed ball remains 3-D. All fits share the same pairs and unchanged intrinsic distance targets. No curvature or metric-unit rescaling is introduced.

The comparison table reports actual iterations, budget, stop category and solver message. The convergence chart shows the selected attempt. JSON audit retains every attempted fit, with at most 501 evenly sampled history points per attempt, including iteration zero and the actual final iterate. Returned coordinates retain the best finite objective found; the final solver iterate can differ from that best iterate. Legacy saved results remain readable; unavailable metadata are labelled legacy.

Iteration budget and dimension address different sources of error. Compare fitting objective, diagnostic-pair error, neighborhood preservation and dimension sensitivity; convergence is not proof of metric fidelity. Diagnostics are not strict out-of-sample estimates because spectral initialization can use the source distances.

Longer fitting can occupy the existing serialized worker for longer. This release does not add mid-fit cancellation or live streaming of optimizer iterations. Histories appear with the completed analysis. Existing node budgets remain in force.

Legacy dimension_sensitivity=true requests still mean an 8-D diagnostic. New requests record diagnostic_dimensions explicitly. Re-analyze a saved checkpoint to apply new settings; model retraining is unnecessary.
