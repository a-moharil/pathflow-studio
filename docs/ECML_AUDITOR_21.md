# ECML manuscript integration · Pathflow Studio 0.13.0

Branch: `integrating-ecml-manuscript`.
Published source baseline: `db75674` (`feat/healthcare-workflow-19`, version 0.12.0).
The additive installer also supports a clean local 0.12.1 checkout and preserves its channel-map and worker fixes. The included full source snapshot is based on the published 0.12.0 branch; use the additive installer to retain unpublished local changes.

## Using the two windows

Open **ECML auditor** from an investigation. The experiment filter follows the page you came from. Select a completed recorded result explicitly: live controls are not assumed to describe a saved computation. Switch between **Interpretation** and **Statistical evidence and fidelity**, or open the other view in a separate window. Each window retains its selected record; changes in another window cannot silently alter it.

Interpretation covers concentration, conditional realisation, geometry and intervention. The expert view supplies recorded reconstruction/conservation residuals, histograms, ECDFs, marginal mass concentration, layer occupation and geometric criteria. A source report SHA-256 and the configuration accompany every export.

Current coverage: native analysis/dose jobs, healthcare case/dose jobs and baseline/intervened residual-DAG heads. Completed cohort, attribution-only and ensemble jobs are not reinterpreted as independently fitted model replicates. Missing quantities remain unavailable.

## Scientific definitions

Occupancy and deficit use the existing engine's complete-path computations. `eta1` is the entropy-equivalent path count divided by the architectural path count; it is not the fraction of nonzero paths. `Sdef = -log_eta1` is a redundant description, not independent evidence. Edge, input-origin and channel marginals retain explicit observational units. The app does not enumerate or invent complete-path histograms.

The manuscript source for this integration is `ECML_PKDD_submission_finalv1.pdf`, Sections 3–6, Appendix C.10 and Appendices D–E, supplied by the author. It is not copied into the repository. No published numerical finding is inserted as a current experiment result.

The numerical status covers only displayed checks. A passing audit is not a certificate of every mathematical assumption. Reconstruction uses a displayed scale-aware threshold of `1e-7 * max(1, abs(delta))`. Native cut and entropy identities use `1e-9`; stored DAG audits retain their recorded tolerance. Existing engine tolerances are not changed.

## Geometry

Legacy `bcan-1` uses 36 scales and its own slope criteria. Its existing profiles are shown explicitly as legacy. The seven Appendix C.10 gates can be inspected on those profiles, without claiming an exact reproduction of the manuscript instrument.

An explicit 64-scale remeasurement is available from saved **full scalar** edge marginals for at most 800 nodes. It uses recorded metric lengths where present; otherwise it uses the native declared enrichment and ruler. Undirected parallel alternatives take minimum length. Disconnected carriers are rejected. No projected distances enter this computation. Operation quotients and induced subgraphs cannot substitute for a full scalar carrier.

The deterministic greedy diameter cover is an approximation to minimum box covering. The radius grid spans 0.95 times the smallest positive distance to 1.005 times the maximum, in 64 geometric steps. Node mass is obtained from incoming marginals, or outgoing marginals at source nodes, and normalised over the geometry-bearing graph. This is a new explicit auditor estimator; reproducing the paper's entire instrument would additionally require its exact archived implementation and settings.

Repeated count plateaus are compressed at their first scale. Candidate contiguous windows require six distinct counts, 0.5 decades, exponent >= 0.25, a fivefold count change, R² >= 0.97, local-slope CV <= 0.25 and linear-minus-quadratic BIC <= 2. The selected credible window has maximum span, then most points. If none passes, the closest candidate's failing gates are shown and the exponent remains null. Window search is exploratory. Adjacent compressed-transition slopes are labelled as this auditor's estimator and are not silently equated with the paper's effective-dimension smoothing.

Flow lacunarity is the second moment of box mass divided by squared mean box mass. Retention is the directed edge-flow fraction between boxes, measured at each scale; the summary averages the nearest recorded resolutions to fractions 0.5, 0.25, 0.125 and 0.0625. The actual resolutions are exported. Radius normalisation by each state's xi50 permits shape comparison where xi50 is bracketed.

## Population inference

Select one record per run. Different checkpoint indices, datasets, model families, readout definitions, reference policies, transforms, stage, gates or conventions are rejected as one comparison stratum. The individual input pairs can vary; the estimand is explicitly conditional on that selected set of run/condition pairs. Available repeated parameter hashes are rejected. Native reports without parameter hashes cannot receive that additional duplicate-check; the auditor must check training provenance.

Inference is disabled until the user declares independent training units and at least six valid pairs exist. Six is a product guard, not a power justification. Finite observations from reports with failed numerical checks are excluded with counts and reasons. Reports whose numerical checks were not recorded are provisional and remain visible as such.

The supported contrast is conditional minus structural within each run. The engine reports the sample median difference and a 95% percentile bootstrap interval using 4000 resamples of entire paired runs, seed 2101. The two-sided Wilcoxon signed-rank test uses `zero_method='wilcox'`, SciPy `method='auto'`; a signed-rank location interpretation assumes symmetric differences. All-zero differences have statistic 0, p=1 and rank-biserial 0. The normal Q–Q plot is descriptive, does not certify symmetry, and does not make normality a Wilcoxon assumption. Exact SciPy/NumPy versions are exported.

Holm correction covers the available tests of `log_eta1` and `log_eta2` in that submitted selection. Repeated interactive selections are exploratory and are not covered by that correction. Normalised deficit remains descriptive because of its relationship to occupancy. The rank-biserial effect uses ranks of absolute nonzero paired differences. Direction counts retain positive, negative and zero observations.

This release does not implement a generic arbitrary-test menu. Optimiser block models, tail-model comparisons, parameter/reuse specialisation, local paired support/mass-growth fits and new occurrence-targeted interventions require additional designs or stored quantities. They are explicitly unavailable rather than inferred from channel aggregates. Existing analyses and interventions remain available in their original investigations.

## Dose semantics

Existing native healthcare and residual jobs use an `alpha` field for the **retained multiplier**: 1 is baseline and 0 removes the selected contribution. The auditor labels it `g`. The manuscript's attenuation strength is `1-g`. Postactivation channel interventions and input-region perturbations retain their recorded semantics; they are not relabelled as primitive-occurrence interventions.

## Engineering

The integration adds a Flask blueprint, template and isolated scripts. It does not change existing numerical kernels, training, job storage schema or checkpoint contents. Remeasurement is an explicit bounded read-only request, and it does not overwrite legacy results. Export includes any new remeasurement. Late responses after a selection change are ignored. Population results are invalidated when selection or independence declaration changes.

API: GET `/api/auditor/catalogue`, GET `/api/auditor/report?record=JOB:STAGE:HEAD`, POST `/api/auditor/population`, POST `/api/auditor/geometry`. Existing CSRF protection applies. Report files must resolve inside the configured workspace.

## Verification

Run from the project directory with the existing environment:

```bash
.venv-pathflow/bin/python -m pytest -q tests/test_auditor21.py tests/test_healthcare.py tests/test_workflow19.py tests/test_regions17.py tests/test_selectors18.py
```

Optional real-browser check, with Playwright and Chromium installed:

```bash
PYTHONPATH=. .venv-pathflow/bin/python tests/browser_auditor21.py
```

The test creates independent temporary engineering data and does not touch production experiments. `PW_CHROMIUM` may name an installed Chromium executable. DOM checks are provided separately and do not validate visual layout.
