# Current validation · 0.5.0 (2026-09-25)

The results below apply to this package. Older results retained later in this
file are historical and do not imply that medical datasets were rerun today.

- 32 pytest tests passed. This includes all inherited training, pause/resume,
  optimizer, scheduling, deletion and path-algebra tests, plus Lorentz constraints,
  Lorentz/Poincare distance agreement, isometries, identical-state fitting,
  undefined conditional measures, identity/rotation/scaling/translation transport,
  boundary loss, and MLP/CNN zero-reference reconstruction with nonzero biases.
- 100 independently enumerated path-field checks passed.
- Node checks passed for camera/tab state, Observatory contexts, and all 79,490
  structural carriers (including zero-mass edges). All-node labels are verified
  against every node identity in the renderer test.
- Chromium browser_smoke passed: deep MLP, preparation, actual training and pause,
  complete geometry, fields, gates, plots and Observatory. No JS exceptions.
- Chromium browser_integration passed: scheduling/backfill and deletion workflows.
- Chromium browser_hyperbolic passed: H3 ball selection, rotation, anchor release,
  labels, intrinsic/embedded/screen pair inspector, zero reference, transformed
  fields, identity reset, two-frame transformation sweep, checkpoint playback,
  geometry video download, and stale-response protection. No JS exceptions.
- Desktop screenshots of the ball and transformed field comparisons were inspected.

Runtime: Linux, Python 3.12, PyTorch 2.6.0+cpu / torchvision 0.21.0+cpu,
Playwright 1.51.0 Chromium. Application dependency pins are unchanged. Numerical
checks used installed NumPy 2.3.5 and SciPy 1.17.0; the exact older pinned NumPy/
SciPy environment was not recreated in this release. Browser tests use synthetic
MLPs; CNN transformation/zero-reference paths have numerical coverage. Real
FashionMNIST and PneumoniaMNIST downloads, CUDA, native Windows/macOS, long
training runs and maximum-size Lorentz fits were not rerun here.

The short 25-step browser fit intentionally exercises diagnostics, not convergence:
conditional held-out stress remained high and the UI reported high distortion.
No test establishes globally optimal embeddings or absence of genuine contraction.
See REPRESENTATION_03.md for numerical budgets and explicit limitations.

Reproduce from the project root with your configured Python environment:

```bash
python -m pytest -q
python tests/input_fields_numerical.py
node tests/geometry_state.cjs
node tests/render_completeness.cjs
node tests/observatory_context.cjs
```

Optional browser tests require Playwright and its Chromium runtime, which are
development dependencies, not requirements for running the app itself.

---

# Historical validation record

# Validation for 0.2.0

Environment: Linux, Python 3.12, CPU PyTorch 2.6.0+cpu, torchvision 0.21.0+cpu. Browser: Playwright 1.51.0 Chromium headless. These checks establish operation on this stack, not identical numerical behavior on every OS, GPU or dependency version.

## Automated backend suite

14 tests pass (`python -m pytest -q` from the project directory):

- Adam, SGD and RMSprop, each on MLP and CNN: real parameter updates, saved optimizer state, pause/resume, and byte-exact CPU parameter agreement with an uninterrupted run using the same minibatches.
- ReLU and tanh, each on MLP and CNN: native and intervened finite-contrast reconstruction below 1e-12, normalized cut sums, and undefined zero-contrast measure handling.
- Dataset preparation required before run creation, independent preparation jobs, validated controls, CSRF checks, analysis completion and layer-covered landmark diagnostics.
- Pause remains pending until checkpoint measurements are committed.
- Log-domain entropy and concentration match an explicitly enumerated small graph.
- Invalid SGD Nesterov parameters are rejected.

## Browser workflows exercised

- Design checks, RMSprop-specific fields, dataset preparation, experiment creation and real training.
- Confirmed pause followed by full-distance geometry and BCANw analysis.
- Three learning-curve traces, true 3-D controls, focus control, and gate analysis.
- Conditional inset and expanded plot visibility toggled independently.
- Logarithmic concentration histograms and a 390-pixel-wide observatory without horizontal document overflow.
- A real PneumoniaMNIST CNN checkpoint, channel intervention, and a downloaded nonempty geometry recording (33,972 bytes in the validation run).
- No browser JavaScript exceptions in these workflows. Static JavaScript syntax checks also pass.

The optional `tests/browser_smoke.py` reproduces the synthetic end-to-end browser check. Screenshots and temporary experiment workspaces used during development are not bundled as user data.

## Real dataset checks

Both datasets were downloaded through their actual torchvision/medmnist loaders, prepared, verified, then used for two RMSprop CNN updates. Training subset size was 32, architecture channels [2,2], default stride-2 convolutions. Full official test sets were evaluated.

| Dataset | Validation count | Test count | Initial train CE | Final train CE | Final validation CE | Final test CE |
|---|---:|---:|---:|---:|---:|---:|
| Fashion-MNIST | 6,000 | 10,000 | 2.30714 | 2.23274 | 2.28533 | 2.28447 |
| PneumoniaMNIST | 524 | 624 | 0.70804 | 0.46538 | 0.57317 | 0.67492 |

These are integration smoke checks, not model-quality results or a medical performance claim. Long runs, maximum supported graph sizes, native Windows, macOS and CUDA were not validated. A low-dimensional embedding can still have high stress or overlapping nodes; the app exposes this rather than altering intrinsic coordinates to conceal it.

## Version 0.3.0 additions

The expanded backend suite passes 16 tests. New MLP and CNN checks prove, numerically, that the all-output partition equals the sum of individual output partitions for both measures. Every output contrast is reconstructed within 1e-12, including gate interventions, while structural measures remain unchanged by those gates.

`node tests/render_completeness.cjs` exercises the actual renderer on 79,490 edges. All are passed through the carrier drawing routine, including all 39,745 zero-mass carriers in that test. Moving markers remain explicitly bounded at 3,000. This checks carrier inclusion, not the perceptual separability of overlapping edges in a 2-D projection.

The browser smoke test now uses an eight-hidden-layer, width-64 MLP and asserts that more than 50,000 execution edges are retained in the structural view under all-output scope.

The updated browser smoke test could not execute in the 0.3 validation environment: Chromium was blocked at startup by a socket/process restriction. Earlier 0.2 browser results above remain historical results, not evidence of a 0.3 visual pass. Current validation includes the API integration job for all-output geometry, numerical tests and the direct renderer inclusion test.

## Version 0.4 validation

20 backend tests pass. Added checks cover independent projection/measurement budgets, backfill deduplication, run deletion including data cleanup and shared-cache preservation, deletion requested during checkpoint saving, and automatic geometry scheduling. The Node Observatory test confirms all-output geometry values appear, selected contexts do not mix those records, and unmeasured checkpoints remain explicitly missing.

A default Fashion-MNIST-shaped CNN with channels [4,6,8] and 2,001 execution nodes was analyzed with synthetic random input contrasts and frozen random weights. With a 500-node projection threshold and 2,500-node measurement budget, the display used landmark projection while both intrinsic BCANw responses were computed. Structural deff=0.88237 and dstar=2.94255; conditional deff=0.24701 and dstar=0.26318. Vector reconstruction residual was 5.38e-17. Wall time in this environment was approximately 6.8 seconds. This is an implementation check, not a trained-model scientific result.

The 0.4 Chromium browser smoke test passed on the eight-layer width-64 MLP. It exercised dataset preparation, training/pause, complete >50,000-edge geometry, intervention, independent inset visibility and a populated all-output geometric-response histogram. Desktop screenshots were inspected; a 390-pixel-wide Observatory had no horizontal document overflow. No JavaScript exceptions were observed. These current results supersede the earlier browser-launch limitation for the tested 0.4 workflow.

A separate browser check confirmed experiment deletion through the registry confirmation, disappearance from the registry, removal of its run folder and preservation of the shared dataset cache.
