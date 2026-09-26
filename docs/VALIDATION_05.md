# Validation of integrity augmentation 05

Executed locally for this release on 26 September 2026:

- Full Python suite: 43 passed. Includes existing MLP/CNN optimizer, path audit,
  training lifecycle, measurement and deletion tests plus embedding regressions.
- Focused metric/hyperbolic suite: 13 passed, rerun after strengthening the
  released-anchor numerical validity check.
- New gradient finite-difference checks at curvature magnitudes 0.01, 1 and 4,
  ordinary separation, 1e-7 separation and inside the numerical floor. A fit
  initialized at 1e-7 separation reaches objective below 1e-8.
- Invalid Lorentz coordinates rejected; exact zero self-distance checked;
  alignment invariance checked; numerical and budget failures covered.
- Neighborhood regression verifies that errors beyond the first 12 sources
  affect the reported statistic and source samples span every layer.
- Production fidelity-renderer regression: unrequested, budget-exceeded,
  numerically invalid and undefined hyperbolic states do not throw.
- JavaScript camera/hidden-resize/anchor/tab state regression passed. Independent
  2-D coordinates remain invariant under changes to the 3-D camera rotation.
- Complete carrier rendering: all 79,490 test edges retained, including all
  39,745 zero-mass carriers; all-node labeling checked.
- Observatory scope regression passed.
- Independent enumeration: 100 path-field cases passed, including channel
  reduction and undefined conditional measures.
- Headless Chromium workflow passed with no page errors: Euclidean/H3 fitting,
  automatic checkpoint change and changed coordinates, fidelity table, intrinsic
  matrix inspection, anchors, labels, orbit, zero reference, field transforms,
  transform/checkpoint playback, recording and stale-request protection.

The bundle installer is exercised against both original delivered baselines,
0.5.0 and 0.5.1, in isolated Git repositories. Checks require exact equality of
installed files with the bundled project, retained experiment-workspace sentinel
content, expected branch creation and refusal of a dirty working tree. The
packaging script also asserts no 0.5.1 source files were removed and checks
scientific-core hashes. See source-integrity.json in the bundle.

Limitations: these checks do not establish faithful low-dimensional embedding
for every trained network, global optimization, or universal low distortion.
The new validity thresholds can intentionally reject extreme coordinates. This
release was not validated on a downloaded medical dataset, CUDA, or the user's
local Python environment. No new dependency is required. The headless workflow
uses a small synthetic-data MLP; the Python suite separately covers CNN behavior.
