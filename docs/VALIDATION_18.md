# Validation for 0.11.4

- 18 pytest cases passed: dataset selector catalog/template integration plus the retained
  source-region additivity, native-shape, zero-contrast, residual carrier, and faithfulness checks.
- Dependency-free selection policy tests passed for classification cycling, missing
  matches, singleton alternatives, invalid indices and regression reference pools.
- DOM interaction tests passed for label/source descriptions, relation-change events,
  successive button clicks, zero-reference disabling/re-enabling, swapping, contour
  visibility, profile visibility, comparison-card visibility, and unchanged mass arrays.
- Existing source-region DOM adapter regression test passed.
- Modified JavaScript files pass Node syntax checks.
- Guarded installers were checked against both supported snapshots, root and nested
  repository layouts, plus dirty-tree refusal. Target file hashes were checked.

The full browser test is included as tests/browser_selectors18.py, but did NOT run to
completion here. Chromium startup was blocked by the execution environment's socket
permission restriction (Operation not permitted). DOM tests use jsdom and drawing
adapters, so they do not verify visual layout, rendering or native browser pointer behaviour.

Commands from the project directory:

    python -m pytest tests/test_selectors18.py tests/test_regions17.py tests/test_faithful15.py -q
    node tests/selectors18.cjs
    node tests/source_region16.cjs

The optional DOM test needs jsdom on NODE_PATH. The optional browser test needs Playwright
and Chromium. Neither is a new application/runtime dependency.
