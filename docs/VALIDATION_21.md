# Validation record · ECML auditor release 21

## Passed

- 8 auditor Python tests: real native and residual-DAG adapters; healthcare dose semantics; undefined measures; report hashes; stored-file preservation; pseudo-replication and incompatible-context rejection; SciPy agreement; Holm correction; scaling gates; exact diamond geometry; API/CSRF/export checks.
- 13 retained Python tests across healthcare, workflow 19, region 17 and selector 18 modules.
- 6 installer scenarios: clean 0.12.0; project nested inside a repository; local 0.12.1 with additional channel code; dirty-tree rejection; existing-branch rejection; wrong-remote rejection. Preflight-only operation and backup refs are also checked.
- Full auditor script executed in JSDOM using real native path computations and full scalar geometry results: report selection, view switch, distribution data, 64-scale remeasurement, radius normalisation, population declaration, exports, linked windows, late-response rejection and experiment filtering.
- JavaScript syntax and Python compilation checks.

The installer initially rejected trailing whitespace in the new stylesheet and rolled back its changes. The whitespace was removed and all six installer scenarios then passed.

## Validation limit

Chromium was attempted. The bundled download failed; the installed system Chromium then failed at startup because the environment denied its singleton socket operation. No rendered browser layout is claimed verified. JSDOM uses a Plotly recording adapter and validates chart data and lifecycle, not rendered graphics. The included `tests/browser_auditor21.py` supports local end-to-end verification with Playwright and Chromium.

Synthetic engineering fixtures are generated in temporary directories. They are real computations used for validation, not research findings or demonstration values substituted into user results. No production checkpoint or user dataset was changed by these tests. No remote branch was pushed.
