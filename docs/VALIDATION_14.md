# Validation for 0.11.0

Executed in a temporary Python 3.12 environment using torch 2.6.0+cpu, torchvision 0.21.0+cpu and the project's pinned scientific dependencies.

- `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q`: **107 passed**, 109.14 seconds in the final complete run.
- New tests compare complete-route entropy and source-region mass with explicit enumeration of a tiny branching graph; check exact forward/signed reconstruction for ReLU/tanh, two/three branches and preactivation; verify CNN hidden head on/off and regression; train a residual regression update; verify full scalar scope/budget rejection; test zero contrast; verify general traced ResNet-50; and verify shared-module execution identity plus average pooling.
- Independent small residual check: signed reconstruction residual approximately 2.6e-17. Default full scalar preset: 1,251 nodes and 8,950 occurrences, with zero-weight admissible dependencies retained.
- `tests/browser_residual14.py`: **passed**, real residual training, full scalar geometry, source-region computation, colour controls, channel attenuation, healthcare navigation, native CNN hidden-head regression training and geometry; zero JavaScript errors. Artifacts inspected visually.
- `tests/browser_healthcare14.py`: **passed**, real synthetic joint-task training, classifier/segmentation views, attenuation, shared geometry and Lorentz/Poincare 3-D controls; zero JavaScript errors.
- `tests/browser_integration.py`: **passed**, existing-run measurement scheduling, all-output backfill and deletion; zero JavaScript errors.
- JavaScript syntax checked and Python sources compiled.

The old catalog test was updated to assert mask dataset visibility at `/design?domain=healthcare`, reflecting the requested move. Numerical acceptance assertions were not loosened.

No clinical performance claim, BUS-UCLM training study, fresh medical-dataset download or ImageNet weight download was performed. Existing standard dataset adapters were exercised using synthetic/cache fixtures. Full ResNet-50 scalar all-pairs geometry was not computed and is not claimed. The raw-affine residual structural reference and topology-relative lengths remain experimental and explicitly labelled.

Browser automation is a developer-only dependency; it is not needed to run the app. A local test browser used Playwright 1.51.0. Ordinary installation dependencies remain unchanged from 0.10.0.

The Git installer was tested in repository-root and nested-project layouts, with existing workspace data preserved. Dirty trees and committed source mismatches were rejected before branch creation. Baseline hashes and patch are grounded in the actual delivered 0.10.0 ZIP.
