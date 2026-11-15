# Validation of Update 15

Executed using the isolated CPU test environment with torch 2.6.0 / torchvision 0.21.0.

- Full regression suite: **119 passed**, 96.29 seconds. This includes all 107 prior tests and 12 new tests.
- Additional positive ReLU-rescaling test, added after that full run: **1 passed**, 4.94 seconds. Both declared structural measures and conditional path marginals remain unchanged under the tested compensated rescaling.
- New checks include hand-calculated residual partition functions, explicit tiny-path enumeration, entropy, unequal-depth He effects, sequential normalization cancellation, scalar comparison-class means, zero classes, minimum parallel-edge distances, full scalar scope, depthwise convolution, optional merge mass, baseline/intervention graph identity, structural invariance under attenuation, contrast reversal, nonfinite rejection and evaluation-mode restoration.
- `tests/browser_faithful15.py`: **passed**, zero JavaScript errors. Real residual training and full scalar geometry; He/epsilon/power controls persisted through analysis; source-region fields; baseline/intervened execution-graph hashes equal; conservation audits passed; healthcare entry; native CNN hidden-head regression training and geometry.
- Earlier browser test attempt failed because it read visible text from a collapsed details panel; the test now reads the panel's actual text content. No production bypass was introduced.
- The rescaling test initially constructed coefficients in float32 before copying to float64; constructing the test coefficients directly in float64 corrected that test setup without relaxing its tolerance.
- JavaScript syntax, Python compilation, ZIP integrity and guarded installer checks were performed before delivery. Installer checks cover both supported baselines, root/nested repositories, dirty/mismatched source rejection and preservation of existing workspace data.

This is implementation evidence for the specified constructions. It is not a clinical validation, an isometric-embedding guarantee or an exhaustive test of arbitrary user-defined PyTorch operators. No medical dataset download or full high-resolution ResNet-50 scalar all-pairs computation was performed. The defined explicit-geometry budget remains in force.
