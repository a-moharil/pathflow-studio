# Validation of Joint healthcare extension 10

Executed on CPU, Python 3.12, torch 2.6.0+cpu, torchvision 0.21.0+cpu, NumPy 2.1.3, pandas 2.2.3. These are engineering checks, not medical performance studies.

- All 72 original Python tests passed on the initial additive integration.
- The combined suite exposed a direct-worker compatibility regression when only the legacy schema had been initialized. Joint polling now returns without work when the additive table is absent. The 83-test combined suite then passed.
- All 13 new tests passed, including two subsequently added dataset-import and deterministic-resume tests. In total, 85 distinct Python tests were exercised successfully.
- Tensor squared path energy, input marginals, branch conservation and positive/negative amplitude moments agree with independent explicit path enumeration on a small graph containing residual addition, ReLU, max pooling, concatenation and nearest upsampling. Structural and topology-reference measures are tested separately.
- Max-pool checks cover winner changes, ties, padding exclusion, reversal and adjoint consistency. Evaluation BatchNorm with negative affine gain is checked.
- ResNet-50 classification encoder output matches torchvision's forward computation. Both classifier and mask readouts reconstruct finite output contrasts from the same recorded execution.
- Zero-contrast conditional measure is unavailable rather than normalized or set to an invented probability; interventions do not mutate model state.
- Adam, SGD and RMSprop step using their torch.optim implementations. A saved/restored two-update run matches uninterrupted training exactly in the deterministic residual-model fixture.
- Dataset checks cover patient-disjoint splits, duplicate detection, letterboxing and RGB-mask BUS-UCLM import contracts. Runtime dependency imports, including PneumoniaMNIST through pandas, passed.
- A real ResNet-50, decoder width 16, 64×64 synthetic input smoke run performed training, validation/test evaluation, checkpoint save, both readout analyses, a channel intervention, scalar dose curves and selected-subgraph geometry. Baseline reconstruction residuals were 4.26e-14 for classification and 5.20e-18 for segmentation. The analysis used four IG intervals; its visible quadrature residual is not confused with the path reconstruction residual.
- Existing browser inspector regression passed with no JavaScript errors, covering visual channel and primitive selection, region diagnostics, compensated rescaling, reversal, old-report compatibility, JSON/CSV/PNG and WebM recording.
- Existing Node checks passed for camera/anchor state, independent 2-D representation, full rendering (including zero-mass carrier edges), all-node labels, Observatory contexts and unavailable-fidelity states.
- New Chromium browser checks cover the registry, restored analyses, both heads, baseline/intervened maps, visual channel selection, independent 2-D/3-D view controls, labels, ghosts, anchor removal JSON/PNG export and WebM recording with directional pulses. A collapsed-plot overlap and a WebGL text-rendering stall were found and corrected; the selected-subgraph renderer uses canvas and orthographic projection.

## Explicit limits

The full BUS-UCLM archive download returned HTTP 403 from the publisher-linked URL here. The adapter's extracted-folder contract is tested with a fixture, not the entire medical archive. No trained medical model, pretrained download, CUDA training, 512-pixel training/analysis benchmark or clinical comparison was validated. The source dataset and model weights are not bundled.

Raw-affine structural and topology-reference metric conventions are declared experimental additions. They do not settle the unresolved He-normalized residual-DAG theory. Full ResNet scalar-graph BCANw and hyperbolic embedding are not implemented. The supplied selected-subgraph fit has explicit distortion diagnostics and cannot stand in for those measurements.

- The Git installer was tested in project-root and nested repositories. It produced the exact packaged file hashes, staged changes on the intended branch, retained a workspace sentinel, and refused a dirty tree without changing its staged patch.
