# Pathflow Studio 0.9.0 · Joint healthcare extension 10

## Scope and compatibility

This release adds `/joint` and `/joint/<experiment>` alongside every existing MLP/CNN page. It does not replace the old graph engine, measurements, geometric estimators, fields, interventions, anchoring, exports or recording. New run records are stored in an additive `joint_runs` table and `workspace/joint/`; shared datasets are under `workspace/joint_data/`. Existing `runs`, measurements and checkpoints keep their definitions. Rollback code can ignore the additional table without deleting any data.

The new architecture is a torchvision ResNet-50 encoder plus a four-level nearest-upsample/concatenate/two-convolution ReLU decoder and a parallel global-average-pooling classification head. It is semantic foreground segmentation, not Mask R-CNN or instance segmentation. Decoder width is 16/32/64. Input resolution is 64/128/256/384/512, with identical letterboxing of image and target. A 64-pixel run is an engineering smoke test, not a high-resolution medical evaluation.

## Dataset

The built-in adapter targets **BUS-UCLM v1**, which contains ultrasound images, expert masks and patient IDs encoded in filenames. Source images are 1024 × 768; model resolution is separately recorded. Download source is the archive linked by the authors' repository. The automatic download returned HTTP 403 during this release's validation. The application reports the failure and provides an extracted-folder import plus a general CSV importer; it never substitutes generated data. Full medical-dataset download/training was not validated in this environment.

- Dataset v1: https://data.mendeley.com/datasets/7fvgj4jsp7/1
- Authors' tools: https://github.com/noeliavallez/BUS-UCLM-Dataset
- Primary paper: https://doi.org/10.1038/s41597-025-04562-3
- **Version-specific rights:** the v1 repository record lists CC BY-NC 3.0. The paper describes a CC BY 4.0 dataset; retain the licence attached to the version actually downloaded. No medical images are redistributed in this package.

The built-in classification rule is explicit: any red lesion makes an image malignant; otherwise a green lesion makes it benign; an empty expert mask makes it normal. Thus the two targets are annotation-derived, not independent evidence. Binary segmentation unions all lesion pixels. It is not a clinical diagnostic validation claim.

Download/extracted folder layout must contain paired `images/XXXX_NNN.png` and `masks/XXXX_NNN.png`. For other data use CSV columns `id,patient,image,mask,label,mask_encoding`, with paths relative to the CSV or absolute, labels 0/1/2, and binary or rgb-lesion masks. A missing mask is not assumed to be normal. At least six groups and all three classes in each partition are required. Patient groups are split deterministically 70/15/15 before resizing. Exact-byte duplicate images crossing partitions are rejected; this is not a perceptual duplicate audit. Source image checksums are retained.

The named synthetic dataset is generated explicitly by its own button for engineering checks, never masquerading as healthcare data.

## Training and recovery

Adam, SGD with momentum 0.9, and RMSprop (alpha 0.99, epsilon 1e-8, momentum 0) use torch.optim directly. Task choices are joint, classification-only and segmentation-only. Loss is weighted classification cross entropy plus binary mask BCE and soft Dice. Each component is stored independently. Hard Dice and IoU use a mask logit threshold of zero; empty prediction/target pairs score one. Letterbox padding participates in segmentation loss and metrics; it is not silently cropped out.

BatchNorm running statistics remain fixed during training in this version; trainable affine parameters may still update if the encoder is unfrozen. Pretrained initialization can download ImageNet V2 weights. Input normalization is the fixed ImageNet affine transform and is inside the analysed graph. Random initialization is also supported.

Latest checkpoint, optimizer state and update are saved after each update. Historical checkpoints and train/validation/test evaluations occur at the selected interval and the target update. Evaluation limit 0 evaluates the full split; a positive limit uses a deterministic leading subset, explicitly labelled with evaluated and population counts. Training loss at every update is a minibatch statistic. The best checkpoint is selected only from validation by the chosen trained-task metric.

A typical Adam checkpoint is approximately 277 MiB with decoder width 16; each historical checkpoint adds storage. Pause is honored after the active update/evaluation. Worker restart pauses training. Analysis queues pin `latest` to its actual update before execution. Deletion is handled by the serialized worker, cancels queued run work by removing its job records, and preserves shared datasets.

CPU bootstrap remains supported. CUDA training requires a separately installed compatible CUDA-enabled PyTorch build; the CPU bootstrap does not create one. Pair analysis currently runs in CPU float64, even for CUDA-trained checkpoints. The analysis budget is a conservative estimate, not an operating-system memory reservation.

## Mathematical conventions

`finite-dag-1` assigns local finite coefficients so that delta h_v = sum_e b_e delta h_u. Convolutions and linear layers use actual coefficients; affine offsets cancel in contrasts while affecting nonlinear endpoint states. Evaluation BN uses gamma/sqrt(running variance+epsilon). ReLU uses endpoint secants, with the positive-side derivative convention when endpoints agree. Residual addition retains each branch with unit coefficient. Concatenation routes coordinates exactly. Nearest upsampling has one source per output. Global average pooling and scalar readouts retain their actual coefficients.

Max pooling uses the local straight interpolation between its two input tensors. Each candidate receives the fraction of the interval on which it is maximal. All pairwise crossing times are considered; ties use the lowest candidate index. Padding candidates are excluded. This provides an exact local finite identity but is not a claim that this local interpolation equals the hidden trajectory induced by IG's input interpolation.

A complete path amplitude is delta x_i times the product of local coefficients. Conditional probability is its square divided by Q, the sum of all squared complete-path amplitudes. Forward/backward nonnegative tensor messages carry logarithmic scale factors, so normalization does not require representing enormous path counts. Marginals are probabilities in the complete readout path space, not normalized per layer. A residual convolution can have total occupation below one. Both heads have separate normalizers and scalar questions.

`raw-affine-carrier-1` is an explicit **new structural convention**, not an assertion that the unresolved He-normalized residual extension has been settled. It keeps learned affine coefficients, fixed preprocessing/pooling/readout coefficients, removes conditional ReLU and intervention attenuation factors, starts with unit input mass, and uses equal 1/K coefficients over K valid max-pool candidates. No fan-in/He factor is silently added. Bias and BN offset effects enter conditional endpoint states, not separate constant-origin paths.

The topology reference gives coefficient one to every permissible scalar occurrence, including currently zero learned edges, and retains readout support. It yields uniform complete-path probability for that readout. The **experimental** metric `topology-reference-1` uses edge length 1/(F_e/r_e + 0.03), where r is this topology marginal. This is separately versioned and does not overwrite the legacy cut-based metric. It must not be conflated with a finalized extension of the research theory.

## Fields and interpretation

Readouts are a classification logit/margin or the mean segmentation logit on a fixed output rectangle (a 1×1 rectangle selects one pixel). Reference is another held-out image or zero in raw [0,1] input coordinates. Region selection is held fixed at both endpoints and across intervention doses.

Field A sums conditional origin mass over image channels. Sign-separated positive/negative squared masses sum to Field A. P and N sum positive/absolute-negative path amplitudes; cancellation is 1-|P-N|/(P+N), undefined when denominator is zero. Export uses -1 only as an explicit undefined cancellation sentinel; zero energy makes the conditional field unavailable. Log Q and original output contrast are displayed alongside normalized occupation.

**Precision:** P and N can be enormously larger than their difference in a deep signed network. Even logarithmic scaling cannot recover significant digits lost when nearly equal P and N are subtracted. Both this subtraction's residual and an independent signed finite-contrast propagation are exported. Use the latter for the output reconstruction audit. This is a floating-point limitation, not proof of exact cancellation.

IG is trapezoidal input-line integration with 4..128 intervals and an explicit completeness residual; low interval counts can have visible quadrature error. Input gradients and native-resolution endpoint Grad-CAM are also exported. They address different questions from normalized path occupation. No invariance superiority claim is made.

A visual gallery selects post-ReLU channels. Alpha attenuates the selected channel at **both** contrast endpoints. Both heads are recomputed; scalar dose curves at alpha 1/.75/.5/.25/0 are measured. Baseline Field B is available in raw input coordinates for selected stem-channel incoming Conv1 occurrences. Deeper selected channels have a native-coordinate occupation footprint, not an invented raw-image Field B. Other primitive/parameter/pixel mechanisms remain available in the original small-network workspace; they are not advertised as implemented for ResNet in this release.

## Geometry limits

Full-model tensor marginals are computed for both heads, but a full scalar ResNet metric embedding and full-network BCANw are **not implemented**. Their absence is not represented by zero-valued dimensions.

The new geometry option materializes all valid Conv1 incoming occurrences into a central 3×3 destination window of one output channel. It retains the complete-model marginals on these edges. A deeper feature selection uses stem channel zero for this separate geometry preview, recorded in the result. The metric graph is undirected for distance calculations. Paths are restricted to this induced subgraph, so these distances need not equal full-network shortest paths.

Two and three dimensions are fitted independently from classical MDS initializations, followed by 100 steps of full-pair, scale-balanced distance-stress refinement. Relative stress and the original subgraph distance matrices are exported. A display warning above 0.2 relative stress is a heuristic visibility aid, not a scientific fit-qualification criterion. Conditional coordinates are aligned by a rigid transform only. There is no repulsive layout distortion or visual spreading to hide collapse. The iteration budget is not a certificate of convergence or fidelity. The common centred reference can be disabled for independent camera controls. Node size, all-node labels, structural ghost and thin square-root mass edge colour/width controls are provided. Independent canvas cameras provide orthographic 3-D rotation, pan and zoom; all-node labels avoid a WebGL text dependency. Optional directional pulses have constant decorative speed, not a measured velocity. PNG export and WebM recording are available. Hyperbolic views and BCANw remain available in the legacy workspace, not extrapolated to the ResNet subgraph.

## Validation and use

See VALIDATION_10.md for executed checks and explicit untested areas. Run `python -m pytest -q` from the project. Browser checks need Playwright, which is intentionally not a runtime dependency.

Start with Joint healthcare → prepare synthetic engineering check → resolution 64 → 1 update → evaluate 3 samples/split → compute both readouts with IG 4. This exercises the complete workflow quickly. Then import the medical dataset, choose meaningful resolution and evaluate all held-out cases. Do not treat engineering checks as evidence of clinical benefit.
