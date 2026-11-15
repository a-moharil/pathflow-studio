# Pathflow Studio 0.11.0 — residual families and source-field interpretation

## Entry points

- `/design`: MLP, CNN, Residual network; classification or regression.
- `/healthcare`: domain workspace entry. Its setup adds BUS-UCLM/imported masks, segmentation and joint classification/segmentation.
- Existing experiment URLs, MLP/CNN healthcare investigation, saved checkpoints and dataset folders remain supported.
- Residual models use the established worker, checkpoint, geometry renderer and field investigation workflow. Storage namespaces remain compatible; this release does not rewrite existing databases or historical numerical results.

## Architectures

Small residual preset: RGB input, stride-2 stem, stage widths 4,8,12, one two-convolution block per stage, explicit projection at shape transitions, optional hidden dense head. Default 8×8 resolution is an engineering scalar-inspection setting, not a medical imaging recommendation. The measured default classification graph has 1,251 scalar nodes and 8,950 occurrences.

Custom residual: stage widths and depths, 2–4 additive branches, ReLU/tanh, preactivation choice, global average pooling or flattening, optional dense hidden widths. Editing block controls switches the UI from small preset to custom. ResNet-50 is a separate standard preset with its standard ReLU/BatchNorm and global average pooling, optional classifier hidden layers and optional ImageNet initialization. Custom stage controls do not change the ResNet-50 preset.

Native CNN: flatten the final feature map, optionally append the user-defined hidden MLP, then project to the task output. The checkbox is enabled for new designs. Missing `cnn_head` in old configurations means disabled, preserving the exact old checkpoint architecture. Channel intervention scope applies only to convolutional layers; dense layers have unit and layer interventions.

## Tasks and datasets

Existing synthetic bars, Fashion-MNIST and PneumoniaMNIST classify their existing labels. Regression uses a new deterministic spatial dataset: target = mean intensity of the left image half minus mean intensity of the right half, on the original 28×28 image. MLP/CNN and residual models use MSE, a single unbounded output, and no classification accuracy. Resizing for residual input does not recompute the target; this is recorded. This release does not add a new external real-world regression dataset.

Residual classification adapters reuse prepared dataset splits and source identities. Images are encoded as 8-bit grayscale, replicated into RGB and letterboxed to chosen size. Small/custom residuals use raw [0,1] inputs. ResNet-50 includes ImageNet affine normalization in the analyzed graph. The older healthcare architecture retains its checkpoint definition and decoder. No mask losses are attached to general regression/classification-only models.

## Execution and numerical contract

New general residual models are traced using torch.fx. An interpreter executes the same traced program for learning, endpoint evaluation, attribution and finite response recording. Module names are labels, not analysis rules. Reused modules receive unique execution names and stable shared weight-tensor identities.

Supported exact rules include Conv2d (including stride, padding, dilation and groups), Linear, eval BatchNorm2d with running statistics, ReLU/tanh finite secants, exact switching MaxPool2d within the existing supported configurations, average pooling without ceil mode, global adaptive average pooling, flatten, identity and additive merges. Existing healthcare handlers retain concatenation and nearest-neighbour resize. Unregistered operators fail explicitly. Arbitrary data-dependent Python control flow and unrestricted custom Python model uploads are not offered by the builder.

Positive tensor messages propagate entrywise squared local coefficients. Branches remain distinct. No path enumeration is used in production. Complete-carrier H1, entropy deficit, eta1 and eta2 are obtained from positive messages and log-weight expectations; Renyi partition uses fourth-power coefficients. The topology count is the count of admissible architectural routes, including zero learned coefficients. Unequal residual route lengths are retained. Summed edge occupation is expected route length, not unit mass. ReLU equal-zero secant is declared zero. A vanishing partition remains undefined.

Signed propagation is audited against actual endpoint contrast; failed reconstruction rejects the analysis. Export includes parameter hash, rule version, dtype, evaluation policy, equal-point convention and tolerance. Signed source contribution is computed by independent signed backward propagation. P and N are separate route-amplitude totals; subtracting large deep-network totals can lose precision. Their cancellation diagnostic must be read alongside the independent residual.

## Structural and geometric scope

The conditional finite-response construction follows the supplied residual carrier guide. The existing raw-affine structural reference remains explicitly experimental. It uses unit source strengths, no activation attenuation, actual fixed affine/weight coefficients and the existing uniform-valid structural max-pool convention. It is not relabelled as a validated He-normalized residual structural measure.

The residual intrinsic edge-length convention remains `1 / (F / F_topology + .03)` when the topology mass is positive, and `1/.03` otherwise. This is recorded, separately from legacy cut-relative lengths. No scientific equivalence between the two conventions is claimed.

- Full scalar mode explicitly materializes requested scalar occurrences under 2,500 nodes and 200,000 occurrences. The implicit tensor engine remains the default computational representation.
- Its distances use the complete scalar carrier, and BCANw profiles/qualification are computed on those intrinsic distances, independently of display coordinates.
- Above budget the result says `budget_exceeded`, never substitutes an operation quotient.
- Conv1 scope is an induced occurrence subgraph and is labelled accordingly.
- Operation view is an aggregated summary; its distances are not full scalar-network distances.
- Existing independent Euclidean 2-D/3-D and optional Lorentz 3-D/Poincare-ball fits, shared anchor toggle, labels, flow controls, ghost controls, local next-pair/update/intervention buttons, motion and recording remain available. Fit stress is visible; a rendered network is not proof of low distortion.

Full all-pairs ResNet-50 scalar metrics and BCANw remain beyond the local explicit budget. Full-model tensor marginals, source fields and branch summaries remain available. There is no fabricated full-network geometry.

## Field A interpretation

Linear, square-root and log-positive display mappings change colour only. Original pixel probability is retained in hover/export. Baseline/intervention share the same maximum. Log display spans eight orders below that maximum; smaller positive mass is visually clipped, not numerically discarded.

50%, 80%, 95% mass sets rank source pixels and include all threshold ties. Boundaries follow pixel-set edges, not interpolated anatomical curves. Actual enclosed mass and occupied image fraction are reported. Effective source-pixel count is exp(source entropy), explicitly distinct from path entropy.

Signed source contributions, sign-separated route mass, P/N and cancellation remain separate maps. Field A measures normalized squared route mass; IG is a signed input-path attribution; Grad-CAM is endpoint gradient-weighted evidence in native feature coordinates; Field B is baseline intervention-support occupation. None is labelled a clinical diagnosis.

Click a Field A pixel or specify a source rectangle to trace its routes. Absolute results retain whole-network normalization; within-region allocation divides by the original source-region mass only when positive. Residual views provide downstream operation bars, native feature-space occupation and incoming branch comparisons. MLP/CNN fields provide exact native destination-coordinate mass by layer. Region controls trigger an actual recomputation. The source rectangle is independent of a mask readout rectangle and remains in the displayed input coordinates under transforms.

Intervened Field B is frozen to baseline support mass. Post-intervention selected-support occupation is exported under a separate name. Fields, geometry and motion remain checkpoint-bound; moving pulses encode direction, never a physical propagation rate.

## Compatibility and validation

See VALIDATION_14.md for executed checks. Existing legacy architecture definitions and measurement settings remain loadable. No dependency versions change. All old feature code stays in place unless the documented navigation move requires a new entry point. Scientific definitions and scope labels are exported, rather than silently revised on loading an old result.
