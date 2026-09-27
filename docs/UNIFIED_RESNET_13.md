# Pathflow Studio 0.10.0 — shared ResNet workflow

## Integration
Design experiment now offers MLP, CNN and ResNet-50 in one page. ResNet controls appear inline. All runs open at /experiment/<id>. Prior /joint and /joint/<id> URLs redirect to shared pages. Storage and existing saved model state remain compatible. The residual engine remains an internal adapter; separate job/config schemas are retained so existing runs are not destructively migrated.

The common geometry renderer and metric-fit-4 solvers now serve ResNet too. Controls include independent 2D/3D cameras, optional Lorentz fitting and Poincare 3D ball, labels, anchors on/off, shared magnification, structural ghosts, deformation interpolation, node and flow size, directional motion, orbit, PNG and recording. Per-panel update, next pair and channel intervention controls use the same analysis request. Fields render safely at large array sizes and resize after tab switches. A disconnected Grad-CAM layer/readout is marked inapplicable, not assigned invented values.

## Data and tasks
Fashion-MNIST, PneumoniaMNIST and synthetic bars can train classification-only ResNet models. The existing preparation/split cache supplies images and original labels/indices. The adapter writes 8-bit grayscale PNGs (rounding cached values to multiples of 1/255), replicates into RGB and letterboxes to the selected resolution. ResNet input normalization remains inside the analyzed graph. Dataset-derived heads contain 10, 2 and 2 classes respectively. No decoder or segmentation loss is constructed for new classification-only runs. Old checkpoints retain their original architecture.

BUS-UCLM and synthetic masks support classification, segmentation and joint tasks. BUS-UCLM still uses the saved local data path, preserves patient splitting, and uses annotation-derived classes as documented. No working replacement for the 403 archive URL is claimed. Unannotated datasets reject mask tasks. The standard import manifest currently retains its original three-class schema.

## Geometry scope and faithfulness
Default: operation quotient, with each displayed edge equal to the sum of complete-path occurrence marginals for that operation connection. It includes residual shortcuts, decoder skips and heads present in the model. Edges unrelated to the scalar readout have zero occupation and remain in the carrier. This is NOT a scalar execution embedding. Its metric is length=1/(F/r+0.03) using corresponding topology-reference mass r, with a finite 1/0.03 length where r=0. No full scalar-network distance is inferred from a quotient distance.

Optional: central Conv1 induced occurrence subgraph, preserving the previously implemented scope and exact full-model marginals. Input-side nodes here are preprocessing outputs, labelled accordingly. No silent receptive-field projection of deep occurrences onto input pixels is used.

Both scopes provide their explicit lengths to the existing pair-covered Euclidean/Lorentz fitters. The recorded display unit is the median unweighted hop distance of the fixed scoped topology. It is independent of learned weights and input contrast and only rescales fit coordinates/targets; explicit metric edge lengths remain available. Ambient curvature applies in these recorded display units. Validation pairs, fitting coverage, stress, local/global diagnostics, deformation fidelity and numerical validity are retained. High distortion and failed hyperbolic fits remain explicit; successful rendering is not a fidelity certificate.

Structural convention remains raw-affine-carrier-1 (experimental); this release does not equate it to He-normalized path theory. Full ResNet scalar-network BCANw dimensions and path entropy/concentration histories remain unavailable. The ResNet Observatory plots computed log-Q/output contrast from matching saved analyses, up to the latest 20 jobs, with separate axes. It does not substitute quotient geometry for unavailable intrinsic observables. MLP/CNN observatories and their definitions are unchanged.

## Investigations
Dataset or zero reference; class-logit/margin and fixed mask-region readouts; signed path audit; Field A/B, signed input gradient, IG and native Grad-CAM; visual channel selection; post-ReLU channel attenuation at both endpoints; dose response for available heads. Baseline and intervened geometry can both be fitted. Field B remains baseline selected Conv1 support, while intervention response is separate.

Translation/rotation/scaling resample the original endpoints through the shared affine-transform implementation. A five-frame horizontal translation sweep recomputes each frame. A mask target is transformed with nearest-neighbor sampling. The mask readout region stays fixed in output coordinates, and clinical label invariance is not assumed. Legacy primitive/shared-parameter targeting and healthcare analyses remain available for the original MLP/CNN adapter; ResNet currently implements channel attenuation, not those additional primitive intervention types.

## Validation
The tests include a real synthetic ResNet classification optimizer update and analysis in Chromium; a joint-task update with both heads, channel intervention, and Poincare 3D view; a 512x512 renderer-only field fixture; disconnected Grad-CAM handling; standard-dataset cache fixtures for class-count verification; and preservation of legacy source behavior. No high-resolution BUS-UCLM clinical training run or live Fashion/Pneumonia download is claimed by these tests.

Release validation completed: 95 Python tests passed; Chromium classification workflow, joint two-head/intervention/Lorentz workflow, high-resolution renderer fixture, and legacy integration workflow passed without JavaScript errors. Git installer verified for 0.9.2 and 0.9.1 in root/nested layouts with workspace sentinels retained.
