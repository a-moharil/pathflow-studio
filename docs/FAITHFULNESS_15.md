# Pathflow Studio 0.11.1 / Update 15

This release corrects the residual geometry rule and intervention topology while preserving the 0.11.0 application features. No training architecture, checkpoint tensor key, optimizer state or dataset format changes. Saved results retain their original definitions. Recompute analyses to obtain the new measurements.

## Source definitions and a correction to the earlier assessment

The supplied **Architecture-General Residual Carrier Measure** defines exact finite conditional response paths and their squared-amplitude probability measure. It does not fix a unique structural reference or flow-to-length ruler.

The manuscript **MLP_Structural_Disposition_to_Conditional_Realization_Exhaustive(1).pdf**, equations 12–26, explicitly defines structural path gain as the product of squared weights. Section 26.4 gives the architecture-general comparison-class enrichment. Therefore raw squared products are not intrinsically unfaithful, and the earlier blanket suggestion that He normalization was required was too broad. In a sequential graph, constant factors common to every route cancel from normalized probabilities. In a residual DAG, unequal route depths can prevent that cancellation.

## Structural measure

The default `raw_affine` convention uses source factor 1 and a product of squared coefficients along each complete route. Learned convolutions and dense maps use their actual weights, evaluated BatchNorm uses its fixed affine slope, and deterministic averaging, preprocessing, readout and additive merge coefficients remain explicit. Activation response coefficients and intervention attenuation are replaced by 1 in the structural calculation. Biases cancel in affine contrasts but remain in checkpoint provenance and affect conditional activation responses.

The selectable `he_normalized` convention replaces each learned convolution/dense coefficient by `w / sqrt(2/fan_in)`. Fan-in is the full kernel area times input channels per group, or linear input width. Fixed affine operators and merge coefficients are not He-scaled. This is an explicit DAG structural convention, not a claim that the residual conditional guide uniquely requires it. It changes neither model predictions nor conditional path amplitudes.

For max-pool structure, amplitude `1/k` over the k valid candidate coordinates is a declared input-independent reference, squared during positive propagation. Conditional max pooling continues to use exact upper-envelope interval coefficients. These two quantities must not be conflated. Neither structural convention claims invariance under arbitrary functionally equivalent architecture rewrites.

## Exact conditional measure and audits

For an edge coefficient R_e obtained from paired execution, complete-route amplitude is input contrast times the product of R_e. Positive forward/backward messages use |R_e|². Additive branches retain separate carrier occurrences before squaring. No complete routes are enumerated in production.

Every new residual/healthcare analysis records:

- independent signed reconstruction and tolerance;
- positive forward/backward partition agreement;
- scalar-node incoming/outgoing marginal conservation, checked with tensor operators;
- normalization on actual source/sink cuts;
- merge visit mass and branch allocations, without assuming optional merges have unit visit probability;
- explicit undefined measure when the partition vanishes, without epsilon normalization;
- parameter, execution-graph and endpoint hashes, numeric dtype, evaluation mode, software/rule versions, structural convention and metric settings.

Direct analysis calls promote an isolated model copy to float64 when necessary, use evaluation state, and restore the original per-module training modes. Invalid nonnegative messages fail explicitly. The audits do not require explicit scalar graph materialization.

## Metric construction

For full scalar geometry, K is every scalar execution occurrence entering one operation instance, across all of its input branches. For example, an additive merge groups all incoming branch occurrences into one class. For positive class mass:

`enrichment(e) = |K| F_e / sum(F_a for a in K)`

`length(e) = (enrichment(e) + epsilon)^(-power)`

This is the section 26.4 rule. Each class has mean enrichment 1, but an optional branch or operation is not falsely claimed to carry unit path probability. Metric normalization does not replace the original displayed marginal F_e. A node-wise nonlinear operation may have its own declared class, since this is the actual unfused execution graph.

If an entire class has zero mass, enrichment is recorded as null. For a finite metric on the fixed carrier, the app explicitly uses the zero extension ONLY in the length rule, giving epsilon^(-power). No positive path mass is invented. A globally undefined conditional measure has no conditional fit. A globally undefined structural measure has no paired geometry.

Epsilon and positive metric power are selectable, validated and recorded. Their defaults remain 0.03 and 1 for continuity, not because they are unique theoretical constants. Geometry is a declared observable family, not uniquely forced by the path measure.

Parallel execution edges retain separate probability marginals. Undirected shortest paths use the minimum length among parallel alternatives. Sparse matrix conversion must not sum these lengths. This correction also applies to the shared embedding backend.

Full scalar scope includes the complete selected-readout carrier, including zero-weight or zero-conditional-mass occurrences that remain architecturally admissible. Edges outside every complete selected-readout route are excluded using topology support, with the excluded count reported. Display scope and comparison classes are exported. Conv1 uses its displayed occurrence family as its declared class and is explicitly an induced metric. Operation quotient geometry remains an aggregated view and never substitutes for full scalar BCANw.

## Intervention and history integrity

An intervention now multiplies the existing activation response coefficient and endpoint activations. It no longer inserts an additional gate node. Thus node/edge identities and the selected-readout structural measure stay fixed during attenuation; the conditional measure and resulting metric can change.

New histories match the analysis rule version plus structural and metric settings. Old topology-reference analyses remain viewable and exportable but are not mixed into new-version history curves. Changing a selector does not rewrite an old result; Compute analysis is required.

## Remaining scope limits

Full scalar materialization still has the declared 2,500-node / 200,000-occurrence budget. Full ResNet-50 tensor marginals are supported, but full high-resolution ResNet-50 all-pairs scalar geometry is not claimed. Euclidean and Lorentz representations remain approximate fits with their diagnostics. BCANw summary values retain the existing qualification rules and expose their profiles; no universal dimension or clinical claim follows from passing these tests.

Preserved controls include both geometry panels, 2-D/3-D and Lorentz options, shared anchor/removal, labels, node sizes, zoom, ghosts, flow controls, recording, checkpoint playback, contrast/reference selection, field transforms, region tracing, channel interventions, IG/Grad-CAM, training controls and deletion. See validation notes for the exercised subset.
