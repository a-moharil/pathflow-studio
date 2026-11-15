# Visual evidence / Pathflow Studio 0.7.0

## Access and workflow

Open an existing binary PneumoniaMNIST or synthetic experiment → Healthcare investigation → Visual evidence. Choose the checkpoint, split, input and reference at the top. No retraining is needed. The readout is g = logit(1) − logit(0), consistently used for attribution, conditional fields and interventions. Evidence jobs start from the original selected images; their transformation settings are independent of the older Intervene & transform controls.

Choose an investigation and Compute measured evidence. Presenter view hides the technical settings, and Exit presenter view restores them. Playback advances saved measured frames. JSON contains the full measured context, arrays, support identities and audits; CSV contains response summaries; PNG and WebM capture the displayed map panels. Plotly charts retain SVG export. Saved investigations can be reopened through the existing history selector. Running jobs are serialized and are not interrupted by Cancel display; queued cancellation and stale-display rejection follow the existing workflow.

## Definitions

Field A is the input-origin marginal of a normalized complete-path distribution. Its conditional version concerns c=(x,x°); it is not a magnitude of the output response. Δg, log QΔ, concentration, cancellation and reconstruction residuals are distinct quantities.

For selected first-convolution occurrence support I, Field B is

    B_occ(s) = sum_{a in I, srcSpatial(a)=s} F_a^c.

For selected shared parameter IDs J, it is

    B_param(s) = sum_{a: rho(a) in J, srcSpatial(a)=s} F_a^c.

F_a^c is computed from the native conditional complete-path measure. Field B records baseline selected occupation, not a post-intervention map or activation. It retains exact layer, source/destination channel and spatial coordinates, kernel offset and shared parameter identity. Only Conv1 is displayed in raw-image coordinates. Deeper source-channel maps are exported, with a channel-summed spatial view in their native feature-map coordinates. Input-image Field B is empty for deeper-only selections. MLPs report Field B as not applicable; their input fields, attribution, pixel interventions, rescaling and metric inspections remain available.

Raw selected mass is the default. Its sum is the captured mass on that cut; totals across multiple cuts must not be interpreted as the probability of a union. Shape-only normalization is optional and explicitly labelled. Baseline Field B stays fixed during attenuation. Undefined conditional measures are retained as undefined.

## Attribution

Vanilla input gradients are ∂g(x)/∂x, displayed signed. Integrated Gradients uses the same x° and g, with composite trapezoidal quadrature over the straight input path, 8–1024 intervals, batched in float64. IG is signed and has readout-contribution units. Its sum, absolute mass and completeness residual are retained. A visible note requests more intervals if the residual exceeds max(1e−8, 1% of max(|Δg|, sum|IG|)). This is a numerical display criterion, not a theoretical accuracy guarantee. Completeness does not bound each pixel's integration error. The existing signed finite-path contribution field remains separately named.

Signed quantities have diverging colours. Probability fields have sequential colours. Each quantity has its own fixed scale over its measured sequence; incompatible physical/numerical quantities are not assigned a common map scale. All maps retain native cells without spatial smoothing.

## Support and intervention

Convolutional support can be chosen by conditional flow, absolute contrast parameter gradient, absolute weight, seeded random selection or manual identity IDs. Budgets apply per selected layer. Parameter gradients sum signed occurrence gradients before absolute value; they differentiate g(x)−g(x°). Ties are broken by identity index. Biases are not targeted.

Occurrence interventions attenuate specified edge uses before activation. Parameter interventions attenuate every valid spatial use of each selected kernel parameter. Both apply the same α to x and x°. Execution topology is retained, including zero-weight edges. Counterfactual path measures use the intervened effective weights; baseline Field B records the original model. This primitive intervention is distinct from the pre-existing postactivation channel/unit gates.

Primitive controls use three seeded random supports with the same selected identity count in each layer. Equal parameter counts can affect unequal occurrence counts because of spatial boundaries; all counts and baseline cut masses are reported. Random supports may overlap the selected support. These small control sets are exploratory, not significance tests.

Pixel comparisons select equal numbers of individual pixels using Field A, absolute IG, absolute input gradients and three seeded random masks. Masks are frozen before the sweep. x interpolates toward x° within the mask, while x° is fixed. These are pixel sets, not contiguous anatomical regions. The existing rectangular region controls remain available. Pixel and internal interventions are not presented as equivalent interventions.

The linked reuse comparison preserves associated parameter identities: an occurrence selection unfolds to all uses of its parameters; a parameter selection is paired with the highest-mass occurrence of each selected parameter. It illustrates scope, not equal-budget intervention performance.

## Rescaling audit

Supported hidden ReLU channels/units are rescaled by a positive a in [0.1,10], including their biases, and all outgoing uses are compensated by 1/a. Model copies are used. Tanh is explicitly rejected. Activations scale by a, their g(x) gradients by 1/a, and their products remain invariant. Input gradients, IG, complete-path marginals and fixed-support fields should be unchanged within numerical tolerance. Reselecting support after scaling is a separate mode; coordinate-dependent ranking rules can select different identities.

Audits retain output and map discrepancies, every marginal discrepancy and exact structural/conditional shortest-path distance discrepancies from input node 0. This is a one-source distance audit, not an all-pairs audit. Disconnected execution graphs produce an explicit diagnostic failure for metric audits. The invariance claim concerns this compensated positive rescaling within fixed topology, not arbitrary functionally equivalent architectures. Rescaling is an invariance check, not evidence of superiority over input gradients or IG.

## Movement and exact metric views

Transformation frames process both x and x°. Affine images use bilinear zero padding. Baseline probability transport is forward mass deposition. Contours show cell boundaries at 25/50/75% of the common Field A scale; they are not particle trajectories. Prediction/contrast curves and Hellinger mismatch use the same measured frames. Boundary loss is reported before renormalization.

Transported Field B uses raw baseline mass. Recomputed fixed-identity and reselected-support Field B are separate. Transporting a footprint is not an exact mapping of convolution occurrence identities under arbitrary rotation or scale. Native convolution kernels are not assumed rotation-equivariant. No transformation is assumed to preserve clinical meaning.

The neighborhood inspector computes exact shortest paths with the existing intrinsic length 1/(number of edges in cut × marginal + 0.03). It displays a union of up to 12 nearest nodes per measure plus the selected node, with structural/conditional distance matrices and a common raw-unit colour scale across native and intervened states. Clicking an image-space Field B pixel populates its input-node ID. Neighbor sets can change under intervention; inspect the node labels. No embedding is involved, and proximity is not by itself causal influence.

## Case comparison

Choose 2–8 explicit case indices with one fixed checkpoint and reference. All pairs are shown ranked by cosine similarity of absolute IG, with output-margin and η₁ gaps. Selecting a pair displays input, IG, Field A and measured intervention curves. CNN curves use the chosen primitive support and matching per-layer identity budgets. MLP curves use equal-count Field A pixels. True labels, probability, input source IDs, exact reference and selection settings are saved. This is a disclosed exploratory comparison; similar attributions and different vulnerability are not guaranteed or preselected outcomes. Cohort evaluation remains available on its own tab.

## Integrity

The only modification to the existing path-math core is a guarded optional occurrence-factor multiplication when constructing the execution weights. Native networks without this attribute use the original computation. Training, model definitions, optimizers, datasets, stored checkpoints, fitting budgets, geometry renderers, anchoring and previous healthcare jobs remain available. The package includes a baseline hash check, Git patch, full source and a changed/preserved-file report.
