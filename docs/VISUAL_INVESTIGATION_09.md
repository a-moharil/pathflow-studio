# Visual investigation 09 / Pathflow Studio 0.8.0

This release augments 0.7.0 without replacing the training, geometry, fitting, anchoring, Observatory or prior healthcare workflows. The main healthcare readout remains g = logit(1) − logit(0). Binary PneumoniaMNIST and synthetic experiments are supported here. This is research software, not clinical evidence of superiority.

## Walkthrough

1. Open Healthcare investigation → Visual evidence. Choose a saved checkpoint, split, input and reference in the existing context panel. Click **Compute maps & channels**.
2. **Case** shows endpoint images, true labels, predictions, threshold, scores and their difference. A zero reference is zero in model coordinates; no reference class is assigned. Same-class and opposite-class contrasts do not predetermine the measured flow.
3. **Maps** shows endpoint Grad-CAM, IG explanation and illustrative input interpolation, and Field A with signed diagnostics. Hover a pixel, click to pin it, or drag a rectangle. The companion selector offers signed path contribution, cancellation, positive squared mass and negative squared mass. Pink outlines identify the selected region. Clear resets to the whole image. A dedicated JSON export records the region, frame, series and context.
4. **Select mechanism** contains a clickable architecture strip and channel/unit gallery. The last convolutional layer is the initial visual layer. Gallery modes are activation at either endpoint, activation difference, conditional node occupation and signed Grad-CAM contribution. Dense units have no Grad-CAM. The gallery is explicitly the frozen baseline used to select support; dose-frame fields are displayed separately.
5. Click a channel/unit to select it. **Run channel/unit attenuation** gates that postactivation feature at both endpoints. **Load exact connections** computes the incoming occurrence identities for the chosen channel. Click a native destination cell to filter them, then use an exact table row as either one occurrence or its shared kernel parameter. The latter unfolds to every valid spatial reuse. Connections are paginated without discarding identities. Numeric/manual ranking remains in Advanced settings.
6. **Test response** shows frozen support/footprint and the selected dose curve for g(x), g(x°), Δg and cancellation. Random channel controls match one channel in the same layer; occurrence and parameter controls match the corresponding identity count per layer. Controls can overlap and are exploratory. Pixel comparisons form a separate intervention track.
7. **Check stability** runs compensated positive ReLU rescaling on the selected channel/unit, or the existing transformation sweep. Rescaling should preserve Grad-CAM, IG, output and fixed-support path fields while activation and hidden gradient scale reciprocally. It is an invariance audit, not a superiority demonstration.

Shared playback/recording/export controls remain close to the investigation. Full maps and support tables are retained in an expandable section. Advanced settings expose exact neighborhoods, explicit case comparisons, ranking, rescaling support policy and affine sweep configuration. Prior saved reports remain readable; missing new diagnostics are marked unavailable and require recomputation, never imputed.

## Signed path diagnostics

For complete finite-contrast path amplitudes aπ, P = Σ max(aπ,0), N = Σ max(−aπ,0), S = P−N, T = P+N, QΔ = Σ aπ² and C = 1−|S|/T for T>0. S reconstructs g(x)−g(x°). When T=0, C is undefined. When QΔ=0, normalized occupation is undefined.

The same quantities are collected at each exact input origin and summed across input image channels. For a region R, aggregate P and N first, then calculate S, T and C. Averaging pixel cancellation indices would conflate within-pixel and between-pixel cancellation and is not used.

A+(s) sums positive-amplitude squared path mass and A−(s) sums negative-amplitude squared path mass, both divided by the same global QΔ. Their sum equals Field A. Their colour scale is shared. The cancellation diagnostic uses linear amplitude totals, not squared totals. Signed finite-path contributions and IG are distinct allocations and remain separately labelled.

Implementation uses sign-separated log-domain backward propagation on the existing execution DAG for powers one and two. It does not enumerate complete paths. This is numerically stable for mass accumulation, but subtracting almost equal positive and negative totals can still lose relative precision in S. Reconstruction residuals expose that limitation. Raw overflow is null with log-origin totals retained; tiny raw values can underflow in float64. None of these conditions is replaced by an invented diagnostic. The legacy signed reconstruction remains available alongside the new audit.

## Attribution and Grad-CAM

IG remains the existing float64 trapezoidal integration along the straight reference-to-input path. Its sum and actual contrast are shown with an integration residual. The illustrative interpolation scrubber does not claim to have separately evaluated every displayed alpha. Intermediate images may be unrealistic; IG is not a deletion response. Absolute IG is explicitly unsigned and does not retain signed completeness.

Grad-CAM uses postactivation convolutional feature maps and spatially averaged gradients of the selected scalar readout. Each endpoint is evaluated separately. The positive rectification is standard Grad-CAM. Individual channel contributions are shown before that rectification, with signed colours. The optional overlay is a bilinear enlargement of native CAM cells; its native dimensions are reported. A layer of a dense MLP has unit activations and occupation, not Grad-CAM.

Map scales are held fixed across measured frames within the same quantity; distinct quantities keep separate units. Baseline Grad-CAM is labelled as baseline if a dose or transformation frame does not contain recomputed attribution. The region inspector does not substitute baseline IG for an intervened frame. Grad-CAM is recomputed in each rescaling frame, enabling its numerical invariance audit.

## Intervention semantics and Field B

Primitive Field B keeps the Evidence 08 definition. It is baseline conditional occurrence mass on selected primitive support, resolved at source coordinates. Only Conv1 gives an exact raw-image footprint. Deeper maps remain in native source coordinates. A postactivation channel intervention has a native node-occupation footprint, explicitly named **channel occupation footprint**, not Field B.

Channel gates enter finite activation secants, including the equal-preactivation limiting convention. The underlying weight-only structural measure is not redefined by a channel gate. Occurrence interventions continue to modify effective occurrence weights as in Evidence 08. Model copies/wrappers do not modify stored checkpoints or training parameters.

## Controlled demonstrations

**Reverse contrast audit** re-evaluates the same frozen model on swapped endpoints. Field A should remain invariant while signed IG and Δg reverse. Absolute IG also stays invariant. The endpoint labels swap with the displayed images. This is orientation invariance, not attribution failure.

**Constructed cancellation example** builds h1=ReLU(x1), h2=ReLU(x1), h3=ReLU(x2), and g=h1±h2+h3, with x=(1,1), reference=(0,0). It computes the same signed diagnostics and numerical IG using the production core. Both networks have Field A=(2/3,1/3), while one contains opposing routes. Their individual and joint attenuation curves are evaluated explicitly. The endpoint ReLU kink can produce a trapezoidal IG residual, which is shown. These are constructed different readouts, not findings on the selected medical checkpoint. Output-gain scaling and readout-sign reversal remain explanatory possibilities, not new automatic experiments in this release.

## Exports and compatibility

Full JSON includes signed-origin arrays, global diagnostics, endpoint channel catalog, exact connection identities, support settings and rescaling audits. Response CSV includes global P/N/S/T/Q/logQ/C and residuals when available. Selected-region JSON records its numerical allocation and selection rectangle. PNG/WebM composites include the legacy measured maps plus diagnostic and Grad-CAM maps. Figures preserve separate units; the composite is not a clinical report. Existing Plotly chart exports remain available.

No dependency additions or database migration are required. Old evidence reports carry their original schema; new fields are never fabricated on reload. Baseline integration is hash-checked and staged on a separate Git branch. Datasets, virtual environments and experiment workspaces are not included in or overwritten by the patch.
