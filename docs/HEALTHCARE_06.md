# Healthcare Investigation · release 06 / version 0.6.0

Open an existing PneumoniaMNIST experiment, then choose **Healthcare investigation**
in its toolbar. Synthetic binary experiments also expose the page for validation.
Both MLP and CNN checkpoints are supported. Train to a committed checkpoint first.
The page is at `/healthcare/<experiment-id>` and adds no runtime dependencies.

## Workflow

1. Choose a saved checkpoint, test or validation split, input index, and either
   another image or the zero reference. Next pair selects a different reference
   in the requested same-class/across-class group and immediately analyzes it.
2. Analyze case. Inspect the input, reference, class labels, uncalibrated model
   probabilities, logit margin, contrast and exact signed reconstruction audit.
3. Fields & pathways shows structural and conditional Field A, their difference,
   signed contributions, intrinsic concentration and layer/channel marginals.
4. Intervene & transform runs internal attenuation with matched controls, input
   region perturbations, or affine transformations with measured frame playback.
5. Cohort evidence evaluates the entire chosen split for predictive metrics and
   a seeded sample for intrinsic observables. Click a sampled case to inspect its
   original-image contrast with the cohort's reference and threshold.
6. Export JSON, CSV or a field-panel PNG. Completed investigations are persistent
   jobs and can be reloaded after a browser/app restart.

The measured context is printed separately from editable controls. Editing an
input invalidates pending display updates. Cancel removes queued jobs or prevents
rendering a running job; running calculations finish and remain in the history.
Training uses the existing serialized worker, so pause/resume remains on the
experiment page. Healthcare jobs do not train, mutate checkpoints or select the
best checkpoint automatically.

## Mathematical definitions

The selected scalar readout is g(x)=f_1(x)-f_0(x). For PneumoniaMNIST these are
pneumonia and normal logits, respectively. The app evaluates both inputs, including
the actual zero input when requested. Contrast is g(x)-g(x0).

Field A at an input location sums the normalized complete-path probabilities for
paths originating there. Structural probabilities use weight-product squares;
conditional probabilities use squared finite path amplitudes, including input
difference and activation secants. Channels are summed at image locations.
Each defined Field A sums to one. Brightness does not establish a large response.

Signed contrast contributions sum signed complete-path amplitudes by input origin.
The sum over pixels reconstructs the selected finite output contrast. The backward
linear path recurrence uses the same finite secants as the exact contrast
reconstruction. This field is signed, unnormalized and is NOT Field B.

Complete-path Hellinger is computed through the Bhattacharyya coefficient. The
cross partition has input factor |dx_i| and edge factor w_e² |s_v|, the geometric
mean of structural and conditional unnormalized path factors. Log partitions
provide BC=Z_cross/sqrt(Z_structural Z_conditional), and H=sqrt(1-BC). This is a
complete-path divergence, not merely a distance between the two input fields.

H1, eta1, eta2, entropy deficit, logQ and cancellation retain the existing core
definitions. Hidden-layer node marginals are destination marginals at that cut.
CNN channel masses sum spatial occurrences; MLP bars correspond to hidden units.
Effective unit count is exp(entropy); participation is 1/sum(mass²). These are
layer-marginal summaries and are distinct from complete-path concentration.
Undefined normalized measures remain null. Constant output contrast can also
result from cancellation even when path mass is defined.

The exact path-math engine is unchanged. The new module is checked independently
by complete-path enumeration for a small positive-ReLU MLP, and by signed
reconstruction for MLP/CNN, ReLU/tanh and attenuated gates.

## Interventions and transformations

Internal interventions target one channel (all spatial units in it) or one MLP
unit. Alpha=1,.75,.5,.25,0 is applied after activation to both members of the pair.
The highest conditional-mass target is selected from the native analyzed contrast,
or an explicit index can be chosen. Up to five other targets from the same layer
are selected without replacement, seed 605, as matched controls. Their unit counts
match the selected target. If the conditional measure is undefined, automatic
selection is refused; manual selection remains possible.

Region perturbation interpolates the transformed input pixels toward corresponding
transformed-reference pixels; the reference stays fixed. Bounds are integer pixel
coordinates, right/bottom exclusive. It reports original regional conditional mass
and signed contribution. It is an input perturbation and may create an unusual
image. The displayed dose field uses the actual perturbed image as its background.

Affine investigations transform both original input and reference from their
original arrays. They use the existing bilinear image resampling and forward
probability-mass transport. H_eq compares recomputed and transported conditional
fields; H_inv compares recomputed and original conditional fields. Retained/lost
mass is reported before renormalization. The four panels share a linear scale
except the signed residual, which has its own symmetric scale. Sweep frames are
computed independently at a fixed checkpoint; stopping does not fabricate frames.
Clinical plausibility of a transformation range is not assumed.

## Cohort statistics

All images in the selected split are evaluated without intervention or transform
for accuracy, sensitivity, specificity, precision, balanced accuracy, ROC-AUC,
average precision, Brier score, NLL and 10-bin ECE. The explicitly selected
probability threshold affects confusion-derived metrics only. ROC uses tied-score
thresholds; AUC uses average ranks; average precision uses the stepwise recall
increments. Single-class undefined metrics remain null. Calibration is evaluated,
not fitted. Test data never drive parameter updates in this page.

Metric intervals are percentile 95% intervals from 200 ordinary image bootstrap
resamples, seed 602. Replicates with undefined statistics are excluded and counted.
These intervals do not account for unknown patient-level clusters.

Intrinsic sampling is uniform without replacement, seed 604, from the split,
with 1..128 selected images. Reference is zero, same-class, or opposite-class;
dataset references exclude the image itself. Reference identities are exported.
Original-image correctness defines TN/FP/FN/TP groups. The configured transform
is additionally evaluated on each sampled pair for H_eq, H_inv and margin change.
Reference-dependent concentration, input difference and contrast are exported.
Group-mean intervals use 200 image-row bootstrap resamples, seed 603, and are
explicitly exploratory because references can repeat and patient IDs are absent.
Groups with fewer than two defined values have no mean interval. Histograms are
normalized within outcome group. Log view excludes nonpositive/undefined values;
each legend reports the retained count. Probabilities are per bin, not densities.

## Scope and interpretation

PneumoniaMNIST is a 28×28 research benchmark, not a clinical deployment validation.
Input-origin path mass is not a lesion segmentation. Another subject's image is
not a patient-specific disease-removal counterfactual. Concentration and divergence
are not calibrated confidence scores or evidence of clinical correctness.

Primitive-occurrence intervention Field B is still explicitly marked pending.
This release does not substitute another map for that theoretical quantity.
BCANw, distance matrices and Euclidean/H3 embeddings remain on the existing
Training & geometry page. For test contrasts the healthcare link preloads the
exact checkpoint, pair, readout and transform; Analyze checkpoint computes the
metric geometry under its existing budget and fidelity diagnostics. Validation
contrasts are explicitly labeled as not transferable to the existing test-only
geometry controls. No healthcare result is silently pooled into the Observatory.

## Compatibility

No existing experiment data is migrated or deleted. Healthcare results use their
own version 1 and job kinds. Existing geometry cache version 6 is retained because
its numerical implementation did not change. Dataset, training, optimizer,
path-math, geometry and embedding modules remain byte-identical to 0.5.2.
The package's integration guide and integrity manifest document supported Git
baselines and changed files. Commit an existing update before running the installer.
