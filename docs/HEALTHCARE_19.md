# Healthcare investigation workflow · 0.12.0

## One investigation, five questions

1. Examine the case: checkpoint, split, two explicitly labelled inputs, fixed binary
   margin and measured predictions in the native MLP/CNN healthcare page.
2. Locate the contrast: raw contrast and image overlays, Field A, a visual source box,
   signed regional amplitude totals and cancellation. IG/Grad-CAM comparisons expand
   on demand rather than competing with the first case view.
3. Follow the computation: exact source-origin route marginals in native layer
   coordinates, visual channel/primitive selection, and a link to metric geometry for
   the same test contrast. Where available, each gallery channel also reports the
   exact occupation originating in the computed source box. This does not change
   the channel's intervention semantics.
4. Test a mechanism: selected support and recorded response settings, measured dose
   curves, baseline/response Field A at shared series scale, and actual change in Δg.
   An alternative channel-ranking or input-region replacement protocol is retained
   in a separate expandable section. Tracing a source box does not itself attenuate it.
5. Assess stability: measured transformations, rescaling/reversal audits and an
   expandable cohort investigation. Transformations are not presumed clinical symmetries.

## Lifecycle and provenance

The native healthcare page keeps completed results in context-indexed slots. The key
contains experiment, checkpoint, split, input/reference, threshold, readout and transform.
Dictionary key order does not affect readout identity. Case, dose and individual evidence
modes remain separate slots. Changed controls label old plots explicitly; selecting a new
support marks an older response as belonging to its recorded support.

Only one request is submitted by the workflow at a time. Edits invalidate display of an
in-flight result, but a completed result remains saved. New reports include the exact
submitted request and workflow version. Loading saved results restores controls and
loads a compatible saved case baseline when one exists. Older reports remain readable;
a missing baseline or missing native maps must be computed explicitly.

Attribution/support evidence uses original input coordinates. The prior implicit reset
of transform settings is replaced by an explicit frontend guard and backend rejection
of a nonidentity case transform for an evidence request. The measured evidence transform
sweep remains supported. No transformed result is relabelled as an original-image result.

Presentation mode prevents new analysis submissions and exposes saved measured frames.
It does not generate or interpolate scientific results. The main response comparison,
endpoint diagnostics and inspector now update together during saved-frame playback.
Research controls, exports, support identities, audits and existing investigations remain
available in expandable sections outside the central presentation.

## Classification plus mask tasks

Joint healthcare experiments use the same five-stage navigation through their existing
DAG engine. Case/mask, contrast, route/channel and matched baseline/intervention views are
separated. The selected and recorded contexts, readout region, channel and attenuation
remain explicit. Memory budgets and metric conventions are expandable. Presentation mode
blocks new computations here too. Ordinary residual classification pages keep their
existing navigation. This is a separate engine adapter, not a claim that the binary
cohort protocol or primitive intervention catalog has been generalized to all joint tasks.

## Numerical integrity and limitations

No changes to path amplitudes, secants, marginal normalization, metric lengths, embedding
solvers, IG or Grad-CAM formulas. Native healthcare source tracing reuses the existing
exact pathmath source-region implementation; it does not crop the network or renormalize
the global Field A. Selected-source cancellation uses summed P and N, not averaged ratios.
Checkpoint files are unchanged by investigation jobs. Field B retains each engine's
existing declared support and spatial scope; route tracing and channel footprints are
not silently renamed Field B. Native binary readout remains logit(1)-logit(0).

The joint adapter retains existing transformation computations; it does not introduce a
new cohort estimator or new cross-frame scale convention. Geometry opens in its existing
workspace with its fit diagnostics. The release does not claim that every projection is
faithful or that healthcare maps are clinical lesion annotations.
