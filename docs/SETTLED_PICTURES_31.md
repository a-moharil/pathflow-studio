# Settled geometry pictures · 0.17.1

## What was wrong

The geometry picture is a stress fit of the measured carrier distances. Until 0.17.1:

- the 2-D and 3-D pictures were two independent fits, each stopped after 120 L-BFGS
  iterations by default, well before the objective levelled off;
- when the unfinished fit still had an objective above 0.06, a random start was also
  fitted for 120 iterations, with seed 710 + dimension, and the smaller of two unfinished
  objectives was kept.

A random start that wins this way still has the round shape of its starting cloud. In the
54-network study of 0.17.0, 14 of the 19 recorded "balls" came from such restarts, and the
2-D and 3-D pictures of one case could be unrelated (Procrustes disparity about 1.0,
against 0.03–0.07 when no restart was used). Refitting the other five to convergence
removed them as well. The measured quantities (H₁, S_def, η₁, η₂, d\*, d_eff, routing
concentration) are computed on the graph and were never affected.

## What 0.17.1 does

Procedure `metric-fit-5` (`lab/core/metricfit.py`):

1. 2-D starts from the spectral layout; L-BFGS stops at its own tolerance or when the
   objective improves by less than 0.2% over 100 iterations (not before iteration 300).
   Ceiling 3,000 iterations per attempt (field "Euclidean max iterations / attempt").
2. 3-D starts from the settled 2-D picture plus the third spectral axis at one tenth of the
   2-D spread, and is rotated (no distance changes) so that its best shadow lies on the
   2-D picture. Diagnostic 8/16/32-D fits continue the chain.
3. A random restart is tried only if a settled objective stays above 0.06. It uses one
   nested draw for all dimensions, the full budget, and is kept only if its own settled
   objective is smaller. It was never triggered in the study below.
4. Each fit records how it started, why it stopped, its last 100-iteration improvement,
   and, for 3-D, the median node shift from the 2-D picture. The geometry card shows these.
5. Geometry requests carry `fit_version`, so a result from an earlier procedure is never
   reused. Saved analyses keep their pictures and are labelled as earlier-procedure.

The Lorentz restart uses the same nested draw. The joint (residual) engine uses the same
procedure. 3-D is oriented after the 2-D picture is aligned across checkpoints.

## Study rerun

The same 54 networks and 70 pair analyses (110 distinct analyses) were refitted with
`metric-fit-5`:

| Quantity | Result |
|---|---|
| Fits settled (plateau or tolerance) | 338 of 440 |
| Fits at the 3,000 ceiling | 102; last 100-iteration gain ≤ 2.8%; continuing five of them for 3,000 more iterations changed their shape by ≤ 6% (Procrustes) |
| Held-out stress, structural | 0.02–0.14 (2-D and 3-D) |
| Held-out stress, conditional | 0.04–0.17 (2-D and 3-D) |
| Random restarts triggered | 0 |
| 2-D → 3-D median node shift, CNN conditional | ≤ 0.07 |
| 2-D → 3-D median node shift, MLP 4×64 conditional | median 0.24, up to 0.98 (rays turning about the hub) |
| Conditional input isotropy | ≤ 0.25 in all analyses (0.17.0 "ball" criterion: > 0.6) |
| Time per analysis | about 15 s (MLP) and 30 s (CNN, 3,231 nodes), on 2 CPU cores |

The forms of settled pictures (About · Field guide) are filament and star for the
structural disposition, fan (every CNN) and hub and rays (every MLP) for the conditional
realisation. The CNN fan opens wider with SGD (57–65°) than with Adam (38–46°) or RMSprop
(38–48°), and its opening rises with routing concentration (ρ = 0.64). The results that
did not depend on the picture stand unchanged: the optimiser effect on CNN routing
(0.63–0.77 against 0.23–0.43, U = 36, p = 0.002 across networks) and the realisation being
more concentrated than the disposition in every analysis.

## Retracted

- The Ball form, and the claims built on it: "Adam CNNs form balls more often" (13 of 30
  against 4 of 30), "a ball means distributed routing", the conditional-ball recipe, the
  PneumoniaMNIST ball suggestions, and the ball section of `SYNTHETIC_FIELDS_27.md`.
- The forms tripod, sheet and lobe, bushy star and core and arms, which were classified on
  unsettled pictures; the guide now classifies settled pictures only.
- "The 2-D display uses the first two axes of the 3-D embedding" (`METHODS.md`).
- The stress ranges 0.57–0.88 (conditional) and 0.17–0.93 (structural) quoted in 0.17.0:
  they measured unsettled pictures.
