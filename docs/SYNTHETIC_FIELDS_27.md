# Synthetic field datasets · 0.16.2

Two synthetic datasets chosen for how their path geometry looks, selectable in **Build**
for MLPs and CNNs (`lab/fields_data.py`). They are 28 × 28 single-channel images, so no
network, measure or geometry code changes; only the number of outputs follows the
dataset (`config.class_count`).

| Dataset | Classes | Image |
|---|---|---|
| Synthetic interference | 3 · grating near 0°, 60°, 120° | windowed grating plus a class-independent second grating |
| Synthetic orbits | 4 · rings, spokes, right spiral, left spiral | polar wave cos(k·r + m·θ) plus a weaker wave of another class |

## Design properties

1. **Continuous latent factors.** Orientation, frequency, pitch and position vary
   continuously, so hidden units tile a continuum rather than a few cues.
2. **Shared components.** All classes come from one family of patterns (gratings; polar
   waves), so classes reuse units and their conditional routes partly overlap.
3. **Calibrated ambiguity.** `ambiguity` (0–0.8, default 0.35) widens the class-defining
   factor and strengthens the class-independent component. Several hypotheses stay
   active, so the conditional realisation of an input spreads across routes instead of
   tracing one crisp path.
4. **Symmetry.** Classes are related by rotation (interference) or by the sign and type of
   the polar wave number (orbits).

Generation is deterministic in (dataset, seed, training size, ambiguity); the cache name
records the ambiguity (`synthetic-orbits_42_512_a35.npz`). Cache names of the existing
datasets are unchanged. Residual networks do not use these datasets.

## Measured comparison

Seed 3, 1,024 training images, 800 Adam updates, ambiguity 0.35. MLP: 5 × 16 ReLU,
learning rate 0.002. CNN: channels 6, 8, 10 with the 1×1 convolution + global average
readout, learning rate 0.003. Analysis at update 800: first test image of class 0 against
the first of class 1, all outputs. s = structural, c = conditional. One run each:
recorded values, not population estimates.

| Run | Test accuracy | H₁ s / c | S_def s / c | d* s / c | d_eff s / c |
|---|---|---|---|---|---|
| Bars · MLP | 1.00 | 17.2 / 14.7 | 4.04 / 6.56 | 2.94 / 0.79 | 2.21 / 0.39 |
| Interference · MLP | 0.98 | 14.9 / 13.2 | 6.73 / 8.42 | 3.20 / 0.87 | 0.49 / 0.29 |
| Orbits · MLP | 0.69 | 14.9 / 13.2 | 7.00 / 8.73 | 3.40 / 0.58 | 0.42 / 0.22 |
| Bars · CNN | 1.00 | 13.8 / 11.8 | 1.98 / 3.94 | — | — |
| Interference · CNN | 0.99 | 13.4 / 12.1 | 2.80 / 4.04 | — | — |
| Orbits · CNN | 0.89 | 13.2 / 11.0 | 3.31 / 5.43 | — | — |

d* is not reported for these CNNs because their carriers exceed the measurement budget
and were measured on landmarks.

In the rendered geometry, the bars MLP's structural disposition collapses towards a thin
line, interference produces a fanned star, and orbits a dense radial star with many
partially shared routes; the conditional realisations stay visibly narrower than their
dispositions. In single 600-update runs of the same networks, interference MLP accuracy
fell with ambiguity (0.99 / 0.90 / 0.77 at 0.2 / 0.35 / 0.6); orbits MLP accuracy was
0.97 / 0.63 / 0.69, so not monotone there.

## Suggested settings for demonstrations

- Interference, MLP 5 × 16, 1,024 images, 800 updates, ambiguity 0.35: learns well and
  shows a symmetric fan.
- Orbits, CNN 6, 8, 10 with the global average readout, 1,024 images, 800 updates,
  ambiguity 0.3: the richest radial structure among these runs.
- Raise ambiguity towards 0.5–0.6 for broader, less defined conditional realisations;
  lower it towards 0.2 for crisper ones.
