# Synthetic field datasets · 0.16.2, revised in 0.16.3

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

## Measured comparison (rendering version 2, 0.16.3)

Seed 3, 1,024 training images, 800 Adam updates, ambiguity 0.35. MLP: 5 × 16 ReLU,
learning rate 0.002. CNN: channels 6, 8, 10 with the 1×1 convolution + global average
readout, learning rate 0.003. Analysis at update 800: first test image of class 0 against
the first of class 1, all outputs. s = structural, c = conditional. "Zero-c edges" is the
share of edges with zero conditional mass; "units carrying c" counts, per hidden layer,
the units with at least one outgoing edge of positive conditional mass. One run each:
recorded values, not population estimates. Every node and edge is drawn in all cases.

| Run | Test accuracy | Zero-c edges | Units carrying c | H₁ s / c | S_def s / c | d* s / c |
|---|---|---|---|---|---|---|
| Bars · MLP 5 × 16 | 1.00 | 21% | 13, 10, 13, 12, 11 | 17.2 / 14.7 | 4.04 / 6.56 | 2.94 / 0.79 |
| Interference · MLP 5 × 16 | 0.98 | 44% | 9, 11, 13, 13, 10 | 17.6 / 15.8 | 3.99 / 5.85 | 1.95 / 0.70 |
| Orbits · MLP 5 × 16 | 0.93 | 39% | 10, 11, 12, 13, 8 | 17.3 / 14.9 | 4.64 / 7.05 | 1.97 / 0.82 |
| Bars · CNN | 1.00 | 20% | — | 13.8 / 11.8 | 1.98 / 3.94 | — |
| Interference · CNN | 0.99 | 32% | — | 13.8 / 12.0 | 2.35 / 4.17 | — |
| Orbits · CNN | 0.89 | 24% | — | 12.7 / 11.2 | 3.79 / 5.22 | — |
| Bars · MLP 4 × 64 | 1.00 | 30% | — | — | — | — |
| Orbits · MLP 4 × 64 | 0.95 | 48% | — | — | — | — |

d* is not reported for the CNNs because their carriers exceed the measurement budget and
were measured on landmarks.

### Why version 1 realisations were sparse

Version 1 (0.16.2) drew the patterns on a mid-grey ground: input mean about 0.5 with
modest contrast. That shared offset decided the sign of most first-layer
pre-activations. During training 10–14 of the 16 first-layer ReLU units of the orbits MLP
became inactive on every test image (3–6 at initialisation). The conditional measure then
correctly assigns zero mass to all edges into those units, and the 784 × 16 input edges
are most of the network: 85% of edges carried no conditional mass for orbits and 72% for
interference, and the orbits MLP reached only 0.69 accuracy. Nothing was hidden or
subsampled; the realisation was sparse because the trained network was.

Version 2 draws the bright positive phase on a dark ground (mean about 0.16, as for the
bars dataset). With the same networks, zero-mass edges fall to 39% (orbits) and 44%
(interference), 9–10 first-layer units carry conditional mass instead of 2–4, and orbits
accuracy rises to 0.93. Runs created with version 1 keep regenerating version-1 data.

### The dense ball (retracted in 0.17.1)

An earlier revision of this note described a ball-shaped conditional realisation for the
bars dataset with an MLP of width 64. That picture came from an unfinished fit (120
iterations, with a random restart that kept the round shape of its starting cloud). With
the settled fits of 0.17.1 the same networks give hub-and-rays realisations; no settled
picture in the 0.17.1 study was a ball. See `docs/SETTLED_PICTURES_31.md` and About ·
Reading a geometry picture.

## Suggested settings for demonstrations

- Interference, MLP 5 × 16, 1,024 images, 800 updates, ambiguity 0.35: learns well. Every MLP in
  the 0.17.1 study realised its cases as a hub with rays.
- Orbits, MLP 4 × 64, 1,024 images, 800 updates, ambiguity 0.35: learns well;
  in wide MLPs the rays turn freely about the hub, so read ray lengths, not angles.
- For a fan, use a CNN with the global average readout; SGD with momentum gives a wider
  fan than Adam (About · Field guide).
- Raise ambiguity towards 0.5–0.6 for broader, less defined conditional realisations;
  lower it towards 0.2 for crisper ones.
