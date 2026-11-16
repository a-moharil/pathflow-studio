# About Pathflow Studio

Pathflow Studio measures how a trained network routes its computation. It computes exact
path measures on the network's execution graph, derives a geometry from them, and records
every result with the conventions under which it was obtained. This page explains the
framework, the views of the studio, and a field guide to the geometric forms we observe,
with what each form does and does not imply.

## 1. The framework

### Execution graph and paths

Every affine or convolutional occurrence of a trained network becomes a node; every weight
occurrence becomes an edge. A path runs from an input occurrence through one occurrence per
layer to a readout. For a selected readout (a class margin, or all outputs), the studio
assigns two masses to every path π:

- **Structural mass** Q_str(π) = W_π², the squared product of the weights along the path. It
  depends only on the trained weights: it describes the network's **structural
  disposition**.
- **Conditional mass** Q_Δ(π) = A_π², with amplitude A_π = Δx_i · W_π · ∏ s, where Δx_i is
  the input contrast between an input x and a reference x° at the path's source and the s
  are the secants of the activations along the path. It depends on the case (x, x°): it
  describes the network's **conditional realisation** of that case.

Normalising either mass gives a probability over paths. Forward and backward log-sum-exp
messages compute the normaliser Z and every edge marginal F_e exactly, without
enumerating paths.

### Path observables

- **H₁** = log Z − E_p[log Q], in nats: the effective number of paths, on a log scale.
- **S_def** = log N_paths − H₁: the entropy deficit, how far the mass is from uniform over
  all paths. Larger means more concentrated.
- **η₁** = exp(H₁) / N_paths and **η₂** = Z² / (N_paths · Σ Q²): occupancy ratios.

In every one of the 124 analyses of the study in section 4, the conditional realisation is
more concentrated than the structural disposition: H₁ is lower and S_def higher.

### Derived geometry

The edge marginals turn the graph into a metric space. Each edge has length
1 / (n_e · F_e + 0.03), where n_e is the number of edges in its layer cut and 0.03 is the
recorded regulariser ε. High-mass edges are short; low-mass and zero-mass edges are long
(at most 1/0.03 ≈ 33). Shortest-path distances on this carrier define the geometry.

- **BCANw** box-covering gives a finite-scale profile; **d\*** is the largest qualified
  local slope and **d_eff** the slope at the half-coverage scale. They are finite-scale
  responses, not universal fractal dimensions, and are reported only when measured within
  the budget.
- **Embedding.** The picture you see is a landmark classical MDS of these distances. The
  2-D view shows the first two axes of the 3-D embedding. **Stress** reports how much the
  picture distorts the measured distances. A high stress means the picture is a strongly
  compressed view of a geometry that does not fit in two or three dimensions.

### Interventions and evidence

Interventions multiply chosen activations on a frozen checkpoint, for both x and x°, and
record the consequence on the readout. Statistics carry an evidential status (identity,
descriptive, exploratory, insufficient, unavailable). Audits reproduce recorded results at
their recorded conventions.

## 2. How the studio is organised

- **Library**: every network in the workspace. **Build**: design a network and prepare
  data. **Queue**: all requested work. **Evidence**: statistics across runs.
- For each run, the stage rail follows the investigation: **Overview** and **Training**;
  for the current case **Geometry** (structural disposition and conditional realisation
  side by side), **Realisation** (Field A over the contrast, routes, transformations) and
  **Observables** (path observables across checkpoints); then **Evidence** and **Audit**.
  **Clinical reading** presents the same case in clinical language for binary runs.
- A **case** is one input pair at one committed checkpoint. It travels between stages, and
  each stage opens its saved result for that case without recomputation.

## 3. Reading a geometry picture

- **Distance is mass.** Two nodes are close when strong-mass edges connect them. Nothing
  else positions a node: there is no force layout, jitter or repulsion.
- **Long spokes are low mass.** A node reached only through weak or zero-mass edges sits
  far out, because those edges are long. Arms and spikes therefore mark the parts of the
  network that carry little mass for this measure, not its main route.
- **The dense centre holds the strong routes.** For the conditional realisation, the core
  around the readout is where the case's mass actually flows.
- **Read the stress first.** In our study, conditional views had stress 0.57–0.88 and
  structural views 0.17–0.93. At these values the overall form is informative, but
  distances inside the picture are not faithful. Use the 3-D view and the measured numbers,
  not the picture alone.
- **Routing concentration.** Under each geometry the studio reports, for the input layer,
  the average share of an input's mass carried by its single strongest first-layer edge
  (1 = every input feeds one unit; low = spread over many), and the share of edges with no
  mass. This is the measured counterpart of the concentrating and diffusing behaviours
  below.

## 4. Field guide to geometric forms

We classify the forms observed in a study of 54 networks: three datasets (bars,
interference, orbits) × three architectures (MLP 4×16, MLP 4×64, CNN with channels 8·12·16
and the 1×1 convolution + global average readout) × three optimisers (SGD with momentum,
Adam, RMSprop) × two seeds, each trained for 800 updates on 1,024 images, plus five input
pairs on 14 of the networks (70 further analyses). The forms are recorded outputs of the
studio; none is constructed.

The forms lie along one axis, from **concentrating** (mass funnelled through few routes) to
**diffusing** (mass spread over many comparable routes).

| Form | Behaviour | Typical routing concentration | Where we saw it |
|---|---|---|---|
| Filament | strongly concentrating | high | structural, MLP trained with SGD |
| Tripod | concentrating | high | structural and conditional, CNN trained with SGD |
| Sheet and lobe | concentrating, split | high (CNN SGD: 0.63–0.77) | conditional, CNN trained with SGD |
| Bushy star | intermediate | moderate | structural, MLP trained with Adam or RMSprop |
| Core and arms | mixed | moderate | conditional, MLPs and CNNs |
| Ball | diffusing | low (CNN Adam: 0.23–0.43) | conditional, CNN trained with Adam; rarely structural |

### Filament

![Filament: structural disposition, bars, MLP 4×64, SGD, seed 2](/static/about/filament.jpg)

**Form.** Nodes stretched along one line, often with a short tuft at one end.

**Measured signature.** Low isotropy; mass concentrated on few routes.

**Interpretation (hypothesis).** Distances vary along a single dominant direction: a few
units carry the mass from layer to layer and everything else hangs off that corridor. The
network's disposition is close to one-dimensional at this checkpoint.

**Where seen.** Structural dispositions of MLPs trained with SGD (bars, interference): four
of the five MLP 4×64 SGD networks inspected show a filament or tripod.

### Tripod

![Tripod: structural disposition, orbits, CNN, SGD, seed 1](/static/about/tripod.jpg)

**Form.** A small core with two to four long, thin arms.

**Measured signature.** High routing concentration; arms are long low-mass corridors.

**Interpretation (hypothesis).** A few route families dominate; the arms are the parts of
the network those families barely use, pushed out by their long edges. The number of arms
suggests the number of distinct weakly coupled regions, not the number of decision routes.

### Sheet and lobe

![Sheet and lobe: conditional realisation, interference, CNN, SGD, seed 2](/static/about/sheet-and-lobe.jpg)

**Form.** An elongated dense sheet with a separate dense lobe, joined by a fan of edges.

**Measured signature.** Each input sends most of its mass through one channel (CNN SGD:
0.63–0.77 of an input's first-layer mass on its strongest edge, in all 30 analyses).

**Interpretation (hypothesis).** The realisation splits into separate route families,
each pixel committed to one channel. The case is decided by a small number of channels,
and inputs are ordered along the sheet by which channel they feed.

### Bushy star

![Bushy star: structural disposition, interference, MLP 4×64, RMSprop, seed 1](/static/about/bushy-star.jpg)

**Form.** A compact dense body with many short spikes in all directions.

**Measured signature.** Moderate concentration; many comparable spikes.

**Interpretation (hypothesis).** Many units carry comparable structural mass and none
dominates; the network's disposition is spread but not uniform. Most MLP 4×64
dispositions trained with Adam or RMSprop in our study took this form.

### Core and arms

![Core and arms: conditional realisation, orbits, MLP 4×64, RMSprop, seed 1](/static/about/core-and-arms.jpg)

**Form.** A round, dense core with a few long arms.

**Measured signature.** Diffuse within the core; many zero-mass edges outside it.

**Interpretation (hypothesis).** The case is realised by a broad, distributed set of routes
(the core), while some units are silent for this case and sit on the arms. This is the
mixed regime between the tripod and the ball.

### Ball

![Ball: conditional realisation, interference, CNN, Adam, seed 1](/static/about/ball.jpg)

**Form.** A filled, nearly isotropic disc (a sphere in 3-D), sometimes fringed with spikes.

**Measured signature.** Low routing concentration (CNN Adam: 0.23–0.43); the input nodes
form a round cloud (isotropy above 0.6).

**Interpretation (hypothesis).** Many routes carry comparable mass, so most nodes lie at
similar distances from one another: a high-dimensional, distributed realisation projected
into the plane. No single unit or channel decides the case.

## 5. What shapes the form

These are the measured relations from the study. Each is a finite-sample observation on
synthetic data and small networks.

1. **The case.** For a fixed network, the conditional form changes with the input pair.
   The interference CNN trained with Adam (seed 1) gave input isotropy 0.94, 0.06, 0.09,
   0.30 and 0.91 for five pairs. A conditional form describes a case, not a network.
2. **The seed.** The orbits CNN trained with Adam formed a ball for five of five pairs
   with seed 1 and for one of five with seed 2.
3. **The optimiser, for CNN routing.** SGD with momentum concentrated each input's mass on
   one channel (0.63–0.77 on the strongest edge); Adam spread it (0.23–0.43). The two
   ranges do not overlap; across networks, Mann–Whitney U = 0, p = 0.002 (6 against 6).
   Balls followed: 13 of 30 pairs with Adam against 4 of 30 with SGD (network level
   p = 0.065, not significant). Within CNNs, more concentrated routing went with less
   ball-like forms (Spearman ρ = −0.43, p < 0.001, 60 analyses clustered in 12 networks).
   SGD and Adam also ran at different learning rates (0.02 and 0.002), so this is an
   optimiser-and-step-size effect until a matched comparison separates them.
4. **The input statistics.** When images sat on a mid-grey ground (mean about 0.5), 10–14
   of 16 first-layer ReLU units of an MLP died in training, and 85% of edges carried no
   conditional mass. On a dark ground (mean about 0.16) this fell to 39%. Dead units make a
   realisation sparse; check the zero-mass share before reading a sparse picture.
5. **The data structure and width.** Balls appeared on all three datasets, and no dataset
   produced them reliably. Width alone did not decide the form; for MLPs the link between
   routing concentration and the ball was weak (ρ = −0.17 across all 70 pair analyses,
   p = 0.16).
6. **Structural balls are rare.** Two of 18 CNNs and none of the 17 MLP 4×64 networks we
   inspected showed a structural ball. The structural disposition depends only on the
   weights, so it is a property of the trained network alone.

## 6. What a form implies, and how to test it

The geometry shows where mass flows; what that implies about the network has to be tested.
Each interpretation above predicts an interventional consequence the studio can measure.

- **Concentrating forms** (filament, tripod, sheet and lobe) predict that gating the few
  units or channels on the dense route changes the readout contrast strongly. Test: select
  those units in Intervention controls, run a dose scan, and compare with gating the same
  number of units on the arms.
- **Diffusing forms** (ball) predict that gating any single unit changes the readout
  little and that the effect grows only as many units are gated together. Test: dose-scan
  increasing sets of units and look for a gradual rather than abrupt change.
- **Disposition against realisation.** A diffuse disposition with a concentrated
  realisation means the network has many available routes but uses few for this case.
  Compare H₁ and S_def of the two panels; in all 124 analyses the realisation was the more
  concentrated.

What a form does not imply: it is not a fractal dimension, not a measure of accuracy or
robustness by itself, and not stable across cases or seeds unless shown to be. Report the
form together with the stress, the routing concentration, the number of cases examined and
how many of them showed it.

## 7. Recipes

### A conditional ball, for demonstration

1. Build a CNN with the 1×1 convolution + global average readout, channels 8, 12, 16.
2. Train with Adam (learning rate 0.002) or RMSprop (0.001); avoid SGD with momentum.
3. Train two or three seeds; 800–1,500 updates on at least 1,024 images.
4. In Geometry, analyse at least five pairs with Next pair. Keep the cases whose routing
   concentration is low (below about 0.4) and whose zero-mass share is moderate.
5. Check the stress and the 3-D view; report how many of the pairs formed a ball.

### PneumoniaMNIST (untested)

These suggestions transfer the findings above; we could not download PneumoniaMNIST in the
environment where the study ran, so none of them has been measured on it.

- **Aim for the conditional ball first**: it was far more common than the structural one.
- **Use a CNN with the global average readout and Adam**, the combination that formed
  conditional balls most often in our study.
- **Expect a high input mean.** Chest radiographs are bright images. In our data a mid-grey
  ground (mean about 0.5) killed most first-layer units of an MLP and left 85% of edges
  without conditional mass; we have not measured whether radiographs do the same.
- **Check dead units before reading the picture.** If the zero-mass share is high, the
  sparse form reflects silent units, not routing.
- **Train several seeds and analyse several pairs**, including normal against pneumonia
  and pairs within a class. Choose by the measured routing concentration, not by the
  picture alone, and report the proportion of cases that formed the ball.
- **For a structural ball**, expect to need several seeds: it appeared in 2 of 18 CNNs in
  our study.
