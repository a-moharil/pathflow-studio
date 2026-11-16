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

In every one of the 110 analyses of the study in section 4, the conditional realisation is
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
- **Embedding.** The picture you see is a stress fit of these distances. The 2-D picture
  starts from the spectral (classical scaling) layout; the 3-D picture starts from the
  settled 2-D picture plus one further axis and is turned so that its best shadow lies on
  the 2-D picture, so switching between 2-D and 3-D adds depth rather than producing a new
  shape. **Stress** reports how much the picture distorts the measured distances; a high
  stress means the geometry does not fit in two or three dimensions. Each card says
  whether its fit settled. Read a shape only from a settled fit (section 3).

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
- **Read only a settled picture.** The picture is a fit, and an unfinished fit keeps the
  shape of its starting layout. Each geometry card states whether its fit settled. In the
  study below 338 of 440 fits settled; the other 102 reached the 3,000-iteration limit
  while still improving by at most 2.8% per 100 iterations, and continuing five of them
  for another 3,000 iterations changed their shape by at most 6%. Held-out stress was
  0.02–0.14 for the structural and 0.04–0.17 for the conditional measure: distances in
  the picture are close to the measured ones.
- **2-D and 3-D show the same geometry.** The 3-D picture is grown from the 2-D one and
  turned so that its shadow lies on it. The card reports the *median node shift* between
  the two. For the conditional pictures of CNNs it stayed below 0.07 of the median radius;
  the 3-D view adds depth to the same form.

![The same conditional fan in 2-D (left) and 3-D (right): interference, CNN, SGD, seed 2](/static/about/fan-2d-3d.jpg)

- **When the shift is large, read lengths, not angles.** In wide MLPs the input pixels sit
  on rays around a hub, and the rays can turn about the hub at little cost: the median
  shift reached 0.98 there. Ray lengths, the hub and which rays carry mass are fixed by the
  distances; the angles between rays are not.
- **Routing concentration.** Under each geometry the studio reports, for the input layer,
  the average share of an input's mass carried by its single strongest first-layer edge
  (1 = every input feeds one unit; low = spread over many), and the share of edges with no
  mass. This is the measured counterpart of concentrating and diffusing behaviour; it is
  computed on the graph, not on the picture.

**Pictures made before 0.17.1.** Until 0.17.1 the 2-D and 3-D pictures were separate fits stopped after 120 iterations,
and a random restart with a different seed for each dimension could replace either. Such
pictures often kept the round cloud of their random start, and the 2-D and 3-D views of
one case could be unrelated. Every "ball" we recorded with that procedure (13 of 54
networks, 19 of 70 pair analyses) disappeared when the same case was fitted to a settled
picture. The measured quantities (H₁, S_def, η, d\*, d_eff, routing) never depended on the
picture and are unchanged. Saved analyses keep their old pictures, labelled on the card;
analyse the case again for a settled one.

## 4. Field guide to geometric forms

The forms below are those of settled pictures in a study of 54 networks: three datasets
(bars, interference, orbits) × three architectures (MLP 4×16, MLP 4×64, CNN with channels
8·12·16 and the 1×1 convolution + global average readout) × three optimisers (SGD with
momentum, Adam, RMSprop) × two seeds, each trained for 800 updates on 1,024 images, plus
five input pairs on 14 of the networks: 110 distinct analyses. They are recorded outputs of
the studio; none is constructed. Two forms describe the structural disposition and two the
conditional realisation.

| Form | Measure | Where we saw it | Measured signature |
|---|---|---|---|
| Filament | structural | 16 of 18 networks trained with SGD; 11 of 36 with Adam or RMSprop | node spread along one axis (minor/major variance ≤ 0.06) |
| Star | structural | 25 of 36 networks trained with Adam or RMSprop, mostly MLPs | compact head of input nodes with long spikes in several directions |
| Fan | conditional | all 18 CNNs, every pair | wedge from the readout; opening 57–65° with SGD, 38–48° with Adam or RMSprop |
| Hub and rays | conditional | all 36 MLPs | compact hub with input pixels on straight rays |

No settled picture in the study was a ball: the input nodes of a conditional realisation
never formed a round cloud (largest minor/major variance 0.25).

### Filament

![Filament: structural disposition, orbits, CNN, SGD, seed 1](/static/about/filament.jpg)

**Form.** Nodes stretched along one line, often densest near one end, with a few short
spikes.

**Measured signature.** The minor/major variance ratio of all node positions is at most
0.06. With SGD, 16 of 18 networks formed a filament against 11 of 36 with Adam or RMSprop;
for MLPs the elongation differs between optimisers (Mann–Whitney U = 30, p < 0.001, 12
against 24 networks), for CNNs it does not clearly (p = 0.05, 6 against 12).

**Interpretation (hypothesis).** Distances vary along one dominant direction: the
disposition is organised as a single corridor of strong mass from the inputs to the
readout, and the rest of the network hangs off it.

### Star

![Star: structural disposition, bars, MLP 4×64, Adam, seed 2](/static/about/star.jpg)

**Form.** A compact head that holds the input nodes, with long spikes (hidden and output
units on long, low-mass edges) in several directions.

**Measured signature.** Minor/major variance above 0.06. On the bars dataset the input
nodes in the head can form a round cluster (isotropy up to 0.92 in 2-D, 0.5 in 3-D): the
inputs are nearly equidistant because each reaches the first layer through weights of
similar size. This is a small cluster inside the head, not a ball of the network.

**Interpretation (hypothesis).** Many units carry comparable structural mass and none
dominates, so the disposition spreads into several low-mass directions.

### Fan

![Fan: conditional realisation, interference, CNN, SGD, seed 2](/static/about/fan.jpg)

**Form.** A wedge opening from the readout: hidden nodes fill the wedge and the input
pixels lie along its rim.

**Measured signature.** The opening (the angle around the readout that holds 90% of the
input nodes) is 57–65° for CNNs trained with SGD and 38–48° with Adam or RMSprop (network
means; SGD against Adam U = 36, p = 0.002, 6 against 6). Within a network it varied by
8–30° across the five pairs; the form itself did not change.

**Interpretation (hypothesis).** The wider the fan, the further apart the inputs are in
the realised geometry. With SGD each pixel sends most of its mass through one channel
(routing concentration 0.63–0.77 in all 30 analyses), so pixels feeding different channels share few strong
routes and spread around the readout; with Adam the mass of a pixel is shared across
channels (0.23–0.43), routes overlap and the pixels draw together. Across the CNN analyses
the opening rises with routing concentration (Spearman ρ = 0.64, p < 0.001, 66 analyses in
18 networks; analyses within a network are not independent).

### Hub and rays

![Hub and rays: conditional realisation, orbits, MLP 4×64, SGD, seed 2](/static/about/hub-and-rays.jpg)

**Form.** A compact hub holding the hidden units and the readout, with the input pixels
on straight rays out of it. Rays that carry conditional mass form one or two sectors;
silent pixels lie on longer rays elsewhere.

**Measured signature.** In MLP 4×16 the 2-D and 3-D pictures agree (median node shift up
to 0.29); in MLP 4×64 the rays turn between the two (median shift 0.24, up to 0.98) while
the hub, the ray lengths and the realised sectors stay.

**Interpretation (hypothesis).** A pixel's distance to the hub records how strongly it is
realised; pixels on one ray reach the hub through the same first-layer units. Because
pixels are linked only through the hub, the directions of the rays are free, so their
arrangement says nothing about the network.

## 5. What shapes the form

These are the measured relations from the study. Each is a finite-sample observation on
synthetic data and small networks.

1. **The architecture decides the conditional form.** Every CNN realisation was a fan and
   every MLP realisation hub and rays, on all three datasets and for all optimisers.
2. **The optimiser shapes CNN routing, and with it the fan.** SGD with momentum
   concentrated each input's mass on one channel (0.63–0.77 on the strongest edge, 30
   analyses); Adam spread it (0.23–0.43, 30 analyses). The ranges do not overlap; across
   networks U = 36, p = 0.002 (6 against 6). SGD fans were correspondingly wider (section 4). SGD and Adam also ran at
   different learning rates (0.02 and 0.002), so this is an optimiser-and-step-size effect
   until a matched comparison separates them.
3. **The optimiser shapes the MLP disposition.** MLPs trained with SGD formed filaments
   (10 of 12); with Adam or RMSprop mostly stars (21 of 24).
4. **The case changes details, not the form.** Across five pairs per network the fan
   opening varied by 8–30°, and the form was the same in every pair.
5. **The input statistics.** When images sat on a mid-grey ground (mean about 0.5), 10–14
   of 16 first-layer ReLU units of an MLP died in training and 85% of edges carried no
   conditional mass. On a dark ground (mean about 0.16) this fell to 39%. Dead units make a
   realisation sparse; check the zero-mass share before reading a sparse picture.

## 6. What a form implies, and how to test it

The geometry shows where mass flows; what that implies about the network has to be tested.
Each interpretation above predicts an interventional consequence the studio can measure.

- **Concentrated routing** (high strongest-edge share, wide CNN fan, filament
  disposition) predicts that gating the few units or channels on the dense route changes
  the readout contrast strongly. Test: select those units in Intervention controls, run a
  dose scan, and compare with gating the same number of units off the route.
- **Diffuse routing** (low strongest-edge share, narrow CNN fan, star disposition)
  predicts that gating any single unit changes the readout little and that the effect
  grows only as many units are gated together. Test: dose-scan increasing sets of units
  and look for a gradual rather than abrupt change.
- **Disposition against realisation.** A diffuse disposition with a concentrated
  realisation means the network has many available routes but uses few for this case.
  Compare H₁ and S_def of the two panels; in all 110 analyses the realisation was the more
  concentrated.

What a form does not imply: it is not a fractal dimension, not a measure of accuracy or
robustness by itself, and not stable across seeds unless shown to be. Report the form
together with the stress, the fit status, the routing concentration, the number of cases
examined and how many of them showed it.

## 7. Recipes

### Comparing concentrated and diffuse routing in a CNN

1. Build a CNN with the 1×1 convolution + global average readout, channels 8, 12, 16.
2. Train one copy with SGD and momentum (learning rate 0.02) and one with Adam (0.002); two
   seeds each; 800–1,500 updates on at least 1,024 images.
3. In Geometry, analyse at least five pairs per network with Next pair, using the default
   iteration limit, and check that every card reports a settled fit.
4. Record the routing concentration and the fan opening for each case, and compare
   networks rather than single cases.

### PneumoniaMNIST (untested)

These suggestions transfer the findings above; we could not download PneumoniaMNIST in the
environment where the study ran, so none of them has been measured on it.

- **Expect a fan for a CNN and hub and rays for an MLP**, with the optimiser setting how
  concentrated the routing is; compare SGD and Adam on the same architecture.
- **Expect a high input mean.** Chest radiographs are bright images. In our data a mid-grey
  ground (mean about 0.5) killed most first-layer units of an MLP and left 85% of edges
  without conditional mass; we have not measured whether radiographs do the same.
- **Check dead units before reading the picture.** If the zero-mass share is high, the
  sparse form reflects silent units, not routing.
- **Train several seeds and analyse several pairs**, including normal against pneumonia
  and pairs within a class, and read each picture only once its card reports a settled fit.
