# Fully convolutional readout and ruler sensitivity · Pathflow Studio 0.15.0

Branch: `efficient-realisation`. Baseline: 0.14.0 (efficient realisation, as delivered).

## Why

Until now, every studio CNN ended by flattening its last feature map into a dense linear
output, and a "downstream hidden MLP" was ticked by default. The research programme's CNN
panels are different. They are fully convolutional: the last kernel is a 1×1 convolution to
the outputs, followed by global spatial averaging, with no dense layer anywhere. For example,
C1_shallow3 is channels [4, 4] with kernels [3, 3, 1], and its 184 parameters are exactly
1→4 (3×3) + 4→4 (3×3) + 4→1 (1×1) without bias.

A dense readout connects every final occurrence to every output. A dense head multiplies the
complete-path carrier: (32, 16) multiplies the number of complete paths by 512 on the default
CNN. The studio therefore could not reproduce the carrier on which the paper's CNN
geometries are measured.

## Readout options (Design → Architecture, CNN)

| Option | Layers after the last convolution | Hidden head |
|---|---|---|
| Flatten + dense output (default; existing runs) | dense linear output | optional, **off by default** |
| 1×1 convolution + global average (fully convolutional) | 1×1 convolution → global spatial average | not available |

- Configurations saved before 0.15.0 have no `cnn_readout` key and keep the dense readout.
  Their checkpoints, layer names and results are unchanged.
- **Kernel sizes per layer** (optional) gives one kernel size per convolution layer, for
  example `3,3` for C1_shallow3. Blank uses *Kernel* for every layer. The 1×1 readout kernel
  is added by the readout option, so it is not listed.
- With the head off, its widths are ignored. A blank widths field no longer blocks creation.
- Choosing the fully convolutional readout with the head on is refused with an explanation.

To match the CNN panels, set: channels and kernels from the panel (without the final 1),
stride 1, padding 0 (valid), bias off, readout = 1×1 convolution + global average. The
studio keeps 28×28 inputs, whereas the panels used 16×16, so execution-node counts are larger
and BCANw can exceed the node budget for deeper panels; the Design estimate reports this.

## Exactness

- The 1×1 convolution is an ordinary convolution layer in the spatial execution graph.
- Global averaging enters as parameter-free edges (k, y, x) → k with coefficient 1/(H·W).
- No activation follows either readout operation, so both carry unit secants.
- Gates and interventions apply to hidden (activated) layers only. The readout map is refused
  as "not a hidden layer". Healthcare layer marginals, Grad-CAM layer lists, Field B support
  layers and rescaling use the same boundary through `hidden_count(model)`, which reads it
  from the layer types.

Verified in `tests/test_readout23.py`:

- complete-path enumeration on a small CNN reproduces log Q, H₁, η₁, log N and every edge
  marginal, including the averaging edges, for ReLU and tanh, structurally and conditionally;
- the signed path sums reproduce the network's actual output contrast (residual < 1e-10);
- parameter and node estimates equal the built network and graph, and C1_shallow3 has 184 parameters;
- analysis with gates, healthcare case, dose, evidence and cohort jobs all complete;
- dense configurations keep their state-dictionary layout.

## Ruler sensitivity (Run Statistics → Scale geometry)

The flow-induced length is the declared ruler λ_e = (|E_ℓ|F_e + ε)^(−p). The recorded
convention is ε = 0.03, p = 1, and it is unchanged. The diagnostic:

- reads the saved edge marginals of one completed native analysis;
- recomputes BCANw for the ε values you give (and optionally another p, including the
  negative sign-reversal test), using the same legacy `bcan-1` estimator (36 radii,
  strength-ordered greedy colouring);
- always recomputes the recorded convention alongside and reports whether it reproduces the
  saved d* and ξ₅₀;
- reports d*, d_eff, ξ₅₀/diameter, qualified windows and the share of edges whose enrichment
  is below ε, where ε rather than flow sets the length.

It runs as an ordinary queued job in the single worker, is refused in presentation mode,
reuses an identical earlier request, and never modifies the source analysis. On a fully
convolutional synthetic CNN at update 60, conditional d* ranged from 0.34 (ε = 0.003) to 5.37
(ε = 0.3), with 67% of conditional edges below the recorded ε. A d* reported for a
concentrated conditional measure should therefore state its ε.

Verified in `tests/test_ruler23.py`: exact reproduction of the saved geometry at the recorded
convention, monotone floor fractions, request validation, deduplication, presentation guard
and refusal of foreign analyses.

## Demonstration workspace

`tools/build_demo_workspace.py` now also trains one fully convolutional synthetic CNN, and
the optional PneumoniaMNIST CNN uses the same readout.
