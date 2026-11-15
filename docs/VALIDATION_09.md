# Validation for 0.8.0

- Full Python suite: 72 tests passed. New tests independently enumerate tiny complete-path graphs for ReLU and tanh and check positive/negative and squared masses, signed reconstruction and A=A++A−.
- Grad-CAM matches direct autograd construction and is invariant under compensated positive ReLU channel scaling.
- Channel wrappers match the original model's explicit gates for both activations, including zero attenuation and equal-endpoint secants. Checkpoint tensors remain unchanged.
- Connection metadata is verified against source/destination/kernel identities. Contrast reversal and the constructed cancellation/intervention example are checked numerically. Reports serialize without NaN/Infinity.
- New browser workflow exercises architecture/channel selection, exact occurrence selection, regional drag aggregation, companion maps, channel/primitive intervention, Grad-CAM rescaling, contrast reversal, constructed demonstration, saved results and PNG/JSON/CSV/WebM export.
- Previous healthcare browser workflow and Evidence 08 browser workflow pass. Existing Node regressions check cameras, anchors, all-node labels, all structural edges including zero-mass carriers, independent 2-D coordinates, fidelity fallbacks and Observatory scope.

Tests use synthetic CPU data and small independent graphs. They do not establish clinical robustness or an empirical advantage over attribution on PneumoniaMNIST. See VISUAL_INVESTIGATION_09.md for numerical and interpretation limits.
