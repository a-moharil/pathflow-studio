"""Sequential MLP and CNN models for exact path analysis.

CNN readouts
------------
``dense`` (default, unchanged behaviour)
    The final feature map is flattened into a dense linear output, optionally after a
    dense hidden head (``cnn_head``).
``gap`` (fully convolutional, as in the research programme's CNN panels)
    A bias-optional 1x1 convolution maps the final channels to one map per output,
    followed by global spatial averaging. Neither readout operation is followed by an
    activation. Averaging is linear with fixed coefficients 1/(H*W); it enters the
    execution graph as parameter-free edges, so complete-path measures stay exact.

Layers before the readout are the hidden (activated, gateable) layers. Code that needs
that boundary uses ``hidden_count(model)``, which reads it from the layer types and so
also holds for wrapper models that share ``model.layers``.
"""
import torch
from torch import nn


class SpatialAverage(nn.Module):
    """Global spatial averaging: (B, K, H, W) -> (B, K). Fixed coefficients, no parameters."""

    def forward(self, x):
        return x.mean(dim=(2, 3))


def hidden_count(model):
    """Number of activated hidden layers; the remaining layers form the linear readout."""
    layers = model.layers
    if len(layers) >= 2 and isinstance(layers[-1], SpatialAverage):
        return len(layers) - 2
    return len(layers) - 1


def conv_kernels(c):
    """Per-layer kernel sizes: explicit ``kernels`` list, else ``kernel`` for every layer."""
    kernels = c.get('kernels')
    return list(kernels) if kernels else [c['kernel']] * len(c['channels'])


class Network(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.layers = nn.ModuleList()
        self.shapes = [(1, 28, 28)]
        n = 784
        classes = 1 if c.get('task') == 'regression' else 10 if c['dataset'] == 'fashion' else 2
        if c['model'] == 'mlp':
            for m in c['widths']:
                self.layers.append(nn.Linear(n, m, bias=c['bias']))
                self.shapes.append((m,))
                n = m
            self.layers.append(nn.Linear(n, classes, bias=c['bias']))
            self.shapes.append((classes,))
            return
        ch, h = 1, 28
        for out, k in zip(c['channels'], conv_kernels(c)):
            self.layers.append(nn.Conv2d(ch, out, k, stride=c['stride'], padding=c['padding'], bias=c['bias']))
            h = (h + 2 * c['padding'] - k) // c['stride'] + 1
            ch = out
            self.shapes.append((ch, h, h))
        if c.get('cnn_readout', 'dense') == 'gap':
            self.layers.append(nn.Conv2d(ch, classes, 1, bias=c['bias']))
            self.shapes.append((classes, h, h))
            self.layers.append(SpatialAverage())
            self.shapes.append((classes,))
            return
        n = ch * h * h
        if c.get('cnn_head', False):
            for m in c.get('head_widths', [32, 16]):
                self.layers.append(nn.Linear(n, m, bias=c['bias']))
                self.shapes.append((m,))
                n = m
        self.layers.append(nn.Linear(n, classes, bias=c['bias']))
        self.shapes.append((classes,))

    def activation(self, x):
        return torch.relu(x) if self.config['activation'] == 'relu' else torch.tanh(x)

    def forward(self, x, gates=None, trace=False):
        hs = [x]
        zs = []
        hidden = hidden_count(self)
        for l, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                x = x.flatten(1)
            z = layer(x)
            zs.append(z)
            x = self.activation(z) if l < hidden else z
            if gates and str(l) in gates and l < hidden:
                g = torch.ones(x[0].numel(), device=x.device, dtype=x.dtype)
                for idx, alpha in gates[str(l)].items():
                    g[int(idx)] = float(alpha)
                x = x * g.reshape(x.shape[1:])
            hs.append(x)
        return (x, hs, zs) if trace else x
