"""Synthetic field datasets for visual investigation of path geometry.

Both datasets are 28 × 28 single-channel images, so every MLP and CNN in the studio trains
on them unchanged. They are designed around four data properties that shape the measured
geometry:

1. Continuous latent factors (orientation, frequency, pitch, position). Hidden units tile a
   continuum rather than a few discrete cues, so structural flow spreads over many routes
   instead of collapsing onto one bottleneck.
2. Shared components across classes. Every class is built from the same family of
   patterns, so classes reuse hidden units and their conditional routes partly overlap.
3. Calibrated ambiguity. A nuisance component drawn independently of the class, and a
   widening of the class-defining factor, keep several hypotheses alive in the network; the
   conditional realisation of an input then differs visibly from the structural disposition
   instead of tracing a single crisp route. ``ambiguity`` in [0, 0.8] sets this overlap.
4. Symmetry. Classes are related by rotation (interference) or by the sign and type of a
   polar wave number (orbits), which gives the geometry a regular, symmetric organisation.

These are demonstration datasets for the studio's visualisations. They make no claim about
natural images, and their labels are generated, not annotated.
"""
from __future__ import annotations

import numpy as np

FIELD_DATASETS = {
    'synthetic-interference': dict(
        classes=['Grating 0°', 'Grating 60°', 'Grating 120°'],
        title='Synthetic interference (3 classes)',
        summary='Windowed gratings whose dominant orientation sits near 0°, 60° or 120°, '
                'over a class-independent second grating.'),
    'synthetic-orbits': dict(
        classes=['Rings', 'Spokes', 'Right spiral', 'Left spiral'],
        title='Synthetic orbits (4 classes)',
        summary='Polar waves cos(k·r + m·θ): rings (m = 0), spokes (k = 0) and spirals of either '
                'chirality, mixed with a weaker wave of another class.'),
}

DEFAULT_AMBIGUITY = 0.35
SPLITS = (('train', None), ('val', 96), ('test', 120))
_GRID = np.linspace(-1.0, 1.0, 28, dtype='float64')
_Y, _X = np.meshgrid(_GRID, _GRID, indexing='ij')


def is_field(name: str) -> bool:
    return name in FIELD_DATASETS


def class_names(name: str) -> list[str]:
    return list(FIELD_DATASETS[name]['classes'])


def _finish(field: np.ndarray, envelope: np.ndarray, rng) -> np.ndarray:
    image = 0.5 + 0.42 * field * envelope + rng.normal(0.0, 0.035, field.shape)
    return np.clip(image, 0.0, 1.0)


def _interference(label: int, ambiguity: float, rng) -> np.ndarray:
    spread = np.deg2rad(6.0 + 24.0 * ambiguity)
    theta = np.deg2rad(60.0 * label) + rng.normal(0.0, spread)
    freq = rng.uniform(2.0, 2.8)
    # Phase stays near a class template, so the classes are learnable by a plain MLP;
    # its spread grows with ambiguity.
    phase = rng.normal(0.0, 0.25 + 0.9 * ambiguity)
    primary = np.cos(np.pi * freq * (_X * np.cos(theta) + _Y * np.sin(theta)) + phase)
    nuisance_theta = rng.uniform(0, np.pi)
    nuisance = np.cos(np.pi * rng.uniform(1.6, 3.6) * (_X * np.cos(nuisance_theta) + _Y * np.sin(nuisance_theta))
                      + rng.uniform(0, 2 * np.pi))
    weight = 0.2 + 0.65 * ambiguity
    cx, cy = rng.uniform(-0.12, 0.12, 2)
    width = rng.uniform(0.6, 0.85)
    envelope = np.exp(-((_X - cx) ** 2 + (_Y - cy) ** 2) / (2 * width ** 2))
    return _finish((primary + weight * nuisance) / (1 + weight), envelope, rng)


def _polar_wave(kind: int, rng, cx: float, cy: float, jitter: float) -> np.ndarray:
    r = np.hypot(_X - cx, _Y - cy)
    theta = np.arctan2(_Y - cy, _X - cx)
    phase = rng.normal(0.0, jitter)
    if kind == 0:  # rings: radial wave only
        return np.cos(2 * np.pi * rng.uniform(1.8, 2.4) * r + phase)
    if kind == 1:  # spokes: angular wave only
        return np.cos(int(rng.integers(5, 7)) * theta + phase)
    m = int(rng.integers(3, 5)) * (1 if kind == 2 else -1)  # spirals: both, chirality = sign of m
    return np.cos(2 * np.pi * rng.uniform(1.4, 2.0) * r + m * theta + phase)


def _orbits(label: int, ambiguity: float, rng) -> np.ndarray:
    cx, cy = rng.uniform(-0.08, 0.08, 2)
    jitter = 0.25 + 0.9 * ambiguity
    primary = _polar_wave(label, rng, cx, cy, jitter)
    other = int(rng.choice([k for k in range(4) if k != label]))
    secondary = _polar_wave(other, rng, cx, cy, np.pi)
    weight = 0.15 + 0.6 * ambiguity
    r = np.hypot(_X - cx, _Y - cy)
    envelope = np.exp(-(r / 0.88) ** 4) * (1 - np.exp(-(r / 0.12) ** 2) * 0.6)
    return _finish((primary + weight * secondary) / (1 + weight), envelope, rng)


def generate(name: str, seed: int, train_size: int, ambiguity: float) -> dict:
    """Deterministic splits: the same (name, seed, size, ambiguity) gives identical arrays."""
    make = {'synthetic-interference': _interference, 'synthetic-orbits': _orbits}[name]
    count = len(FIELD_DATASETS[name]['classes'])
    out = {}
    for index, (split, n) in enumerate(SPLITS):
        n = train_size if n is None else n
        rng = np.random.default_rng([seed, index, int(round(ambiguity * 1000))])
        y = np.arange(n) % count
        x = np.stack([make(int(label), ambiguity, rng) for label in y])[:, None].astype('float32')
        out.update({split + '_x': x, split + '_y': y.astype('int64'), split + '_ids': np.arange(n)})
    return out
