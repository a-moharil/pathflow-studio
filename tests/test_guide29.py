"""About page and geometry guide (0.17.0), and the eigen-solver fallback of the embedding."""
import re
from pathlib import Path

import numpy as np

import lab.core.pathmath as pm
from lab import create_app
from lab.guide import SOURCE, render


def test_about_page_renders_guide_and_images(tmp_path):
    client = create_app(tmp_path).test_client()
    page = client.get('/about')
    assert page.status_code == 200
    text = page.text
    for heading in ('The framework', 'Field guide to geometric forms', 'What shapes the form', 'Recipes'):
        assert heading in text
    for form in ('Filament', 'Star', 'Fan', 'Hub and rays'):
        assert f'>{form}</h3>' in text
    assert '>Ball</h3>' not in text
    images = re.findall(r'src="(/static/about/[^"]+)"', text)
    assert len(images) == 5
    for src in images:
        assert client.get(src).status_code == 200
    assert '>About</a>' in client.get('/').text


def test_renderer_escapes_markup():
    body, toc = render('## Title\n\nA <script>x</script> **bold** and d\\* and [link](javascript:alert(1)).\n\n- one\n- two\n')
    assert '<script>' not in body and '&lt;script&gt;' in body
    assert '<strong>bold</strong>' in body and 'd&#42;' in body and 'href="javascript' not in body
    assert '<ul><li>one</li><li>two</li></ul>' in body and toc == [('title', 'Title')]


def test_guide_source_is_the_doc():
    assert SOURCE == Path(__file__).resolve().parent.parent / 'docs' / 'GEOMETRY_GUIDE.md'


def test_embedding_falls_back_when_partial_solver_returns_nothing(monkeypatch):
    rng = np.random.default_rng(0)
    points = rng.normal(size=(40, 3))
    D = np.linalg.norm(points[:, None] - points[None], axis=2)
    anchors = np.arange(12)
    reference, stress, _ = pm.embed_distances(D[anchors], anchors)
    calls = {'n': 0}
    real = pm.eigh

    def empty(B, subset_by_index=None):
        calls['n'] += 1
        return np.zeros(0), np.zeros((len(B), 0))

    monkeypatch.setattr(pm, 'eigh', empty)
    Y, stress2, vals = pm.embed_distances(D[anchors], anchors)
    assert calls['n'] == 1 and len(vals) == 3 and np.isfinite(Y).all()
    # Same configuration up to an orthogonal transform: identical pairwise distances.
    d1 = np.linalg.norm(reference[:, None] - reference[None], axis=2)
    d2 = np.linalg.norm(Y[:, None] - Y[None], axis=2)
    assert np.allclose(d1, d2, atol=1e-8) and abs(stress - stress2) < 1e-10
    monkeypatch.setattr(pm, 'eigh', real)
