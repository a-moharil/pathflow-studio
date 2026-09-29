# Validation · Update 19

34 Python tests passed across healthcare, evidence, source-region and carrier-faithfulness
checks plus the new workflow worker test. The new test verifies exact request provenance,
source-box validation, unchanged global Field A/output contrast, explicit rejection of
nonidentity attribution contexts, and byte-identical model checkpoints.

The complete native healthcare JavaScript stack was executed against a Flask-rendered
page and real worker outputs from a trained small CNN. DOM lifecycle checks passed for
case → catalog → attenuation, region/channel rendering, stage navigation, stale-context
blocking, edits during a request, presentation-mode computation blocking and saved-context
restoration. Canvas and Plotly are drawing adapters in this test, not browser renderers.
A joint-page DOM test checks stage placement, original element-ID retention, selected vs
recorded context and the presentation guard.

Full browser rendering and visual layout were not verified in this environment. The
previous Chromium attempt was blocked by socket permissions. These DOM checks do not
substitute for browser visual verification. Existing browser scripts targeting old tab
names need adaptation to the five-stage navigation.

Installers were checked against delivered 0.11.4 and 0.11.3, at repository root and inside
a nested pathflow_studio directory, with target hashes verified. Dirty-tree refusal was
checked. The ZIP is CRC-checked and all project manifest hashes are verified.

Reproduce backend checks from the project directory:

    python -m pytest tests/test_healthcare.py tests/test_evidence08.py tests/test_regions17.py tests/test_faithful15.py tests/test_workflow19.py -q

Optional DOM lifecycle check (jsdom is a development-only dependency):

    python tests/workflow19_fixture.py /tmp/pathflow-workflow19
    NODE_PATH=/path/to/node_modules node tests/dom_workflow19.cjs /tmp/pathflow-workflow19

No new application dependencies are required.
