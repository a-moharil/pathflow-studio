# Structural overhaul · 0.16.0

Branch `structural-overhaul`, built on `efficient-realisation` 0.15.1. The scientific
measures, path mathematics, conventions and stored results are unchanged. This release
changes how the studio is organised, how an investigation travels between pages, and how
the studio looks.

## 1. Organisation and names

The studio is organised around three objects:

| Object | What it is | Where it lives |
|---|---|---|
| Workspace | All networks, cohorts and datasets | Library · Build · Queue · Evidence (header) |
| Run | One trained network and its history | Context bar and stage rail |
| Case | One input pair at one committed checkpoint | The case chip; `?case=<job>` on every stage link |

The run tabs became a **stage rail** in three groups, plus a reading:

| Group | Stage | Opens | Former name |
|---|---|---|---|
| Run | Overview | `/run/<rid>` | Overview |
| Run | Training | `/run/<rid>/training` | Training |
| Case | Geometry | structural disposition and conditional realisation panels | Geometry |
| Case | Realisation | Field A over the contrast, routes, transformations | Fields / Investigation |
| Case | Observables | path observables across checkpoints | Observatory |
| Accountability | Evidence | run statistics | Statistics |
| Accountability | Audit | manuscript audit of the case | ECML auditor |
| Reading | Clinical reading | the healthcare investigation of the same case (binary native runs) | Healthcare |

Header: **Library** (was Experiments), **Build** (was Design), **Queue** (new), **Evidence**
(cross-run statistics), **Present** (was Presentation). The healthcare set-up hub is linked
from the Library.

`/run/<rid>/<stage>` is the canonical address of every stage. Every earlier page address
still works, and the earlier aliases (`/run/<rid>/observatory`, `/healthcare`, `/auditor`)
redirect as before. On the experiment page the rail switches sections in place; the
duplicate in-page tabs are hidden, and Configuration remains as a small sub-tab.

## 2. Cases

A case is named by the job that recorded it. Its coordinates are the checkpoint, split,
input x, reference type and reference (`lab/cases.py`). Each stage computes its own result
for a case; a stage opens:

1. the latest completed result of its own kind whose case coordinates match **exactly**;
2. if the case has no result of that kind, nothing is substituted: the stage pre-fills its
   controls with the case and says that the result has not been computed;
3. with no case, the stage opens its latest recorded result.

The case travels as `?case=<job>` on every stage link and is remembered per run within a
browser tab. Computing or loading a result moves the case to it. The chip in the context
bar names the case; × releases it.

Endpoints, all read-only and available in presentation mode:

* `GET /api/run/<rid>/cases` — recorded cases, newest first.
* `GET /api/case/<job>` — coordinates, label and the stage results recorded for that case.
* `GET /api/run/<rid>/case/open?kind=<kinds>&case=<job>` — the result a stage should open.
* `GET /api/queue`, `/queue` — recent work across the workspace with Open links.

Pages that now open saved results on arrival: Geometry and Realisation (native and
residual), Clinical reading, Audit. The experiment job log offers **Open** for every
completed analysis.

## 3. Admission gate

Every native measurement request passes `storage.admit`:

* `found` — an identical request already completed; its stored result is reused;
* `pending` — an identical request is queued or running;
* `queued` — new work.

Reuse is by exact match of the complete request, including the analysis version, so a
changed convention or code version is never answered with an older result. A request for a
checkpoint that was never committed is rejected at submission with a clear message, rather
than failing later in the worker.

## 4. Fixes

1. Dataset cache writes use a private temporary name per writer. Previously concurrent
   writers shared `<dataset>.tmp.npz` and one could fail a training run. `atomic_json` uses
   the same pattern.
2. Clinical history labels cohorts by size and reference policy (was "input undefined").
3. Residual-DAG Overview: geometry responses of scalar-scope residual analyses are indexed
   (index version 2, re-indexed automatically) and the tile notes no longer mention a fixed
   probe. The operation quotient has no BCANw, so d* stays unmeasured there and the tile
   says so.
4. Build shows a residual network estimate: parameters, activation units and operations of
   one forward pass (`POST /api/joint/estimate`; never downloads pretrained weights).
5. Uncommitted checkpoints are rejected at submission (see §3).
6. The dead `auditor_link.js` is removed.
7. One presentation state: the header switch, the clinical presenter and the evidence
   presenter now set and follow the same server-enforced mode.

## 5. Design system (revised in 0.16.1)

`lab/static/atelier.css`, style "instrument": cool porcelain plates on a pale steel ground,
an ink control strip as the masthead, hairline rules, 2 px corners, and a short navy
registration tick at the top-left of every plate. Square bullet markers precede eyebrows.

Type: **Archivo** (Omnibus-Type, variable width and weight) for titles and interface;
titles are set expanded (width 112–125 %), the interface at normal width. **Martian Mono**
(Evil Martians, variable width) set condensed for every measured number, table header,
status and eyebrow. Both are bundled under `lab/static/vendor/fonts` with their SIL Open
Font Licences; no system or hosted fonts are required. 0.16.0's serif and warm-paper
styling is retired.

Colour roles: navy for structure and disposition; signal red only for the current place
(active stage, focus, presenting); sky for the current case; ochre for realisation and
Field A (data only); rose for the clinical reading; audit green (#86BC25) only for verified
results. Data series use a re-validated categorical set: navy #2D5DA6, ochre #C17A14,
sage-emerald #1F9D6E and Bosch rose #A0457E. All pass the lightness, chroma, colour-vision
separation and 3:1 contrast checks.

## 6. Geometry rendering

A **Rendering** control (Refined light · Classic) on every geometry panel. Refined changes
only the lighting of the measured carrier: a deep ground with a soft glass gradient,
additive light where flows overlap, atmospheric depth fading in 3-D, and a vignette. Node
positions, every edge, flow widths (√F) and the conditional hue encoding are identical to
Classic.

## 7. Not in this release

The three implementations of the investigation (Fields, the healthcare stack, the residual
page) still exist underneath; this release unifies how they are reached and carried, not
their code. The healthcare page keeps its own pair controls. These remain the next step.

## Tests

`tests/test_overhaul25.py` (admission, cases, queue, aliases, cache race, concurrent JSON
writes, residual geometry indexing, residual estimate) and `tests/dom_case25.cjs` (case
carrying). The cache-race test fails on 0.15.1 and passes here.
