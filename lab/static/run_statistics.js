/* Run statistics: temporal, realisation, coupling, geometry and cohort evidence. */
(() => {
  const host = document.getElementById('statsContext');
  const rid = host.dataset.run;
  const C = StatsUI.colors;
  const fmt = StatsUI.fmt;
  let contexts = [], stats = null, cohort = null;

  function contextLabel(c) {
    const x = c.context || {};
    if (x.kind === 'probe') return `Fixed probe · test[${x.x}] vs test[${x.reference}] · ${x.readout?.scope === 'all' ? 'all outputs' : 'selected readout'} · every checkpoint`;
    if (x.kind === 'joint_analysis') return `Residual-DAG analysis · ${x.head} · input ${x.x} vs ${x.reference_type === 'zero' ? 'zero' : x.reference}`;
    const gates = x.gates && Object.keys(x.gates).length ? ' · gated' : '';
    const transform = x.transform && Object.entries(x.transform).some(([k, v]) => v !== (k === 'scale' ? 1 : 0)) ? ' · transformed' : '';
    return `${c.source === 'dose' ? 'Dose analysis' : 'Analysis'} · ${x.readout?.scope === 'all' ? 'all outputs' : 'selected readout'} · test[${x.x}] vs ${x.reference_type === 'zero' ? 'zero' : 'test[' + x.reference + ']'}${gates}${transform}${x.measurement_limit ? ' · budget ' + x.measurement_limit : ''}`;
  }

  /** Match an Observatory-style context (carried in the URL) to an indexed context. */
  function matchCarried(carried) {
    try {
      const c = JSON.parse(carried);
      return contexts.find(k => {
        const x = k.context;
        return x.kind !== 'probe' && x.x === c.x && (c.reference_type === 'zero' ? x.reference_type === 'zero' : x.reference === c.reference)
          && (x.readout?.scope || 'selected') === (c.readout?.scope || 'selected') && JSON.stringify(x.gates || {}) === JSON.stringify(c.gates || {});
      });
    } catch (e) { return null; }
  }

  async function loadContexts() {
    const d = await api(`/api/stats/run/${rid}/contexts`);
    contexts = d.contexts;
    const select = document.getElementById('contextSelect');
    const previous = select.value;
    select.replaceChildren(...contexts.map(c => new Option(`${contextLabel(c)} — ${c.records} record${c.records === 1 ? '' : 's'}`, c.key)));
    if (!contexts.length) select.append(new Option('No recorded observables yet', ''));
    const carried = new URLSearchParams(location.search).get('context');
    const match = carried ? matchCarried(carried) : null;
    select.value = previous && contexts.some(c => c.key === previous) ? previous : match ? match.key : d.default;
  }

  async function loadStats() {
    const key = document.getElementById('contextSelect').value;
    stats = await api(`/api/stats/run/${rid}?context=${encodeURIComponent(key)}`);
    const c = contexts.find(k => k.key === key);
    StatsUI.note('contextNote', c ? `${stats.unit}. ${c.records} recorded ${c.source} record(s), updates ${c.first}–${c.last}. Training metrics are attached to every context.` : '');
    renderRealisation();
    renderTemporal();
    renderCoupling();
    renderGeometry();
    renderResponse();
    await renderLedger();
  }

  // ------------------------------------------------------------ realisation
  function renderRealisation() {
    const items = stats.realisation;
    const eta1 = items.find(i => i.name === 'log_eta1');
    const eta2 = items.find(i => i.name === 'log_eta2');
    document.getElementById('realisationTag').replaceChildren(StatsUI.tag(eta1?.n ? 'descriptive' : 'unavailable'));
    const reading = document.getElementById('realisationReading');
    if (eta1?.n) {
      const lower = eta1.negative, total = eta1.n;
      reading.textContent = `Across ${total} checkpoint${total === 1 ? '' : 's'}, the conditional path measure is more concentrated than the structural measure at ${lower} of ${total}. ` +
        `Conditional realisation retains a median ${StatsUI.pct(eta1.retention?.median)} of structural effective occupancy (IQR ${StatsUI.pct(eta1.retention?.q25)}–${StatsUI.pct(eta1.retention?.q75)})` +
        (eta2?.n ? ` and ${StatsUI.pct(eta2.retention?.median)} of collision occupancy.` : '.') +
        (lower === total && eta1.retention?.median < .5 ? ' At every recorded checkpoint, the declared contrast realises a smaller support than the available structural disposition.'
          : lower === 0 ? ' The conditional measure is not more concentrated at any recorded checkpoint in this context.'
          : ' The direction of the difference varies across checkpoints.');
    } else reading.textContent = 'No matched structural and conditional measurements in this context. Choose another context or measure a checkpoint.';
    if (eta1) {
      StatsUI.plot('realisationChart', [
        StatsUI.line('Structural disposition', eta1.steps, eta1.structural, C.structural),
        StatsUI.line('Conditional realisation', eta1.steps, eta1.conditional, C.conditional)
      ], { xaxis: { title: 'Optimizer update' }, yaxis: { title: 'log η₁' } });
      const traces = [];
      for (const [item, name, dash] of [[eta1, 'η₁ retained', 'solid'], [eta2, 'η₂ retained', 'dash']]) {
        if (item?.n) traces.push(StatsUI.line(name, item.steps, item.difference.map(Math.exp), C.third,
          { line: { color: C.third, width: 2, dash }, hovertemplate: '%{y:.3%}<extra>' + name + '</extra>' }));
      }
      const smallest = Math.min(...traces.flatMap(t => t.y).filter(v => v > 0));
      const tiny = smallest < 1e-4;
      for (const t of traces) if (tiny) t.hovertemplate = '%{y:.3e}<extra>' + t.name + '</extra>';
      StatsUI.plot('retentionChart', traces, { xaxis: { title: 'Optimizer update' },
        yaxis: tiny ? { title: 'Conditional ÷ structural occupancy (fraction)', type: 'log', exponentformat: 'power' }
                    : { title: 'Conditional ÷ structural occupancy', type: 'log', tickformat: '.1~%' } });
    }
    StatsUI.table('realisationTable', [
      { label: 'Observable', render: r => r.label },
      { label: 'Checkpoints', key: 'n', num: true },
      { label: 'Median gap', num: true, render: r => fmt(r.summary.median) },
      { label: 'IQR', num: true, render: r => `${fmt(r.summary.q25)} – ${fmt(r.summary.q75)}` },
      { label: 'More concentrated / less', num: true, render: r => r.name === 'normalised_deficit' ? `${r.positive} / ${r.negative}` : `${r.negative} / ${r.positive}` },
      { label: 'Median retention', num: true, render: r => r.retention ? StatsUI.pct(r.retention.median) : '—' },
      { label: 'Status', render: r => StatsUI.tag(r.status) },
      { label: 'Note', render: r => r.note || r.unit }
    ], items.filter(i => i.n));
  }

  // --------------------------------------------------------------- temporal
  function pairedSmallMultiples(hostId, list, names) {
    const container = document.getElementById(hostId);
    container.replaceChildren();
    let drawn = 0;
    for (const name of names) {
      const order = side => side.includes('structural') ? 0 : side.includes('conditional') ? 1 : 2;
      const rows = list.filter(t => t.name === name && t.n).sort((a, b) => order(a.side) - order(b.side));
      if (!rows.length) continue;
      const id = `${hostId}_${name}`;
      const box = document.createElement('div');
      box.innerHTML = `<h3></h3><div id="${id}" class="chart"></div>`;
      box.querySelector('h3').textContent = rows[0].label;
      container.append(box);
      const traces = rows.map(t => {
        const conditional = t.side.includes('conditional');
        const color = t.side === 'response' || t.side === 'training' ? C.third : conditional ? C.conditional : C.structural;
        return StatsUI.line(t.side_label, t.steps, t.values, color);
      });
      const shapes = [];
      for (const t of rows) for (const tp of t.turning_points || []) {
        shapes.push({ type: 'line', x0: tp.step, x1: tp.step, yref: 'paper', y0: 0, y1: 1, line: { color: '#c8d2df', width: 1, dash: 'dot' } });
      }
      StatsUI.plot(id, traces, { xaxis: { title: 'Optimizer update' }, yaxis: { title: '' }, shapes, margin: { l: 56, r: 10, t: 30, b: 42 }, showlegend: rows.length > 1 });
      drawn++;
    }
    return drawn;
  }

  function renderTemporal() {
    const path = stats.trajectories.path || [];
    pairedSmallMultiples('temporalCharts', path, ['log_eta1', 'log_eta2', 'normalised_deficit', 'H1', 'logZ']);
    const rank = t => ({ structural: 0, conditional: 1, training: 2 })[t.side] ?? 3;
    const rows = [...path, ...(stats.trajectories.training || [])].filter(t => t.n).sort((a, b) => rank(a) - rank(b));
    StatsUI.table('temporalTable', [
      { label: 'Measure', render: r => r.side_label },
      { label: 'Observable', render: r => r.label },
      { label: 'n', key: 'n', num: true },
      { label: 'First', num: true, render: r => `${fmt(r.first)} @${r.first_step}` },
      { label: 'Last', num: true, render: r => `${fmt(r.last)} @${r.last_step}` },
      { label: 'Change', num: true, render: r => fmt(r.change) },
      { label: 'Trend ρ', num: true, render: r => fmt(r.trend_rho, 3) },
      { label: 'Turning points', render: r => (r.turning_points || []).map(t => `${t.kind} @${t.step}`).join(', ') || '—' },
      { label: 'Status', render: r => StatsUI.tag(r.status) }
    ], rows);
  }

  // --------------------------------------------------------------- coupling
  function renderCoupling() {
    const c = stats.coupling;
    StatsUI.note('couplingNote', c.keys.length > 1 ? c.note + ' Each observable appears once per identity class: log η₁ stands for S_def and H₁.' :
      'Coupling needs at least two observables recorded at four or more shared checkpoints.');
    if (c.keys.length < 2) { Plotly.purge(document.getElementById('couplingChart')); return; }
    const labels = c.keys.map(k => k.short || k.label);
    const text = c.rho.map(row => row.map(v => v === null ? '' : v.toFixed(2)));
    const custom = c.n.map(row => row.map(n => n));
    StatsUI.plot('couplingChart', [{
      type: 'heatmap', z: c.rho, x: labels, y: labels, text, texttemplate: '%{text}', customdata: custom,
      zmin: -1, zmax: 1, xgap: 2, ygap: 2,
      colorscale: [[0, C.structural], [0.5, '#f2f1ee'], [1, C.conditional]],
      colorbar: { title: { text: 'ρ' }, thickness: 12 },
      hovertemplate: '%{y}<br>%{x}<br>ρ = %{z:.3f} · n = %{customdata}<extra></extra>'
    }], { margin: { l: 210, r: 20, t: 10, b: 130 }, xaxis: { tickangle: -30, gridcolor: 'rgba(0,0,0,0)' },
          yaxis: { autorange: 'reversed', gridcolor: 'rgba(0,0,0,0)' }, hovermode: 'closest' });
  }

  // --------------------------------------------------------------- geometry
  function renderGeometry() {
    const g = stats.trajectories.geometry || [];
    const drawn = pairedSmallMultiples('geometryCharts', g, ['dstar', 'xi50', 'deff', 'qualified_fraction']);
    StatsUI.note('geometryNote', drawn ? 'Saved legacy bcan-1 profiles (36 scales). A missing d* means no qualified scaling window; it is never imputed. ' +
      'For the manuscript gates and 64-scale remeasurement, open the ECML auditor.' :
      'This context has no geometry measurements. Choose a scheduled analysis context (all outputs) to see d*, ξ₅₀ and d_eff.');
    StatsUI.table('geometryTable', [
      { label: 'Measure', render: r => r.side_label }, { label: 'Observable', render: r => r.label }, { label: 'n', key: 'n', num: true },
      { label: 'Median', num: true, render: r => fmt(r.summary.median) }, { label: 'First → last', num: true, render: r => `${fmt(r.first)} → ${fmt(r.last)}` },
      { label: 'Trend ρ', num: true, render: r => fmt(r.trend_rho, 3) }, { label: 'Status', render: r => StatsUI.tag(r.status) }
    ], g.filter(t => t.n), '');
  }

  function renderResponse() {
    const drawn = pairedSmallMultiples('responseCharts', stats.trajectories.response || [], ['delta', 'cancellation', 'contrast_l2', 'path_hellinger']);
    if (!drawn) document.getElementById('responseCharts').textContent = 'No response observables in this context.';
  }

  // ----------------------------------------------------------------- cohort
  async function loadCohorts() {
    const d = await api(`/api/stats/run/${rid}/cohorts`);
    const select = document.getElementById('cohortSelect');
    const previous = select.value;
    select.replaceChildren(...d.cohorts.map(c => new Option(`Update ${c.step} · ${c.split} · ${c.policy} reference · n=${c.size} · threshold ${c.threshold}`, c.job)));
    const empty = !d.cohorts.length;
    const applicable = host.dataset.engine === 'native' && ['pneumonia', 'synthetic'].includes(host.dataset.dataset);
    document.getElementById('secCohort').hidden = empty && !applicable;
    document.getElementById('cohortNavLink').hidden = empty && !applicable;
    document.getElementById('cohortBody').hidden = empty;
    StatsUI.note('cohortEmpty', empty ? (['pneumonia', 'synthetic'].includes(host.dataset.dataset)
      ? 'No cohort evaluation recorded yet. Run one in Healthcare → Cohort evidence; its statistics appear here.'
      : 'Cohort evidence applies to binary healthcare experiments.') : '');
    if (empty) return;
    if (previous && d.cohorts.some(c => c.job === previous)) select.value = previous;
    await loadCohort();
  }

  async function loadCohort() {
    const job = document.getElementById('cohortSelect').value;
    if (!job) return;
    cohort = await api(`/api/stats/cohort/${job}`);
    const obsSelect = document.getElementById('cohortObservable');
    const names = cohort.errors.map(e => [e.observable, e.label]);
    const previous = obsSelect.value;
    obsSelect.replaceChildren(...names.map(([k, l]) => new Option(l, k)));
    obsSelect.value = names.some(n => n[0] === previous) ? previous : (names.find(n => n[0] === 'path_hellinger') || names[0] || [])[0];
    const m = cohort.metrics || {};
    const tiles = document.getElementById('cohortTiles');
    const tile = (label, value, note) => { const el = document.createElement('div'); el.className = 'tile'; el.innerHTML = '<div class="tileLabel"></div><div class="tileValue"></div><div class="tileNote"></div>'; el.children[0].textContent = label; el.children[1].textContent = value; el.children[2].textContent = note; return el; };
    tiles.replaceChildren(
      tile('Predictive accuracy', fmt(m.accuracy, 3), `${m.n ?? '—'} images · threshold ${m.threshold ?? '—'}`),
      tile('Sensitivity · specificity', `${fmt(m.sensitivity, 3)} · ${fmt(m.specificity, 3)}`, 'Whole split'),
      tile('ROC AUC (probability)', fmt(m.roc_auc, 3), `Brier ${fmt(m.brier, 3)} · ECE ${fmt(m.ece, 3)}`),
      tile('Intrinsic sample', String(cohort.sampled), `${cohort.policy} reference · of ${cohort.population ?? '—'} images`)
    );
    StatsUI.note('cohortDependence', `${cohort.unit}. ${cohort.dependence}`);
    StatsUI.note('cohortHolm', `${cohort.holm} ${cohort.redundancy}${cohort.excluded_observables?.length ? ' Not tested: ' + cohort.excluded_observables.map(e => e.name).join(', ') + ' (identity transform).' : ''}`);
    StatsUI.note('cohortConfounding', cohort.confounding || '');
    renderCohortCharts();
    const cmp = [
      { label: 'Observable', render: r => r.label },
      { label: 'n', num: true, render: r => r.n.join(' / ') },
      { label: 'Medians', num: true, render: r => r.summaries.map(s => fmt(s.median)).join(' / ') },
      { label: 'AUROC', num: true, render: r => fmt(r.auroc, 3) },
      { label: '95% CI', num: true, render: r => r.auroc_interval ? `${fmt(r.auroc_interval[0], 3)} – ${fmt(r.auroc_interval[1], 3)}` : '—' },
      { label: 'r_rb', num: true, render: r => fmt(r.rank_biserial, 3) },
      { label: 'p', num: true, render: r => StatsUI.p(r.p) },
      { label: 'p (Holm)', num: true, render: r => StatsUI.p(r.p_holm) },
      { label: 'Status', render: r => { const s = StatsUI.tag(r.status); if (r.reason) s.title = r.reason; return s; } }
    ];
    const errorCount = (cohort.groups.FP?.n || 0) + (cohort.groups.FN?.n || 0);
    if (!errorCount) document.getElementById('cohortErrors').textContent = `No misclassified images in this sample (0 of ${cohort.sampled}); the error comparison is undefined. Choose an earlier checkpoint or a harder split.`;
    else StatsUI.table('cohortErrors', cmp, cohort.errors);
    StatsUI.table('cohortLabels', cmp, cohort.labels);
    StatsUI.table('cohortOmnibus', [
      { label: 'Observable', render: r => r.label }, { label: 'Groups', render: r => (r.groups || []).join(' · ') },
      { label: 'H', num: true, render: r => fmt(r.H, 3) }, { label: 'ε²', num: true, render: r => fmt(r.epsilon_squared, 3) },
      { label: 'p', num: true, render: r => StatsUI.p(r.p) }, { label: 'p (Holm)', num: true, render: r => StatsUI.p(r.p_holm) },
      { label: 'Status', render: r => { const s = StatsUI.tag(r.status); if (r.reason) s.title = r.reason; return s; } }
    ], cohort.omnibus);
  }

  function renderCohortCharts() {
    if (!cohort) return;
    const name = document.getElementById('cohortObservable').value;
    const traces = ['TN', 'FP', 'FN', 'TP'].map((g, i) => ({
      type: 'box', name: `${g} (n=${cohort.groups[g].n})`, y: cohort.groups[g].values[name] || [],
      boxpoints: 'all', jitter: .35, pointpos: 0, marker: { color: C.groups[i], size: 7, line: { color: '#fff', width: 1 } },
      line: { color: C.groups[i], width: 1.5 }, fillcolor: 'rgba(255,255,255,0)', hovertemplate: '%{y:.5g}<extra>' + g + '</extra>'
    }));
    StatsUI.plot('cohortGroups', traces, { showlegend: false, xaxis: { title: 'Prediction outcome' }, yaxis: { title: cohort.errors.find(e => e.observable === name)?.label || name }, hovermode: 'closest' });
    const tested = cohort.errors.filter(e => e.auroc !== undefined && e.auroc !== null);
    const forest = document.getElementById('cohortForest');
    if (!tested.length) {
      Plotly.purge(forest);
      forest.dataset.message = '1';
      const errors = (cohort.groups.FP?.n || 0) + (cohort.groups.FN?.n || 0);
      document.getElementById('cohortForest').textContent = errors ? `No comparison: ${errors} misclassified image(s); at least five per group are required.` : 'No misclassified images in this sample.';
      return;
    }
    // Clear a previous message only; clearing a live plot's DOM would orphan Plotly's state.
    if (forest.dataset.message) { forest.textContent = ''; delete forest.dataset.message; }
    StatsUI.plot('cohortForest', [{
      type: 'scatter', mode: 'markers', x: tested.map(e => e.auroc), y: tested.map(e => e.label),
      error_x: { type: 'data', symmetric: false, array: tested.map(e => e.auroc_interval[1] - e.auroc), arrayminus: tested.map(e => e.auroc - e.auroc_interval[0]), color: '#8794a8', thickness: 1.5, width: 0 },
      marker: { color: C.structural, size: 9, line: { color: '#fff', width: 2 } },
      hovertemplate: '%{y}<br>AUROC %{x:.3f}<extra></extra>'
    }], { xaxis: { title: 'AUROC · misclassified ranked above correct', range: [0, 1] },
          shapes: [{ type: 'line', x0: .5, x1: .5, yref: 'paper', y0: 0, y1: 1, line: { color: '#9aa6b8', dash: 'dot', width: 1 } }],
          hovermode: 'closest', margin: { l: 190, r: 16, t: 16, b: 48 }, yaxis: { automargin: true, autorange: 'reversed' } });
  }

  // -------------------------------------------------------- ruler sensitivity
  let rulerRecords = [];

  async function loadRulerRecords() {
    const d = await api(`/api/stats/run/${rid}/geometry-records`);
    rulerRecords = d.records;
    const card = document.getElementById('rulerCard');
    card.hidden = host.dataset.engine !== 'native';
    const select = document.getElementById('rulerRecord');
    const previous = select.value;
    select.replaceChildren(...d.records.map(r => new Option(
      `Update ${r.step} · ${r.scope === 'all' ? 'all outputs' : 'selected readout'} · test[${r.x}] vs ${r.reference_type === 'zero' ? 'zero' : 'test[' + r.reference + ']'}${r.gated ? ' · gated' : ''} · ${r.nodes} nodes${r.measured ? '' : ' · BCANw not measured'}`, r.job)));
    if (!d.records.length) select.append(new Option('No saved geometry analysis yet', ''));
    if (previous && d.records.some(r => r.job === previous)) select.value = previous;
    const done = (d.previous || []).find(p => p.status === 'completed');
    if (done && !document.getElementById('rulerTable').childElementCount) {
      select.value = done.request.source_job;
      document.getElementById('rulerEps').value = done.request.epsilons.join(', ');
      document.getElementById('rulerPower').value = done.request.power;
      renderRuler((await api('/api/job/' + done.job)).data);
    }
  }

  async function runRuler() {
    const source = document.getElementById('rulerRecord').value;
    if (!source) { showError(Error('Choose a saved geometry analysis.')); return; }
    const state = document.getElementById('rulerState');
    try {
      clearError();
      document.getElementById('rulerRun').disabled = true;
      const job = await api('/api/stats/ruler', { run: rid, source_job: source, epsilons: document.getElementById('rulerEps').value, power: +document.getElementById('rulerPower').value });
      const data = await awaitJob(job.id, t => { state.textContent = t; });
      state.textContent = `Computed in ${data.seconds} s · ${data.nodes} nodes`;
      renderRuler(data);
    } catch (e) { showError(e); state.textContent = ''; }
    finally { document.getElementById('rulerRun').disabled = false; }
  }

  function renderRuler(data) {
    const traces = (key, transform = v => v) => data.sides.filter(s => s.status === 'available').map(s => StatsUI.line(
      s.side === 'structural' ? 'Structural' : 'Conditional', s.rows.map(r => r.epsilon), s.rows.map(r => transform(r[key])),
      s.side === 'structural' ? C.structural : C.conditional, { hovertemplate: 'ε %{x}<br>%{y:.4g}<extra>' + s.side + '</extra>' }));
    const recorded = [{ type: 'line', x0: data.recorded.epsilon, x1: data.recorded.epsilon, yref: 'paper', y0: 0, y1: 1, line: { color: '#9aa6b8', dash: 'dot', width: 1 } }];
    const note = [{ x: Math.log10(data.recorded.epsilon), yref: 'paper', y: 1.02, text: 'recorded ε', showarrow: false, font: { size: 10, color: '#65748a' } }];
    StatsUI.plot('rulerDstar', traces('dstar'), { xaxis: { title: 'ε (log scale)', type: 'log' }, yaxis: { title: 'd* · maximal qualified exponent' }, shapes: recorded, annotations: note, hovermode: 'closest' });
    StatsUI.plot('rulerXi', traces('floor_fraction'), { xaxis: { title: 'ε (log scale)', type: 'log' }, yaxis: { title: 'Edges at the ε floor (|E_ℓ|F < ε)', tickformat: '.0%', range: [0, 1] }, shapes: recorded, annotations: note, hovermode: 'closest' });
    const rows = data.sides.flatMap(s => s.rows.map(r => ({ ...r, side: s.side })));
    StatsUI.table('rulerTable', [
      { label: 'Measure', render: r => r.side === 'structural' ? 'Structural' : 'Conditional' },
      { label: 'ε', num: true, render: r => fmt(r.epsilon) + (r.recorded_convention ? ' (recorded)' : '') },
      { label: 'p', num: true, render: r => fmt(r.power) },
      { label: 'd*', num: true, render: r => fmt(r.dstar, 3) },
      { label: 'd_eff', num: true, render: r => fmt(r.deff, 3) },
      { label: 'ξ₅₀ / diameter', num: true, render: r => fmt(r.xi50_over_diameter, 3) },
      { label: 'Qualified windows', num: true, render: r => `${r.qualified_windows} / ${r.scales}` },
      { label: 'Edges at floor', num: true, render: r => StatsUI.pct(r.floor_fraction) }
    ], rows);
    const parts = [];
    for (const s of data.sides) {
      if (s.status !== 'available') { parts.push(`${s.side}: measure undefined.`); continue; }
      const d = s.rows.map(r => r.dstar).filter(v => v !== null);
      const span = d.length ? `d* ranges ${fmt(Math.min(...d), 3)}–${fmt(Math.max(...d), 3)} over ε ${fmt(s.rows[0].epsilon)}–${fmt(s.rows.at(-1).epsilon)}` : 'no qualified window at any ε';
      const repro = s.reproduction ? (s.reproduction.reproduced ? 'recomputation at the recorded convention reproduces the saved d* and ξ₅₀' : `recomputation at the recorded convention gives d* ${fmt(s.reproduction.recomputed_dstar, 4)} against the saved ${fmt(s.reproduction.saved_dstar, 4)}`) : 'p differs from the recorded convention, so no reproduction check applies';
      parts.push(`${s.side === 'structural' ? 'Structural' : 'Conditional'}: ${span}; ${repro}.`);
    }
    document.getElementById('rulerReading').textContent = `Update ${data.step}, ${data.nodes} nodes, p = ${fmt(data.request.power)}. ` + parts.join(' ') +
      ' A d* that moves strongly with ε is conditional on the declared ruler; report the ε used alongside it.';
  }

  document.getElementById('rulerRun').onclick = runRuler;

  // ----------------------------------------------------------------- ledger
  function ledgerRequest(format) {
    return { run: rid, context: document.getElementById('contextSelect').value, cohort: document.getElementById('cohortSelect').value || null, format };
  }

  async function renderLedger() {
    const d = await api('/api/stats/ledger', ledgerRequest('json'));
    StatsUI.table('ledgerTable', [
      { label: 'Section', key: 'section', render: r => r.section }, { label: 'Quantity', render: r => r.quantity },
      { label: 'Unit', render: r => r.unit || '' }, { label: 'n', num: true, render: r => r.n ?? '—' },
      { label: 'Estimate', num: true, render: r => fmt(r.estimate) }, { label: 'Interval', num: true, render: r => r.interval ? fmt(r.interval) : '—' },
      { label: 'p (Holm)', num: true, render: r => StatsUI.p(r.p_holm ?? r.p) }, { label: 'Effect', num: true, render: r => typeof r.effect === 'number' ? fmt(r.effect, 3) : (r.effect || '—') },
      { label: 'Status', render: r => StatsUI.tag(r.status) }
    ], d.rows);
  }

  async function exportLedger(format, extension, mime) {
    try {
      const response = await fetch('/api/stats/ledger', { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-CSRF-Token': document.querySelector('meta[name=csrf]').content }, body: JSON.stringify(ledgerRequest(format)) });
      if (!response.ok) throw Error((await response.json()).error);
      StatsUI.download(await response.text(), `pathflow_ledger_${rid}.${extension}`, mime);
    } catch (e) { showError(e); }
  }

  document.getElementById('ledgerCsv').onclick = () => exportLedger('csv', 'csv', 'text/csv');
  document.getElementById('ledgerTex').onclick = () => exportLedger('tex', 'tex', 'text/x-tex');
  document.getElementById('ledgerJson').onclick = () => saveJSON({ statistics: stats, cohort }, `pathflow_statistics_${rid}.json`);
  document.getElementById('contextSelect').onchange = () => loadStats().catch(showError);
  document.getElementById('cohortSelect').onchange = () => loadCohort().then(renderLedger).catch(showError);
  document.getElementById('cohortObservable').onchange = renderCohortCharts;
  document.getElementById('refreshStats').onclick = () => init().catch(showError);

  async function init() {
    loadRulerRecords().catch(showError);
    await loadContexts();
    await loadCohorts();
    await loadStats();
  }
  init().catch(showError);
})();
