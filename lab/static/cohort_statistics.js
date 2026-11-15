/* Cross-run statistics: one value per independently trained run, grouped by a factor. */
(() => {
  const C = StatsUI.colors;
  const fmt = StatsUI.fmt;
  const NAMES = {
    conditional: ['log_eta1', 'log_eta2', 'normalised_deficit', 'H1', 'logZ'],
    structural: ['log_eta1', 'log_eta2', 'normalised_deficit', 'H1', 'logZ'],
    geometry_structural: ['dstar', 'xi50', 'deff', 'qualified_fraction'],
    geometry_conditional: ['dstar', 'xi50', 'deff', 'qualified_fraction'],
    response: ['delta', 'cancellation', 'contrast_l2'],
    training: ['val_loss', 'val_accuracy', 'train_loss', 'test_accuracy', 'test_loss']
  };
  const LABELS = {
    log_eta1: 'log η₁', log_eta2: 'log η₂', normalised_deficit: 'Normalised entropy deficit', H1: 'Path entropy H₁', logZ: 'log Q',
    dstar: 'd*', xi50: 'ξ₅₀', deff: 'd_eff', qualified_fraction: 'Qualified window fraction', delta: 'Δf', cancellation: 'Cancellation',
    contrast_l2: '‖Δf‖₂', val_loss: 'Validation loss', val_accuracy: 'Validation accuracy', train_loss: 'Training loss',
    test_accuracy: 'Test accuracy', test_loss: 'Test loss'
  };
  let runs = [], last = null;

  const architecture = r => r.config.model === 'cnn' ? `channels ${(r.config.channels || []).join('-')}${r.config.cnn_head ? ' + head ' + (r.config.head_widths || []).join('-') : ''}` : `widths ${(r.config.widths || []).join('-')}`;
  // Strata share dataset, family, task and architecture; choose an architecture factor to compare across them.
  const stratumKey = r => `${r.config.dataset} · ${String(r.config.model).toUpperCase()} · ${architecture(r)} · ${r.config.task || 'classification'}`;

  function setNames() {
    const side = $('side').value, select = $('name'), previous = select.value;
    select.replaceChildren(...NAMES[side].map(n => new Option(LABELS[n], n)));
    if (NAMES[side].includes(previous)) select.value = previous;
    if (side.startsWith('geometry')) $('contextKind').value = 'analysis';
  }

  function renderPicker() {
    const stratum = $('stratum').value;
    const picker = $('runPicker');
    picker.replaceChildren();
    // Grouping by an architecture factor spans architectures: show all runs of that dataset and family.
    const spans = ['widths', 'channels'].includes($('factor').value);
    const base = stratum.split(' · ').slice(0, 2).join(' · ');
    for (const r of runs.filter(r => spans ? stratumKey(r).startsWith(base) : stratumKey(r) === stratum)) {
      const label = document.createElement('label');
      const box = document.createElement('input');
      box.type = 'checkbox'; box.value = r.id; box.checked = true;
      label.append(box, `${r.config.name} · ${r.config.optimizer} · ${r.config.activation} · seed ${r.config.seed} · update ${r.step} · ${r.status}`);
      picker.append(label);
    }
  }

  async function loadRuns() {
    const d = await api('/api/runs/summary');
    runs = d.runs.filter(r => r.engine === 'native');
    const strata = [...new Set(runs.map(stratumKey))];
    const counts = Object.fromEntries(strata.map(s => [s, runs.filter(r => stratumKey(r) === s).length]));
    $('stratum').replaceChildren(...strata.sort((a, b) => counts[b] - counts[a]).map(s => new Option(`${s} · ${counts[s]} runs`, s)));
    renderPicker();
    if (!runs.length) $('runPicker').textContent = 'No MLP/CNN experiments yet.';
  }

  function request() {
    const state = $('state').value === 'update' ? String($('update').value) : $('state').value;
    return {
      runs: [...$('runPicker').querySelectorAll('input:checked')].map(i => i.value),
      side: $('side').value, name: $('name').value, state, factor: $('factor').value,
      context_kind: $('contextKind').value, independent: $('independent').checked
    };
  }

  function reading(d) {
    const inf = d.inference;
    const groups = d.groups.map(g => `${g.label}: median ${fmt(g.summary.median)} (n=${g.n})`).join('; ');
    if (inf.status !== 'exploratory') return `${groups}. ${inf.reason || ''}`;
    if (inf.kendall_w !== undefined) {
      return `${groups}. Across ${inf.blocks} matched blocks the ${d.factor} ordering is ${inf.kendall_w >= .7 ? 'highly' : inf.kendall_w >= .3 ? 'moderately' : 'weakly'} concordant: ` +
        `Kendall W = ${fmt(inf.kendall_w, 3)}, Friedman χ² = ${fmt(inf.statistic, 3)}, p = ${StatsUI.p(inf.p)}. With few blocks, W can be large while p remains above conventional thresholds.`;
    }
    if (inf.rank_biserial !== undefined) return `${groups}. Matched-pair Wilcoxon over ${inf.blocks} blocks: rank-biserial ${fmt(inf.rank_biserial, 3)}, Hodges–Lehmann shift ${fmt(inf.hodges_lehmann)}, p = ${StatsUI.p(inf.p)}.`;
    return `${groups}. Kruskal–Wallis H = ${fmt(inf.statistic, 3)}, ε² = ${fmt(inf.epsilon_squared, 3)}, p = ${StatsUI.p(inf.p)}.`;
  }

  async function compare() {
    const body = request();
    if (body.runs.length < 2) { showError(Error('Select at least two runs.')); return; }
    clearError();
    $('compareState').textContent = 'Computing from recorded observables…';
    const d = await api('/api/stats/cross', body);
    last = { request: body, result: d };
    $('compareState').textContent = `${d.runs.length} runs · ${d.seconds}s`;
    $('results').hidden = false;
    $('resultTitle').textContent = `${d.label} · ${({ structural: 'structural', conditional: 'conditional', geometry_structural: 'structural geometry', geometry_conditional: 'conditional geometry', response: 'response', training: 'training' })[d.side]} · by ${d.factor} · ${d.state === 'final' ? 'final checkpoint' : d.state === 'initial' ? 'initialisation' : 'update ' + d.state}`;
    $('inferenceTag').replaceChildren(StatsUI.tag(d.inference.status));
    $('inferenceReading').textContent = (d.warnings?.length ? d.warnings.join(' ') + ' ' : '') + reading(d) + (d.rank_retention?.rho !== null && d.rank_retention ? ` Rank retention from initialisation: ρ = ${fmt(d.rank_retention.rho, 3)} (n=${d.rank_retention.n}).` : '');
    const band = [], strip = [];
    d.groups.forEach((g, i) => {
      const color = C.groups[i % C.groups.length];
      if (g.band.steps.length) {
        band.push({ type: 'scatter', mode: 'lines', x: g.band.steps.concat([...g.band.steps].reverse()), y: g.band.q75.concat([...g.band.q25].reverse()),
                    fill: 'toself', fillcolor: color + '22', line: { width: 0 }, hoverinfo: 'skip', showlegend: false, legendgroup: g.label });
        band.push(StatsUI.line(`${g.label} (n=${g.n})`, g.band.steps, g.band.median, color, { legendgroup: g.label }));
      }
      const members = d.runs.filter(r => r.level === g.level && r.value !== null);
      strip.push({ type: 'box', name: g.label, y: members.map(r => r.value), text: members.map(r => `${r.name}`), boxpoints: 'all', jitter: .3, pointpos: 0,
                   marker: { color, size: 9, line: { color: '#fff', width: 2 } }, line: { color, width: 1.5 }, fillcolor: 'rgba(255,255,255,0)',
                   hovertemplate: '%{text}<br>%{y:.5g}<extra></extra>' });
    });
    StatsUI.plot('bandChart', band, { xaxis: { title: 'Optimizer update' }, yaxis: { title: d.label } });
    StatsUI.plot('stripChart', strip, { showlegend: false, xaxis: { title: d.factor }, yaxis: { title: d.label }, hovermode: 'closest' });
    StatsUI.table('groupTable', [
      { label: d.factor, render: r => r.label }, { label: 'Runs', key: 'n', num: true },
      { label: 'Median', num: true, render: r => fmt(r.summary.median) }, { label: 'IQR', num: true, render: r => `${fmt(r.summary.q25)} – ${fmt(r.summary.q75)}` },
      { label: 'Range', num: true, render: r => `${fmt(r.summary.minimum)} – ${fmt(r.summary.maximum)}` }
    ], d.groups);
    StatsUI.note('blockNote', `${d.unit}. ${d.block_definition} ${d.inference.design ? 'Design: ' + d.inference.design + '. ' + d.inference.test + '.' : ''}` +
      (d.excluded.length ? ` Excluded: ${d.excluded.length} run(s) without a defined value.` : ''));
    StatsUI.table('runTable', [
      { label: 'Run', render: r => { const a = document.createElement('a'); a.href = `/run/${r.run}/statistics`; a.textContent = r.name; return a; } },
      { label: d.factor, render: r => r.level_label }, { label: 'Seed', key: 'seed', num: true },
      { label: 'State update', num: true, render: r => r.step ?? '—' }, { label: 'Value', num: true, render: r => fmt(r.value) },
      { label: 'Initial', num: true, render: r => fmt(r.initial) }
    ], d.runs);
  }

  async function exportLedger(format, extension, mime) {
    if (!last) { showError(Error('Compare runs first.')); return; }
    try {
      const response = await fetch('/api/stats/ledger', { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-CSRF-Token': document.querySelector('meta[name=csrf]').content }, body: JSON.stringify({ cross: last.request, format }) });
      if (!response.ok) throw Error((await response.json()).error);
      StatsUI.download(await response.text(), `pathflow_cross_run.${extension}`, mime);
    } catch (e) { showError(e); }
  }

  $('side').onchange = setNames;
  $('stratum').onchange = renderPicker;
  $('factor').onchange = renderPicker;
  $('state').onchange = () => { $('updateLabel').hidden = $('state').value !== 'update'; };
  $('selectAll').onclick = () => $('runPicker').querySelectorAll('input').forEach(i => { i.checked = true; });
  $('selectNone').onclick = () => $('runPicker').querySelectorAll('input').forEach(i => { i.checked = false; });
  $('compare').onclick = () => compare().catch(e => { $('compareState').textContent = ''; showError(e); });
  $('exportCsv').onclick = () => exportLedger('csv', 'csv', 'text/csv');
  $('exportTex').onclick = () => exportLedger('tex', 'tex', 'text/x-tex');
  setNames();
  loadRuns().catch(showError);
})();
