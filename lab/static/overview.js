/* Run overview: recorded state, headline observables and the views of this run. */
(() => {
  const host = document.getElementById('overviewContext');
  const rid = host.dataset.run;
  const C = StatsUI.colors;
  const fmt = StatsUI.fmt;

  const VIEW_TEXT = {
    training: 'Learning curves, run controls, checkpoints, schedule and job log.',
    geometry: 'Structural and conditional geometries, metric fidelity and hyperbolic audits.',
    investigation: 'Fields, routes, interventions and the five-stage healthcare workflow.',
    observatory: 'Observable trajectories, temporal histograms, scale profiles and contrast ensembles.',
    healthcare: 'Case, contrast, routes, mechanism tests and cohort evidence.',
    statistics: 'Disposition → realisation, temporal regimes, coupling, geometry and cohort tests.',
    auditor: 'ECML evidence: fidelity checks, distributions, 64-scale remeasurement and population audit.'
  };

  function tile(label, value, note) {
    const el = document.createElement('div');
    el.className = 'tile';
    el.innerHTML = '<div class="tileLabel"></div><div class="tileValue"></div><div class="tileNote"></div>';
    el.querySelector('.tileLabel').textContent = label;
    const v = el.querySelector('.tileValue');
    if (value instanceof Node) { v.replaceWith(value); } else v.textContent = value;
    el.querySelector('.tileNote').textContent = note || '';
    return el;
  }

  function pair(structural, conditional, transform = x => x) {
    const wrap = document.createElement('div');
    wrap.className = 'pairValue tileValue';
    for (const [cls, item] of [['structural', structural], ['conditional', conditional]]) {
      const span = document.createElement('span');
      span.innerHTML = `<i class="swatch ${cls}"></i>`;
      span.append(item ? fmt(transform(item.value), 3) : '—');
      span.title = cls === 'structural' ? 'Structural disposition' : 'Conditional realisation';
      wrap.append(span);
    }
    return wrap;
  }

  function contextLabel(c) {
    if (!c) return '';
    if (c.kind === 'probe') return `Fixed probe · test[${c.x}] vs test[${c.reference}] · readout ${c.readout?.a}${c.readout?.b !== undefined && c.readout?.b !== null ? '−' + c.readout.b : ''}`;
    if (c.kind === 'joint_analysis') return `Residual-DAG analysis · ${c.head}`;
    return `Scheduled analysis · ${c.readout?.scope === 'all' ? 'all outputs' : 'selected readout'} · test[${c.x}] vs ${c.reference_type === 'zero' ? 'zero' : 'test[' + c.reference + ']'}`;
  }

  async function load() {
    const d = await api(`/api/stats/run/${rid}/overview`);
    const t = d.tiles;
    document.getElementById('ovState').textContent =
      `${d.configuration.dataset} · ${d.engine === 'joint' ? (d.configuration.architecture || 'Residual DAG') : String(d.configuration.model).toUpperCase()} · ${d.configuration.optimizer ?? ''} · ${d.status} · committed update ${d.step}`;
    StatsUI.note('ovContext', contextLabel(d.probe_context));
    const tiles = document.getElementById('ovTiles');
    tiles.replaceChildren(
      tile('Committed update', String(d.step), `${d.status}${d.configuration.updates ? ` · ${d.configuration.updates} configured` : ''}`),
      tile('Validation loss', t.val_loss ? fmt(t.val_loss.value, 4) : '—', t.val_accuracy ? `Accuracy ${fmt(t.val_accuracy.value, 3)} at update ${t.val_accuracy.step}` : 'No validation record'),
      tile('Effective occupancy η₁', pair(t.structural_log_eta1, t.conditional_log_eta1, Math.exp),
           t.retention ? `Conditional retains ${StatsUI.pct(t.retention.value)} of structural η₁ at update ${t.retention.step}` : 'Structural · conditional'),
      tile('Normalised entropy deficit', pair(t.structural_deficit, t.conditional_deficit), 'Structural · conditional (S_def / log N)'),
      tile('Maximal qualified exponent d*', pair(t.structural_dstar, t.conditional_dstar),
           t.structural_dstar ? `Scheduled geometry at update ${t.structural_dstar.step}` : 'No scheduled geometry yet'),
      tile('Recorded jobs', String(Object.values(d.jobs).reduce((a, j) => a + j.completed, 0)),
           Object.entries(d.jobs).map(([k, j]) => `${k} ${j.completed}${j.pending ? ` (+${j.pending})` : ''}`).join(' · ') || 'None yet')
    );
    StatsUI.note('ovTileNote', 'η and deficit come from the fixed probe recorded at every committed checkpoint; d* from the scheduled all-output analyses. ' +
      'Undefined measures are shown as —. These are recorded values of one run, not population estimates.');

    const tr = d.series.training;
    const learning = [];
    for (const [key, name, color] of [['train_loss', 'Training loss', C.third], ['val_loss', 'Validation loss', C.structural], ['test_loss', 'Test loss', C.conditional]]) {
      if (tr[key]?.steps.length) learning.push(StatsUI.line(name, tr[key].steps, tr[key].values, color));
    }
    StatsUI.plot('ovLearning', learning, { xaxis: { title: 'Optimizer update' }, yaxis: { title: 'Loss' } });

    const s = d.series.structural_log_eta1, c = d.series.conditional_log_eta1;
    StatsUI.plot('ovRealisation', [
      StatsUI.line('Structural disposition', s.steps, s.values, C.structural),
      StatsUI.line('Conditional realisation', c.steps, c.values, C.conditional)
    ], { xaxis: { title: 'Optimizer update' }, yaxis: { title: 'log η₁' } });
    StatsUI.note('ovRealisationNote', s.steps.length ? 'The vertical gap is the log fraction of structural occupancy that the declared contrast realises.' :
      'No path observables have been recorded for this run yet.');

    const views = document.getElementById('ovViews');
    views.replaceChildren();
    for (const a of document.querySelectorAll('#runBar .runTabs a')) {
      if (a.dataset.tab === 'overview') continue;
      const card = document.createElement('a');
      card.className = 'viewCard';
      card.href = a.getAttribute('href');
      card.innerHTML = '<strong></strong><span></span>';
      card.querySelector('strong').textContent = a.textContent;
      card.querySelector('span').textContent = VIEW_TEXT[a.dataset.tab] || '';
      views.append(card);
    }

    StatsUI.table('ovConfig', [{ key: 'k', label: 'Setting' }, { key: 'v', label: 'Value', render: r => Array.isArray(r.v) ? r.v.join(', ') : String(r.v) }],
      Object.entries(d.configuration).map(([k, v]) => ({ k, v })));
    StatsUI.table('ovJobs', [{ key: 'kind', label: 'Job kind' }, { key: 'completed', label: 'Completed', num: true },
      { key: 'pending', label: 'Pending', num: true }, { key: 'failed', label: 'Failed', num: true }],
      Object.entries(d.jobs).map(([kind, j]) => ({ kind, ...j })));
    return d.status === 'training' || Object.values(d.jobs).some(j => j.pending);
  }

  Poller.every(() => load().catch(e => { showError(e); return false; }), { active: 5000, idle: 30000 });
})();
