/* Shared helpers for the overview and statistics views. */
const StatsUI = {
  colors: { structural: '#2D5DA6', conditional: '#C17A14', third: '#1F9D6E', groups: ['#2D5DA6', '#C17A14', '#1F9D6E', '#A0457E'] },

  fmt(v, digits = 4) {
    if (v === null || v === undefined || Number.isNaN(v)) return '—';
    if (Array.isArray(v)) return '[' + v.map(x => StatsUI.fmt(x, digits)).join(', ') + ']';
    if (typeof v !== 'number') return String(v);
    if (v === 0) return '0';
    const a = Math.abs(v);
    return a < 1e-3 || a >= 1e5 ? v.toExponential(2) : Number(v.toPrecision(digits)).toString();
  },

  /** Percentages for readable fractions; scientific notation (as a fraction) below 0.01%. */
  pct(v) { return v === null || v === undefined ? '—' : v > 0 && v < 1e-4 ? `${v.toExponential(2)} (fraction)` : `${StatsUI.fmt(100 * v, 3)}%`; },

  p(v) { return v === null || v === undefined ? '—' : v < 1e-4 ? v.toExponential(1) : v.toFixed(4); },

  tag(status) {
    const s = document.createElement('span');
    s.className = 'statusTag ' + (status || '');
    s.textContent = status || 'unavailable';
    s.title = {
      descriptive: 'Summary of recorded values; no sampling model is assumed.',
      exploratory: 'Test or interval under the declared design; not preregistered; not corrected across interactive selections.',
      identity: 'Exact numerical relation of recorded quantities.',
      insufficient: 'The selected design does not support this quantity.',
      unavailable: 'Not recorded for this selection.'
    }[status] || '';
    return s;
  },

  /** Build a table. columns: [{key, label, num, render}] */
  table(host, columns, rows, empty = 'No records for this selection.') {
    const el = typeof host === 'string' ? document.getElementById(host) : host;
    if (!rows.length) { el.textContent = empty; return; }
    const table = document.createElement('table');
    table.className = 'statTable';
    const head = document.createElement('tr');
    for (const c of columns) { const th = document.createElement('th'); th.textContent = c.label; head.append(th); }
    table.createTHead().append(head);
    const body = table.createTBody();
    for (const row of rows) {
      const tr = document.createElement('tr');
      for (const c of columns) {
        const td = document.createElement('td');
        if (c.num) td.className = 'num';
        const value = c.render ? c.render(row) : StatsUI.fmt(row[c.key]);
        if (value instanceof Node) td.append(value); else td.textContent = value;
        tr.append(td);
      }
      body.append(tr);
    }
    el.replaceChildren(table);
  },

  /** Plotly with the studio's scientific styling; x title must be supplied. */
  plot(id, traces, layout = {}) {
    const host = document.getElementById(id);
    if (!host || typeof Plotly === 'undefined') return;
    const base = {
      autosize: true, paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: '#ffffff',
      font: { family: '\"Archivo Instrument\",Archivo,\"Helvetica Neue\",Arial,sans-serif', color: '#3A4556', size: 12 },
      margin: { l: 62, r: 16, t: 26, b: 48 },
      legend: { orientation: 'h', y: 1.16, x: 0, font: { size: 11 } },
      xaxis: { gridcolor: '#ECEAE4', zerolinecolor: '#C9C5BA', automargin: true, linecolor: '#C9C5BA' },
      yaxis: { gridcolor: '#ECEAE4', zerolinecolor: '#C9C5BA', automargin: true, linecolor: '#C9C5BA' },
      hovermode: 'x unified', hoverlabel: { bgcolor: '#ffffff', bordercolor: '#C9C5BA', font: { color: '#0E1A2B' } },
      uirevision: id
    };
    const final = { ...base, ...layout, xaxis: { ...base.xaxis, ...layout.xaxis }, yaxis: { ...base.yaxis, ...layout.yaxis } };
    Plotly.react(host, traces, final, {
      responsive: true, displaylogo: false, displayModeBar: 'hover',
      toImageButtonOptions: { format: 'svg', filename: 'pathflow_' + id }, modeBarButtonsToRemove: ['lasso2d', 'select2d']
    });
    if (!host._observed && typeof ResizeObserver !== 'undefined') {
      new ResizeObserver(() => { if (host.clientWidth && host.data) Plotly.Plots.resize(host); }).observe(host);
      host._observed = true;
    }
  },

  line(name, steps, values, color, extra = {}) {
    // Dense series: smaller markers without rings, so the line reads as continuous.
    const dense = (steps || []).length > 14;
    return { type: 'scatter', mode: 'lines+markers', name, x: steps, y: values, connectgaps: false,
             line: { color, width: 2 }, marker: dense ? { size: 5, color } : { size: 8, color, line: { color: '#ffffff', width: 2 } },
             hovertemplate: '%{y:.5g}<extra>' + name + '</extra>', ...extra };
  },

  note(id, text) { const el = document.getElementById(id); if (el) el.textContent = text; },

  download(text, name, mime) { saveText(text, name, mime); }
};
if (typeof module !== 'undefined') module.exports = StatsUI;
