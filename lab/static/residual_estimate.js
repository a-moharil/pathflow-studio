/* Work estimate for residual networks in Build: parameters and activation units of one
   forward pass, refreshed as the form changes. The general MLP/CNN estimate does not apply. */
(() => {
  const form = document.getElementById('jointDesign');
  if (!form) return;
  const out = document.createElement('p');
  out.id = 'jointEstimate';
  out.className = 'estimateLine';
  out.setAttribute('aria-live', 'polite');
  (document.getElementById('jointReadiness') || form).after(out);
  let timer = null, ticket = 0;
  const fmt = n => Number(n).toLocaleString('en-GB');
  async function update() {
    const mine = ++ticket;
    const d = Object.fromEntries(new FormData(form));
    if (form.elements.task) d.task = form.elements.task.value;
    d.pretrained = false;
    out.textContent = 'Estimating network size…';
    try {
      const e = await api('/api/joint/estimate', d);
      if (mine !== ticket) return;
      out.textContent = `${fmt(e.parameters)} parameters · ${fmt(e.activation_units)} activation units over ${fmt(e.operations)} operations at ${e.size} × ${e.size}` +
        (e.blocks ? ` · ${e.blocks} residual blocks` : '') +
        '. Path analysis runs on the operation quotient or a bounded occurrence subgraph, so this sizes the network, not the analysis.';
    } catch (err) {
      if (mine === ticket) out.textContent = `Estimate unavailable: ${err.message}`;
    }
  }
  const schedule = () => { clearTimeout(timer); timer = setTimeout(update, 450); };
  form.addEventListener('input', schedule);
  form.addEventListener('change', schedule);
  window.addEventListener('load', schedule);
})();
