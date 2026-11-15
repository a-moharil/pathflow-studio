/* Run workspace bar, carried measurement context and global presentation mode. */
(() => {
  const bar = document.getElementById('runBar');
  const params = new URLSearchParams(location.search);

  // Carry the declared measurement context between run views when one is present.
  const context = params.get('context');
  if (bar && context) {
    for (const a of bar.querySelectorAll('.runTabs a')) {
      if (['observatory', 'statistics'].includes(a.dataset.tab)) {
        const url = new URL(a.href, location.origin);
        url.searchParams.set('context', context);
        a.href = url.pathname + url.search;
      }
    }
  }

  // Keep the active tab in step with in-page sections (experiment page tabs).
  window.RunNav = {
    setSection(name) {
      if (!bar) return;
      const key = String(name).toLowerCase();
      const tab = { training: 'training', geometry: 'geometry', fields: 'geometry', configuration: 'training',
                    fieldspanel: 'investigation', resnetobservatory: 'observatory', provenance: 'training' }[key];
      if (!tab) return;
      for (const a of bar.querySelectorAll('.runTabs a')) {
        if (a.dataset.tab === tab) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current');
      }
      const url = new URL(location.href);
      url.searchParams.set('section', name);
      history.replaceState(null, '', url.pathname + url.search + url.hash);
    }
  };

  // Live status in the bar: frequent while active, slow when idle, paused when hidden.
  if (bar) {
    const status = document.getElementById('runBarStatus');
    Poller.every(async () => {
      const d = await api(`/api/stats/run/${bar.dataset.run}/status`);
      status.dataset.status = d.status;
      status.textContent = `${d.status} · update ${d.step}${d.pending ? ` · ${d.pending} queued` : ''}`;
      return d.active;
    }, { active: 3000, idle: 15000, immediate: false });
  }

  // Presentation mode: a workspace-wide switch enforced by the server.
  const toggle = document.getElementById('presentationToggle');
  const apply = on => {
    document.body.dataset.presentation = on ? 'on' : 'off';
    toggle?.setAttribute('aria-pressed', String(on));
    if (toggle) toggle.textContent = on ? 'Presentation on' : 'Presentation';
    const hw = window.healthWorkflow;
    const presenter = document.getElementById('hwPresenter');
    if (hw && presenter && hw.presenting !== on) presenter.click();
    document.dispatchEvent(new CustomEvent('pathflow:presentation', { detail: { on } }));
  };
  window.Presentation = { get on() { return document.body.dataset.presentation === 'on'; } };
  if (toggle) {
    toggle.onclick = async () => {
      try { const d = await api('/api/presentation', { on: !window.Presentation.on }); apply(d.on); }
      catch (e) { showError(e); }
    };
  }
  // The healthcare presenter hides the header; leaving it from its own exit control also
  // turns the workspace switch off, so the studio never stays locked without a visible toggle.
  document.addEventListener('click', e => {
    if (!e.target.closest('#hwPresenter, #evExit')) return;
    setTimeout(async () => {
      const hw = window.healthWorkflow;
      if (hw && !hw.presenting && window.Presentation.on) {
        try { await api('/api/presentation', { on: false }); apply(false); } catch (err) { showError(err); }
      }
    }, 0);
  });
  // Apply after page scripts have created their own presenter controls.
  window.addEventListener('load', () => { if (window.Presentation.on) apply(true); });
})();
