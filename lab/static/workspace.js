/* Context bar, stage rail, carried measurement context, queue badge and the one presentation mode. */
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
      const tab = { training: 'training', geometry: 'geometry', fields: 'realisation', configuration: 'training',
                    fieldspanel: 'realisation', resnetobservatory: 'observables', provenance: 'training' }[key];
      if (!tab) return;
      for (const a of bar.querySelectorAll('.runTabs a')) {
        if (a.dataset.tab === tab) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current');
      }
      const url = new URL(location.href);
      url.searchParams.set('section', name);
      history.replaceState(null, '', url.pathname + url.search + url.hash);
    }
  };

  // On the experiment page the stage rail switches in-page sections in place, and the
  // duplicate in-page tabs it names are hidden; sections the rail does not name stay as tabs.
  const SECTION_OF = {
    native: { training: 'Training', geometry: 'Geometry', realisation: 'Fields' },
    joint: { training: 'training', geometry: 'geometry', realisation: 'fieldsPanel', observables: 'resnetObservatory' }
  };
  window.addEventListener('load', () => {
    if (!bar || !location.pathname.startsWith('/experiment/')) return;
    const buttons = [...document.querySelectorAll('main .sectionTabs:not([role=tablist]) button')];
    for (const [key, section] of Object.entries(SECTION_OF[bar.dataset.engine] || {})) {
      const button = buttons.find(b => (b.dataset.section || b.textContent.trim()) === section);
      const link = bar.querySelector(`.runTabs a[data-tab="${key}"]`);
      if (!button || !link) continue;
      button.hidden = true;
      link.addEventListener('click', e => {
        if (e.metaKey || e.ctrlKey || e.shiftKey || e.button) return;
        e.preventDefault();
        button.click();
        window.scrollTo({ top: 0, behavior: 'smooth' });
      });
    }
    const strip = buttons[0]?.parentElement;
    if (strip) strip.classList.toggle('subSections', buttons.some(b => !b.hidden));
  });

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
    if (toggle) toggle.textContent = on ? 'Presenting' : 'Present';
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
  // Entering it from the page presenter turns the switch on, so there is one presentation state.
  document.addEventListener('click', e => {
    if (!e.target.closest('#hwPresenter, #evExit, #evPresenter')) return;
    setTimeout(async () => {
      const hw = window.healthWorkflow;
      if (!hw || hw.presenting === window.Presentation.on) return;
      try { const d = await api('/api/presentation', { on: hw.presenting }); apply(d.on); } catch (err) { showError(err); }
    }, 0);
  });

  // Queue badge: count of queued and running jobs across the workspace.
  const badge = document.getElementById('queueBadge');
  if (badge) {
    Poller.every(async () => {
      const d = await api('/api/queue?limit=1');
      const pending = (d.counts.queued || 0) + (d.counts.running || 0);
      badge.hidden = !pending;
      badge.textContent = pending;
      return pending > 0;
    }, { active: 4000, idle: 20000, immediate: true });
  }
  // Apply after page scripts have created their own presenter controls.
  window.addEventListener('load', () => { if (window.Presentation.on) apply(true); });
})();
