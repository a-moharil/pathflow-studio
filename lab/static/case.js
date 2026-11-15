/* The current case of a run, carried between stages.

   A case is one input pair at one committed checkpoint. It is named by a recorded job and
   travels as ?case=<job> on every stage link; within a browser tab it is also remembered
   per run, so a stage opened without the parameter resumes the case last investigated.
   Pages call Case.open(kind) to obtain the result to show, and Case.set(job) when they
   display or compute a result, which moves the whole rail to that case. */
window.Case = (() => {
  const bar = document.getElementById('runBar');
  const rid = bar ? bar.dataset.run : null;
  const key = rid ? 'pathflow.case.' + rid : null;
  const store = {
    get() { try { return key ? sessionStorage.getItem(key) : null; } catch { return null; } },
    set(v) { try { if (key) v ? sessionStorage.setItem(key, v) : sessionStorage.removeItem(key); } catch { /* storage unavailable */ } }
  };
  let id = new URLSearchParams(location.search).get('case') || store.get();
  let summary = null;

  function decorate() {
    for (const a of document.querySelectorAll('#runBar .runTabs a, a[data-carry-case]')) {
      const url = new URL(a.getAttribute('href'), location.origin);
      if (id) url.searchParams.set('case', id); else url.searchParams.delete('case');
      a.setAttribute('href', url.pathname + url.search + url.hash);
    }
  }

  function writeUrl() {
    const url = new URL(location.href);
    if (id) url.searchParams.set('case', id); else url.searchParams.delete('case');
    history.replaceState(history.state, '', url.pathname + url.search + url.hash);
  }

  async function chip() {
    const el = document.getElementById('caseChip');
    if (!el) return;
    if (!id) { el.hidden = true; summary = null; return; }
    try {
      summary = await api('/api/case/' + encodeURIComponent(id));
      if (summary.run !== rid) throw Error('Case belongs to another experiment');
      document.getElementById('caseText').textContent = summary.label;
      el.title = `Recorded results for this case: ${Object.keys(summary.stages).join(', ') || 'none'}`;
      el.hidden = false;
    } catch {
      // A deleted or foreign record is released silently; the stage falls back to the latest result.
      id = null; summary = null; store.set(null); el.hidden = true; writeUrl(); decorate();
    }
  }

  function announce() {
    document.dispatchEvent(new CustomEvent('pathflow:case', { detail: { id, summary } }));
  }

  const api_ = {
    get id() { return id; },
    get summary() { return summary; },
    rid,
    /** Make ``job`` the current case of this run. */
    set(job) {
      if (!rid || !job || job === id) return;
      id = String(job); store.set(id); writeUrl(); decorate(); chip().then(announce);
    },
    clear() { id = null; store.set(null); writeUrl(); decorate(); chip().then(announce); },
    /** The recorded result of ``kind`` (comma-separated kinds allowed) for the current case,
        or the latest result of that kind when no case is set. */
    async open(kind) {
      if (!rid) return { match: null };
      const q = new URLSearchParams({ kind });
      if (id) q.set('case', id);
      return api(`/api/run/${rid}/case/open?${q}`);
    }
  };

  document.getElementById('caseClear')?.addEventListener('click', () => api_.clear());
  if (rid) { if (id) store.set(id); writeUrl(); decorate(); chip(); }
  return api_;
})();
