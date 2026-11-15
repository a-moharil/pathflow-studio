/* Adaptive polling and incremental run feeds.
 *
 * Poller.every(task, {active, idle}) runs `task` repeatedly. The task returns true
 * while something is changing (training, queued jobs) and false when idle; the
 * interval follows. Polling pauses while the tab is hidden and resumes, with an
 * immediate refresh, when it becomes visible again.
 *
 * RunFeed keeps a client-side copy of a native run's checkpoint measurements and
 * observation summaries, and asks the server only for records it has not seen
 * (lean form, without input fields and source diagnostics).
 */
const Poller = {
  every(task, { active = 2000, idle = 10000, immediate = true } = {}) {
    let timer = null, busy = false, last = true, stopped = false;
    const schedule = () => {
      clearTimeout(timer);
      if (stopped || document.hidden) return;
      timer = setTimeout(run, last ? active : idle);
    };
    const run = async () => {
      if (busy || stopped) return;
      busy = true;
      try { last = (await task()) !== false; } catch (e) { last = false; }
      finally { busy = false; schedule(); }
    };
    document.addEventListener('visibilitychange', () => { if (!document.hidden) run(); else clearTimeout(timer); });
    if (immediate) run(); else schedule();
    return { refresh: run, stop() { stopped = true; clearTimeout(timer); } };
  }
};

class RunFeed {
  constructor(rid) {
    this.rid = rid;
    this.metrics = [];
    this.lastStep = null;
    this.observationsByJob = new Map();
    this.lastRow = 0;
  }

  /** Run detail with all checkpoint metrics, fetching only new checkpoints. */
  async run() {
    const since = this.lastStep === null ? '' : `&since=${this.lastStep}`;
    const d = await api(`/api/run/${this.rid}?lean=1${since}`);
    if (this.lastStep === null) this.metrics = d.metrics;
    else this.metrics = this.metrics.concat(d.metrics);
    if (this.metrics.length !== d.metric_count) {
      // A checkpoint was replaced or removed: resynchronise once in full.
      const full = await api(`/api/run/${this.rid}?lean=1`);
      this.metrics = full.metrics;
    }
    this.lastStep = this.metrics.length ? this.metrics[this.metrics.length - 1].step : -1;
    d.metrics = this.metrics;
    return d;
  }

  /** Observation summaries in the original order (step, then record order). */
  async observations() {
    const d = await api(`/api/run/${this.rid}/observations?lean=1&after=${this.lastRow}`);
    for (const record of d.records) {
      this.observationsByJob.set(record._job, record);
      this.lastRow = Math.max(this.lastRow, record._rowid);
    }
    if (this.observationsByJob.size !== d.total) {
      this.observationsByJob.clear();
      this.lastRow = 0;
      const full = await api(`/api/run/${this.rid}/observations?lean=1&after=0`);
      for (const record of full.records) { this.observationsByJob.set(record._job, record); this.lastRow = Math.max(this.lastRow, record._rowid); }
    }
    return [...this.observationsByJob.values()].sort((a, b) => a.step - b.step || a._rowid - b._rowid);
  }
}

if (typeof module !== 'undefined') module.exports = { Poller, RunFeed };
