"""Measured progress for the single serialized worker. Telemetry never changes results."""
import json, os, threading, time
from pathlib import Path
from contextlib import contextmanager
from . import storage as st
ACTIVE = None

def duration(s):
    if s is None: return 'estimating'
    s=max(0,int(s));return f'{s//3600:02d}:{s//60%60:02d}:{s%60:02d}'

class Reporter:
    def __init__(self,root,label,run=None,job=None):
        self.root=Path(root);self.label=label;self.run=run;self.job=job
        self.started=time.monotonic();self.stage_started=self.started;self.stage='Starting'
        self.done=None;self.total=None;self.unit=None;self.status='running';self.stop=threading.Event()
        self.lock=threading.RLock();self.last_write=0.;self.last_print=0.;self.error=None;self.training=None
    def snapshot(self):
        with self.lock:
            now=time.monotonic();elapsed=now-self.stage_started
            eta=elapsed*(self.total-self.done)/self.done if self.done and self.total is not None else None
            return dict(label=self.label,run=self.run,job=self.job,pid=os.getpid(),status=self.status,stage=self.stage,
                completed=self.done,total=self.total,unit=self.unit,elapsed_seconds=now-self.started,
                stage_elapsed_seconds=elapsed,stage_eta_seconds=eta if self.status=='running' else None,
                eta_scope='current stage only; later stages are not included',updated_at=time.time(),error=self.error,
                training=self.training)
    def publish(self,force=False):
        with self.lock:
            now=time.monotonic()
            if not force and now-self.last_write<.5:return
            v=self.snapshot();self.last_write=now
            try:st.atomic_json(self.root/'compute_progress.json',v)
            except OSError:return
            if force or now-self.last_print>=5:
                self.last_print=now
                count='' if v['total'] is None else f" | {v['completed']}/{v['total']} {v['unit']}"
                training='' if not self.training else f" | training ETA {duration(self.training['eta_seconds'])} ({self.training['note']})"
                print(f"[Pathflow {self.status}] {self.label} | {self.stage}{count} | elapsed {duration(v['elapsed_seconds'])} | stage ETA {duration(v['stage_eta_seconds'])}{training}",flush=True)
    def update(self,stage,done=None,total=None,unit=None):
        with self.lock:
            changed=stage!=self.stage
            if changed:self.stage_started=time.monotonic()
            self.stage=stage;self.done=done;self.total=total;self.unit=unit
            self.publish(changed)
    def beat(self):
        while not self.stop.wait(2):self.publish()

def training_estimate(value):
    if ACTIVE:
        with ACTIVE.lock:ACTIVE.training=value

def stage(message,done=None,total=None,unit=None):
    if ACTIVE:ACTIVE.update(message,done,total,unit)

@contextmanager
def task(root,label,run=None,job=None):
    global ACTIVE
    reporter=Reporter(root,label,run,job);ACTIVE=reporter
    reporter.publish(True);thread=threading.Thread(target=reporter.beat,daemon=True);thread.start()
    try:
        yield reporter
    except BaseException as exc:
        reporter.status='failed';reporter.error=str(exc);raise
    else:reporter.status='completed'
    finally:
        reporter.stop.set();thread.join(timeout=3);reporter.publish(True);ACTIVE=None

def snapshot(root):
    try:
        value=json.loads((Path(root)/'compute_progress.json').read_text())
        value['stale']=value.get('status')=='running' and time.time()-value['updated_at']>15
        return value
    except (OSError,ValueError):return dict(status='idle',stale=False)
