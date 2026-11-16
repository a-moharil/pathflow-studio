"""Persistent run state and jobs; each operation uses its own SQLite connection."""
import json,sqlite3,time,uuid
from pathlib import Path

def db(root):
    Path(root).mkdir(parents=True,exist_ok=True);c=sqlite3.connect(Path(root)/'studio.sqlite',timeout=30);c.row_factory=sqlite3.Row;c.execute('PRAGMA journal_mode=WAL');return c

def init(root):
    with db(root) as c:c.executescript('''CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY,config TEXT,status TEXT,step INTEGER DEFAULT 0,live_step INTEGER DEFAULT 0,request TEXT DEFAULT '',error TEXT DEFAULT '',created REAL);
    CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,run TEXT,payload TEXT,status TEXT,progress TEXT DEFAULT '',result TEXT DEFAULT '',error TEXT DEFAULT '',created REAL);
    CREATE TABLE IF NOT EXISTS measurements(run TEXT,step INTEGER,payload TEXT,PRIMARY KEY(run,step));
    CREATE TABLE IF NOT EXISTS observations(job TEXT PRIMARY KEY,run TEXT,step INTEGER,payload TEXT);''')

def create(root,config):
    rid=uuid.uuid4().hex[:16];(Path(root)/rid).mkdir()
    with db(root) as c:c.execute('INSERT INTO runs(id,config,status,created) VALUES(?,?,?,?)',(rid,json.dumps(config),'ready',time.time()))
    return rid

def run(root,rid):
    with db(root) as c:r=c.execute('SELECT * FROM runs WHERE id=?',(rid,)).fetchone()
    if r is None:raise KeyError('Run not found')
    r=dict(r);r['config']=json.loads(r['config']);return r

def job(root,payload,rid=None):
    jid=uuid.uuid4().hex
    with db(root) as c:c.execute('INSERT INTO jobs(id,run,payload,status,created) VALUES(?,?,?,?,?)',(jid,rid,json.dumps(payload),'queued',time.time()))
    return jid

def progress(root,jid,message):
    from .progress import stage
    stage(message)
    with db(root) as c:c.execute('UPDATE jobs SET progress=? WHERE id=?',(message,jid))

def atomic_json(path,obj):
    # Unique temporary name per writer, so concurrent writers never share a partial file.
    path=Path(path);p=path.with_name(f'{path.name}.{uuid.uuid4().hex[:8]}.tmp')
    try:
        p.write_text(json.dumps(obj,allow_nan=False),encoding='utf8');p.replace(path)
    finally:
        p.unlink(missing_ok=True)


def delete_run(root,rid):
    """Worker-only deletion after current serial work has returned. Retryable cleanup."""
    import shutil
    base=Path(root).resolve();source=base/rid;trash=base/'trash'/rid
    if source.parent!=base or len(rid)!=16 or any(ch not in '0123456789abcdef' for ch in rid):raise ValueError('Invalid experiment identifier')
    trash.parent.mkdir(exist_ok=True)
    if source.exists():source.rename(trash)
    if trash.exists():shutil.rmtree(trash)
    with db(root) as c:
        c.execute('BEGIN IMMEDIATE')
        for table,key in [('observations','run'),('measurements','run'),('jobs','run'),('runs','id')]:c.execute(f'DELETE FROM {table} WHERE {key}=?',(rid,))


def admit(root,rid,payload):
    """Admission gate for every native measurement request.

    Returns ``(job_id, state)`` where state is ``'found'`` (an identical request already
    completed: its stored result is reused), ``'pending'`` (an identical request is queued or
    running) or ``'queued'`` (new work). Reuse is by exact match of the full request,
    including the analysis version, so a changed convention or code version never reuses an
    older result. Requests for a checkpoint that was never committed are rejected here,
    before they reach the worker.
    """
    payload=dict(payload,analysis_version=6)
    if payload.get('kind','analysis') in ('analysis','dose'):
        # Geometry requests also carry the picture-fitting version, so a request computed
        # under an earlier fitting procedure is never reused for the current one.
        from .core.fitsettings import FIT_VERSION
        payload['fit_version']=FIT_VERSION
    encoded=json.dumps(payload,sort_keys=True)
    with db(root) as c:
        c.execute('BEGIN IMMEDIATE')
        r=c.execute('SELECT request,status FROM runs WHERE id=?',(rid,)).fetchone()
        if not r or r['request']=='delete' or r['status'].startswith('delet'):raise ValueError('Experiment is being deleted')
        step=payload.get('step')
        if step is not None:
            try:step=int(step)
            except (TypeError,ValueError):raise ValueError('Checkpoint must be a committed update number')
            if not c.execute('SELECT 1 FROM measurements WHERE run=? AND step=?',(rid,step)).fetchone():
                raise ValueError(f'Checkpoint {step} was never committed for this experiment. Choose a saved checkpoint.')
        for j in c.execute("SELECT id,payload,status FROM jobs WHERE run=? AND status IN ('queued','running','completed') ORDER BY status='completed' DESC",(rid,)):
            if json.dumps(json.loads(j['payload']),sort_keys=True)==encoded:
                return j['id'],('found' if j['status']=='completed' else 'pending')
        jid=uuid.uuid4().hex;c.execute('INSERT INTO jobs(id,run,payload,status,created) VALUES(?,?,?,?,?)',(jid,rid,encoded,'queued',time.time()));return jid,'queued'


def queue_measurement(root,rid,payload):
    """Avoid repeated queued/running/completed jobs for the same measurement request."""
    return admit(root,rid,payload)[0]
