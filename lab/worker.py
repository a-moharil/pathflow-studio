"""Serialized local worker; explicit run control and dataset jobs independent of runs."""
import argparse,json,os,threading,time,traceback,sys
from pathlib import Path
import torch
from filelock import FileLock
from . import storage as st
from .datasets import prepare
from .engine import quantum
from .analysis_jobs import process_job

def tick(root):
    from .joint.engine import tick as joint_control
    if joint_control(root,controls_only=True):return True
    with st.db(root) as db:deleting=db.execute("SELECT id FROM runs WHERE request='delete' ORDER BY created LIMIT 1").fetchone()
    if deleting:
        try:st.delete_run(root,deleting[0])
        except Exception:
            with st.db(root) as db:db.execute("UPDATE runs SET status='delete_failed',request='',error=? WHERE id=?",(traceback.format_exc(),deleting[0]))
        return True
    # Honour a pending pause before starting any other work.
    with st.db(root) as db:
        r=db.execute("SELECT id FROM runs WHERE request<>'' ORDER BY created LIMIT 1").fetchone()
    if r:
        try:quantum(root,st.run(root,r[0]))
        except Exception:
            with st.db(root) as db:db.execute("UPDATE runs SET status=CASE WHEN request='delete' THEN 'delete_requested' ELSE 'failed' END,error=?,request=CASE WHEN request='delete' THEN 'delete' ELSE '' END WHERE id=?",(traceback.format_exc(),r[0]))
        return True
    with st.db(root) as db:
        j=db.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY created LIMIT 1").fetchone()
        if j:db.execute("UPDATE jobs SET status='running' WHERE id=?",(j['id'],))
    if j:
        try:
            p=json.loads(j['payload'])
            if p['kind'].startswith('joint_'):
                from .joint.jobs import process as joint_process
                joint_process(root,dict(j))
            elif p['kind']=='prepare':
                result=prepare(root,p['config'],lambda msg:st.progress(root,j['id'],msg))
                folder=Path(root)/'dataset_jobs';folder.mkdir(exist_ok=True);path=folder/(j['id']+'.json');st.atomic_json(path,result)
                with st.db(root) as db:db.execute("UPDATE jobs SET status='completed',result=?,progress='Ready' WHERE id=?",(str(path),j['id']))
            else:process_job(root,dict(j))
        except Exception:
            with st.db(root) as db:db.execute("UPDATE jobs SET status='failed',error=? WHERE id=?",(traceback.format_exc(),j['id']))
        return True
    from .joint.engine import tick as joint_tick
    if joint_tick(root):return True
    with st.db(root) as db:r=db.execute("SELECT id FROM runs WHERE status='training' ORDER BY created LIMIT 1").fetchone()
    if r:
        try:quantum(root,st.run(root,r[0]))
        except Exception:
            with st.db(root) as db:db.execute("UPDATE runs SET status=CASE WHEN request='delete' THEN 'delete_requested' ELSE 'failed' END,error=?,request=CASE WHEN request='delete' THEN 'delete' ELSE '' END WHERE id=?",(traceback.format_exc(),r[0]))
        return True
    return False

def main(root,parent=False):
    root=Path(root);st.init(root)
    from .joint.engine import init as joint_init
    joint_init(root)
    torch.set_num_threads(max(1,min(4,os.cpu_count() or 1)));stop=threading.Event()
    if parent:
        def watch():sys.stdin.buffer.read();stop.set()
        threading.Thread(target=watch,daemon=True).start()
    with FileLock(str(root/'worker.lock'),timeout=0):
        with st.db(root) as db:
            db.execute("UPDATE jobs SET status='queued' WHERE status='running'")
            db.execute("UPDATE joint_runs SET status='paused',request='' WHERE request<>'delete' AND (status='training' OR request<>'')")
            # A restart never unexpectedly restarts an experiment.
            db.execute("UPDATE runs SET status='paused',request='' WHERE request<>'delete' AND status IN ('training','saving','pause_requested','stop_requested')")
            db.execute("UPDATE runs SET status='delete_requested' WHERE request='delete'")
        while not stop.is_set():
            st.atomic_json(root/'worker.json',{'pid':os.getpid(),'heartbeat':time.time()})
            if not tick(root):stop.wait(.25)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--parent',action='store_true');a=p.parse_args();main(a.root,a.parent)
