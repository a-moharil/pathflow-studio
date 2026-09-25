"""Launch from PyCharm or an Ubuntu/WSL terminal with this interpreter."""
import os,subprocess,sys,time,json
from pathlib import Path
from waitress import serve
from lab import create_app
if __name__=='__main__':
    project=Path(__file__).resolve().parent;root=Path(os.environ.get('PATHFLOW_HOME',project/'workspace')).resolve()
    app=create_app(root);worker=subprocess.Popen([sys.executable,'-m','lab.worker','--root',str(root),'--parent'],cwd=project,stdin=subprocess.PIPE)
    try:
        # Do not accept run-control requests before restart recovery has finished.
        for _ in range(240):
            if worker.poll() is not None:raise RuntimeError('The worker could not start. See the console error above.')
            try:
                if json.loads((root/'worker.json').read_text())['pid']==worker.pid:break
            except (FileNotFoundError,ValueError,KeyError):pass
            time.sleep(.25)
        else:raise RuntimeError('Worker startup timed out.')
        print(f"Pathflow Studio: http://127.0.0.1:{os.environ.get('PORT','5000')}",flush=True)
        serve(app,host='127.0.0.1',port=int(os.environ.get('PORT','5000')),threads=6)
    finally:
        worker.terminate()
        try:worker.wait(timeout=15)
        except subprocess.TimeoutExpired:worker.kill();worker.wait()
