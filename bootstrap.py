"""Install the tested CPU stack into the current interpreter (use a virtualenv)."""
import platform,subprocess,sys
from pathlib import Path
if sys.version_info[:2] not in [(3,10),(3,11),(3,12)]:
    raise SystemExit('Use Python 3.10, 3.11 or 3.12. Tested with 3.12.')
try:
    import pip
except ImportError:
    subprocess.check_call([sys.executable,'-m','ensurepip','--upgrade'])
cmd=[sys.executable,'-m','pip','install','torch==2.6.0','torchvision==0.21.0']
if platform.system() in ['Linux','Windows']:cmd+=['--index-url','https://download.pytorch.org/whl/cpu']
subprocess.check_call(cmd)
subprocess.check_call([sys.executable,'-m','pip','install','-r',str(Path(__file__).parent/'requirements.txt')])
print('Installation complete. Run: python -m pytest -q, then python run.py')
