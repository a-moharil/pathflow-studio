"""Run with ECML_PACKAGE=/path/to/extracted/pathflow-ecml-21."""
import io,json,os,subprocess,sys,tarfile
from pathlib import Path
import pytest

PACKAGE=Path(os.environ.get('ECML_PACKAGE','/nonexistent-package'))
pytestmark=pytest.mark.skipif(not (PACKAGE/'apply.py').is_file(),reason='Set ECML_PACKAGE to the extracted release package')
PROJECT=Path(__file__).resolve().parents[1]

def git(path,*args):
    return subprocess.check_output(['git','-C',str(path),*args],text=True).strip()

def setup(tmp_path,nested=False):
    root=tmp_path/'checkout';root.mkdir();project=root/'pathflow_studio' if nested else root;project.mkdir(exist_ok=True)
    archive=subprocess.check_output(['git','archive','db75674'],cwd=PROJECT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as t:t.extractall(project,filter='data')
    git(root,'init','-q','-b','working');git(root,'config','user.email','validation@example.invalid');git(root,'config','user.name','Installer validation');git(root,'add','.');git(root,'commit','-qm','Baseline')
    git(root,'remote','add','origin','https://github.com/a-moharil/pathflow-studio.git')
    (project/'workspace').mkdir();(project/'workspace'/'sentinel').write_text('Preserve experiments')
    return root,project

def run(project,*args):return subprocess.run([sys.executable,str(PACKAGE/'apply.py'),*args],cwd=project,text=True,capture_output=True)

@pytest.mark.parametrize('nested',[False,True])
def test_supported_install_and_dryrun(tmp_path,nested):
    root,p=setup(tmp_path,nested);head=git(root,'rev-parse','HEAD')
    r=run(p,'--check');assert r.returncode==0,r.stderr;assert git(root,'branch','--show-current')=='working';assert not git(root,'status','--porcelain')
    r=run(p);assert r.returncode==0,r.stderr;assert git(root,'branch','--show-current')=='integrating-ecml-manuscript';assert git(root,'rev-parse','HEAD')==head
    assert (p/'VERSION').read_text().strip()=='0.13.0';assert (p/'workspace'/'sentinel').read_text()=='Preserve experiments'
    assert git(root,'diff','--cached','--name-only');assert not git(root,'diff','--name-only')
    assert 'backup/before-ecml-' in git(root,'branch','--list')


def test_preserves_unpublished_channel_release(tmp_path):
    root,p=setup(tmp_path);(p/'VERSION').write_text('0.12.1\n');(p/'lab/static/base_channel_gallery.js').write_text('// local channel implementation\n')
    base=p/'lab/templates/base.html';base.write_text(base.read_text().replace('0.12.0','0.12.1').replace('</body>','<script src="/static/base_channel_gallery.js"></script></body>'))
    git(root,'add','.');git(root,'commit','-qm','Unpublished channel release');original=(p/'lab/static/base_channel_gallery.js').read_bytes()
    r=run(p);assert r.returncode==0,r.stderr;assert (p/'lab/static/base_channel_gallery.js').read_bytes()==original;assert 'base_channel_gallery.js' in base.read_text();assert '?v=0.12.1' in base.read_text()


def test_dirty_tree_is_untouched(tmp_path):
    root,p=setup(tmp_path);file=p/'README.md';file.write_text(file.read_text()+'\nUser work\n');before=file.read_bytes();head=git(root,'rev-parse','HEAD')
    r=run(p);assert r.returncode!=0 and 'not clean' in r.stderr;assert file.read_bytes()==before;assert git(root,'rev-parse','HEAD')==head;assert git(root,'branch','--show-current')=='working'


def test_existing_branch_is_untouched(tmp_path):
    root,p=setup(tmp_path);git(root,'branch','integrating-ecml-manuscript');r=run(p)
    assert r.returncode!=0 and 'already exists' in r.stderr;assert not git(root,'status','--porcelain')


def test_wrong_remote_is_untouched(tmp_path):
    root,p=setup(tmp_path);git(root,'remote','set-url','origin','https://github.com/other/other.git');r=run(p)
    assert r.returncode!=0 and 'origin' in r.stderr;assert not git(root,'status','--porcelain')
