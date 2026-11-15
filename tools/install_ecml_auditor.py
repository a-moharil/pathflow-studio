#!/usr/bin/env python3
"""Install the additive ECML release on a clean, supported local Pathflow checkout."""
import argparse,datetime,hashlib,json,subprocess,sys
from pathlib import Path

BRANCH='integrating-ecml-manuscript'

def git(project,*args,check=True):
    p=subprocess.run(['git','-C',str(project),*args],text=True,capture_output=True)
    if check and p.returncode:raise RuntimeError(p.stderr.strip() or p.stdout.strip())
    return p.stdout.strip()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',default='.',help='Pathflow project directory (default: current directory)')
    parser.add_argument('--check',action='store_true',help='Preflight only; write nothing')
    args=parser.parse_args();project=Path(args.project).resolve();package=Path(__file__).resolve().parent
    manifest_path=package/'manifest.json'
    if not manifest_path.exists():raise RuntimeError('Run apply.py from the extracted release package, which includes manifest.json and payload/.')
    manifest=json.loads(manifest_path.read_text());git(project,'rev-parse','--show-toplevel')
    if git(project,'status','--porcelain'):raise RuntimeError('Working tree is not clean. Review and commit your existing work before installing. Nothing was changed.')
    origin=git(project,'remote','get-url','origin',check=False)
    if origin.rstrip('/').removesuffix('.git') not in ['https://github.com/a-moharil/pathflow-studio','git@github.com:a-moharil/pathflow-studio','ssh://git@github.com/a-moharil/pathflow-studio']:
        raise RuntimeError('origin does not identify a-moharil/pathflow-studio. Verify this is the intended checkout; no remote was changed.')
    version=(project/'VERSION').read_text().strip()
    if version not in ['0.12.0','0.12.1']:
        raise RuntimeError(f'Installed version {version} is not a supported baseline. This additive installer accepts 0.12.0 or 0.12.1 only.')
    if git(project,'branch','--list',BRANCH):raise RuntimeError(f'Branch {BRANCH} already exists. It will not be overwritten. Review that branch before installing again.')
    oldbranch=git(project,'symbolic-ref','--quiet','--short','HEAD',check=False)
    if not oldbranch:raise RuntimeError('Check out your working branch before installing; detached HEAD is not supported.')
    baseline=git(project,'rev-parse','HEAD');writes={}
    for name,digest in manifest['files'].items():
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts:raise RuntimeError('Invalid payload path')
        source=package/'payload'/relative;dest=project/relative
        if not dest.resolve().is_relative_to(project):raise RuntimeError('Target escapes the project through a symlink')
        content=source.read_bytes()
        if hashlib.sha256(content).hexdigest()!=digest:raise RuntimeError('Package checksum mismatch: '+name)
        if dest.exists():raise RuntimeError('A new integration file already exists: '+name+'; nothing was overwritten.')
        writes[relative]=content
    init=project/'lab/__init__.py';text=init.read_text();anchor='    app.register_blueprint(joint_web)'
    if text.count(anchor)!=1 or 'auditor_web' in text:raise RuntimeError('App registration has changed; automatic integration is unsafe.')
    writes[Path('lab/__init__.py')]=text.replace(anchor,anchor+'\n    from .auditor.web import web as auditor_web\n    app.register_blueprint(auditor_web)').encode()
    base=project/'lab/templates/base.html';text=base.read_text()
    if text.count('<nav>')!=1 or text.count('{% block scripts %}')!=1 or 'auditorNav' in text:raise RuntimeError('Base template has changed; automatic integration is unsafe.')
    text=text.replace('<nav>','<nav><a href="/auditor" id="auditorNav">ECML auditor</a>',1).replace('{% block scripts %}','<script src="/static/auditor_link.js?v=1"></script>{% block scripts %}',1)
    writes[Path('lab/templates/base.html')]=text.encode();writes[Path('VERSION')]=b'0.13.0\n'
    print(f'Preflight passed: {version}, {baseline}, {len(writes)} source files. Existing branch: {oldbranch}.')
    if args.check:return
    original={name:(project/name).read_bytes() if (project/name).exists() else None for name in writes}
    backup='backup/before-ecml-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    git(project,'branch',backup,baseline);git(project,'switch','--no-track','-c',BRANCH)
    try:
        for name,content in writes.items():
            path=project/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)
        git(project,'add','--',*[str(p) for p in writes])
        git(project,'diff','--cached','--check')
    except Exception as error:
        git(project,'reset','--quiet','HEAD','--',*[str(p) for p in writes],check=False)
        for name,content in original.items():
            path=project/name
            if content is None:path.unlink(missing_ok=True)
            else:path.write_bytes(content)
        git(project,'switch',oldbranch,check=False)
        raise RuntimeError('Installation failed: '+str(error)+'. Source changes were rolled back. The backup branch remains: '+backup)
    print('Applied and staged ECML auditor 0.13.0 on '+BRANCH)
    print('Rollback reference: '+backup)
    print('No commit or push performed. Existing investigation code, local channel updates, checkpoints and environments retained.')
    print('Review: git diff --cached --stat && git diff --cached')
    print('Commit: git commit -m "Integrate ECML manuscript audit views"')
    print('Dry run: git push --dry-run origin HEAD:refs/heads/'+BRANCH)
    print('Push: git push -u origin HEAD:refs/heads/'+BRANCH)
    print('Launch with your existing interpreter and port; open ECML auditor. Do not force-push.')

if __name__=='__main__':
    try:main()
    except (RuntimeError,OSError,ValueError) as e:print('STOP: '+str(e),file=sys.stderr);sys.exit(1)
