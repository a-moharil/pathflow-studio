"""End-to-end browser checks using real native path computations, no invented maps."""
import os,json,tempfile,threading
from pathlib import Path
import numpy as np
import torch
from werkzeug.serving import make_server
from playwright.sync_api import sync_playwright
from lab import create_app,storage
from lab.config import validate
from lab.core.network import Network
from lab.core.pathmath import analyze

root=Path(tempfile.mkdtemp(prefix='pathflow_auditor21_'));app=create_app(root);torch.set_num_threads(2)
records=[]
for seed in range(6):
    torch.manual_seed(seed)
    c=validate(dict(name='Engineering validation seed '+str(seed),widths=[2],activation='tanh',seed=seed,geometry_every=0,measurement_limit=800,geometry_limit=500))
    model=Network(c);x=np.random.default_rng(81).random((1,28,28));d=analyze(model,x,np.zeros_like(x),dict(scope='selected',a=1,b=0),geometry_enabled=seed==0);d.update(step=0)
    rid=storage.create(root,c);jid=storage.job(root,dict(kind='analysis',step=0,x=0,reference_type='zero'),rid)
    path=root/rid/(jid+'.json');storage.atomic_json(path,d)
    with storage.db(root) as db:db.execute("UPDATE jobs SET status='completed',result=? WHERE id=?",(str(path),jid))
    records.append(jid+':analysis:native')
server=make_server('127.0.0.1',0,app,threaded=True);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
try:
    with sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True,executable_path=os.environ.get('PW_CHROMIUM'))
        page=browser.new_page(viewport={'width':1500,'height':1050});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto(f'http://127.0.0.1:{server.server_port}/auditor?record='+records[0])
        page.wait_for_function("document.getElementById('auditContent').hidden===false")
        assert 'passed' in page.locator('#auditStatus').inner_text()
        assert page.locator('#auditRecord option').count()==7
        page.screenshot(path=str(root/'interpretation.png'),full_page=True)
        page.locator('#evidenceTab').click();assert page.locator('#fidelityTable tr').count()>=3
        page.locator('#auditDistribution').select_option(index=1)
        assert 'input pixel' in page.locator('#distributionScope').inner_text()
        page.locator('#remeasureGeometry').click();page.wait_for_function("document.getElementById('geometryScope').textContent.includes('64 scales')",timeout=180000)
        assert page.locator('#geometryTexture').is_visible()
        page.locator('#normaliseRadius').check()
        page.locator('#geometrySide').select_option('conditional')
        page.screenshot(path=str(root/'evidence.png'),full_page=True)
        page.locator('#populationSelector summary').click()
        for checkbox in page.locator('#populationRecords input').all():checkbox.check()
        page.locator('#computePopulation').click();page.wait_for_function("document.getElementById('populationResults').hidden===false")
        assert 'Descriptive' in page.locator('#populationStatus').inner_text()
        page.locator('#independentRuns').check();assert not page.locator('#populationResults').is_visible()
        page.locator('#computePopulation').click();page.wait_for_function("document.getElementById('populationResults').hidden===false")
        assert 'Exploratory' in page.locator('#populationStatus').inner_text()
        assert page.locator('#populationTable tr').count()==3
        with page.expect_download() as download:page.locator('#exportPopulation').click()
        download.value.save_as(str(root/'population.json'));p=json.loads((root/'population.json').read_text());assert p['results'][0]['p'] is not None
        with page.expect_popup() as new:page.locator('#auditOpenWindow').click()
        other=new.value;other.wait_for_load_state();other.wait_for_function("document.getElementById('auditContent').hidden===false")
        assert 'view=interpretation' in other.url
        # A new selected report clears the previous geometry audit.
        page.locator('#auditRecord').select_option(records[1]);page.wait_for_function("document.getElementById('auditContext').textContent.includes('seed 1')")
        assert page.locator('#remeasureGeometry').is_disabled()
        page.set_viewport_size({'width':550,'height':950});page.screenshot(path=str(root/'mobile.png'),full_page=True)
        assert not errors,errors
        browser.close()
    print(json.dumps(dict(status='passed',artifacts=str(root),records=records,errors=errors)))
finally:server.shutdown()
