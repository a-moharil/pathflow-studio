"""Integrated healthcare page on a trained small CNN, including exports and persistence."""
import os,sys,time,tempfile,subprocess,urllib.request,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from lab import storage as st,datasets
from lab.config import validate
from lab.engine import quantum
from playwright.sync_api import sync_playwright

torch.set_num_threads(2)
project=Path(__file__).resolve().parents[1];root=Path(tempfile.mkdtemp(prefix='healthcare_browser_'));st.init(root)
c=validate(dict(model='cnn',activation='tanh',channels=[2,2],train_size=32,updates=1,checkpoint_every=1));datasets.prepare(root,c);rid=st.create(root,c)
with st.db(root) as db:db.execute("UPDATE runs SET status='training' WHERE id=?",(rid,))
quantum(root,st.run(root,rid));log=open(root/'server.log','w');server=subprocess.Popen([sys.executable,'run.py'],cwd=project,env={**os.environ,'PATHFLOW_HOME':str(root),'PORT':'5091'},stdout=log,stderr=log)
try:
 for _ in range(100):
  try:urllib.request.urlopen('http://127.0.0.1:5091').read();break
  except Exception:time.sleep(.2)
 with sync_playwright() as pw:
  browser=pw.chromium.launch(headless=True);page=browser.new_page(viewport={'width':1440,'height':1050});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
  page.goto('http://127.0.0.1:5091/experiment/'+rid);page.locator('#openHealthcare').click();page.wait_for_url('**/healthcare/**');page.wait_for_function("document.getElementById('hcStep')?.options.length>0")
  page.locator('#hcReferenceType').select_option('zero');page.locator('#hcAnalyze').click();page.wait_for_function("healthCase?.case?.signed_field_residual<1e-10")
  assert page.evaluate('healthCase.context.checkpoint')==1
  page.screenshot(path=str(root/'case.png'))
  page.locator('[data-health-tab="fields"]').click();page.wait_for_timeout(200)
  assert page.evaluate("document.getElementById('hcLayerChart').data.length===2")
  page.screenshot(path=str(root/'fields.png'))
  with page.expect_download() as download:page.locator('#hcSaveFields').click()
  assert download.value.suggested_filename=='healthcare_fields.png'
  page.locator('[data-health-tab="intervene"]').click();page.locator('#hcDose').click();page.wait_for_function("healthCase?.dose?.length===5")
  assert page.evaluate('healthCase.controls.length')==1
  page.locator('#hcDoseFrame').select_option('4');page.screenshot(path=str(root/'dose.png'))
  page.locator('#hcIntervention').select_option('region');page.locator('#hcDose').click();page.wait_for_function("healthCase?.intervention?.mode==='region'")
  assert page.evaluate("healthCase.dose[4].image[0][10][10]===0")
  page.locator('#hcDx').fill('1');page.locator('#hcTransform').click();page.wait_for_function("healthCase?.kind==='healthcare_case' && healthCase.context.transform.dx===1")
  assert page.evaluate('Number.isFinite(healthCase.transformation.H_eq)')
  page.locator('#hcSweepFrames').fill('2');page.locator('#hcSweepTo').fill('2');page.locator('#hcPlay').click();page.wait_for_function("document.getElementById('hcStatus').textContent.includes('Measured transform frame 2/2')",timeout=90000)
  page.locator('#hcStop').click()
  page.locator('[data-health-tab="cohort"]').click();page.locator('#hcCohortSize').fill('4');page.locator('#hcCohort').click();page.wait_for_function("healthCohort?.rows?.length===4",timeout=90000)
  assert page.evaluate('healthCohort.metrics.n')>4
  page.screenshot(path=str(root/'cohort.png'))
  with page.expect_download() as download:page.locator('#hcExport').click()
  assert download.value.suggested_filename.endswith('.json')
  with page.expect_download() as download:page.locator('#hcCSV').click()
  assert download.value.suggested_filename.endswith('.csv')
  page.locator('#hcObservable').select_option('eta1');page.locator('#hcCohortScale').select_option('log')
  assert page.evaluate('''async()=>{const old=healthLatest;const p=healthSubmit('healthcare_case');document.getElementById('hcX').dispatchEvent(new Event('change'));await p;return healthLatest===old}''')
  page.reload();page.wait_for_function("document.getElementById('hcHistory').options.length>0");page.locator('#hcLoad').click();page.wait_for_function('healthLatest!==null')
  assert not errors,errors
  print(json.dumps(dict(browser_healthcare='passed',root=str(root),errors=errors)));browser.close()
finally:server.terminate();server.wait(timeout=20);log.close()
