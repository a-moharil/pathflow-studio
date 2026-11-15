"""End-to-end measured evidence, exports and saved context."""
import os,sys,time,tempfile,subprocess,urllib.request,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from lab import storage as st,datasets
from lab.config import validate
from lab.engine import quantum
from playwright.sync_api import sync_playwright

torch.set_num_threads(2)
project=Path(__file__).resolve().parents[1];root=Path(tempfile.mkdtemp(prefix='evidence08_browser_'));st.init(root)
c=validate(dict(model='cnn',activation='relu',channels=[2,2],train_size=32,updates=1,checkpoint_every=1));datasets.prepare(root,c);rid=st.create(root,c)
with st.db(root) as db:db.execute("UPDATE runs SET status='training' WHERE id=?",(rid,))
quantum(root,st.run(root,rid));log=open(root/'server.log','w');server=subprocess.Popen([sys.executable,'run.py'],cwd=project,env={**os.environ,'PATHFLOW_HOME':str(root),'PORT':'5093'},stdout=log,stderr=log)
try:
 for _ in range(100):
  try:urllib.request.urlopen('http://127.0.0.1:5093').read();break
  except Exception:time.sleep(.2)
 with sync_playwright() as pw:
  browser=pw.chromium.launch(headless=True);page=browser.new_page(viewport={'width':1500,'height':1050});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
  page.goto('http://127.0.0.1:5093/healthcare/'+rid);page.wait_for_function("document.getElementById('hcStep')?.options.length>0")
  page.locator('#hcReferenceType').select_option('zero');page.locator('[data-health-tab="evidence"]').click();page.evaluate("document.getElementById('ivAdvanced').open=true");page.locator('#evIG').fill('8');page.locator('#evBudget').fill('2')
  def compute(mode):
   page.locator('#evMode').select_option(mode);page.evaluate('evidenceReport=null');page.locator('#evRun').click();page.wait_for_function("evidenceReport!==null",timeout=120000)
   assert page.evaluate('evidenceReport.evidence.mode')==mode
   page.evaluate("document.getElementById('ivLegacy').open=true")
  compute('compare');assert page.locator('#evMaps canvas').count()==6
  assert page.evaluate('evidenceReport.evidence.baseline.field_b.total_mass')>=0
  page.locator('#evMaps').scroll_into_view_if_needed();page.screenshot(path=str(root/'comparison.png'))
  page.locator('#evNormalizeB').check();assert 'normalized shape' in page.locator('#evMaps').inner_text();page.locator('#evNormalizeB').uncheck()
  compute('intervene');assert page.evaluate('evidenceReport.evidence.curves.length')==4
  page.locator('#evFrame').fill('4');page.locator('#evSeries').select_option('1');assert page.locator('#evFrame').input_value()=='0'
  page.locator('#evResponse').scroll_into_view_if_needed();page.screenshot(path=str(root/'intervention.png'))
  page.locator('summary').filter(has_text='Intervention, rescaling').click();page.locator('#evIntervention').select_option('pixels');compute('intervene');assert page.evaluate('evidenceReport.evidence.curves.length')==6
  compute('rescale');page.locator('#evFrame').fill('4');assert 'Hidden gradient' in page.locator('#evMaps').inner_text()
  assert page.evaluate('evidenceReport.evidence.frames.every(f=>f.audit.input_gradient_max_change<1e-10)')
  page.locator('#evFrames').fill('2');compute('transform');assert page.locator('#evMaps canvas').count()==7
  page.locator('#evFrame').fill('1');page.locator('#evMaps').scroll_into_view_if_needed();page.screenshot(path=str(root/'movement.png'))
  page.locator('#evPlay').click();page.wait_for_timeout(1900)
  compute('neighborhood');assert page.evaluate("document.getElementById('evMetricS').data[0].z.length>0")
  page.locator('#evMetricState').select_option('intervened');assert 'intervened Δg' in page.locator('#evMetricNote').inner_text()
  compute('cases');assert page.locator('#evCaseMaps canvas').count()==6 and page.locator('#evPairs tr').count()==2
  page.locator('#evCaseResponse').scroll_into_view_if_needed();page.screenshot(path=str(root/'cases.png'))
  compute('compare')
  for button,filename in [('evJSON','pathflow_visual_evidence.json'),('evCSV','pathflow_evidence.csv'),('evPNG','pathflow_visual_evidence.png')]:
   with page.expect_download() as download:page.locator('#'+button).click()
   assert download.value.suggested_filename==filename
  page.locator('#evPresenter').click();assert page.locator('#evRun').is_hidden();page.locator('#evExit').click();assert page.locator('#evRun').is_visible()
  with page.expect_download(timeout=20000) as download:page.locator('#evRecord').click()
  assert download.value.suggested_filename=='pathflow_evidence.webm'
  page.reload();page.wait_for_function("document.getElementById('hcHistory').options.length>0");page.locator('#hcLoad').click();page.wait_for_function('evidenceReport!==null')
  assert page.locator('#evWorkspace').is_visible()
  assert page.evaluate('''async()=>{const old=evidenceReport;const pending=healthSubmit('healthcare_evidence',evidencePayload());document.getElementById('evBudget').dispatchEvent(new Event('change'));await pending;return old===evidenceReport}''')
  assert not errors,errors
  print(json.dumps(dict(browser_evidence08='passed',root=str(root),errors=errors)));browser.close()
finally:server.terminate();server.wait(timeout=20);log.close()
