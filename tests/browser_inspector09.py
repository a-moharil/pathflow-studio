"""Visual selector, diagnostic region and evidence workflow regression."""
import os,sys,time,tempfile,subprocess,urllib.request,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from lab import storage as st,datasets
from lab.config import validate
from lab.engine import quantum
from playwright.sync_api import sync_playwright

torch.set_num_threads(2)
project=Path(__file__).resolve().parents[1];root=Path(tempfile.mkdtemp(prefix='inspector09_browser_'));st.init(root)
c=validate(dict(model='cnn',activation='relu',channels=[2,2],train_size=32,updates=1,checkpoint_every=1));datasets.prepare(root,c);rid=st.create(root,c)
with st.db(root) as db:db.execute("UPDATE runs SET status='training' WHERE id=?",(rid,))
quantum(root,st.run(root,rid));log=open(root/'server.log','w');server=subprocess.Popen([sys.executable,'run.py'],cwd=project,env={**os.environ,'PATHFLOW_HOME':str(root),'PORT':'5094'},stdout=log,stderr=log)
try:
 for _ in range(100):
  try:urllib.request.urlopen('http://127.0.0.1:5094').read();break
  except Exception:time.sleep(.2)
 with sync_playwright() as pw:
  browser=pw.chromium.launch(headless=True);page=browser.new_page(viewport={'width':1500,'height':1100});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
  page.goto('http://127.0.0.1:5094/healthcare/'+rid);page.wait_for_function("document.getElementById('hcStep')?.options.length>0")
  page.locator('#hcReferenceType').select_option('zero');page.locator('[data-health-tab="evidence"]').click()
  page.evaluate("document.getElementById('evIG').value=8;document.getElementById('evBudget').value=2")
  def click_job(selector):
   page.evaluate('evidenceReport=null');page.locator(selector).click();page.wait_for_function('evidenceReport!==null',timeout=120000)
  click_job('#ivCompute');assert page.locator('#ivCamMaps canvas').count()==2;assert page.locator('#ivDiagnosticMaps canvas').count()==2
  page.locator('#ivOverlay').check();assert page.locator('#ivCamMaps canvas').count()==4
  page.locator('#ivDiagnosticMaps').scroll_into_view_if_needed();page.wait_for_timeout(700);page.screenshot(path=str(root/'diagnostics.png'))
  canvas=page.locator('#ivDiagnosticMaps canvas').first;box=canvas.bounding_box();page.mouse.move(box['x']+10,box['y']+10);page.mouse.down();page.mouse.move(box['x']+70,box['y']+70,steps=4);page.mouse.up()
  assert page.evaluate('ivRegionValue.rectangle[2]>ivRegionValue.rectangle[0]')
  assert page.evaluate('Math.abs(ivRegionValue.S-(ivRegionValue.P-ivRegionValue.N))<1e-10')
  with page.expect_download() as dl:page.locator('#ivRegionExport').click()
  assert dl.value.suggested_filename=='pathflow_selected_region.json'
  page.locator('#ivCompanion').select_option('C');assert page.evaluate('ivPinned!==null')
  page.locator('[data-stage="mechanism"]').click();assert page.locator('#ivGallery article').count()==2
  page.locator('#ivArchitecture button').first.click();page.locator('#ivGallery article').nth(1).click();assert page.evaluate('ivLayer===0&&ivChannel===1')
  click_job('#ivConnections');assert page.locator('#ivConnectionsTable tbody tr').count()>0
  page.locator('#ivConnectionsTable button').first.click();assert page.evaluate("document.getElementById('evRule').value==='manual'")
  page.screenshot(path=str(root/'selection.png'))
  click_job('#ivPrimitiveTest');assert page.evaluate("evidenceReport.evidence.intervention==='primitive'")
  page.locator('[data-stage="mechanism"]').click();click_job('#ivChannelTest');assert page.evaluate("evidenceReport.evidence.curves[0].channel_support.index===1")
  page.locator('#evFrame').fill('4');assert 'not Field B' in page.locator('#ivFootprint').inner_text()
  assert page.evaluate('evidenceReport.evidence.curves[0].frames[4].diagnostics.global_values.reconstruction_residual<1e-9')
  page.locator('[data-stage="stability"]').click();click_job('#ivRescale');assert page.evaluate('evidenceReport.evidence.frames.every(f=>f.audit.gradcam_max_change<1e-10)')
  click_job('#ivReverse');assert page.evaluate('evidenceReport.evidence.reverse_audit.field_a_max_change<1e-10')
  click_job('#ivDemo');assert page.locator('#ivDemoMaps canvas').count()==4
  page.locator('#ivDemoPanel').scroll_into_view_if_needed();page.screenshot(path=str(root/'demo.png'))
  page.evaluate("document.getElementById('ivAdvanced').open=true")
  for button,filename in [('evJSON','pathflow_visual_evidence.json'),('evCSV','pathflow_evidence.csv'),('evPNG','pathflow_visual_evidence.png')]:
   with page.expect_download() as download:page.locator('#'+button).click()
   assert download.value.suggested_filename==filename
  with page.expect_download(timeout=20000) as download:page.locator('#evRecord').click()
  assert download.value.suggested_filename=='pathflow_evidence.webm'
  page.locator('#evPresenter').click();assert page.locator('#evRun').is_hidden();page.locator('#evExit').click()
  # Old report compatibility: preserve original arrays, report new fields unavailable.
  page.evaluate("const old=JSON.parse(JSON.stringify(evidenceReport));delete old.evidence.baseline.catalog;delete old.evidence.baseline.connections;delete old.evidence.baseline.diagnostics;evidenceDisplay(old)")
  assert 'unavailable' in page.locator('#ivPrecision').inner_text()
  page.reload();page.wait_for_function("document.getElementById('hcHistory').options.length>0");page.locator('#hcLoad').click();page.wait_for_function('evidenceReport!==null')
  assert page.evaluate('evidenceReport.evidence.version')=='evidence-2'
  assert page.evaluate("document.getElementById('evManual').value===String(evidenceReport.evidence.settings.manual_ids.join(','))")
  assert not errors,errors
  print(json.dumps(dict(browser_inspector09='passed',root=str(root),errors=errors)));browser.close()
finally:server.terminate();server.wait(timeout=20);log.close()
