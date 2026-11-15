"""Real synthetic classification training and analysis through the shared workflow."""
import os,sys,time,tempfile,subprocess,urllib.request,json
from pathlib import Path
from playwright.sync_api import sync_playwright
project=Path(__file__).resolve().parents[1];root=Path(tempfile.mkdtemp(prefix='joint_ui13_'));log=(root/'server.log').open('w')
server=subprocess.Popen([sys.executable,'run.py'],cwd=project,env={**os.environ,'PATHFLOW_HOME':str(root),'PORT':'5098'},stdout=log,stderr=log)
try:
 for _ in range(100):
  try:urllib.request.urlopen('http://127.0.0.1:5098');break
  except Exception:time.sleep(.2)
 with sync_playwright() as p:
  b=p.chromium.launch(headless=True);page=b.new_page(viewport={'width':1500,'height':1100});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
  page.goto('http://127.0.0.1:5098/design?domain=healthcare');page.locator('#design [name=seed]').fill('17');page.locator('#design [name=dataset]').select_option('synthetic-joint');page.wait_for_function("!document.getElementById('resnetDesign').hidden")
  assert '/design?' in page.url;page.wait_for_function("document.getElementById('dataset').value==='synthetic'")
  page.locator('#jointDesign [name=size]').select_option('64');page.locator('#jointDesign [name=width]').select_option('16');page.locator('#jointDesign [name=evaluation_count]').fill('3');page.locator('#jointDesign [name=save_every]').fill('1');page.locator('#prepareSelected').click();page.wait_for_function("!document.getElementById('jointCreate').disabled",timeout=60000)
  page.locator('#jointCreate').click();page.wait_for_url('**/experiment/*');rid=page.url.rsplit('/',1)[1]
  page.locator('#updates').fill('1');page.locator('#train').click();page.wait_for_function("document.getElementById('runStatus').textContent.includes('paused · update 1')",timeout=120000)
  page.locator('[data-section=geometry]').click();page.locator('#igSteps').fill('4');page.locator('#memory').fill('8');page.locator('#fitHyperbolic').check();page.locator('#alpha').fill('0.5');page.locator('#computeGeometry').click()
  page.wait_for_function("document.getElementById('geometryStatus').textContent.includes('Operation quotient')",timeout=240000)
  assert not page.locator('#error').is_visible(),page.locator('#error').inner_text()
  page.locator('#representation1').select_option('hyperbolic');page.wait_for_function("document.getElementById('hyperStatus1').textContent.includes('Hyperbolic 3-D')");page.locator('#dimension1').select_option('3');page.locator('#labels1').select_option('all');page.locator('#geometry').scroll_into_view_if_needed();page.screenshot(path=str(root/'geometry.png'))
  page.locator('[data-section=fieldsPanel]').click();page.wait_for_function("document.querySelector('#fieldA .plot-container')!==null")
  page.locator('#fieldsPanel').scroll_into_view_if_needed();page.wait_for_timeout(700);page.screenshot(path=str(root/'fields.png'))
  assert page.locator('#head option').count()==2
  page.locator('#head').select_option('segmentation');page.locator('#state').select_option('intervened');page.wait_for_function("document.getElementById('predictionMap').data?.[0]?.z.length===64")
  assert not errors,errors
  print(json.dumps(dict(browser_joint_ui13='passed',root=str(root),rid=rid,errors=errors)));b.close()
finally:server.terminate();server.wait(timeout=20);log.close()
