"""Real synthetic classification training and analysis through the shared workflow."""
import os,sys,time,tempfile,subprocess,urllib.request,json
from pathlib import Path
from playwright.sync_api import sync_playwright
project=Path(__file__).resolve().parents[1];root=Path(tempfile.mkdtemp(prefix='unified13_'));log=(root/'server.log').open('w')
server=subprocess.Popen([sys.executable,'run.py'],cwd=project,env={**os.environ,'PATHFLOW_HOME':str(root),'PORT':'5098'},stdout=log,stderr=log)
try:
 for _ in range(100):
  try:urllib.request.urlopen('http://127.0.0.1:5098');break
  except Exception:time.sleep(.2)
 with sync_playwright() as p:
  b=p.chromium.launch(headless=True);page=b.new_page(viewport={'width':1500,'height':1100});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
  page.goto('http://127.0.0.1:5098/design');page.locator('#design [name=train_size]').fill('32');page.locator('#design [name=model]').select_option('resnet50_joint');page.wait_for_function("!document.getElementById('resnetDesign').hidden")
  assert page.url.endswith('/design');page.wait_for_function("document.getElementById('dataset').value==='synthetic-42-32'")
  page.locator('#jointDesign [name=size]').select_option('64');page.locator('#jointDesign [name=width]').select_option('16');page.locator('#jointDesign [name=evaluation_count]').fill('3');page.locator('#jointDesign [name=save_every]').fill('1');page.locator('#prepareSelected').click();page.wait_for_function("!document.getElementById('jointCreate').disabled",timeout=60000)
  page.locator('#jointCreate').click();page.wait_for_url('**/experiment/*');rid=page.url.rsplit('/',1)[1]
  page.locator('#updates').fill('1');page.locator('#train').click();page.wait_for_function("document.getElementById('runStatus').textContent.includes('paused · update 1')",timeout=120000)
  page.locator('[data-section=geometry]').click();page.locator('#igSteps').fill('4');page.locator('#memory').fill('8');page.locator('#computeGeometry').click()
  page.wait_for_function("document.getElementById('geometryStatus').textContent.includes('Operation quotient')",timeout=240000)
  assert not page.locator('#error').is_visible(),page.locator('#error').inner_text()
  page.locator('#dimension1').select_option('3');page.locator('#labels1').select_option('all');page.locator('#geometry').scroll_into_view_if_needed();page.screenshot(path=str(root/'geometry.png'))
  page.locator('[data-section=fieldsPanel]').click();page.wait_for_function("document.querySelector('#fieldA .plot-container')!==null")
  page.locator('#fieldsPanel').scroll_into_view_if_needed();page.wait_for_timeout(700);page.screenshot(path=str(root/'fields.png'))
  assert page.locator('#head option').count()==1
  # Renderer-only stress fixture: enlarge a computed map in the response, never persisted.
  def highres(route):
   response=route.fetch();d=response.json()
   if d.get('data'):
    h=d['data']['heads']['classification']
    for key in ['FieldA','IG']:
     z=h[key];h[key]=[[z[y//8][x//8]/64 for x in range(512)] for y in range(512)]
   route.fulfill(response=response,json=d)
  page.route('**/api/job/*',highres)
  page.locator('[data-section=provenance]').click();page.locator('#savedJobs button').first.click();page.wait_for_function("document.getElementById('fieldA').data?.[0]?.z.length===512",timeout=60000)
  page.locator('[data-section=fieldsPanel]').click();page.wait_for_timeout(500);assert page.locator('#fieldA').evaluate('(el)=>el.data[0].z[0].length')==512
  page.unroute('**/api/job/*',highres)
  assert not errors,errors
  print(json.dumps(dict(browser_unified13='passed',root=str(root),rid=rid,errors=errors)));b.close()
finally:server.terminate();server.wait(timeout=20);log.close()
