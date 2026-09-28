"""Real end-to-end residual scalar analysis and CNN regression, no mocked maps."""
import os,sys,time,tempfile,subprocess,urllib.request,json
from pathlib import Path
from playwright.sync_api import sync_playwright
project=Path(__file__).resolve().parents[1];root=Path(tempfile.mkdtemp(prefix='residual14_browser_'));log=(root/'server.log').open('w')
server=subprocess.Popen([sys.executable,'run.py'],cwd=project,env={**os.environ,'PATHFLOW_HOME':str(root),'PORT':'5098','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},stdout=log,stderr=log)
try:
 for _ in range(100):
  try:urllib.request.urlopen('http://127.0.0.1:5098');break
  except Exception:time.sleep(.2)
 with sync_playwright() as p:
  b=p.chromium.launch(headless=True);page=b.new_page(viewport={'width':1550,'height':1150});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
  page.goto('http://127.0.0.1:5098/design');page.locator('#design [name=train_size]').fill('32');page.locator('#design [name=model]').select_option('residual');page.wait_for_function("document.getElementById('dataset').value==='synthetic-42-32'")
  page.locator('#jointDesign [name=evaluation_count]').fill('2');page.locator('#jointDesign [name=save_every]').fill('1');page.locator('#prepareSelected').click();page.wait_for_function("!document.getElementById('jointCreate').disabled",timeout=60000)
  assert page.locator('#jointArchitecture').input_value()=='residual_small'
  page.locator('#jointCreate').click();page.wait_for_url('**/experiment/*');rid=page.url.rsplit('/',1)[1]
  page.locator('#updates').fill('1');page.locator('#train').click();page.wait_for_function("document.getElementById('runStatus').textContent.includes('paused · update 1')",timeout=120000)
  page.locator('[data-section=geometry]').click();page.locator('#igSteps').fill('4');page.locator('#computeGeometry').click();page.wait_for_function("document.getElementById('geometryStatus').textContent.includes('Full scalar occurrence')",timeout=180000)
  assert not page.locator('#error').is_visible(),page.locator('#error').inner_text()
  page.locator('#dimension1').select_option('3');page.locator('#labels1').select_option('all');page.locator('#geometry').scroll_into_view_if_needed();page.screenshot(path=str(root/'scalar_geometry.png'))
  page.locator('[data-section=fieldsPanel]').click();page.locator('#fieldMapping').select_option('log');page.locator('#rx1').fill('4');page.locator('#ry1').fill('4');page.locator('#geometryMode').select_option('operations');page.locator('#traceRegion').click();page.wait_for_function("document.getElementById('regionStatus').textContent.includes('available')",timeout=120000)
  page.locator('#regionScale').select_option('allocation');page.locator('#fieldsPanel').scroll_into_view_if_needed();page.screenshot(path=str(root/'source_fields.png'))
  assert 'effective source pixels' in page.locator('#massSummary').inner_text()
  page.locator('#alpha').fill('0.5');page.locator('#analyze').click();page.wait_for_function("document.getElementById('analyze').disabled===false",timeout=120000)
  page.locator('#state').select_option('intervened');assert not page.locator('#error').is_visible(),page.locator('#error').inner_text()
  page.goto('http://127.0.0.1:5098/healthcare');page.locator('a[href*="domain=healthcare"]').click();assert page.locator('#jointArchitecture option[value=resnet50_joint]').count()==1
  assert page.locator('#jointDesign [name=task] option[value=joint]').count()==1
  page.goto('http://127.0.0.1:5098/design');page.locator('#design [name=dataset]').select_option('synthetic-regression');page.locator('#design [name=train_size]').fill('32');page.locator('#design [name=model]').select_option('cnn');page.locator('#design [name=channels]').fill('2,3');page.locator('#design [name=head_widths]').fill('4,3');page.locator('#design [name=updates]').fill('1');page.locator('#design [name=checkpoint_every]').fill('1');page.locator('#design [name=geometry_every]').fill('0');page.locator('#prepareData').click();page.wait_for_function("!document.getElementById('create').disabled",timeout=30000);page.locator('#create').click();page.wait_for_url('**/experiment/*');cnn=page.url.rsplit('/',1)[1]
  page.locator('#start').click();page.wait_for_function("document.getElementById('state').textContent.includes('completed')",timeout=60000)
  page.get_by_role('button',name='Geometry',exact=True).click();page.locator('#analyze').click();page.wait_for_function("document.getElementById('geometryState').textContent.includes('Checkpoint 1')",timeout=120000)
  assert not page.locator('#error').is_visible(),page.locator('#error').inner_text()
  assert 'predictions' in page.locator('#pairInfo').inner_text();assert not errors,errors
  print(json.dumps(dict(status='passed',root=str(root),residual=rid,cnn_regression=cnn,errors=errors)));b.close()
finally:server.terminate();server.wait(timeout=20);log.close();print('Artifacts:',root)
