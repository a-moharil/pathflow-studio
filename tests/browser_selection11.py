"""Fresh-workspace regression: discover, prepare, create and find joint experiments."""
import os,sys,time,tempfile,subprocess,urllib.request,json
from pathlib import Path
from playwright.sync_api import sync_playwright
project=Path(__file__).resolve().parents[1];root=Path(tempfile.mkdtemp(prefix='selection11_'));log=(root/'server.log').open('w')
server=subprocess.Popen([sys.executable,'run.py'],cwd=project,env={**os.environ,'PATHFLOW_HOME':str(root),'PORT':'5098'},stdout=log,stderr=log)
try:
 for _ in range(100):
  try:urllib.request.urlopen('http://127.0.0.1:5098');break
  except Exception:time.sleep(.2)
 with sync_playwright() as p:
  b=p.chromium.launch(headless=True);page=b.new_page(viewport={'width':1450,'height':1100});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
  page.goto('http://127.0.0.1:5098/design');page.locator('select[name=dataset]').select_option('bus-uclm-v1');page.wait_for_url('**/joint?*');page.wait_for_function("document.querySelector('#dataset')?.options.length>=2")
  assert page.locator('#dataset').input_value()=='bus-uclm-v1';assert 'not downloaded' in page.locator('#dataset').inner_text();assert page.locator('#jointCreate').is_disabled()
  assert page.locator('#jointArchitecture').input_value()=='resnet50_joint'
  page.locator('#jointPreset').select_option('finetune256');assert page.locator('[name=pretrained]').input_value()=='true';assert page.locator('[name=evaluation_count]').input_value()=='0'
  page.locator('#jointPreset').select_option('engineering64');assert page.locator('[name=size]').input_value()=='64';assert page.locator('#dataset').input_value()=='synthetic'
  page.locator('#prepareSelected').click();page.wait_for_function("!document.getElementById('jointCreate').disabled",timeout=60000)
  page.locator('#jointDesign').scroll_into_view_if_needed();page.screenshot(path=str(root/'presets.png'))
  page.locator('#jointCreate').click();page.wait_for_url('**/joint/*');rid=page.url.rsplit('/',1)[1]
  page.goto('http://127.0.0.1:5098/');page.wait_for_selector(f'a[href="/joint/{rid}"]');assert 'ResNet-50' in page.locator('#experiments').inner_text()
  page.goto('http://127.0.0.1:5098/design');page.locator('select[name=model]').select_option('resnet50_joint');page.wait_for_url('**/joint?*');page.wait_for_function("document.querySelector('#dataset')?.options.length>=2")
  assert page.locator('#dataset').input_value()=='bus-uclm-v1';assert page.locator('#jointCreate').is_disabled()
  page.goto('http://127.0.0.1:5098/design');assert page.locator('select[name=model]').input_value()=='mlp';assert page.locator('select[name=dataset]').input_value()=='synthetic'
  assert not errors,errors;print(json.dumps(dict(browser_selection11='passed',root=str(root),errors=errors)));b.close()
finally:server.terminate();server.wait(timeout=20);log.close()
