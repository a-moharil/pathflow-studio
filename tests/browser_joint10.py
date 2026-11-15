"""Browser integration with a real ResNet checkpoint and saved analysis.
Run tests/smoke_joint10.py first. JOINT_SMOKE_ROOT can override its output.
"""
import json,os,sys,time,subprocess,urllib.request
from pathlib import Path
from playwright.sync_api import sync_playwright
project=Path(__file__).resolve().parents[1];root=Path(os.environ.get('JOINT_SMOKE_ROOT','/tmp/joint10-smoke'));info=json.loads((root/'smoke.json').read_text());log=(root/'browser.log').open('w')
server=subprocess.Popen([sys.executable,'run.py'],cwd=project,env={**os.environ,'PATHFLOW_HOME':str(root),'PORT':'5096'},stdout=log,stderr=log)
try:
 for _ in range(100):
  try:urllib.request.urlopen('http://127.0.0.1:5096/joint');break
  except Exception:time.sleep(.2)
 with sync_playwright() as p:
  b=p.chromium.launch(headless=True);page=b.new_page(viewport={'width':1500,'height':1100});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
  page.goto('http://127.0.0.1:5096/joint');page.wait_for_selector('#dataset option',state='attached');assert 'Integration smoke' in page.locator('#jointRegistry').inner_text()
  page.goto('http://127.0.0.1:5096/joint/'+info['rid']);page.wait_for_selector('#savedJobs button:not([disabled])');page.locator('#savedJobs button:not([disabled])').first.click();page.wait_for_selector('#gallery button',timeout=60000)
  assert 'update 1' in page.locator('#resultContext').inner_text();assert 'Reconstruction residual' in page.locator('#audit').inner_text()
  page.locator('#fieldA').scroll_into_view_if_needed();page.wait_for_timeout(1000);page.screenshot(path=str(root/'fields.png'))
  page.locator('#gallery button').nth(1).click();assert page.locator('#channel').input_value()=='1'
  page.locator('#head').select_option('segmentation');page.locator('#state').select_option('intervened');assert 'No intervened result' not in page.locator('#audit').inner_text()
  page.locator('#state').select_option('baseline');page.locator('#dimensions').select_option('3');page.locator('#labels').check();page.locator('#ghost').uncheck();page.locator('#anchor').uncheck();page.locator('#geometry').scroll_into_view_if_needed();page.wait_for_timeout(1500);page.screenshot(path=str(root/'geometry.png'))
  page.locator('#labels').uncheck();page.locator('#motion').check();page.locator('#recordGeometry').click();page.wait_for_timeout(800)
  with page.expect_download(timeout=10000) as recording:page.locator('#recordGeometry').click()
  assert recording.value.suggested_filename=='joint-geometry.webm'
  with page.expect_download(timeout=10000) as png:page.locator('#geometryPNG').click()
  assert png.value.suggested_filename=='joint-selected-geometry.png'
  with page.expect_download(timeout=30000) as dl:page.locator('#export').click()
  assert dl.value.suggested_filename.endswith('.json')
  page.locator('#refType').select_option('zero');assert not errors,errors
  page.set_viewport_size({'width':600,'height':900});page.locator('#investigation').scroll_into_view_if_needed();page.wait_for_timeout(700);page.screenshot(path=str(root/'mobile.png'))
  print(json.dumps(dict(browser_joint10='passed',errors=errors,root=str(root))),flush=True);b.close()
finally:server.terminate();server.wait(timeout=20);log.close()
