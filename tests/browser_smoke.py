import os,subprocess,time,tempfile,urllib.request,json
from pathlib import Path
from playwright.sync_api import sync_playwright
project=Path(__file__).resolve().parents[1]
root=tempfile.mkdtemp(prefix='studio_browser_')
log=open(Path(root)/'browser.log','w')
p=subprocess.Popen(['python','run.py'],cwd=project,env={**os.environ,'PATHFLOW_HOME':root,'PORT':'5089'},stdout=log,stderr=log)
try:
 for _ in range(80):
  try:urllib.request.urlopen('http://127.0.0.1:5089').read();break
  except Exception:time.sleep(.25)
 with sync_playwright() as w:
  b=w.chromium.launch(headless=True);page=b.new_page(viewport={'width':1440,'height':1000},device_scale_factor=1)
  errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
  page.goto('http://127.0.0.1:5089/design');page.locator('[name=train_size]').fill('32');page.locator('[name=widths]').fill('64,64,64,64,64,64,64,64');page.locator('[name=optimizer]').select_option('RMSprop');page.locator('[name=updates]').fill('1000');page.locator('[name=checkpoint_every]').fill('10');page.locator('[name=geometry_limit]').fill('500');page.locator('[name=landmarks]').fill('16')
  page.wait_for_timeout(600);assert page.locator('#create').is_disabled();assert page.locator('[name=alpha]').is_visible();assert not page.locator('[name=beta1]').is_visible()
  page.locator('#prepareData').click();page.wait_for_function("!document.getElementById('create').disabled");page.locator('#create').click();page.wait_for_url('**/experiment/*');rid=page.url.split('/')[-1]
  page.locator('#start').click();page.wait_for_function("!document.getElementById('pause').disabled && /live update ([1-9][0-9]*)/.test(document.getElementById('state').textContent)");page.locator('#pause').click();page.wait_for_function("document.getElementById('controlNote').textContent.includes('Ready to inspect')",timeout=60000)
  page.wait_for_function("document.getElementById('geometryState').textContent.includes('nodes')",timeout=120000)
  assert page.evaluate("analysis.scope==='all' && views[0].edges.length===analysis.edges.length && analysis.edges.length>50000")
  assert page.locator('#lossChart').evaluate('(el)=>el.data.length')==3
  page.locator('.focus').nth(0).click();page.locator('.focus').nth(1).click();page.locator('#geometry0').scroll_into_view_if_needed();page.screenshot(path=str(Path(root)/'geometry.png'))
  page.locator('.sectionTabs button').filter(has_text='Geometry').click();page.locator('#dimension1').select_option('3');page.locator('#node1').fill('3');page.locator('#mix').fill('0.5');page.locator('#mix').fill('1')
  page.locator('.geomCard').nth(1).locator('summary').filter(has_text='Local observables').click();page.locator('#showInset1').check();page.locator('#showExpanded1').uncheck();assert page.locator('#inset1').is_visible();assert not page.locator('#localChart1').is_visible();page.locator('#showInset1').uncheck();page.locator('#showExpanded1').check();assert not page.locator('#inset1').is_visible();assert page.locator('#localChart1').is_visible()
  page.locator('.geomCard').nth(1).locator('summary').filter(has_text='Intervention controls').click();page.locator('.applyGate').nth(1).click();page.wait_for_function("document.getElementById('auditSummary').textContent.includes('native')",timeout=120000)
  page.goto('http://127.0.0.1:5089/observatory/'+rid);page.wait_for_timeout(1000);page.locator('#observable').select_option('deff');page.locator('#mode').select_option('histogram');page.locator('#scale').select_option('log');page.screenshot(path=str(Path(root)/'observatory.png'));assert not errors,errors
  page.set_viewport_size({'width':390,'height':844});page.wait_for_timeout(500);assert page.evaluate('document.documentElement.scrollWidth <= innerWidth+2')
  print(json.dumps({'browser':'passed','run':rid,'root':root,'errors':errors}));b.close()
finally:
 p.terminate();p.wait(timeout=20);log.close()
