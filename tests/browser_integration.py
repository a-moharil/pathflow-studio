"""Browser check of existing-run scheduling, all-output backfill and deletion."""
import os,sys,time,tempfile,subprocess,urllib.request,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lab import storage as st,datasets
from lab.config import validate
from lab.engine import quantum
from playwright.sync_api import sync_playwright
import torch
torch.set_num_threads(2)
project=Path(__file__).resolve().parents[1];root=Path(tempfile.mkdtemp(prefix='integration_browser_'))
st.init(root);c=validate(dict(widths=[3],train_size=32,updates=100000,checkpoint_every=1,measurement_limit=50));datasets.prepare(root,c);rid=st.create(root,c)
with st.db(root) as db:db.execute("UPDATE runs SET status='training' WHERE id=?",(rid,))
quantum(root,st.run(root,rid))
with st.db(root) as db:db.execute("UPDATE runs SET status='paused' WHERE id=?",(rid,))
log=open(root/'server.log','w');p=subprocess.Popen([sys.executable,'run.py'],cwd=project,env={**os.environ,'PATHFLOW_HOME':str(root),'PORT':'5090'},stdout=log,stderr=log)
try:
 for _ in range(100):
  try:urllib.request.urlopen('http://127.0.0.1:5090').read();break
  except Exception:time.sleep(.2)
 with sync_playwright() as w:
  browser=w.chromium.launch(headless=True);page=browser.new_page(viewport={'width':1440,'height':1000});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
  page.goto('http://127.0.0.1:5090/experiment/'+rid)
  page.locator('.sectionTabs button').filter(has_text='Configuration').click()
  page.locator('summary').filter(has_text='Configuration & job log').click()
  page.locator('#existingSchedule').fill('20');page.locator('#existingBudget').fill('1000');page.locator('#saveMeasurementSettings').click()
  page.wait_for_function("document.getElementById('scheduleStatus').textContent.startsWith('Saved:')")
  assert st.run(root,rid)['config']['geometry_every']==20
  page.goto('http://127.0.0.1:5090/observatory/'+rid)
  page.locator('#budget').fill('1000');page.locator('#backfill').click()
  page.wait_for_function("document.getElementById('measurementStatus').textContent.includes('2 available')",timeout=120000)
  page.locator('#observable').select_option('dstar')
  assert 'all path space' in page.locator('#contextNote').inner_text()
  page.screenshot(path=str(root/'observatory.png'))
  page.goto('http://127.0.0.1:5090/experiment/'+rid);page.locator('#start').click()
  page.wait_for_function("document.getElementById('state').textContent.includes('training')",timeout=30000)
  page.on('dialog',lambda d:d.accept());page.locator('#deleteRun').click();page.wait_for_url('http://127.0.0.1:5090/')
  page.wait_for_function("document.getElementById('experiments').textContent.includes('No experiments yet')",timeout=120000)
  assert not (root/rid).exists();assert (root/'datasets').exists();assert not errors,errors
  print(json.dumps({'browser_integration':'passed','root':str(root),'errors':errors}));browser.close()
finally:p.terminate();p.wait(timeout=20);log.close()
