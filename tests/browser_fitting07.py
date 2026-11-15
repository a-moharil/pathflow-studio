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
  page.locator('.sectionTabs button').filter(has_text='Geometry').click()
  assert page.locator('#embeddingSteps').get_attribute('max')=='5000'
  assert page.locator('#hyperSteps').get_attribute('max')=='5000'
  page.locator('#dimensionSensitivity').check();page.locator('#diagnosticDimensions').select_option(['8','16','32'])
  page.locator('#embeddingSteps').fill('501');page.locator('#fitHyperbolic').check();page.locator('#hyperSteps').fill('501');page.locator('#measurementBudget').fill('50')
  assert page.evaluate("payload().embedding_steps===501 && payload().hyperbolic.steps===501")
  page.locator('#embeddingSteps').fill('25');page.locator('#hyperSteps').fill('25');page.locator('#analyze').click()
  page.wait_for_function("typeof analysis!=='undefined' && analysis?.hyperbolic?.status==='available'",timeout=120000)
  assert page.evaluate("Object.keys(analysis.hyperbolic.dimensions).join(',')==='3,8,16,32'")
  assert page.evaluate("analysis.hyperbolic.dimensions['3'][0].optimization.max_iterations===25")
  assert page.locator('#fitHistory .plot-container').count()==1
  for k in [0,1]:page.locator('#representation'+str(k)).select_option('hyperbolic')
  page.wait_for_timeout(1200)
  assert page.evaluate("views[0].hyper() && views[1].hyper()")
  assert page.evaluate("analysis.euclidean.pairs.unconstrained_nodes===0 && analysis.hyperbolic.minimum_training_degree>=4")
  assert page.locator('#fidelityTable tr').count()>=7
  page.locator('#inspectMatrix').click();page.wait_for_function("!document.getElementById('matrixPlot').hidden")
  assert page.locator('#matrixStatus').inner_text().startswith('sampled_intrinsic_submatrix')
  old_geometry=page.evaluate('JSON.stringify(analysis.euclidean.dimensions)');previous=page.evaluate('analysis.step');target='0' if previous else '1'
  page.locator('#checkpoint').select_option(target)
  page.wait_for_function('(step)=>analysis?.step===step',arg=int(target),timeout=120000)
  assert page.evaluate("Object.keys(analysis.temporal_deformation).length>0")
  assert page.evaluate('JSON.stringify(analysis.euclidean.dimensions)')!=old_geometry
  page.locator('#fidelityTable').scroll_into_view_if_needed();page.screenshot(path=str(root/'fidelity.png'))
  page.locator('#anchor1').uncheck();page.wait_for_timeout(400)
  assert 'held-out stress' in page.locator('#hyperStatus1').inner_text()
  page.locator('summary').filter(has_text='Hyperbolic fidelity').click()
  page.locator('#inspectPair').click();page.wait_for_function("document.getElementById('pairDistances').textContent.includes('hyperbolic_3')")
  page.locator('#geometry0').scroll_into_view_if_needed();page.screenshot(path=str(root/'hyperbolic.png'))
  page.locator('#labels0').select_option('all');page.locator('#orbit0').check()
  old=page.locator('#yaw0').input_value();page.wait_for_timeout(500);assert page.locator('#yaw0').input_value()!=old
  page.locator('#orbit0').uncheck()
  page.locator('.sectionTabs button').filter(has_text='Fields').click()
  page.locator('#fitHyperbolic').uncheck();page.locator('#referenceType').select_option('zero');page.locator('#transformDx').fill('2');page.locator('#applyTransform').click()
  page.wait_for_function("analysis?.reference_type==='zero' && analysis?.transform.dx===2",timeout=120000)
  assert page.evaluate("analysis.reference_logits.length>0 && analysis.investigation.transported.retained_mass>0")
  assert page.locator('#transformPanels canvas').count()==4
  page.locator('#transformPanels').scroll_into_view_if_needed();page.screenshot(path=str(root/'transform.png'))
  page.locator('#resetTransform').click();page.wait_for_function("analysis?.transform.dx===0",timeout=120000)
  assert page.evaluate("analysis.investigation.H_eq<1e-8")
  page.locator('#sweepFrames').fill('2');page.locator('#sweepFrom').fill('0');page.locator('#sweepTo').fill('1');page.locator('#transformPlay').click()
  page.wait_for_function("analysis?.transform.dx===1 && !analyzing",timeout=120000)
  page.locator('#stopTransform').click()
  page.locator('.sectionTabs button').filter(has_text='Geometry').click()
  page.locator('#checkpointPlay').click();page.wait_for_function("document.getElementById('playbackState').textContent.includes('Measured frame 2/2')",timeout=120000)
  page.locator('#stopPlayback').click()
  page.locator('#record').click();page.wait_for_timeout(600)
  with page.expect_download() as download:page.locator('#record').click()
  assert download.value.suggested_filename.startswith('pathflow_geometry.')
  assert page.evaluate('''async()=>{const previous=analysis;const pending=compute();document.getElementById('x').dispatchEvent(new Event('change'));await pending;return analysis===previous}''')
  assert not errors,errors
  print(json.dumps({'browser_fitting07':'passed','root':str(root),'errors':errors}));browser.close()
finally:p.terminate();p.wait(timeout=20);log.close()
