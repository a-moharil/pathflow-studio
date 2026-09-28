"""Real end-to-end residual scalar analysis and CNN regression, no mocked maps."""
import os,sys,time,tempfile,subprocess,urllib.request,json
from pathlib import Path
from playwright.sync_api import sync_playwright
project=Path(__file__).resolve().parents[1];root=Path(tempfile.mkdtemp(prefix='selectors18_browser_'));log=(root/'server.log').open('w')
server=subprocess.Popen([sys.executable,'run.py'],cwd=project,env={**os.environ,'PATHFLOW_HOME':str(root),'PORT':'5098','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},stdout=log,stderr=log)
try:
 for _ in range(100):
  try:urllib.request.urlopen('http://127.0.0.1:5098');break
  except Exception:time.sleep(.2)
 with sync_playwright() as p:
  b=p.chromium.launch(headless=True,executable_path=os.environ.get("PW_CHROMIUM"));page=b.new_page(viewport={'width':1550,'height':1150});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
  page.goto('http://127.0.0.1:5098/design');page.locator('#design [name=train_size]').fill('32');page.locator('#design [name=model]').select_option('residual');page.wait_for_function("document.getElementById('dataset').value==='synthetic-42-32'")
  page.locator('#jointDesign [name=evaluation_count]').fill('2');page.locator('#jointDesign [name=save_every]').fill('1');page.locator('#prepareSelected').click();page.wait_for_function("!document.getElementById('jointCreate').disabled",timeout=60000)
  assert page.locator('#jointArchitecture').input_value()=='residual_small'
  page.locator('#jointCreate').click();page.wait_for_url('**/experiment/*');rid=page.url.rsplit('/',1)[1]
  page.locator('#updates').fill('1');page.locator('#train').click();page.wait_for_function("document.getElementById('runStatus').textContent.includes('paused · update 1')",timeout=120000)
  page.locator('[data-section=geometry]').click();page.locator('#igSteps').fill('4');page.locator('#computeGeometry').click();page.wait_for_function("document.getElementById('geometryStatus').textContent.includes('Full selected-readout scalar carrier')",timeout=180000)
  assert not page.locator('#error').is_visible(),page.locator('#error').inner_text()
  assert 'true label' in page.locator('#case option').first.inner_text()
  page.locator('#dimension1').select_option('3');page.locator('#labels1').select_option('all');page.locator('#geometry').scroll_into_view_if_needed();page.screenshot(path=str(root/'scalar_geometry.png'))
  page.locator('[data-section=fieldsPanel]').click();page.locator('#fieldMapping').select_option('log');page.locator('#structuralRule').select_option('he_normalized');page.locator('#metricEpsilon').fill('0.02');page.locator('#metricPower').fill('1.2');page.locator('#rx1').fill('4');page.locator('#ry1').fill('4');page.locator('#geometryMode').select_option('operations');page.locator('#traceRegion').click();page.wait_for_function("document.getElementById('regionStatus').textContent.includes('available')",timeout=120000)
  page.locator('#regionScale').select_option('allocation');page.locator('#fieldsPanel').scroll_into_view_if_needed();page.screenshot(path=str(root/'source_fields.png'))
  lab=page.locator('.regionExplorer');assert lab.locator('[data-delta]').is_visible()
  lab.locator('[data-layer]').select_option(index=1);lab.locator('[data-mask]').select_option('0.8')
  assert 'Mask contains' in lab.locator('[data-native-status]').inner_text()
  lab.locator('[data-overlay]').select_option('field');lab.screenshot(path=str(root/'residual_regions.png'))
  assert lab.locator('[data-layer-toggles] input').count()>1
  count=lab.locator('[data-layer-cards] article').count();lab.locator('[data-layer-toggles] input').first.uncheck();assert lab.locator('[data-layer-cards] article').count()==count-1
  page.get_by_role('checkbox',name='80% mass contour',exact=True).uncheck()
  assert 'effective source pixels' in page.locator('#massSummary').inner_text()
  audit=json.loads(page.locator('#carrierAudit').text_content());assert audit['audit']['conditional']['status']=='passed';assert audit['conventions']['structural_rule']=='he_normalized';assert audit['conventions']['metric_epsilon']==.02;assert audit['conventions']['metric_power']==1.2
  baseline_hash=audit['provenance']['architecture_graph_hash']
  page.locator('#alpha').fill('0.5');page.locator('#analyze').click();page.wait_for_function("document.getElementById('analyze').disabled===false",timeout=120000)
  page.locator('#state').select_option('intervened');post=json.loads(page.locator('#carrierAudit').text_content());assert post['provenance']['architecture_graph_hash']==baseline_hash;assert post['audit']['conditional']['status']=='passed';assert not page.locator('#error').is_visible(),page.locator('#error').inner_text()
  page.goto('http://127.0.0.1:5098/healthcare');page.locator('a[href*="domain=healthcare"]').click();assert page.locator('#jointArchitecture option[value=resnet50_joint]').count()==1
  assert page.locator('#jointDesign [name=task] option[value=joint]').count()==1
  page.goto('http://127.0.0.1:5098/design');page.locator('#design [name=dataset]').select_option('synthetic-regression');page.locator('#design [name=train_size]').fill('32');page.locator('#design [name=model]').select_option('cnn');page.locator('#design [name=channels]').fill('2,3');page.locator('#design [name=activation]').select_option('tanh');page.locator('#design [name=head_widths]').fill('4,3');page.locator('#design [name=updates]').fill('1');page.locator('#design [name=checkpoint_every]').fill('1');page.locator('#design [name=geometry_every]').fill('0');page.locator('#prepareData').click();page.wait_for_function("!document.getElementById('create').disabled",timeout=30000);page.locator('#create').click();page.wait_for_url('**/experiment/*');cnn=page.url.rsplit('/',1)[1]
  page.locator('#start').click();page.wait_for_function("document.getElementById('state').textContent.includes('completed')",timeout=60000)
  page.get_by_role('button',name='Geometry',exact=True).click();page.locator('#analyze').click();page.wait_for_function("document.getElementById('geometryState').textContent.includes('Checkpoint 1')",timeout=120000)
  assert not page.locator('#error').is_visible(),page.locator('#error').inner_text()
  page.get_by_role('button',name='Fields',exact=True).click()
  assert 'target' in page.locator('#x option').first.inner_text()
  before=page.locator('#reference').input_value()
  page.locator('.pairSelector [data-same]').click();page.wait_for_function("!sampleSelector.running",timeout=120000)
  first=page.locator('#reference').input_value();assert first!=before
  page.locator('.pairSelector [data-same]').click();page.wait_for_function("!sampleSelector.running",timeout=120000)
  assert page.locator('#reference').input_value()!=first
  page.locator('#referenceType').select_option('zero');assert page.locator('#reference').is_disabled()
  page.locator('#referenceType').select_option('dataset');assert page.locator('#reference').is_enabled()
  assert page.locator('.toolbar.sticky').evaluate('el=>getComputedStyle(el).position')=='static'
  lab=page.locator('.regionExplorer');lab.locator('[data-scope]').select_option('global')
  assert lab.locator('[data-layer] option').count()>=4
  canvas=lab.locator('[data-delta]');canvas.scroll_into_view_if_needed();bb=canvas.bounding_box()
  page.mouse.move(bb['x']+220,bb['y']+220);page.mouse.down();page.mouse.move(bb['x']+30,bb['y']+30,steps=5);page.mouse.up()
  assert int(page.locator('#sx0').input_value())<int(page.locator('#sx1').input_value())
  page.locator('#sourceCompute').click();page.wait_for_function("document.getElementById('analyze').disabled===false",timeout=120000)
  lab.locator('[data-scope]').select_option('local');lab.locator('[data-layer]').select_option(index=0);lab.locator('[data-mask]').select_option('0.8')
  assert 'Mask contains' in lab.locator('[data-native-status]').inner_text()
  native=lab.locator('[data-native]');native.scroll_into_view_if_needed();bb=native.bounding_box()
  page.mouse.move(bb['x']+140,bb['y']+140);page.mouse.down();page.mouse.move(bb['x']+240,bb['y']+240);page.mouse.up()
  lab.screenshot(path=str(root/'native_drag_debug.png'))
  print('Native drag:',page.evaluate("({roi:regionLab.roi,status:regionLab.el('native-status').textContent})"),bb,flush=True)
  assert 'Feature rectangle' in lab.locator('[data-native-status]').inner_text()
  lab.locator('[data-overlay]').select_option('field')
  assert lab.locator('[data-colour-key]').inner_text().startswith('Mask key')
  count=lab.locator('[data-layer-cards] article').count();lab.locator('[data-layer-toggles] input').first.uncheck();assert lab.locator('[data-layer-cards] article').count()==count-1
  lab.screenshot(path=str(root/'cnn_regions.png'))
  lab.locator('[data-layer]').select_option(index=2);assert 'Dense units' in lab.locator('[data-native-status]').inner_text()
  assert lab.locator('[data-units]').is_visible()
  page.locator('#sourcePicker [data-clear]').click();assert 'selection differs' in lab.locator('[data-layer-status]').inner_text()
  assert not errors,errors
  # A real classification experiment: repeated relation actions reach new true-label pairs.
  page.goto('http://127.0.0.1:5098/design');page.locator('#design [name=dataset]').select_option('synthetic');page.locator('#design [name=train_size]').fill('32');page.locator('#design [name=model]').select_option('mlp');page.locator('#design [name=widths]').fill('3,3');page.locator('#design [name=updates]').fill('1');page.locator('#design [name=checkpoint_every]').fill('1');page.locator('#design [name=geometry_every]').fill('0');page.locator('#prepareData').click();page.wait_for_function("!document.getElementById('create').disabled",timeout=30000);page.locator('#create').click();page.wait_for_url('**/experiment/*');page.locator('#start').click();page.wait_for_function("document.getElementById('state').textContent.includes('completed')",timeout=60000)
  page.get_by_role('button',name='Fields',exact=True).click();page.wait_for_function('sampleSelector!==null');assert 'Vertical stripe' in page.locator('#x option').first.inner_text()
  for relation in ['same','same','cross','cross']:
   before=page.locator('#reference').input_value();page.locator('.pairSelector [data-'+relation+']').click();page.wait_for_function('!sampleSelector.running',timeout=120000);assert page.locator('#reference').input_value()!=before
   assert page.evaluate("(data.test.labels[+$('x').value]===data.test.labels[+$('reference').value])") == (relation=='same')
   assert page.evaluate("analysis.x===+$('x').value && analysis.reference===+$('reference').value")
  page.locator('#x').select_option('3');page.locator('#reference').select_option('5');assert 'same true class' in page.locator('.pairSelector [data-pair-status]').inner_text()
  page.locator('.pairSelector').screenshot(path=str(root/'pair_selector.png'));assert not errors,errors

  print(json.dumps(dict(status='passed',root=str(root),residual=rid,cnn_regression=cnn,errors=errors)));b.close()
finally:server.terminate();server.wait(timeout=20);log.close();print('Artifacts:',root)
