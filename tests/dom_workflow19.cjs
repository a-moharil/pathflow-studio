/* Execute the rendered healthcare page with real worker data. Canvas/Plotly are adapters. */
const {JSDOM}=require('jsdom'),fs=require('fs'),path=require('path'),assert=require('assert');
const dir=process.argv[2],fixture=JSON.parse(fs.readFileSync(path.join(dir,'fixtures.json'))),html=fs.readFileSync(path.join(dir,'page.html'),'utf8');
const dom=new JSDOM(html,{runScripts:'outside-only',pretendToBeVisual:true,url:'http://localhost/healthcare/'+fixture.run.id}),w=dom.window,$=id=>w.document.getElementById(id),errors=[];
w.$=$;w.number=v=>Number.isFinite(v)?v.toPrecision(5):'undefined';w.structuredClone=structuredClone;w.showError=e=>errors.push(String(e));w.clearError=()=>{};w.saveJSON=()=>{};w.saveText=()=>{};w.MediaRecorder=function(){};
w.HTMLCanvasElement.prototype.getContext=function(){return new Proxy({getImageData:()=>({data:new Uint8ClampedArray(280*280*4)}),createImageData:(a,b)=>({data:new Uint8ClampedArray(a*b*4)})},{get:(o,k)=>o[k]||(()=>{}),set:(o,k,v)=>(o[k]=v,true)})};w.HTMLCanvasElement.prototype.toDataURL=()=>'';w.HTMLCanvasElement.prototype.setPointerCapture=()=>{};w.HTMLCanvasElement.prototype.hasPointerCapture=()=>false;w.HTMLElement.prototype.scrollIntoView=()=>{};
w.Charts={render:(id,t,l)=>{$(id)._plot={t,l}}};w.Plotly={relayout:()=>{}};
let request=null,requests=0,delay=null;
w.api=async(url,p)=>{if(p){request=p;requests++;return{id:'new-job'}}if(url.endsWith('/dataset'))return fixture.dataset;if(url.endsWith('/measurement-jobs'))return fixture.jobs;if(url.startsWith('/api/job/')){const k=fixture.jobs.findIndex(j=>url.endsWith(j.id));return{status:'completed',data:Object.values(fixture.reports)[k]}}return fixture.run};
w.awaitJob=async()=>{if(delay)await delay;const name=request.kind==='healthcare_case'?'case':request.evidence_mode==='intervene'?'intervene':'compare';return structuredClone(fixture.reports[name])};
const scripts=['pair_selector','field_controls','source_region','region_explorer','healthcare','evidence','inspector','health_workflow'].map(n=>fs.readFileSync(path.join(__dirname,'../lab/static/'+n+'.js'),'utf8')).join('\n');
w.eval(scripts+'\nwindow.test19={submit:healthSubmit,display:healthDisplay,edit:healthEdited,case:()=>healthCase,evidence:()=>evidenceReport,context:hwContext,payload:()=>evidencePayload()};');
(async()=>{await new Promise(r=>setTimeout(r,20));assert.equal($('hcX').options.length,80);assert.equal(w.document.querySelectorAll('[data-hw-stage]').length,5);
await w.test19.submit('healthcare_case');assert.equal(w.test19.case().context.x,0);assert($('hwState').textContent.includes('Baseline available'));
w.healthWorkflow.open(2);$('evMode').value='compare';await w.test19.submit('healthcare_evidence',w.test19.payload());assert(w.test19.evidence());assert($('hwEvidenceState').textContent.includes('Compatible'));assert.equal($('ivMaps').closest('[data-hw-panel]').dataset.hwPanel,'2');assert.equal($('ivMechanism').closest('[data-hw-panel]').dataset.hwPanel,'3');
w.healthWorkflow.open(3);assert($('ivGallery').children.length>0);$('evIntervention').value='channel';$('evMode').value='intervene';await w.test19.submit('healthcare_evidence',w.test19.payload());assert.equal(w.healthWorkflow.stage,4);assert.equal(w.test19.evidence().evidence.mode,'intervene');
// Raw result survives selection edits, but incompatible intervention execution is blocked.
$('hcX').value='2';$('hcX').dispatchEvent(new w.Event('change',{bubbles:true}));assert($('hwState').textContent.includes('not the selected contrast'));const old=requests;await w.test19.submit('healthcare_evidence',w.test19.payload());assert.equal(requests,old);assert.equal(w.healthWorkflow.stage,1);
// Presentation cannot enqueue computations.
$('hwPresenter').click();await w.test19.submit('healthcare_case');assert.equal(requests,old);$('hwPresenter').click();
// Load a saved result restores its case and joins a compatible baseline.
$('hcHistory').value=fixture.jobs[2].id;await $('hcLoad').onclick();assert.equal($('hcX').value,'0');assert($('hwEvidenceState').textContent.includes('Compatible'));
// Edits while awaiting a job prevent substitution, but retain the completed result.
let finish;delay=new Promise(r=>finish=r);const running=w.test19.submit('healthcare_case');await new Promise(r=>setTimeout(r,0));$('hcX').value='4';$('hcX').dispatchEvent(new w.Event('change',{bubbles:true}));finish();await running;assert($('hcStatus').textContent.includes('not substituted'));delay=null;
assert.deepEqual(errors,[]);console.log('PASS: full healthcare scripts, real result rendering, staged navigation, compatible support, stale-job suppression, presentation guard and saved-context restore.');w.close();})().catch(e=>{console.error(e);w.close();process.exit(1)});
