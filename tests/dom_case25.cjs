// Case carrying across stages: rail links take ?case=, the chip names the case, a missing
// record is released, and Case.set moves every stage link.
const {JSDOM}=require('jsdom'),fs=require('fs'),path=require('path'),assert=require('assert');
const src=fs.readFileSync(path.join(__dirname,'../lab/static/case.js'),'utf8');
const html=`<body><div id="runBar" data-run="r1"><span id="caseChip" hidden><span id="caseText"></span><button id="caseClear"></button></span>
<nav class="runTabs"><a data-tab="geometry" href="/run/r1/geometry">G</a><a data-tab="audit" href="/auditor?run=r1">A</a></nav></div>
<a id="carry" data-carry-case href="/run/r1/evidence">E</a><a id="plain" href="/queue">Q</a></body>`;
const cases={j1:{run:'r1',label:'Update 4 · test · input 0 vs reference 1',stages:{analysis:'j1'}},j2:{run:'r1',label:'Update 6 · test · input 3 vs zero reference',stages:{healthcare_case:'j2'}}};
async function page(url,storage={}){
  const dom=new JSDOM(html,{url,runScripts:'outside-only'}),w=dom.window;
  for(const [k,v] of Object.entries(storage))w.sessionStorage.setItem(k,v);
  w.api=async u=>{const id=decodeURIComponent(u.split('/').pop().split('?')[0]);if(u.includes('/case/open'))return{match:'m',url:u};if(cases[id])return cases[id];throw Error('Case not found')};
  w.eval(src);await new Promise(r=>setTimeout(r,20));return w;
}
(async()=>{
  let w=await page('http://localhost/experiment/r1?section=Geometry&case=j1');
  const $=id=>w.document.getElementById(id);
  assert.equal(w.document.querySelector('[data-tab=geometry]').getAttribute('href'),'/run/r1/geometry?case=j1');
  assert.equal(w.document.querySelector('[data-tab=audit]').getAttribute('href'),'/auditor?run=r1&case=j1');
  assert.equal($('carry').getAttribute('href'),'/run/r1/evidence?case=j1');
  assert.equal($('plain').getAttribute('href'),'/queue');
  assert(!$('caseChip').hidden && $('caseText').textContent.startsWith('Update 4'));
  assert.equal(w.sessionStorage.getItem('pathflow.case.r1'),'j1');
  const opened=await w.Case.open('analysis');assert(opened.url.includes('kind=analysis')&&opened.url.includes('case=j1'));
  let events=0;w.document.addEventListener('pathflow:case',()=>events++);
  w.Case.set('j2');await new Promise(r=>setTimeout(r,20));
  assert.equal(w.location.search,'?section=Geometry&case=j2');
  assert.equal(w.document.querySelector('[data-tab=geometry]').getAttribute('href'),'/run/r1/geometry?case=j2');
  assert($('caseText').textContent.includes('zero reference')&&events===1);
  $('caseClear').click();await new Promise(r=>setTimeout(r,20));
  assert($('caseChip').hidden&&w.document.querySelector('[data-tab=geometry]').getAttribute('href')==='/run/r1/geometry'&&w.location.search==='?section=Geometry');
  // A stage opened without ?case= resumes the remembered case of this run.
  w=await page('http://localhost/observatory/r1',{'pathflow.case.r1':'j1'});
  assert.equal(w.location.search,'?case=j1');
  // A deleted record is released and the stage falls back to the latest result.
  w=await page('http://localhost/observatory/r1?case=gone');
  assert(w.document.getElementById('caseChip').hidden&&w.Case.id===null&&w.location.search==='');
  assert.equal(w.sessionStorage.getItem('pathflow.case.r1'),null);
  console.log('case carrying: ok');
})().catch(e=>{console.error(e);process.exit(1)});
