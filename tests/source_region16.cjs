// Dependency-free interaction regression tests; DOM/canvas adapters, no model mocks.
const fs=require('fs'),vm=require('vm'),assert=require('assert'),path=require('path');
const els={};class Element{constructor(){this.value='0';this.checked=false;this.disabled=false;this.width=280;this.height=280;this.style={};this.textContent='';this.listeners={};this.children={};}set value(v){this._value=String(v)}get value(){return this._value}addEventListener(n,f){this.listeners[n]=f}querySelector(n){return this.children[n]||(this.children[n]=new Element())}getContext(){return new Proxy({},{get:()=>()=>{}})}getBoundingClientRect(){return {left:0,top:0,width:280,height:280}}setPointerCapture(){}releasePointerCapture(){}}
const $=id=>els[id]||(els[id]=new Element());const plots={};let computed=0;
const ctx=vm.createContext({document:{getElementById:$},$,console,number:v=>String(v),compute:()=>computed++,RegionExplorer:class{update(){}static chw(a){return a}},Charts:{render:(id,traces,layout)=>plots[id]={traces,layout}}});
const dir=path.join(__dirname,'../lab/static');vm.runInContext(fs.readFileSync(path.join(dir,'source_region.js'),'utf8'),ctx);vm.runInContext(fs.readFileSync(path.join(dir,'fields.js'),'utf8'),ctx);
const run=s=>vm.runInContext(s,ctx);const a=Array.from({length:4},()=>Array(4).fill(1/16));ctx.data={step:1,x:0,reference:1,scope:'all',readout:{a:0,b:1},gates:{},input_fields:{shape:[1,4,4],conditional:a,structural:a},source_diagnostics:{signed:a},source_region:null};$('fieldMapping').value='sqrt';run('InputFields.render(data)');
assert.equal(plots.sourceDiagnostics.layout.xaxis.title,'Image x (pixel)');assert.deepEqual([...plots.sourceDiagnostics.layout.xaxis.range],[-.5,3.5]);assert.deepEqual([...plots.sourceDiagnostics.layout.yaxis.range],[3.5,-.5]);assert.equal(plots.sourceDiagnostics.layout.xaxis.constrain,'domain');assert.equal(plots.sourceDiagnostics.layout.yaxis.scaleanchor,'x');assert(plots.sourceDiagnostics.layout.title.text.includes('sum of output contrasts'));
const c=$('sourcePicker').querySelector('canvas'),status=$('sourcePicker').querySelector('[data-status]');const ev=(x,y)=>({clientX:x,clientY:y,button:0,pointerId:1,preventDefault(){}});
c.onpointerdown(ev(220,220));c.onpointermove(ev(80,80));c.onpointerup(ev(80,80));assert.deepEqual(Array.from(run('sourcePicker.values()')),[1,1,4,4]);assert(status.textContent.includes('56.25%'));assert(status.textContent.includes('Selection changed'));
c.onpointerdown(ev(80,80));c.onpointerup(ev(220,220));assert.deepEqual(Array.from(run('sourcePicker.values()')),[1,1,4,4]);
c.onpointerdown(ev(10,10));c.onpointerup(ev(10,10));assert.deepEqual(Array.from(run('sourcePicker.values()')),[0,0,1,1]);assert(status.textContent.includes('6.250%'));
$('sourcePicker').querySelector('[data-full]').onclick();assert.deepEqual(Array.from(run('sourcePicker.values()')),[0,0,4,4]);assert(status.textContent.includes('100.0%'));$('sourceCompute').onclick();assert.equal(computed,1);
$('sourcePicker').querySelector('[data-clear]').onclick();assert(!$('sourceTrace').checked);assert($('sourceCompute').disabled);
$('sx0').value='';$('sx0').listeners.input();assert($('sourceCompute').disabled);$('sourceCompute').onclick();assert.equal(computed,1);
$('sx0').value='5';$('sx0').listeners.input();assert($('sourceCompute').disabled);
run('sourcePicker.set([0,0,1,1]);sourcePicker.update(data.input_fields.conditional,{region:[0,0,1,1]})');assert(status.textContent.includes('Matches computed region'));
ctx.data.scope='readout';run('InputFields.render(data)');assert(plots.sourceDiagnostics.layout.title.text.includes('selected readout'));
run('sourcePicker.update(null,null)');assert($('sourceCompute').disabled);assert(status.textContent.includes('Compute a defined Field A'));
ctx.data.input_fields=null;run('InputFields.render(data)');assert($('sourceCompute').disabled);
console.log('PASS: selector interactions, exact mass, invalid input, stale state, pixel axes and readout scope');
