/* Field A overlay on the raw-contrast panel. Real RegionExplorer in jsdom; canvas fills are recorded. */
const {JSDOM}=require('jsdom'),fs=require('fs'),path=require('path'),assert=require('assert');
const dom=new JSDOM(`<!doctype html><body><div id="sourcePicker"></div><input id="sx0"><input id="sy0"><input id="sx1"><input id="sy1"><input id="enabled" type="checkbox"><button id="compute"></button></body>`,{runScripts:'outside-only'});
const w=dom.window,$=id=>w.document.getElementById(id);w.$=$;w.Charts={render:()=>{}};
// Record every fillStyle written to each canvas, in drawing order.
w.HTMLCanvasElement.prototype.getContext=function(){const canvas=this;canvas.fills=canvas.fills||[];return new Proxy({},{get:(o,k)=>k==='clearRect'?()=>{canvas.fills=[]}:(o[k]||(()=>{})),set:(o,k,v)=>{if(k==='fillStyle')canvas.fills.push(v);o[k]=v;return true}})};
for(const file of ['field_controls','source_region','region_explorer'])w.eval(fs.readFileSync(path.join(__dirname,'../lab/static/'+file+'.js'),'utf8')+'\n'+({field_controls:'window.MassContourControls=MassContourControls;window.LayerTraceControls=LayerTraceControls;',source_region:'window.SourceRegionPicker=SourceRegionPicker;',region_explorer:'window.RegionExplorer=RegionExplorer;'}[file]));
const picker=new w.SourceRegionPicker('sourcePicker',['sx0','sy0','sx1','sy1'],'enabled','compute'),lab=new w.RegionExplorer(picker);
// 2x2 input: x and x° differ only in the left column; Field A mass sits only at the top-right pixel.
const x=[[[1],[0]],[[1],[0]]],ref=[[[0],[0]],[[0],[0]]],field=[[0,1],[0,0]];
picker.update(field,null);lab.update(x,ref,field,[],null);
const rgb=s=>s.match(/\d+/g).map(Number),delta=()=>lab.el('delta').fills.map(rgb);
const set=(name,value)=>{lab.el(name).value=value;lab.el(name).dispatchEvent(new w.Event('input'))};
// Raw contrast mode: the contrast panel shows raw |Δx| in red, independent of the opacity slider.
set('overlay','contrast');const raw=delta();assert.equal(raw.length,4);
assert(raw[0][0]>200&&raw[0][1]<80,'left-column contrast is red');set('opacity','0.2');assert.deepEqual(delta(),raw,'raw contrast panel is fixed');
assert.equal(lab.el('delta-caption').textContent,'Raw contrast · mean channel |x − x°|');
// Field A mode: raw contrast becomes a greyscale base under the gold Field A overlay.
set('overlay','field');set('opacity','1');let f=delta();
assert.equal(lab.el('delta-caption').textContent,'Raw contrast |x − x°| · Field A probability overlay');
assert.deepEqual(f[0],[255,255,255],'contrast pixel without Field A mass stays as greyscale contrast');
assert.deepEqual(f[2],[255,255,255]);assert.deepEqual(f[3],[0,0,0],'no contrast and no mass is black');
assert.deepEqual(f[1],[255,184,51],'Field A mass is gold at full opacity');
// The opacity slider now acts on the contrast panel too.
set('opacity','0.5');f=delta();assert.deepEqual(f[1],[128,92,26],'half-opacity gold over a zero-contrast base');assert.deepEqual(f[0],[255,255,255]);
// Raw image only: contrast panel returns to raw contrast.
set('overlay','none');assert.deepEqual(delta(),raw);
console.log('PASS: Field A overlay on the raw-contrast panel, greyscale contrast base, opacity, captions and fixed raw mode.');
