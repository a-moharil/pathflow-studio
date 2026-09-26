const fs=require('fs'),vm=require('vm'),assert=require('assert'),path=require('path');
const controls={};global.$=id=>controls[id]??={value:'0',checked:false,addEventListener(){}};global.window={devicePixelRatio:1};
vm.runInThisContext(fs.readFileSync(path.join(__dirname,'../lab/static/geometry.js'),'utf8')+'\nglobal.GeometryView=GeometryView;');
const v=Object.create(GeometryView.prototype);Object.assign(v,{k:0,w:800,h:500,base:50,pan:[45,-22],selected:0,fitted:true,canvas:{width:800,height:500,getBoundingClientRect:()=>({width:0,height:0})},ctx:{setTransform(){}}});
$('zoom0').value='.7';$('dimension0').value='2';$('anchor0').checked=true;
const d={positions:[[[1,2,0],[0,0,0]],[[1,2,0],[0,0,0]]],nodes:[{},{}],edges:[]};
v.resize();assert.equal(v.w,800);assert.equal(v.canvas.width,800);
v.setData(d);assert.equal(v.base,50);assert.deepEqual(v.pan,[45,-22]);assert.equal($('zoom0').value,'.7');
$('autoFit0').checked=true;v.setData(d);assert.equal($('zoom0').value,0);assert.deepEqual(v.pan,[0,0]);
$('anchor0').checked=true;assert.deepEqual(v.positions(false,false).at(-1),[0,0,0]);
// Execute the actual tab setup with DOM stubs and verify repeated navigation.
const buttons=[];const tabs={children:buttons,append(b){buttons.push(b)}};
const el=()=>({hidden:false,closest(){return this},before(){}});
for(const id of ['lossChart','geometry','record','fieldsWorkspace','analyze'])controls[id]=el();
controls.analyze.onclick=()=>{};let fits=0,resizes=0;
global.views=[{data:d,resize(){resizes++},fit(){fits++}}];global.requestAnimationFrame=f=>f();
global.document={createElement:()=>tabs,querySelector:s=>s==='a[href="#geometry"]'?el():el(),querySelectorAll:()=>[]};
// createElement also creates buttons.
document.createElement=tag=>tag==='div'?tabs:{setAttribute(){}};
const script=fs.readFileSync(path.join(__dirname,'../lab/static/experiment.js'),'utf8');
vm.runInThisContext(script.slice(script.indexOf('// Keep the task controls')));
for(let i=0;i<3;i++){window.activateStudioSection('Geometry');window.activateStudioSection('Fields');assert.equal(controls.geometry.hidden,false);assert.equal(controls.fieldsWorkspace.hidden,false);window.activateStudioSection('Training')}
assert.equal(fits,0);assert.equal(resizes,3);
console.log('Camera state, hidden resize, explicit autofit, anchor, and tab navigation passed.');

const independent={positions:[[[8,9,10],[0,0,0]],[[8,9,10],[0,0,0]]],euclidean:{dimensions:{'2':[{positions:[[1,2],[0,0]]},{positions:[[1,2],[0,0]]}],'3':[{positions:[[8,9,10],[0,0,0]]},{positions:[[8,9,10],[0,0,0]]}]}},nodes:[{},{}],edges:[]};
v.data=independent;$('dimension0').value='2';assert.deepEqual(v.positions(false,false)[0],[1,2,0]);$('yaw0').value='2';$('pitch0').value='1';assert.deepEqual(v.positions(false,false)[0],[1,2,0]);console.log('Independent 2-D coordinates are invariant to 3-D camera rotation.');
