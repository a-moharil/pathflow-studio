/* Run: node tests/render_completeness.cjs */
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const controls={};global.$=id=>controls[id]??={value:'1',checked:false,textContent:''};
global.number=String;global.window={devicePixelRatio:1};
const context=new Proxy({}, {get:(o,k)=>o[k]??(()=>{}),set:(o,k,v)=>(o[k]=v,true)});
global.document={createElement:()=>({getContext:()=>context})};
vm.runInThisContext(fs.readFileSync(require('path').join(__dirname,'../lab/static/geometry.js'),'utf8')+'\nglobal.GeometryView=GeometryView;');
const v=Object.create(GeometryView.prototype);Object.assign(v,{k:0,w:500,h:500,canvas:{width:500,height:500},base:1,selected:0,pan:[0,0]});
v.project=()=>[[100,100],[300,300]];v.edges=Array.from({length:79490},(_,i)=>({u:0,v:1,s:i%2?1/40000:0,c:0}));v.data={edges:v.edges,scope:'all'};
let carriers=0;v.edge=(c,a,b,i,color)=>{if(color==='#7797aa')carriers++;return[200,200]};v.renderCarrier();
assert.equal(carriers,79490);assert(v.markers.length<=3000);assert(controls.routeCount0.textContent.includes('79,490 / 79,490'));assert(controls.routeCount0.textContent.includes('39,745 zero-mass'));
console.log('Complete rendering verified: 79,490 edges, including all 39,745 zero-mass carriers.');
