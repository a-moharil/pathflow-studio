// Routing concentration readout: strongest-edge share and zero-mass share from edge marginals.
const {JSDOM}=require('jsdom'),fs=require('fs'),path=require('path'),assert=require('assert');
const src=fs.readFileSync(path.join(__dirname,'../lab/static/experiment.js'),'utf8');
const start=src.indexOf('window.showRouting=');const fn=src.slice(start);  // the readout is the last statement of experiment.js
const dom=new JSDOM('<p id="routingNote" hidden></p>',{runScripts:'outside-only'}),w=dom.window;
w.$=id=>w.document.getElementById(id);w.eval(fn);
// inputs 0,1 -> hidden 2,3 -> output 4. Input 0: s split 1:3, c all on one edge; input 1: no conditional mass.
w.showRouting({nodes:[{layer:0},{layer:0},{layer:1},{layer:1},{layer:2}],edges:[
 {u:0,v:2,s:1,c:2},{u:0,v:3,s:3,c:0},{u:1,v:2,s:1,c:0},{u:1,v:3,s:1,c:0},{u:2,v:4,s:1,c:1},{u:3,v:4,s:1,c:0}]});
const t=w.$('routingNote').textContent;
assert(!w.$('routingNote').hidden);
assert(t.includes('structural 0.63'),t);   // (0.75 + 0.5)/2
assert(t.includes('conditional 1.00'),t);
assert(t.includes('Edges without mass: structural 0% · conditional 67%'),t);
assert(t.includes('Inputs carrying conditional mass: 1/2'),t);
console.log('routing readout: ok');
