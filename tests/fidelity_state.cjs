// Exercise the production renderer with a minimal DOM, including unavailable H3.
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const text=fs.readFileSync('lab/static/investigation.js','utf8');
const fn=text.slice(text.indexOf('function renderFidelity(d)'),text.indexOf("$('inspectMatrix')"));
const elements={},element=()=>({append(){},replaceChildren(){}});
const ctx={document:{createElement:element},$:id=>elements[id]??=(element()),Charts:{render(){}},number:String};
vm.createContext(ctx);vm.runInContext(fn,ctx);
for(const h of [undefined,{status:'budget_exceeded',reason:'4001 nodes exceed budget'},{status:'numerical_invalid',reason:'Constraint failure',dimensions:{}},{status:'available',dimensions:{'3':[{status:'undefined_measure'}]}}]){
 ctx.renderFidelity({hyperbolic:h,euclidean:{dimensions:{},pairs:{training:100,validation:20}}});
 const audit=JSON.parse(elements.fidelityAudit.textContent);
 assert.equal(audit.hyperbolic_status,h?.status);
 if(h&&h.status!=='available')assert(elements.fidelityContext.textContent.includes(h.reason));
}
console.log('Fidelity rendering: unrequested, budget-exceeded, invalid, and undefined states passed.');
