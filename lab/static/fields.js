/* Pixel coordinates, no image-space interpolation or invented field values. */
const InputFields={
 data:null,
 render(d){this.data=d;const fields=d.input_fields;
 $('fieldContext').textContent=`Checkpoint ${d.step} · pair ${d.x} / ${d.reference} · scope ${d.scope} · readout ${JSON.stringify(d.readout)} · gates ${JSON.stringify(d.gates)}`;
 if(!fields){$('fieldLegend').textContent='This saved result predates input fields. Analyze the checkpoint again.';for(const id of ['fieldStructural','fieldConditional'])$(id).getContext('2d').clearRect(0,0,280,280);return}
 const max=$('fieldScale').value==='unit'?1:Math.max(...(fields.structural||[]).flat(),...(fields.conditional||[]).flat(),Number.MIN_VALUE);
 for(const [side,id] of [['structural','fieldStructural'],['conditional','fieldConditional']]){
 const canvas=$(id),ctx=canvas.getContext('2d'),a=fields[side];canvas.onpointermove=null;ctx.clearRect(0,0,canvas.width,canvas.height);ctx.fillStyle='#091621';ctx.fillRect(0,0,canvas.width,canvas.height);
 if(!a){ctx.fillStyle='#ffffff';ctx.font='14px system-ui';ctx.fillText('Measure undefined',20,140);continue}
 const h=a.length,w=a[0].length;
 a.forEach((row,y)=>row.forEach((v,x)=>{const t=Math.max(0,Math.min(1,v/max));ctx.fillStyle=`rgb(${Math.round(12+230*t)},${Math.round(25+168*t)},${Math.round(40+45*t)})`;ctx.fillRect(x*canvas.width/w,y*canvas.height/h,canvas.width/w,canvas.height/h)}));
 canvas.onpointermove=e=>{const r=canvas.getBoundingClientRect(),x=Math.min(w-1,Math.max(0,Math.floor((e.clientX-r.left)*w/r.width))),y=Math.min(h-1,Math.max(0,Math.floor((e.clientY-r.top)*h/r.height)));$('fieldPixel').textContent=`Pixel (x=${x}, y=${y}) · structural ${fields.structural?.[y]?.[x]?.toExponential(6)??'undefined'} · conditional ${fields.conditional?.[y]?.[x]?.toExponential(6)??'undefined'}`};
 }
 $('fieldLegend').textContent=`Linear color scale: dark = 0; gold = ${max.toExponential(4)} probability per pixel. Structural sum ${fields.structural?fields.structural.flat().reduce((a,b)=>a+b,0).toFixed(9):'undefined'}; conditional sum ${fields.conditional?fields.conditional.flat().reduce((a,b)=>a+b,0).toFixed(9):'undefined'}.`;
 }
};
document.getElementById('fieldScale').onchange=()=>{if(InputFields.data)InputFields.render(InputFields.data)};
