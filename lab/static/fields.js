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
 a.forEach((row,y)=>row.forEach((v,x)=>{const q=Math.max(0,Math.min(1,v/max)),mode=$('fieldMapping').value,t=mode==='sqrt'?Math.sqrt(q):mode==='log'?(q>0?Math.max(0,1+Math.log10(q)/8):0):q;ctx.fillStyle=`rgb(${Math.round(12+230*t)},${Math.round(25+168*t)},${Math.round(40+45*t)})`;ctx.fillRect(x*canvas.width/w,y*canvas.height/h,canvas.width/w,canvas.height/h)}));
 if($('fieldContours').checked){const summary=side==='conditional'?d.source_summary:null;for(const [i,c] of (summary?.contours||[]).entries()){ctx.strokeStyle=['white','#ffc857','#f08c9a'][i];ctx.lineWidth=1;for(let y=0;y<h;y++)for(let x=0;x<w;x++)if(a[y][x]>=c.threshold){let xx=x*canvas.width/w,yy=y*canvas.height/h,ww=canvas.width/w,hh=canvas.height/h;ctx.beginPath();if(y===0||a[y-1][x]<c.threshold){ctx.moveTo(xx,yy);ctx.lineTo(xx+ww,yy)}if(x===0||a[y][x-1]<c.threshold){ctx.moveTo(xx,yy);ctx.lineTo(xx,yy+hh)}if(y===h-1||a[y+1][x]<c.threshold){ctx.moveTo(xx,yy+hh);ctx.lineTo(xx+ww,yy+hh)}if(x===w-1||a[y][x+1]<c.threshold){ctx.moveTo(xx+ww,yy);ctx.lineTo(xx+ww,yy+hh)}ctx.stroke()}}}
 canvas.onclick=e=>{const b=canvas.getBoundingClientRect(),x=Math.max(0,Math.min(w-1,Math.floor((e.clientX-b.left)*w/b.width))),y=Math.max(0,Math.min(h-1,Math.floor((e.clientY-b.top)*h/b.height)));$('sx0').value=x;$('sy0').value=y;$('sx1').value=x+1;$('sy1').value=y+1;$('sourceTrace').checked=true;const q=d.source_diagnostics;$('fieldPixel').textContent+=q?` · P ${number(q.P[y][x])} · N ${number(q.N[y][x])} · net ${number(q.signed[y][x])} · cancellation ${number(q.cancellation[y][x])}`:''};
 canvas.onpointermove=e=>{const r=canvas.getBoundingClientRect(),x=Math.min(w-1,Math.max(0,Math.floor((e.clientX-r.left)*w/r.width))),y=Math.min(h-1,Math.max(0,Math.floor((e.clientY-r.top)*h/r.height)));$('fieldPixel').textContent=`Pixel (x=${x}, y=${y}) · structural ${fields.structural?.[y]?.[x]?.toExponential(6)??'undefined'} · conditional ${fields.conditional?.[y]?.[x]?.toExponential(6)??'undefined'}`};
 }
 const s=d.source_summary;$('sourceSummary').textContent=s?.status==='available'?`Effective source pixels ${number(s.effective_source_pixels)}; source entropy ${number(s.source_entropy)}. `+s.contours.map(c=>`${Math.round(c.target*100)}% mass set covers ${(100*c.fraction).toFixed(1)}% of pixels`).join(' · '):'Recompute for source diagnostics';
 const diag=d.source_diagnostics;Charts.render('sourceDiagnostics',diag?[{z:diag.signed,type:'heatmap',colorscale:'RdBu',zmid:0}]:[],{title:'Signed source contribution · separate from normalized Field A',yaxis:{autorange:'reversed',scaleanchor:'x'}});
 const r=d.source_region;Charts.render('sourceRoutes',r?r.layers.map(l=>({x:l.absolute_mass.map((_,i)=>i),y:l.absolute_mass,mode:'lines',name:'Layer '+l.layer})):[],{title:r?`Selected region downstream mass · source mass ${number(r.source_mass)}`:'Select a source rectangle and compute',xaxis:{title:'Native destination coordinate index'},yaxis:{title:'Absolute probability mass'}});
 $('fieldLegend').textContent=`${$('fieldMapping').value} colour mapping (raw masses unchanged): dark = 0; gold = ${max.toExponential(4)} probability per pixel. Structural sum ${fields.structural?fields.structural.flat().reduce((a,b)=>a+b,0).toFixed(9):'undefined'}; conditional sum ${fields.conditional?fields.conditional.flat().reduce((a,b)=>a+b,0).toFixed(9):'undefined'}.`;
 }
};
document.getElementById('fieldScale').onchange=()=>{if(InputFields.data)InputFields.render(InputFields.data)};

for(const id of ['fieldMapping','fieldContours'])$(id).onchange=()=>{if(InputFields.data)InputFields.render(InputFields.data)};
$('sourceCompute').onclick=()=>{$('sourceTrace').checked=true;compute()};
