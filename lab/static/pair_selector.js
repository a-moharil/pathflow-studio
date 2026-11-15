/* Sample identity comes from the experiment's saved dataset split, never predictions. */
class PairSelector {
 static candidates(labels,x,relation,regression=false){
  if(!Number.isInteger(x)||x<0||x>=labels.length)throw Error('Choose an available input sample.');
  const ids=labels.map((_,i)=>i).filter(i=>i!==x);
  if(relation==='any')return ids;
  if(regression){ids.sort((a,b)=>Math.abs(labels[a]-labels[x])-Math.abs(labels[b]-labels[x])||a-b);const n=Math.ceil(ids.length/2);return relation==='same'?ids.slice(0,n):ids.slice(-n).reverse()}
  return ids.filter(i=>(labels[i]===labels[x])===(relation==='same'));
 }
 static next(labels,x,current,relation,regression=false){const ids=this.candidates(labels,x,relation,regression);if(!ids.length)throw Error('No distinct reference satisfies this relation in the available split. Pair unchanged.');if(ids.length===1&&ids[0]===current)throw Error('Only one eligible reference exists and it is already selected. Pair unchanged.');const k=ids.indexOf(current);return ids[(k+1)%ids.length]}
 constructor({x,reference,type,relation=null,labels,sourceIds=[],names=[],regression=false,onCompute,isBusy=()=>false}){
  Object.assign(this,{x,reference,type,labels,sourceIds,names,regression,onCompute,isBusy});this.running=false;
  for(const el of [x,reference]){const old=el.value;el.replaceChildren(...labels.map((v,i)=>new Option(this.label(i),i)));el.value=labels[+old]!==undefined?old:'0';el.addEventListener('change',()=>this.describe())}
  this.host=document.createElement('div');this.host.className='pairSelector';reference.closest('label').parentElement.after(this.host);
  this.host.innerHTML='<div class="toolbar"><span data-relation-host></span><button type="button" data-same></button><button type="button" data-cross></button><button type="button" data-swap>Swap x and x°</button></div><p data-pair-status role="status"></p><p class="help">Selection uses ground-truth labels from this saved split. Relation buttons hold x fixed and cycle distinct references. Manual dropdown selection is unrestricted. Changed controls require analysis; maps retain their computed context until it finishes.</p>';
  this.el=k=>this.host.querySelector('[data-'+k+']');
  this.relation=relation;if(!relation){const l=document.createElement('label');l.textContent='Reference relation ';this.relation=document.createElement('select');this.relation.append(new Option('Any distinct sample','any'),new Option('Same class','same'),new Option('Different class','cross'));l.append(this.relation);this.el('relation-host').append(l)}
  for(const o of this.relation.options){if(o.value==='same')o.textContent=regression?'Nearer target half':'Same true class';if(o.value==='cross')o.textContent=regression?'Farther target half':'Different true class'}
  if(regression)this.host.querySelector('.help').textContent+=' Nearer/farther pools split the distinct examples ranked by absolute target difference; ties use sample index, and odd pool sizes share the middle example.';
  this.el('same').textContent=regression?'Next nearer-target reference':'Next same-class reference';this.el('cross').textContent=regression?'Next farther-target reference':'Next different-class reference';
  this.el('same').onclick=()=>this.advance('same').catch(e=>this.message(e.message));this.el('cross').onclick=()=>this.advance('cross').catch(e=>this.message(e.message));
  this.relation.addEventListener('change',()=>this.advance().catch(e=>this.message(e.message)));
  this.el('swap').onclick=()=>{if(this.running||this.isBusy()){this.message('Wait for the current analysis before swapping.');return}if(this.type.value==='zero'){this.message('A zero reference has no dataset index to swap.');return}const old=x.value;x.value=reference.value;reference.value=old;this.changed();this.describe()};
  type.addEventListener('change',()=>this.describe());this.describe();
 }
 label(i){const v=this.labels[i],name=this.names[v];return `Index ${i} · ${this.regression?'target '+Number(v).toPrecision(6):'true label '+v+(name?' ('+name+')':'')}${this.sourceIds[i]!==undefined?' · source '+this.sourceIds[i]:''}`}
 message(s){this.el('pair-status').textContent=s}
 changed(){for(const el of [this.x,this.reference,this.type])el.dispatchEvent(new Event('change'))}
 describe(){this.reference.disabled=this.type.value==='zero';const x=+this.x.value,r=+this.reference.value;const relation=this.type.value==='zero'?'zero reference has no true class':this.regression?'absolute target difference '+Math.abs(this.labels[x]-this.labels[r]).toPrecision(5):x===r?'identical sample':this.labels[x]===this.labels[r]?'same true class':'different true classes';this.message(`Selected x: ${this.label(x)}. x°: ${this.type.value==='zero'?'zero input':this.label(r)}. ${relation}.`)}
 async advance(relation=null){if(this.running||this.isBusy()){this.message('Wait for the current analysis before selecting another reference.');return}const desired=relation||this.relation.value;const next=PairSelector.next(this.labels,+this.x.value,+this.reference.value,desired,this.regression);this.relation.value=desired;this.reference.value=String(next);this.type.value='dataset';this.changed();this.describe();this.running=true;try{await this.onCompute()}finally{this.running=false}}
}
if(typeof module!=='undefined')module.exports=PairSelector;
