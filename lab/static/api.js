const $=id=>document.getElementById(id);
async function api(url,data){let r=await fetch(url,data===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json','X-CSRF-Token':document.querySelector('meta[name=csrf]').content},body:JSON.stringify(data)});let d=await r.json().catch(()=>({error:r.statusText}));if(!r.ok)throw Error(d.error||r.statusText);return d}
function showError(e){$('error').textContent=e.message||String(e);$('error').hidden=false}
function clearError(){$('error').hidden=true}
function saveJSON(obj,name){saveText(JSON.stringify(obj,null,2),name,'application/json')}
function saveText(text,name,mime='text/plain'){let u=URL.createObjectURL(new Blob([text],{type:mime})),a=document.createElement('a');a.href=u;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)}
async function awaitJob(id,onProgress=()=>{}){for(;;){let j=await api('/api/job/'+id);onProgress(j.progress||j.status);if(j.status==='completed')return j.data;if(['failed','cancelled'].includes(j.status))throw Error(j.error);await new Promise(r=>setTimeout(r,700))}}
function number(v){return Number.isFinite(v)?(Math.abs(v)>0&&Math.abs(v)<.001?v.toExponential(3):v.toPrecision(5)):'undefined'}

async function deleteExperiment(id){try{const p=await api('/api/run/'+id+'/delete-preview');if(!confirm(`Delete “${p.name}”?\n${p.files} files, ${(p.bytes/1048576).toFixed(1)} MiB plus saved measurements and jobs will be removed.\nActive work will finish its current operation before deletion. Shared datasets are retained.`))return;await api('/api/run/'+id+'/delete',{});location.href='/'}catch(e){showError(e)}}
