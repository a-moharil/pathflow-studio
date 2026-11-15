"""Adapters retain the scope of native, healthcare and residual-DAG reports."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .. import storage
from .statistics import finite
from .geometry import window_audit


def measures(raw):
    out={k:raw.get(k) if finite(raw.get(k)) else None for k in ['H1','Sdef','eta1','eta2','log_eta1','log_eta2','log_path_count']}
    out['defined']=raw.get('defined',raw.get('status')=='available')
    for q in [1,2]:
        if out[f'log_eta{q}'] is None and out[f'eta{q}'] is not None and out[f'eta{q}']>0:
            out[f'log_eta{q}']=float(np.log(out[f'eta{q}']))
    out['normalised_deficit']=out['Sdef']/out['log_path_count'] if out['Sdef'] is not None and out['log_path_count'] and out['log_path_count']>0 else None
    return out


def distribution(name,values,unit,scope):
    a=np.asarray(values,float).ravel(); a=a[np.isfinite(a)]
    if not len(a): return None
    # Exact binned counts and sorted ECDF knots; never label marginals as paths.
    counts,bins=np.histogram(a,bins=min(40,max(1,int(np.sqrt(len(a))))))
    ordered=np.sort(a); ids=np.unique(np.linspace(0,len(a)-1,min(600,len(a)),dtype=int))
    return dict(name=name,unit=unit,scope=scope,n=len(a),zeros=int(np.sum(a==0)),total=float(a.sum()),
                median=float(np.median(a)),mean=float(a.mean()),
                cv=float(a.std()/a.mean()) if a.mean()>0 else None,
                histogram=dict(counts=counts.tolist(),edges=bins.tolist()),
                ecdf=dict(x=ordered[ids].tolist(),y=((ids+1)/len(a)).tolist(),display='Up to 600 exact order-statistic knots; histogram uses all finite observations'),
                mass_curve=dict(fraction=((ids+1)/len(a)).tolist(),mass=(np.cumsum(ordered[::-1])[ids]/a.sum()).tolist()) if a.sum()>0 and a.min()>=0 else None)


def load_record(root,jid):
    with storage.db(root) as db:
        row=db.execute("SELECT * FROM jobs WHERE id=? AND status='completed'",(jid,)).fetchone()
        if not row: raise KeyError('Completed analysis not found')
        run=db.execute('SELECT config FROM runs WHERE id=?',(row['run'],)).fetchone()
        if run is None: run=db.execute('SELECT config FROM joint_runs WHERE id=?',(row['run'],)).fetchone()
    path=Path(row['result']).resolve()
    if not path.is_relative_to(Path(root).resolve()): raise ValueError('Analysis artefact is outside this workspace.')
    if not path.is_file(): raise ValueError('Saved analysis file is missing. Recompute this investigation.')
    blob=path.read_bytes()
    return dict(row),json.loads(blob),json.loads(run[0]) if run else {},hashlib.sha256(blob).hexdigest()


def variants(data):
    if 'heads' in data:
        result=[('baseline',head) for head in data['heads']]
        result.extend(('intervention',head) for head in (data.get('intervention') or {}).get('heads',{}))
        return result
    if 'case' in data and isinstance(data['case'],dict): return [('case','native')]
    if 'structural' in data and 'conditional' in data:return [('analysis','native')]
    return []


def catalogue(root,rid=None):
    with storage.db(root) as db:
        query="SELECT id,run,payload,created FROM jobs WHERE status='completed'"
        args=()
        if rid:query+=' AND run=?';args=(rid,)
        rows=db.execute(query+' ORDER BY created DESC',args).fetchall()
    records=[]; unsupported=0; missing=0
    for row in rows:
        p=json.loads(row['payload']); kind=p.get('kind','')
        if kind not in ['analysis','dose','healthcare_case','healthcare_dose','joint_analysis']: unsupported+=1;continue
        try:
            job,data,config,digest=load_record(root,row['id'])
            for stage,head in variants(data):
                records.append(dict(id=f"{row['id']}:{stage}:{head}",job=row['id'],run=row['run'],stage=stage,head=head,
                                    name=config.get('name',row['run']),checkpoint=data.get('step',data.get('checkpoint',data.get('context',{}).get('checkpoint'))),
                                    kind=kind,created=row['created'],family=config.get('model',config.get('architecture','unknown'))))
        except (OSError,ValueError,KeyError): missing+=1
    return dict(records=records,unsupported=unsupported,missing=missing,
                note='Only completed case, geometry and intervention artefacts are listed. Cohorts and pair ensembles are not independent model replicates.')


def report(root,record_id,include_graph=False):
    parts=record_id.split(':')
    if len(parts)!=3:raise ValueError('Select a recorded analysis.')
    jid,stage,head=parts; job,data,config,digest=load_record(root,jid)
    if (stage,head) not in variants(data):raise ValueError('Selected state or output is unavailable in this record.')
    joint='heads' in data
    if joint: payload=data['heads'][head] if stage=='baseline' else data['intervention']['heads'][head]
    else:payload=data['case'] if stage=='case' else data
    s=measures(payload.get('structural_observables',payload.get('structural',{})))
    c=measures(payload.get('path_observables',payload.get('conditional',{})))
    request=json.loads(job['payload']); context=dict(data.get('context',{}))
    context.update(experiment=job['run'],checkpoint=data.get('step',data.get('checkpoint',context.get('checkpoint',request.get('step')))),
                   stage=stage,head=head,readout=payload.get('readout',context.get('readout',request.get('readout'))),
                   x=request.get('x',request.get('index',context.get('x'))),reference_type=request.get('reference_type','dataset'),
                   reference=request.get('reference',context.get('reference')),transform=request.get('transform',context.get('transform',{})),
                   dataset=config.get('dataset'),family=config.get('model',config.get('architecture')),seed=config.get('seed'))
    metric=payload.get('metric',payload.get('geometry',{}).get('shared',{}).get('metric',{}))
    checks=[]
    def check(name,value,tolerance,role):
        if finite(value): checks.append(dict(name=name,value=float(value),tolerance=tolerance,status='passed' if abs(value)<=tolerance else 'failed',role=role))
    delta=payload.get('delta');tol=1e-7*max(1.,abs(delta)) if finite(delta) else 1e-7
    check('Signed output reconstruction',payload.get('reconstruction_residual',payload.get('residual')),tol,'Scale-aware display audit: 1e-7 × max(1, |Δf|); not a statistical test')
    check('Signed source reconstruction',payload.get('signed_source_residual',payload.get('signed_field_residual')),tol,'Independent signed-source sum where recorded')
    for side,raw in [('structural',payload.get('structural',{})),('conditional',payload.get('conditional',{}))]:
        cuts=raw.get('cut_sums')
        if cuts and raw.get('defined'):check(side+' maximum cut residual',max(abs(v-1) for v in cuts),1e-9,'Actual layered complete-carrier cuts')
    for side,audit in payload.get('audit',{}).items():
        if isinstance(audit,dict):check(side+' DAG conservation residual',audit.get('max_error'),audit.get('tolerance',1e-9),'Node conservation and actual source/sink cuts; operation positions need not be cuts')
    for side,m in [('structural',s),('conditional',c)]:
        if finite(m['Sdef']) and finite(m['log_eta1']):check(side+' entropy identity residual',m['Sdef']+m['log_eta1'],1e-9,'Sdeficit = −log eta1')
    fidelity=dict(status='failed' if any(v['status']=='failed' for v in checks) else 'passed_recorded_checks' if checks else 'not_recorded',checks=checks,
                  scope='Only these numerical checks were evaluated. Passing them is not proof of every manuscript theorem or estimator assumption.')
    distributions=[]; fields=payload.get('input_fields',{})
    for side,value in [('structural',payload.get('structural_FieldA',fields.get('structural'))),('conditional',payload.get('FieldA',fields.get('conditional')))]:
        if value is not None:
            d=distribution(side+' input-origin mass',value,'input pixel','Input-origin marginal, summed over input channels; not the complete-path distribution')
            if d:distributions.append(d)
    layers=[]
    for row in payload.get('layer_marginals',[]):
        layers.append(dict(name='Layer '+str(row['layer']),structural=row.get('structural_masses'),conditional=row.get('conditional_masses'),unit=row.get('kind','unit')))
    if not layers:
        for row in payload.get('layer_fields') or []:
            value=row.get('unit_absolute')
            if value is None:value=row.get('native_absolute')
            if value is not None:layers.append(dict(name=row.get('name','Layer '+str(row.get('layer'))),structural=None,conditional=np.asarray(value).ravel().tolist(),unit='unit' if row.get('unit_absolute') is not None else 'feature position (channels summed)'))
    graph={}; legacy={}
    if joint:
        geo=payload.get('geometry',{}); shared=geo.get('shared',{})
        graph=dict(nodes=shared.get('nodes',[]),edges=shared.get('edges',[]),metric=shared.get('metric',{}),
                   scope='full_scalar' if geo.get('scalar_audit',{}).get('status')=='available' else 'scoped_quotient',defined=dict(structural=s['defined'],conditional=c['defined']))
        legacy=geo.get('responses',{})
    else:
        graph=dict(nodes=payload.get('nodes',[]),edges=payload.get('edges',[]),metric=metric,scope='full_scalar',defined=dict(structural=s['defined'],conditional=c['defined']))
        legacy={side:payload.get(side+'_geometry',{}).get('responses',{}) for side in ['structural','conditional']}
    edges=graph['edges']; nodes=graph['nodes']
    if edges:
        groups={}
        for e in edges:
            key=str(e.get('comparison_class',nodes[e['u']].get('layer','unknown')))
            groups.setdefault(key,[]).append(e)
        for key,group in groups.items():
            for side,short in [('structural','s'),('conditional','c')]:
                if not graph['defined'][side]:continue
                values=[e.get('enrichment_'+short,len(group)*e[short]) for e in group]
                d=distribution(f'{side} edge enrichment · {key}',values,'edge occurrence' if graph['scope']=='full_scalar' else 'quotient/subgraph edge',f"Scope: {graph['scope']}; comparison class {key}; declared metric enrichment")
                if d:distributions.append(d)
    geometry={}
    for side,g in legacy.items():
        if not g: continue
        profile=g.get('profile',[])
        geometry[side]=dict(status=g.get('status'),profile=profile,xi50=g.get('xi50'),legacy_dstar=g.get('dstar'),
                            scaling=window_audit([p['radius'] for p in profile],[p['boxes'] for p in profile]) if len(profile)>=6 else None,
                            note='Saved legacy bcan-1 profile; typically 36 scales. Applying the seven criteria here does not reproduce the manuscript 64-scale estimator.')
    signature=dict(checkpoint=context['checkpoint'],dataset=context['dataset'],family=context['family'],readout=context['readout'],reference_type=context['reference_type'],
                   transform=context['transform'],stage=stage,gates=payload.get('gates',request.get('gates',{})),
                   conventions=payload.get('conventions',{}),scope=payload.get('scope'),metric=metric and {k:metric.get(k) for k in ['epsilon','power','enrichment','version']})
    if stage=='intervention':signature['intervention']={k:request.get(k) for k in ['layer','channel','alpha']}
    out=dict(version='ecml-audit-1',record_id=record_id,run_id=job['run'],name=config.get('name',job['run']),context=context,
             structural=s,conditional=c,fidelity=fidelity,distributions=distributions,layers=layers,geometry=geometry,
             geometry_available=bool(edges) and graph['scope']=='full_scalar',geometry_nodes=len(nodes),metric=metric,
             parameter_hash=payload.get('provenance',{}).get('parameter_hash'),
             comparison_signature=json.dumps(signature,sort_keys=True),
             provenance=dict(job=jid,sha256=digest,recorded_configuration=config,analysis_version=payload.get('analysis_version',data.get('version')),engine_provenance=payload.get('provenance',{})),
             response=dict(delta=delta,cancellation=payload.get('global_cancellation',payload.get('cancellation')) if not isinstance(payload.get('cancellation'),list) else payload.get('global_cancellation'),
                           path_hellinger=payload.get('path_hellinger')),
             limitations=['Complete-path histograms are not recorded; marginal histograms must not be read as path histograms.',
                          'Parameter-identity/reuse entropy decomposition and local mass-growth fits are not recorded by these adapters; no values are inferred from channel sums.',
                          'All population selections are exploratory. Paper results are contextual evidence, never substituted for this experiment.'])
    if joint and data.get('intervention'):
        original=data['heads'][head].get('delta'); changed=data['intervention']['heads'].get(head,{}).get('delta')
        out['intervention']=dict(dose_parameter='Retained contribution multiplier g (1 = baseline, 0 = removed); manuscript attenuation is 1 − g',baseline_delta=original,intervened_delta=changed,absolute_response_displacement=abs(changed-original) if finite(original) and finite(changed) else None,semantics=data['intervention'].get('semantics'),dose=data['intervention'].get('dose_response'))
    elif data.get('dose'):
        out['intervention']=dict(dose_parameter='Retained contribution multiplier g (1 = baseline, 0 = removed); manuscript attenuation is 1 − g',dose=data['dose'],support=data.get('intervention'),semantics='Saved healthcare intervention; inspect original request for operator and controls',controls=data.get('controls',[]))
    if include_graph:out['_graph']=graph
    return out
