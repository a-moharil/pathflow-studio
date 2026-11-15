"""Design-explicit statistics. No checkpoints, edges or pixels become model replicates."""
import math
import numpy as np
from scipy import stats

METRICS = {
    'log_eta1': ('Log effective occupancy', 'log fraction'),
    'log_eta2': ('Log collision occupancy', 'log fraction'),
    'normalised_deficit': ('Normalised entropy deficit', 'fraction'),
}

def finite(value):
    return isinstance(value, (int, float, np.number)) and not isinstance(value, bool) and math.isfinite(value)

def summarise(values):
    a = np.asarray([v for v in values if finite(v)], dtype=float)
    if not len(a):
        return dict(n=0, median=None, mean=None, q25=None, q75=None)
    return dict(n=len(a), median=float(np.median(a)), mean=float(a.mean()),
                q25=float(np.quantile(a, .25)), q75=float(np.quantile(a, .75)),
                minimum=float(a.min()), maximum=float(a.max()))

def holm(pvalues):
    order = np.argsort(pvalues); adjusted = np.zeros(len(pvalues)); previous = 0.
    for rank, index in enumerate(order):
        previous = max(previous, min(1., (len(pvalues)-rank)*pvalues[index]))
        adjusted[index] = previous
    return adjusted.tolist()

def population(reports, independent=False):
    if len(reports) > 100: raise ValueError('Select at most 100 independent runs.')
    runs = [r['run_id'] for r in reports]
    if len(runs) != len(set(runs)):
        raise ValueError('Select only one recorded comparison per run. Checkpoints and conditions within a run are repeated observations.')
    signatures = {r['comparison_signature'] for r in reports}
    if len(signatures) > 1:
        raise ValueError('Selected records have different checkpoints, datasets, families, readouts, reference policies, transformations, gates or measure conventions. Select a compatible stratum.')
    hashes = [r.get('parameter_hash') for r in reports if r.get('parameter_hash')]
    if len(hashes) != len(set(hashes)):
        raise ValueError('Repeated parameter hashes identify duplicated fitted models, not independent runs.')
    results = []
    for metric, (label, unit) in METRICS.items():
        pairs=[]; excluded=[]
        for r in reports:
            a=r['structural'].get(metric); b=r['conditional'].get(metric)
            if finite(a) and finite(b) and r['fidelity']['status'] != 'failed':
                pairs.append(dict(run=r['run_id'], record=r['record_id'], checkpoint=r['context'].get('checkpoint'), a=a, b=b, difference=b-a))
            else: excluded.append(dict(run=r['run_id'], reason='Undefined observable or failed numerical audit'))
        d=np.asarray([p['difference'] for p in pairs]); n=len(d)
        item=dict(metric=metric,label=label,unit=unit,estimand='Median within-run conditional minus structural difference',
                  pairs=pairs,excluded=excluded,summary=summarise(d),positive=int(np.sum(d>0)),negative=int(np.sum(d<0)),zero=int(np.sum(d==0)),
                  inference_status='descriptive',p=None,p_adjusted=None,interval=None)
        if n >= 2:
            theoretical, ordered = stats.probplot(d, dist='norm', fit=False)
            item['normal_qq'] = dict(theoretical=theoretical.tolist(), observed=ordered.tolist(), note='Visual distribution diagnostic; normality is not required by Wilcoxon and this does not certify symmetry.')
        # eta1 and entropy deficit encode the same ordering. Keep deficit descriptive;
        # the declared Holm family contains log eta1 and log eta2 only.
        if independent and n >= 6 and metric != 'normalised_deficit':
            rng=np.random.default_rng(2101)
            medians=np.median(rng.choice(d,size=(4000,n),replace=True),axis=1)
            nonzero=d[d!=0]
            if len(nonzero):
                test=stats.wilcoxon(d,zero_method='wilcox',alternative='two-sided',method='auto')
                ranks=stats.rankdata(abs(nonzero)); effect=float(np.sum(np.sign(nonzero)*ranks)/ranks.sum())
                statistic=float(test.statistic); p=float(test.pvalue)
            else: statistic=0.; p=1.; effect=0.
            item.update(inference_status='exploratory',p=p,statistic=statistic,rank_biserial=effect,
                        interval=[float(v) for v in np.quantile(medians,[.025,.975])],
                        interval_method='95% percentile bootstrap of the median difference; 4000 resamples of whole runs; seed 2101',
                        test='Two-sided Wilcoxon signed-rank; zero_method=wilcox; method=auto',
                        null='Paired differences have a distribution symmetric about zero; this is not a general test of a median alone.')
        elif independent and n < 6: item['inference_status']='insufficient_runs'
        if metric == 'normalised_deficit': item['note']='Descriptive companion to log eta1; not independent corroboration.'
        results.append(item)
    tested=[r for r in results if r['p'] is not None]
    for r,p in zip(tested,holm([r['p'] for r in tested])): r['p_adjusted']=p
    return dict(unit='Independently trained model run, paired across structural and conditional states',
                independent_runs_declared=independent,selection='User-selected exploratory population; no claim of random architecture sampling',
                family='Holm across available tests of log eta1 and log eta2 in this submitted selection. Other selections are not included.',
                assumptions='Independence of fitted runs must be justified by their provenance. Wilcoxon requires symmetric paired differences for a location interpretation. Repeated references do not supply model replication.',
                count=len(reports),results=results,
                software=dict(numpy=np.__version__,scipy=__import__('scipy').__version__))
