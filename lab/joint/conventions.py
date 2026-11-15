"""Versioned analysis definitions. Settings affect measurement, never training."""
import math

VERSION='carrier-metric-15'
DEFAULTS=dict(structural_rule='raw_affine',metric_epsilon=0.03,metric_power=1.0)

def settings(value=None):
    q=dict(DEFAULTS);q.update(value or {})
    if q['structural_rule'] not in ['raw_affine','he_normalized']:raise ValueError('Unknown structural rule')
    for key in ['metric_epsilon','metric_power']:
        q[key]=float(q[key])
        if not math.isfinite(q[key]):raise ValueError('Non-finite metric setting')
    if not 1e-8<=q['metric_epsilon']<=1:raise ValueError('Metric epsilon must be between 1e-8 and 1')
    if not 0.1<=q['metric_power']<=4:raise ValueError('Metric power must be between 0.1 and 4')
    return {k:q[k] for k in DEFAULTS}

def structural_mode(q):return 'structural_he' if q['structural_rule']=='he_normalized' else 'structural'

def description(q):
    return dict(version=VERSION,**q,
        structural_definition='Unit source masses; product of squared carrier coefficients. Activations and intervention gates use coefficient 1. Fixed affine, pooling, merge and readout coefficients retained.',
        learned_coefficient='w/sqrt(2/fan_in)' if q['structural_rule']=='he_normalized' else 'w',
        fan_in='Linear input width or kernel area times input channels per group; fixed architectural fan-in, including padded kernel slots',
        maxpool_structural='Uniform amplitude 1/k over k valid candidates, then squared. Declared input-independent extension, not a finite max-pool response.',
        comparison_class='All scalar occurrences feeding one operation instance, across all its input branches',
        enrichment='|K| F_e / sum(F_a for a in K)',
        zero_class='If class mass is zero, enrichment is undefined; explicit zero-extension is used ONLY for regularized metric length epsilon^(-power)',
        metric='(enrichment + epsilon)^(-power); undirected shortest paths; parallel lengths combined by minimum',
        evidence='MLP manuscript equations 12-26 and section 26.4; supplied residual guide for finite conditional response. He normalization and fixed max-pool structural coefficients are explicitly declared DAG extensions.',
        normalization_note='He scaling cancels from normalized measures when every complete path traverses the same weighted stages. It generally changes residual route probabilities when routes have unequal weighted depths.')
