"""Design-explicit statistics for Pathflow Studio.

Every result carries its observational unit, its sample size and an evidential
status. The statuses follow the project's distinction between kinds of claim:

``identity``      an exact numerical relation of the recorded quantities
``descriptive``   a summary of recorded values; no sampling model is assumed
``exploratory``   a test or interval under a declared sampling design, not
                  preregistered and not corrected across interactive selections
``insufficient``  the design does not support the requested quantity
``unavailable``   the quantity was not recorded for this selection

Checkpoints of one run are repeated observations of one fitted trajectory. They are
summarised descriptively and never treated as replicates; no p-values are attached
to temporal statistics. Images in a healthcare cohort are observations conditional
on one fitted model. Independently trained runs are the unit for cross-run claims.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict

import numpy as np
from scipy import stats as sps

from . import index as ix

# Observables carried into temporal summaries, in display order.
LABELS = {
    'log_eta1': 'log η₁ · effective path occupancy',
    'log_eta2': 'log η₂ · collision occupancy',
    'normalised_deficit': 'Normalised entropy deficit',
    'Sdef': 'Entropy deficit S_def',
    'H1': 'Path entropy H₁ (nats)',
    'logZ': 'log Q · log partition',
    'eta1': 'η₁ · effective path occupancy',
    'eta2': 'η₂ · collision occupancy',
    'dstar': 'd* · maximal qualified exponent',
    'deff': 'd_eff at ξ₅₀',
    'xi50': 'ξ₅₀ · half-cover radius',
    'qualified_fraction': 'Qualified scaling-window fraction',
    'delta': 'Δf · output contrast',
    'cancellation': 'Cancellation',
    'path_hellinger': 'Complete-path Hellinger',
    'contrast_l2': '‖Δf‖₂ · vector contrast',
    'val_loss': 'Validation loss',
    'val_accuracy': 'Validation accuracy',
    'train_loss': 'Training loss',
    'test_accuracy': 'Test accuracy',
    'test_loss': 'Test loss',
    'train_accuracy': 'Training accuracy',
    'val_dice': 'Validation Dice',
}
PATH_NAMES = ['log_eta1', 'log_eta2', 'normalised_deficit', 'H1', 'logZ']
GEOMETRY_NAMES = ['dstar', 'xi50', 'deff', 'qualified_fraction']
RESPONSE_NAMES = ['delta', 'cancellation', 'contrast_l2']
COHORT_NAMES = ['path_hellinger', 'eta1', 'eta2', 'H1', 'Sdef', 'logZ', 'delta', 'cancellation', 'H_eq', 'H_inv']
SEED = 701


def label(name: str) -> str:
    return LABELS.get(name, name)


def finite(v) -> bool:
    return ix.finite(v)


def describe(values) -> dict:
    a = np.asarray([v for v in values if finite(v)], float)
    if not len(a):
        return dict(n=0, median=None, q25=None, q75=None, minimum=None, maximum=None, mean=None)
    return dict(n=int(len(a)), median=float(np.median(a)), q25=float(np.quantile(a, .25)),
                q75=float(np.quantile(a, .75)), minimum=float(a.min()), maximum=float(a.max()),
                mean=float(a.mean()))


def holm(pvalues: list[float]) -> list[float]:
    m = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def spearman(x, y) -> float | None:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 4 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(sps.spearmanr(x, y).statistic)


def _clean(value):
    """JSON-safe floats (NaN and infinities become None)."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


# --------------------------------------------------------------- within-run (temporal)

def _per_step(points) -> tuple[list[int], list[float], int]:
    """Median per checkpoint. Returns steps, values and the largest multiplicity."""
    grouped = defaultdict(list)
    for step, value, _ in points:
        if step is not None and finite(value):
            grouped[int(step)].append(value)
    steps = sorted(grouped)
    multiplicity = max((len(v) for v in grouped.values()), default=0)
    return steps, [float(np.median(grouped[s])) for s in steps], multiplicity


def trajectory_summary(points) -> dict:
    steps, values, multiplicity = _per_step(points)
    out = dict(steps=steps, values=values, n=len(steps), multiplicity=multiplicity,
               status='descriptive' if steps else 'unavailable', summary=describe(values))
    if not steps:
        return out
    first, last = values[0], values[-1]
    out.update(first_step=steps[0], last_step=steps[-1], first=first, last=last, change=last - first)
    if len(values) >= 3:
        # An interior extremum counts as a turning point only when it lies beyond both
        # endpoints by at least 10% of the trajectory's range (plateau noise is ignored).
        lo, hi = int(np.argmin(values)), int(np.argmax(values))
        span = max(values) - min(values)
        points = []
        for i, kind in ((lo, 'minimum'), (hi, 'maximum')):
            if not 0 < i < len(values) - 1 or span <= 0:
                continue
            margin = min(values[0], values[-1]) - values[i] if kind == 'minimum' else values[i] - max(values[0], values[-1])
            if margin >= .1 * span:
                points.append(dict(step=steps[i], value=values[i], kind=kind, margin=margin / span))
        out['turning_points'] = points
    out['trend_rho'] = spearman(steps, values)
    return out


def realisation_gap(series: dict) -> list[dict]:
    """Conditional minus structural, matched at each checkpoint of one context."""
    results = []
    for name in ['log_eta1', 'log_eta2', 'normalised_deficit']:
        s_steps, s_vals, _ = _per_step(series.get(('structural', name), []))
        c_steps, c_vals, _ = _per_step(series.get(('conditional', name), []))
        s_map, c_map = dict(zip(s_steps, s_vals)), dict(zip(c_steps, c_vals))
        common = sorted(set(s_map) & set(c_map))
        diff = [c_map[k] - s_map[k] for k in common]
        item = dict(name=name, label=label(name), steps=common, structural=[s_map[k] for k in common],
                    conditional=[c_map[k] for k in common], difference=diff, n=len(common),
                    status='descriptive' if common else 'unavailable', summary=describe(diff),
                    positive=int(sum(d > 0 for d in diff)), negative=int(sum(d < 0 for d in diff)),
                    unit='checkpoints of one run (repeated observations; no test is attached)')
        if name.startswith('log_eta') and diff:
            ratios = np.exp(diff)
            item['retention'] = describe(ratios.tolist())
            item['retention_note'] = (f'η_cond/η_struct; the conditional measure retains a median '
                                      f'{100 * float(np.median(ratios)):.3g}% of the structural '
                                      f'{"effective" if name == "log_eta1" else "collision"} occupancy.')
        if name == 'normalised_deficit':
            item['note'] = ('Descriptive companion to log η₁: S_def = −log η₁ exactly, so this is a rescaled '
                            'restatement, not independent corroboration. The H₁ gap equals the log η₁ gap.')
            item['status'] = 'descriptive' if common else 'unavailable'
        results.append(item)
    return results


def coupling(series: dict, keys: list[tuple[str, str]]) -> dict:
    """Spearman ρ between observable trajectories across shared checkpoints."""
    per = {}
    for key in keys:
        steps, values, _ = _per_step(series.get(key, []))
        if steps:
            per[key] = dict(zip(steps, values))
    keys = [k for k in keys if k in per]
    matrix, counts = [], []
    for a in keys:
        row, nrow = [], []
        for b in keys:
            common = sorted(set(per[a]) & set(per[b]))
            row.append(spearman([per[a][s] for s in common], [per[b][s] for s in common]) if a != b else 1.0)
            nrow.append(len(common))
        matrix.append(row)
        counts.append(nrow)
    return dict(keys=[dict(side=s, name=n, label=f'{side_label(s)} · {label(n)}',
                           short=f"{side_label(s)} {label(n).split(' · ')[0]}") for s, n in keys],
                rho=matrix, n=counts, status='descriptive',
                note='Spearman ρ across the checkpoints both trajectories share. Checkpoints are serially '
                     'dependent observations of one run, so no p-value is attached.')


def side_label(side: str) -> str:
    return dict(structural='Structural', conditional='Conditional', response='Response',
                geometry_structural='Structural geometry', geometry_conditional='Conditional geometry',
                training='Training').get(side, side)


def run_statistics(root, run: str, context: str) -> dict:
    series = ix.series(root, run, context)
    blocks = defaultdict(list)
    for (side, name), points in sorted(series.items()):
        group = ('training' if side == 'training' else 'geometry' if side.startswith('geometry')
                 else 'response' if side == 'response' else 'path')
        if group == 'path' and name not in PATH_NAMES:
            continue
        if group == 'training' and not (name.endswith('loss') or name.endswith('accuracy') or name.startswith('val')):
            continue
        summary = trajectory_summary(points)
        summary.update(side=side, name=name, label=label(name), side_label=side_label(side))
        blocks[group].append(summary)
    # One representative per identity class: log η₁ (≡ −S_def, ≡ H₁ − log N) and log η₂.
    coupling_keys = [(s, n) for s in ('structural', 'conditional') for n in ('log_eta1', 'log_eta2')]
    coupling_keys += [('geometry_structural', 'dstar'), ('geometry_conditional', 'dstar'),
                      ('geometry_structural', 'xi50'), ('geometry_conditional', 'xi50'),
                      ('response', 'cancellation'), ('training', 'val_loss'), ('training', 'val_accuracy')]
    return _clean(dict(
        run=run, context=json.loads(context) if context else None, context_key=context,
        unit='Committed checkpoints of one fitted trajectory',
        trajectories=dict(blocks), realisation=realisation_gap(series),
        coupling=coupling(series, coupling_keys)))


# --------------------------------------------------------------- healthcare cohort

def auroc(positive, negative) -> tuple[float, float]:
    """AUROC = P(positive > negative) + ½P(tie), with its Mann–Whitney U."""
    u = sps.mannwhitneyu(positive, negative, alternative='two-sided').statistic
    return float(u / (len(positive) * len(negative))), float(u)


def bootstrap_auroc(positive, negative, replicates=1000) -> list[float]:
    rng = np.random.default_rng(SEED)
    positive, negative = np.asarray(positive), np.asarray(negative)
    values = []
    for _ in range(replicates):
        p = positive[rng.integers(len(positive), size=len(positive))]
        n = negative[rng.integers(len(negative), size=len(negative))]
        gt = (p[:, None] > n[None, :]).mean() + .5 * (p[:, None] == n[None, :]).mean()
        values.append(gt)
    return np.quantile(values, [.025, .975]).tolist()


def _comparison(rows, name, a_rows, b_rows, a_label, b_label, minimum=5) -> dict:
    a = [r[name] for r in a_rows if finite(r.get(name))]
    b = [r[name] for r in b_rows if finite(r.get(name))]
    item = dict(observable=name, label=label(name), groups=[a_label, b_label], n=[len(a), len(b)],
                summaries=[describe(a), describe(b)])
    if len(a) < minimum or len(b) < minimum:
        item.update(status='insufficient', reason=f'Requires at least {minimum} defined values per group.')
        return item
    if np.ptp(a + b) == 0:
        item.update(status='descriptive', reason='All values are equal; no ordering to test.')
        return item
    area, u = auroc(a, b)
    test = sps.mannwhitneyu(a, b, alternative='two-sided')
    item.update(status='exploratory', auroc=area, auroc_interval=bootstrap_auroc(a, b),
                rank_biserial=2 * area - 1, U=u, p=float(test.pvalue),
                test='Two-sided Mann–Whitney U; AUROC = U/(n₁n₂) with 1000 image-level bootstrap resamples, seed 701',
                reading=f'AUROC is the probability that a {a_label} image has the larger value than a {b_label} image.')
    return item


def cohort_statistics(result: dict) -> dict:
    rows = [r for r in result.get('rows', []) if r.get('status') == 'available']
    policy = (result.get('sampling') or {}).get('policy') or (result.get('context') or {}).get('cohort_reference')
    transform = (result.get('request') or {}).get('transform') or (result.get('context') or {}).get('transform') or {}
    identity = all(float(transform.get(k, v)) == v for k, v in (('dx', 0), ('dy', 0), ('angle', 0), ('scale', 1)))
    # Transformation observables are floating-point zeros under the identity transform.
    excluded = ['H_eq', 'H_inv'] if identity else []
    names = [n for n in COHORT_NAMES if n not in excluded and any(finite(r.get(n)) for r in rows)]
    groups = {}
    for g in ('TN', 'FP', 'FN', 'TP'):
        subset = [r for r in rows if r.get('group') == g]
        groups[g] = dict(n=len(subset), observables={n: describe([r.get(n) for r in subset]) for n in names},
                         values={n: [r[n] for r in subset if finite(r.get(n))] for n in names})
    omnibus = []
    for n in names:
        samples = [groups[g]['values'][n] for g in groups if len(groups[g]['values'][n]) >= 2]
        item = dict(observable=n, label=label(n), groups=[g for g in groups if len(groups[g]['values'][n]) >= 2])
        if len(samples) < 2:
            item.update(status='insufficient', reason='Fewer than two outcome groups with at least two images.')
        elif np.ptp(np.concatenate(samples)) == 0:
            item.update(status='descriptive', reason='All values are equal.')
        else:
            h = sps.kruskal(*samples)
            total, k = sum(len(s) for s in samples), len(samples)
            item.update(status='exploratory', H=float(h.statistic), p=float(h.pvalue),
                        epsilon_squared=max(0.0, float((h.statistic - k + 1) / (total - k))) if total > k else None,
                        test='Kruskal–Wallis across outcome groups; ε² = (H − k + 1)/(n − k)')
        omnibus.append(item)
    correct = [r for r in rows if r.get('group') in ('TN', 'TP')]
    errors = [r for r in rows if r.get('group') in ('FP', 'FN')]
    positive = [r for r in rows if r.get('label') == 1]
    negative = [r for r in rows if r.get('label') == 0]
    error_family = [_comparison(rows, n, errors, correct, 'misclassified', 'correctly classified') for n in names]
    label_family = [_comparison(rows, n, positive, negative, 'class 1', 'class 0') for n in names]
    for family in (omnibus, error_family, label_family):
        tested = [x for x in family if x.get('p') is not None]
        for x, adj in zip(tested, holm([x['p'] for x in tested])):
            x['p_holm'] = adj
    error_classes = sorted({r.get('label') for r in errors})
    confounding = None
    if errors and len(error_classes) == 1:
        kind = 'false positives' if error_classes[0] == 0 else 'false negatives'
        confounding = (f'All {len(errors)} misclassified images are {kind} (true class {error_classes[0]}), so the '
                       f'error comparison is confounded with class. Read it beside the class comparison; '
                       f'within class {error_classes[0]}, compare {"TN with FP" if error_classes[0] == 0 else "TP with FN"} in the outcome-group view.')
    dependence = ('Zero reference: one input per image.' if policy == 'zero' else
                  'Pair-derived observables: references are drawn from the same split and may repeat, '
                  'so images are not fully independent. Treat intervals and p-values as exploratory.')
    return _clean(dict(
        unit='Images of one split, conditional on one fitted model and checkpoint',
        population=(result.get('sampling') or {}).get('population'), sampled=len(rows), policy=policy,
        dependence=dependence, metrics={k: v for k, v in (result.get('metrics') or {}).items()
                                        if k in ('accuracy', 'sensitivity', 'specificity', 'roc_auc', 'balanced_accuracy', 'brier', 'ece', 'n', 'threshold')},
        groups={g: dict(n=v['n'], observables=v['observables'], values=v['values']) for g, v in groups.items()},
        omnibus=omnibus, errors=error_family, labels=label_family, confounding=confounding,
        excluded_observables=[dict(name=n, reason='Identity transform: defined as zero up to floating-point error') for n in excluded],
        redundancy='Within one architecture, η₁, H₁ and S_def carry the same ordering (S_def = log N − H₁ = −log η₁); '
                   'their tests are one piece of evidence, not three.',
        holm='Holm correction within each family (outcome omnibus, error comparison, class comparison) of this selection only.'))


# ------------------------------------------------------------------------ cross-run

FACTORS = ['optimizer', 'activation', 'lr', 'seed', 'widths', 'channels', 'model', 'batch_size', 'decay', 'momentum', 'train_size']
IDENTITY_KEYS = ['dataset', 'model', 'task']
ARCHITECTURE_KEYS = ['widths', 'channels', 'head_widths', 'cnn_head', 'cnn_readout', 'kernel', 'kernels', 'stride', 'padding', 'bias']
# Hyperparameters that belong to one level of a factor; they cannot match across levels.
COUPLED = {'optimizer': {'lr', 'momentum', 'beta1', 'beta2', 'eps', 'alpha', 'centered', 'amsgrad', 'dampening', 'nesterov'}}


def _run_value(series, side, name, state):
    steps, values, _ = _per_step(series.get((side, name), []))
    if not steps:
        return None, None
    if state == 'initial':
        return steps[0], values[0]
    if state == 'final':
        return steps[-1], values[-1]
    target = int(state)
    if target in steps:
        return target, values[steps.index(target)]
    return None, None


def cross_run(root, runs: list[dict], side: str, name: str, state: str, factor: str,
              context_kind: str = 'probe', independent: bool = False) -> dict:
    """Compare independently trained native runs grouped by one configuration factor."""
    if factor not in FACTORS:
        raise ValueError('Unsupported grouping factor')
    if len(runs) < 2:
        raise ValueError('Select at least two runs.')
    keys = IDENTITY_KEYS + [k for k in ARCHITECTURE_KEYS if k != factor]
    identities = {json.dumps({k: r['config'].get(k) for k in keys}, sort_keys=True) for r in runs}
    if len(identities) > 1:
        raise ValueError('Selected runs differ in dataset, model family, task or architecture. '
                         'Compare one stratum at a time, or group by the architecture factor itself.')
    entries = []
    for r in runs:
        contexts = [c for c in ix.contexts(root, r['id']) if c['context'].get('kind') == context_kind]
        if context_kind == 'analysis':
            contexts = [c for c in contexts if c['context'].get('x') == 0 and c['context'].get('reference') == 1
                        and not c['context'].get('gates') and c['context'].get('reference_type') == 'dataset'
                        and (c['context'].get('readout') or {}).get('scope') == 'all']
        context = contexts[0]['key'] if contexts else ''
        series = ix.series(root, r['id'], context)
        step, value = _run_value(series, side, name, state)
        steps, values, _ = _per_step(series.get((side, name), []))
        init_step, init_value = _run_value(series, side, name, 'initial')
        level = r['config'].get(factor)
        ignored = {factor, 'name'} | COUPLED.get(factor, set())
        block = {k: v for k, v in r['config'].items() if k not in ignored}
        entries.append(dict(run=r['id'], name=r['config'].get('name'), level=json.dumps(level), level_label=str(level),
                            seed=r['config'].get('seed'), step=step, value=value, initial=init_value,
                            trajectory=dict(steps=steps, values=values), block=json.dumps(block, sort_keys=True)))
    defined = [e for e in entries if finite(e['value'])]
    warnings = []
    state_steps = sorted({e['step'] for e in defined if e['step'] is not None})
    if state == 'final' and len(state_steps) > 1:
        warnings.append(f'Final checkpoints differ across runs (updates {state_steps[0]}–{state_steps[-1]}); '
                        'training lengths are not matched. Select a common update to compare a matched state.')
    levels = sorted({e['level'] for e in entries}, key=lambda s: (len(s), s))
    groups = []
    for level in levels:
        members = [e for e in defined if e['level'] == level]
        all_steps = [set(e['trajectory']['steps']) for e in members]
        common = sorted(set.intersection(*all_steps)) if all_steps else []
        band = dict(steps=common, median=[], q25=[], q75=[])
        for s in common:
            vals = [e['trajectory']['values'][e['trajectory']['steps'].index(s)] for e in members]
            band['median'].append(float(np.median(vals)))
            band['q25'].append(float(np.quantile(vals, .25)))
            band['q75'].append(float(np.quantile(vals, .75)))
        groups.append(dict(level=level, label=next((e['level_label'] for e in entries if e['level'] == level), level),
                           n=len(members), summary=describe([e['value'] for e in members]), band=band))
    out = dict(unit='Independently trained runs', side=side, name=name, label=label(name), state=state,
               factor=factor, context_kind=context_kind, runs=entries, groups=groups,
               block_definition=('Blocks share every configuration value except the factor'
                                 + (', its coupled hyperparameters (' + ', '.join(sorted(COUPLED[factor])) + ')' if factor in COUPLED else '')
                                 + ' and the run name.'),
               warnings=warnings,
               excluded=[dict(run=e['run'], name=e['name'], reason='Observable undefined or not measured at this state')
                         for e in entries if not finite(e['value'])],
               inference=dict(status='descriptive', reason='Declare independent training units to enable tests.'))
    # Rank retention (initial to selected state), descriptive.
    pairs = [(e['initial'], e['value']) for e in defined if finite(e['initial'])]
    if len(pairs) >= 4 and state != 'initial':
        out['rank_retention'] = dict(rho=spearman([p[0] for p in pairs], [p[1] for p in pairs]), n=len(pairs),
                                     status='descriptive', note='Spearman ρ between run ranks at initialisation and at the selected state.')
    if not independent:
        return _clean(out)
    if len(defined) != len({e['run'] for e in defined}):
        raise ValueError('Each run must contribute exactly one value.')
    k = len(groups)
    if k < 2:
        out['inference'] = dict(status='insufficient', reason='Only one factor level is present.')
        return _clean(out)
    # Matched blocks: identical configuration apart from the factor (seed matched).
    blocks = defaultdict(dict)
    for e in defined:
        blocks[e['block']][e['level']] = e['value']
    complete = [b for b in blocks.values() if len(b) == k]
    if len(complete) >= 3 and k >= 3:
        matrix = np.asarray([[b[level] for level in levels] for b in complete])
        test = sps.friedmanchisquare(*matrix.T)
        n = len(complete)
        out['inference'] = dict(status='exploratory', design='Matched blocks (seed-matched, otherwise identical configuration)',
                                test='Friedman χ²; Kendall W = χ²/(n(k − 1))', blocks=n, levels=k,
                                statistic=float(test.statistic), p=float(test.pvalue),
                                kendall_w=float(test.statistic / (n * (k - 1))))
    elif len(complete) >= 6 and k == 2:
        d = np.asarray([b[levels[1]] - b[levels[0]] for b in complete])
        nonzero = d[d != 0]
        test = sps.wilcoxon(d, zero_method='wilcox')
        ranks = sps.rankdata(np.abs(nonzero))
        out['inference'] = dict(status='exploratory', design='Matched pairs', blocks=len(complete), levels=2,
                                test='Two-sided Wilcoxon signed-rank on matched differences', statistic=float(test.statistic),
                                p=float(test.pvalue), rank_biserial=float(np.sum(np.sign(nonzero) * ranks) / ranks.sum()) if len(nonzero) else 0.0,
                                hodges_lehmann=float(np.median([(a + b) / 2 for i, a in enumerate(d) for b in d[i:]])))
    elif all(g['n'] >= 3 for g in groups):
        samples = [[e['value'] for e in defined if e['level'] == level] for level in levels]
        h = sps.kruskal(*samples)
        total = sum(len(s) for s in samples)
        out['inference'] = dict(status='exploratory', design='Unmatched groups', test='Kruskal–Wallis; ε² = (H − k + 1)/(n − k)',
                                statistic=float(h.statistic), p=float(h.pvalue), levels=k,
                                epsilon_squared=max(0.0, float((h.statistic - k + 1) / (total - k))) if total > k else None)
    else:
        out['inference'] = dict(status='insufficient', reason='Needs ≥3 complete matched blocks (k ≥ 3), ≥6 matched pairs (k = 2), '
                                                              'or ≥3 runs in every group.')
    return _clean(out)


# -------------------------------------------------------------------------- ledger

LEDGER_COLUMNS = ['section', 'quantity', 'unit', 'n', 'estimate', 'interval', 'statistic', 'p', 'p_holm', 'effect', 'status']


def ledger(run_stats: dict | None = None, cohort: dict | None = None, cross: dict | None = None) -> list[dict]:
    rows = []
    if run_stats:
        for item in run_stats.get('realisation', []):
            if not item['n']:
                continue
            rows.append(dict(section='Disposition → realisation', quantity=f"{item['label']}: conditional − structural (median)",
                             unit='checkpoints', n=item['n'], estimate=item['summary']['median'],
                             interval=[item['summary']['q25'], item['summary']['q75']], effect=f"{item['negative']}↓ / {item['positive']}↑",
                             status='descriptive'))
        for group in ('path', 'geometry', 'training'):
            for t in run_stats.get('trajectories', {}).get(group, []):
                if t['n'] < 2:
                    continue
                rows.append(dict(section='Temporal', quantity=f"{t['side_label']} · {t['label']}: first → last change",
                                 unit='checkpoints', n=t['n'], estimate=t.get('change'), effect=t.get('trend_rho'),
                                 status='descriptive'))
    if cohort:
        for family, title in (('errors', 'Cohort · misclassified vs correct'), ('labels', 'Cohort · class 1 vs class 0')):
            for item in cohort.get(family, []):
                rows.append(dict(section=title, quantity=item['label'], unit='images', n=sum(item['n']),
                                 estimate=item.get('auroc'), interval=item.get('auroc_interval'), statistic=item.get('U'),
                                 p=item.get('p'), p_holm=item.get('p_holm'), effect=item.get('rank_biserial'), status=item['status']))
        for item in cohort.get('omnibus', []):
            rows.append(dict(section='Cohort · outcome groups', quantity=item['label'], unit='images',
                             statistic=item.get('H'), p=item.get('p'), p_holm=item.get('p_holm'),
                             effect=item.get('epsilon_squared'), status=item['status']))
    if cross:
        inf = cross.get('inference', {})
        rows.append(dict(section=f"Cross-run · {cross['factor']}", quantity=f"{side_label(cross['side'])} · {cross['label']} at {cross['state']}",
                         unit='runs', n=sum(g['n'] for g in cross['groups']), statistic=inf.get('statistic'), p=inf.get('p'),
                         effect=inf.get('kendall_w', inf.get('epsilon_squared', inf.get('rank_biserial'))), status=inf.get('status')))
    return _clean(rows)


def _fmt(v) -> str:
    if v is None:
        return ''
    if isinstance(v, list):
        return '[' + ', '.join(_fmt(x) for x in v) + ']'
    if isinstance(v, float):
        return f'{v:.4g}'
    return str(v)


def ledger_csv(rows: list[dict]) -> str:
    import csv
    import io
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(LEDGER_COLUMNS)
    for row in rows:
        writer.writerow([_fmt(row.get(c)) for c in LEDGER_COLUMNS])
    return buffer.getvalue()


def ledger_latex(rows: list[dict]) -> str:
    def esc(s: str) -> str:
        for a, b in [('\\', r'\textbackslash{}'), ('&', r'\&'), ('%', r'\%'), ('_', r'\_'), ('#', r'\#')]:
            s = s.replace(a, b)
        return s
    lines = [r'\begin{tabular}{llrlllll}', r'\toprule',
             r'Section & Quantity & $n$ & Estimate & Interval & $p_{\mathrm{Holm}}$ & Effect & Status \\', r'\midrule']
    for row in rows:
        lines.append(' & '.join(esc(_fmt(row.get(c))) for c in ['section', 'quantity', 'n', 'estimate', 'interval', 'p_holm', 'effect', 'status']) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines) + '\n'
