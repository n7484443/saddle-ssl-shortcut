"""Run the two predeclared extensions, then freeze all K comparisons and diagnostics.

Waits for the primary pipeline; never modifies its numerical code or caches.
"""
import ctypes
import hashlib
import json
import msvcrt
import os
from pathlib import Path
import sys
import time
import numpy as np
import scaling_cache as sc
import train_scaling_pipeline as tp

ROOT = sc.ROOT
STATUS = ROOT / 'train_sensitivity_status.json'
MANIFEST = ROOT / 'cache/train_study_results.json'
BOOT_SEED = 20260924
BOOT_REPEATS = 2000
KS = [10, 20, 50, 100]
STATE = dict(pid=os.getpid(), status='starting')


def status(**changes):
    STATE.update(changes, updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    sc.write_json(STATUS, STATE)


def load_arrays(outcome, root=None):
    directory = (root or ROOT / 'cache/train_runs') / outcome['key']
    arrays = {}
    for cls in outcome['mode_counts']:
        with np.load(directory / f'{cls}.npz', allow_pickle=False) as saved:
            arrays[cls] = {k: saved[k] for k in saved.files}
    return arrays


def bootstrap(values):
    values = np.asarray(values)
    idx = np.random.default_rng(BOOT_SEED).integers(0, len(values), (BOOT_REPEATS, len(values)))
    return values[idx].mean(axis=1)


def variant(outcome, arrays, k):
    truncated = {c: {key: a[:, :k] for key, a in r.items()} for c, r in arrays.items()}
    summary, curves = sc.summarize(truncated, outcome['config'])
    per_class, intervals = {}, {}
    for label in ('amplitude', 'extent'):
        slopes, r2 = sc.fit_power(curves[label]['x'], curves[label]['group_median_curves'])
        per_class[label] = {c: dict(exponent=float(s), r2=float(q))
                            for c, s, q in zip(truncated, slopes, r2)}
        intervals[label] = np.quantile(bootstrap(slopes), [.025, .975]).tolist()
        np.testing.assert_allclose(slopes.mean(), summary[label]['mean_group_median_curve_exponent'], atol=1e-12)
    return dict(name=outcome['name'], key=outcome['key'], k=k, summary=summary,
                audit=outcome['audit'], config=outcome['config'], class_exponents=per_class,
                conditional_class_bootstrap_95=intervals,
                actual_modes={c: int(r['f_alpha'].shape[1]) for c, r in truncated.items()},
                amplitude_distance_from_2=abs(summary['amplitude']['mean_group_median_curve_exponent']-2),
                extent_distance_from_1=abs(summary['extent']['mean_group_median_curve_exponent']-1))


def paired_improvement(current, reference):
    a, b = current['class_exponents']['amplitude'], reference['class_exponents']['amplitude']
    common = sorted(set(a) & set(b))
    ca, cb = np.asarray([a[c]['exponent'] for c in common]), np.asarray([b[c]['exponent'] for c in common])
    idx = np.random.default_rng(BOOT_SEED).integers(0, len(common), (BOOT_REPEATS, len(common)))
    differences = abs(cb[idx].mean(axis=1)-2) - abs(ca[idx].mean(axis=1)-2)
    return dict(reference_name=reference['name'], reference_k=reference['k'], paired_classes=common,
                distance_reduction=float(abs(cb.mean()-2)-abs(ca.mean()-2)),
                bootstrap_95=np.quantile(differences, [.025, .975]).tolist(),
                interpretation='Positive is closer to 2 on the same classes; exploratory, not a corrected hypothesis test.')


def quadratic(outcome, arrays, destination):
    x = np.asarray(outcome['config']['amplitudes'])
    design = np.column_stack((x*x, x, np.ones_like(x)))
    rows = []
    for cls, result in arrays.items():
        f = result['f_alpha']
        coef = np.linalg.lstsq(design, f, rcond=None)[0]
        relative_error = np.max(abs(design @ coef-f), axis=0)/np.max(abs(f), axis=0)
        proportions = coef / coef.sum(axis=0)
        target = destination / outcome['key'] / f'{cls}.npz'
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix('.npz.tmp')
        with temp.open('wb') as stream:
            np.savez_compressed(stream, coefficients=coef, terms_at_alpha_1=proportions,
                                scaled_residual=relative_error,
                                linear_to_quadratic=coef[1][None,:]/(x[:,None]*coef[0][None,:]),
                                constant_to_quadratic=coef[2][None,:]/(x[:,None]**2*coef[0][None,:]))
        temp.replace(target)
        rows.append(dict(class_id=cls, max_scaled_residual=float(relative_error.max()),
                         median_quadratic_linear_constant_at_1=np.median(proportions, axis=1).tolist()))
    return dict(name=outcome['name'], key=outcome['key'], classes=rows,
                max_scaled_residual=max(r['max_scaled_residual'] for r in rows),
                coefficient_order=['lambda', '2q', 'r'],
                note='Decomposed before aggregation; normalized component medians need not sum to one.')


def report(outcomes, variants, diagnostics):
    baseline = next(v for v in variants if v['name']=='train_45_55_classnorm' and v['k']==100)
    for v in variants:
        v['paired_vs_primary'] = paired_improvement(v, baseline)
    references = []
    for o in outcomes:
        protocol = ROOT / 'cache/train_runs' / o['key'] / 'protocol.json'
        references.append(dict(name=o['name'], key=o['key'], path=str(protocol),
                               protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
                               protocol=json.loads(protocol.read_text(encoding='utf-8'))))
    sc.write_json(ROOT / 'cache/train_sensitivity_results.json', variants)
    sc.write_json(ROOT / 'cache/train_quadratic_diagnostics.json', diagnostics)
    candidates = sorted([v for v in variants if v['amplitude_distance_from_2'] < abs(1.59-2)],
                        key=lambda v: v['amplitude_distance_from_2'])
    sc.write_json(ROOT / 'cache/train_candidate_settings.json', dict(
        status='saved_before_structural_controls', baseline=baseline, candidates=candidates,
        all_settings=references, bootstrap_seed=BOOT_SEED, bootstrap_repeats=BOOT_REPEATS,
        analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        note='Candidates are exploratory point estimates closer to 2 than 1.59; not proof of historical identity or independent validation. Bootstrap conditions on selected classes; no multiple-comparison correction.'))
    lines = ['# ImageNet train — sensitivity and frozen settings', '',
             'All settings are reported. K is capped by each class sample count. The ranking uses one fixed aggregation: mean class-median-curve exponent.', '',
             '| Setting | K | Classes / images | Amplitude | Extent | Median R² amp / ext | Amplitude class-bootstrap 95% |',
             '|---|---:|---:|---:|---:|---|---|']
    for v in variants:
        a,e = v['summary']['amplitude'], v['summary']['extent']
        ci = v['conditional_class_bootstrap_95']['amplitude']
        lines.append(f"| {v['name']} | {v['k']} | {v['audit']['computed_groups']} / {v['audit']['computed_images']} | "
                     f"{a['mean_group_median_curve_exponent']:.6f} | {e['mean_group_median_curve_exponent']:.6f} | "
                     f"{a['median_group_curve_r2']:.5f} / {e['median_group_curve_r2']:.5f} | [{ci[0]:.5f}, {ci[1]:.5f}] |")
    lines += ['', 'Bootstrap uses 2,000 resamples of the fixed selected classes, seed 20260924. It does not account for outcome-guided selection or generalize from the first 200 sorted categories to all ImageNet. Paired comparisons in JSON use only common classes, so their baseline may differ from the full-table baseline.', '',
              'Rationales: area and normalization are existing notebook options; K tests leading-mode sensitivity; resolution112 tests spatial-detail sensitivity; noalign tests centering sensitivity. No sweep range, sample seed, or aggregation was selected to force an exponent.', '',
              '## Per-mode quadratic decomposition', '',
              'For fixed extent and object direction, f(alpha)=lambda alpha²+2q alpha+r. These are object, cross, and background terms. The identity is built into the current measurement; its accuracy is a numerical check, not independent empirical evidence.']
    for d in diagnostics:
        lines.append(f"- {d['name']}: maximum scaled quadratic residual {d['max_scaled_residual']:.3g}.")
    lines += ['', 'Frozen configurations and manifests: `Imagenet/cache/train_candidate_settings.json`.',
              'Per-class distributions and paired bootstrap: `Imagenet/cache/train_sensitivity_results.json`.',
              'Per-mode term arrays: `Imagenet/cache/train_diagnostics/<run key>/<class>.npz`.', '']
    (ROOT.parent / 'output/imagenet_train_sensitivity.md').write_text('\n'.join(lines), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, label, theory in zip(axes, ('amplitude','extent'), (2,1)):
        matrix = np.asarray([[next(v for v in variants if v['name']==o['name'] and v['k']==k)['summary'][label]['mean_group_median_curve_exponent'] for k in KS] for o in outcomes])
        im = ax.imshow(matrix, aspect='auto', cmap='viridis')
        ax.set_xticks(range(len(KS)), KS); ax.set_xlabel('Maximum K')
        ax.set_yticks(range(len(outcomes)), [o['name'].replace('train_','') for o in outcomes], fontsize=8)
        ax.set_title(f'{label.capitalize()} exponent (theory {theory})')
        for i in range(len(outcomes)):
            for j in range(len(KS)): ax.text(j,i,f'{matrix[i,j]:.3f}',ha='center',va='center',color='white',fontsize=9)
        fig.colorbar(im, ax=ax, shrink=.7)
    for ext in ('png','pdf'): fig.savefig(ROOT.parent / f'output/imagenet_train_sensitivity.{ext}')
    plt.close(fig)


def main():
    status(status='waiting_for_primary')
    while True:
        source = json.loads((ROOT / 'train_scaling_status.json').read_text(encoding='utf-8'))
        if source.get('status') == 'ready_for_review': break
        status(primary_status=source.get('status'), primary_updated_at=source.get('updated_at'))
        time.sleep(30)
    outcomes = json.loads(MANIFEST.read_text(encoding='utf-8'))
    primary = next(o for o in outcomes if o['name']=='train_45_55_classnorm')
    data = json.loads(tp.RECORDS.read_text(encoding='utf-8'))
    tp.STATUS = STATUS
    tp.STATE = STATE
    for name, changes in [('train_45_55_classnorm_resolution112',dict(resolution=112)),
                          ('train_45_55_classnorm_noalign',dict(align=False))]:
        if not any(o['name']==name for o in outcomes):
            outcomes.append(tp.compute_run({**primary['config'], **changes}, data, name))
            sc.write_json(MANIFEST, outcomes)
    status(status='summarizing_sensitivity')
    variants, diagnostics = [], []
    for o in outcomes:
        arrays = load_arrays(o)
        variants.extend(variant(o, arrays, k) for k in KS)
        diagnostics.append(quadratic(o, arrays, ROOT / 'cache/train_diagnostics'))
    report(outcomes, variants, diagnostics)
    status(status='ready_for_structural_controls', completed_runs=len(outcomes), completed_k_variants=len(variants))
    print('Sensitivity and quadratic diagnostics complete. Settings frozen; structural controls are next.', flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    with (ROOT / 'train_sensitivity.lock').open('a+b') as lock:
        lock.seek(0)
        if not lock.read(1): lock.write(b'0'); lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        try:
            main()
        except Exception as exc:
            status(status='failed', error=repr(exc))
            raise
        finally:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
