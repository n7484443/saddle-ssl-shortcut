import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    import csv
    import hashlib
    import json
    import math
    import statistics
    from pathlib import Path
    import marimo as mo
    import matplotlib.pyplot as plt
    import seaborn as sns
    import torch

    device = torch.device('cuda')
    sns.set_theme(style='whitegrid', font='cmr10', font_scale=1.9)
    plt.rcParams['axes.formatter.use_mathtext'] = True
    plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'pdf.use14corefonts': False})
    colors = sns.color_palette()
    out_dir = Path(__file__).resolve().parent
    project_root = next(p for p in Path(__file__).resolve().parents if (p / "common.py").is_file())
    source_dir = project_root / 'frequency_shortcut/practical/mnist_backdoor_others'
    return colors, csv, hashlib, json, math, mo, out_dir, plt, source_dir, statistics


@app.cell
def _(csv, hashlib, math, source_dir, statistics):
    methods = ('SimCLR', 'VICReg')
    seeds = (0, 1, 2)
    checkpoint = 935
    raw_rows, summary, source_files = [], [], []
    for _method in methods:
        _seed_grids = []
        for _seed in seeds:
            _path = source_dir / 'result' / _method / 'None' / f'{_seed}_seed' / '1_5step_0.2subset.csv'
            _bytes = _path.read_bytes()
            _records = list(csv.DictReader(_bytes.decode('utf-8-sig').splitlines()))
            assert _records and set(_records[0]) == {'step', 'intensity', 'test_acc'}
            _keys = set()
            for _record in _records:
                _step = int(_record['step'])
                _x, _y = float(_record['intensity']), float(_record['test_acc'])
                assert _step == checkpoint, 'Do not pool checkpoints'
                assert math.isfinite(_x) and _x > 0 and math.isfinite(_y) and 0 <= _y <= 100
                assert (_step, _x) not in _keys, 'Duplicate run/intensity row'
                _keys.add((_step, _x))
                raw_rows.append(dict(method=_method, seed=_seed, step=_step, intensity=_x, test_acc=_y))
            _seed_grids.append(sorted(_x[1] for _x in _keys))
            source_files.append(dict(path=str(_path), sha256=hashlib.sha256(_bytes).hexdigest(), rows=len(_records)))
        assert _seed_grids[0] == _seed_grids[1] == _seed_grids[2], 'Incomplete seed grid'
        for _intensity in _seed_grids[0]:
            _values = [r['test_acc'] for r in raw_rows if r['method'] == _method and r['intensity'] == _intensity]
            assert len(_values) == len(seeds)
            summary.append(dict(method=_method, step=checkpoint, intensity=_intensity, n=len(_values),
                                mean=statistics.mean(_values), sample_sd=statistics.stdev(_values)))
    assert len(raw_rows) == 99 and len(summary) == 33
    code_files = []
    for _relative in ['mnist_backdoor.py', 'mnist_backdoor_generation.ipynb',
                      'configs/conf_simclr.yaml', 'configs/conf_vicreg.yaml']:
        _path = source_dir / _relative
        code_files.append(dict(path=str(_path), sha256=hashlib.sha256(_path.read_bytes()).hexdigest()))
    return checkpoint, code_files, methods, raw_rows, seeds, source_files, summary


@app.cell
def _(summary):
    rebuttal_x = [0.5623, 3.1623, 11.5478, 17.7828, 42.1697, 100.0]
    rebuttal_values = {'SimCLR': [90.34, 89.99, 87.07, 83.97, 25.62, 10.47],
                       'VICReg': [85.55, 85.51, 45.32, 16.59, 11.16, 10.50]}
    reconciliation = []
    for _method, _values in rebuttal_values.items():
        for _x, _reported in zip(rebuttal_x, _values):
            _row = next(r for r in summary if r['method'] == _method and r['intensity'] == _x)
            _delta = _row['mean'] - _reported
            reconciliation.append(dict(method=_method, intensity=_x, saved_mean=_row['mean'],
                rebuttal_mean=_reported, delta_percentage_points=_delta,
                matches_rounding=abs(_delta) <= 0.00500001))
    assert all(r['matches_rounding'] for r in reconciliation), 'Rebuttal discrepancy requires review'
    return (reconciliation,)


@app.cell
def _(checkpoint, code_files, csv, json, methods, out_dir, raw_rows, reconciliation, seeds, source_files, summary):
    provenance = dict(checkpoint=checkpoint, seeds=list(seeds), methods=list(methods),
        spurious_correlation='random trigger-label association (10% expected)',
        aggregation='Arithmetic mean and sample standard deviation (ddof=1) across all three seeds',
        missing='SimCLR intensity 0.1 absent for all seeds; no interpolation or imputation',
        source_files=source_files, current_code_files=code_files, rebuttal_reconciliation=reconciliation,
        code_provenance_limit='Current checked-in code/configs, not historical per-run snapshots. Config default conf.yaml absent; select method config explicitly for future reruns.',
        scope='Saved CSV replot; not a new training run or a controlled algorithm-ranking benchmark')
    (out_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    for _name, _rows in [('source_rows.csv', raw_rows), ('summary.csv', summary),
                         ('rebuttal_reconciliation.csv', reconciliation)]:
        with (out_dir / _name).open('w', newline='', encoding='utf-8') as _stream:
            _writer = csv.DictWriter(_stream, fieldnames=list(_rows[0]))
            _writer.writeheader()
            _writer.writerows(_rows)
    return (provenance,)


@app.cell
def _(colors, methods, out_dir, plt, summary):
    def make_figure():
        fig, ax = plt.subplots(figsize=(12, 7.2), layout='constrained')
        for index, method in enumerate(methods):
            rows = [r for r in summary if r['method'] == method]
            x = [r['intensity'] for r in rows]
            mean = [r['mean'] for r in rows]
            sd = [r['sample_sd'] for r in rows]
            ax.errorbar(x, mean, yerr=sd, fmt='o-', color=colors[index], linewidth=2.7,
                        markersize=6, elinewidth=1.6, capsize=4, capthick=1.6, label=method)
        ax.axhline(10, color='0.35', linestyle='--', linewidth=1.6, label='10% reference')
        ax.set_xscale('log')
        ax.set_xlim(0.085, 118)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 10, 30, 50, 70, 90, 100])
        ax.set_xlabel('Trigger intensity')
        ax.set_ylabel('Linear-probe test accuracy (%)')
        ax.legend(loc='lower left', framealpha=0.96)
        ax.grid(axis='x', which='minor', alpha=0.15)
        return fig

    figure = make_figure()
    pdf_path = out_dir / 'ssl_backdoor.pdf'
    png_path = out_dir / 'ssl_backdoor.png'
    figure.savefig(pdf_path, bbox_inches='tight', pad_inches=0.2)
    figure.savefig(png_path, bbox_inches='tight', pad_inches=0.2)
    figure
    return figure, pdf_path, png_path


@app.cell
def _(mo, pdf_path):
    mo.vstack([
        mo.md('''**SimCLR / VICReg backdoored MNIST.** Random trigger-label association (10% expected),
        step 935, all seeds 0, 1, 2. Points show the arithmetic mean. Error bars show one sample SD.
        All available intensities are shown: 16 for SimCLR and 17 for VICReg.
        SimCLR has no saved intensity 0.1 result. This notebook only reads saved results.'''),
        mo.download(pdf_path.read_bytes(), filename=pdf_path.name, mimetype='application/pdf', label='Download plot PDF'),
    ])
    return


@app.cell
def _(mo, reconciliation, summary):
    mo.accordion({'Summary (separate from the figure)': mo.ui.table(summary, selection=None),
                  'Final rebuttal comparison': mo.ui.table(reconciliation, selection=None)})
    return


if __name__ == '__main__':
    app.run()
