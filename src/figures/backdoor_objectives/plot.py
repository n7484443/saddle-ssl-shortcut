import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    import csv
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
    return colors, csv, mo, out_dir, plt, source_dir, statistics


@app.cell
def _(csv, source_dir, statistics):
    methods = ('SimCLR', 'VICReg')
    seeds = (0, 1, 2)
    checkpoint = 935
    raw_rows, summary = [], []
    for _method in methods:
        _seed_grids = []
        for _seed in seeds:
            _path = source_dir / 'result' / _method / 'None' / f'{_seed}_seed' / '1_5step_0.2subset.csv'
            _bytes = _path.read_bytes()
            _records = list(csv.DictReader(_bytes.decode('utf-8-sig').splitlines()))
            _keys = set()
            for _record in _records:
                _step = int(_record['step'])
                _x, _y = float(_record['intensity']), float(_record['test_acc'])
                _keys.add((_step, _x))
                raw_rows.append(dict(method=_method, seed=_seed, step=_step, intensity=_x, test_acc=_y))
            _seed_grids.append(sorted(_x[1] for _x in _keys))
        for _intensity in _seed_grids[0]:
            _values = [r['test_acc'] for r in raw_rows if r['method'] == _method and r['intensity'] == _intensity]
            summary.append(dict(method=_method, step=checkpoint, intensity=_intensity, n=len(_values),
                                mean=statistics.mean(_values), sample_sd=statistics.stdev(_values)))
    return methods, summary


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
    return pdf_path,


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


if __name__ == '__main__':
    app.run()
