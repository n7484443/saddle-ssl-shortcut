import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    from pathlib import Path
    import marimo as mo
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    import torch
    from matplotlib.collections import LineCollection
    device = torch.device("cuda")
    sns.set_theme(style="whitegrid", font="cmr10", font_scale=1.9)
    plt.rcParams["axes.formatter.use_mathtext"] = True
    colors = sns.color_palette()
    root = Path(mo.notebook_dir())
    return LineCollection, json, mo, np, plt, root, sns


@app.cell
def _(LineCollection, json, np, plt, root, sns):
    ROOT = root
    OUTPUT = root.parent / 'output/pdf'
    def fit_modes(x, ratios, reference):
        assert np.isfinite(ratios).all() and (ratios > 0).all(), 'Invalid modes must be diagnosed, not dropped'
        lx, ly = np.log(np.asarray(x) / reference), np.log(ratios)
        slope, intercept = np.polyfit(lx, ly, 1)
        residual = np.sum((ly - lx[:, None] * slope - intercept) ** 2, axis=0)
        total = np.sum((ly - ly.mean(axis=0)) ** 2, axis=0)
        assert (total > 0).all()
        r2 = 1 - residual / total
        return slope, intercept, r2

    def load_result():
        outcome = json.loads((ROOT / 'cache/historical_20point_result.json').read_text(encoding='utf-8'))
        directory = ROOT / 'cache/train_runs' / outcome['key']
        curves = {'amplitude': [], 'extent': []}
        for cls in sorted(outcome['mode_counts']):
            with np.load(directory / f'{cls}.npz', allow_pickle=False) as data:
                curves['amplitude'].append(data['f_alpha'])
                curves['extent'].append(data['f_extent'])
        result = {'outcome': outcome, 'directory': directory}
        for name, grid, target in [('amplitude', 'amplitudes', 2.), ('extent', 'extents', 1.)]:
            x = np.asarray(outcome['config'][grid])
            assert x.size == 20
            index = int(np.argmin(abs(x - 1)))
            reference = float(x[index])
            values = np.concatenate(curves[name], axis=1)
            ratios = values / values[index]
            slope, intercept, r2 = fit_modes(x, ratios, reference)
            stats = dict(median_exponent=float(np.median(slope)), exponent_iqr=np.quantile(slope, [.25, .75]).tolist(),
                         median_r2=float(np.median(r2)), r2_iqr=np.quantile(r2, [.25, .75]).tolist(),
                         median_abs_exponent_error=float(np.median(abs(slope - target))),
                         mean_abs_exponent_error=float(np.mean(abs(slope - target))),
                         median_abs_intercept=float(np.median(abs(intercept))), valid_fits=len(slope))
            assert len(slope) == 9063
            result[name] = dict(x=x, reference=reference, ratios=ratios, slopes=slope, intercepts=intercept,
                                r2=r2, target=target, stats=stats)
        return result

    def statistic_rows(result):
        a, e = result['amplitude']['stats'], result['extent']['stats']
        interval = lambda values: f'[{values[0]:.4f}, {values[1]:.4f}]'
        return [
            ['Target exponent', '2.0000', '1.0000'],
            ['Median exponent k', f"{a['median_exponent']:.4f}", f"{e['median_exponent']:.4f}"],
            ['Exponent IQR', interval(a['exponent_iqr']), interval(e['exponent_iqr'])],
            ['Median |k - target|', f"{a['median_abs_exponent_error']:.4f}", f"{e['median_abs_exponent_error']:.4f}"],
            ['Mean |k - target|', f"{a['mean_abs_exponent_error']:.4f}", f"{e['mean_abs_exponent_error']:.4f}"],
            ['Median R²', f"{a['median_r2']:.4f}", f"{e['median_r2']:.4f}"],
            ['R² IQR', interval(a['r2_iqr']), interval(e['r2_iqr'])],
            ['Median |intercept|', f"{a['median_abs_intercept']:.4f}", f"{e['median_abs_intercept']:.4f}"],
            ['Valid (class, mode/rank) fits', f"{a['valid_fits']:,}", f"{e['valid_fits']:,}"],
        ]

    def make_figure(result, width=14., font_scale=1.9, show_modes=True, panel='scaling'):
        assert panel in ('scaling', 'fits')
        with plt.rc_context():
            sns.set_theme(style="whitegrid", font="cmr10", font_scale=font_scale)
            plt.rcParams["axes.formatter.use_mathtext"] = True
            plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'mathtext.fontset': 'cm',
                                 'axes.unicode_minus': False, 'savefig.facecolor': 'white'})
            palette = sns.color_palette()
            fig = plt.figure(figsize=(width, width * .43), layout='constrained')
            fig.get_layout_engine().set(w_pad=.18, h_pad=.22, wspace=.12)
            grid = fig.add_gridspec(1, 2)
            for col, name in enumerate(['amplitude', 'extent']):
                d = result[name]
                x, ratio = d['x'], d['ratios']
                axis = fig.add_subplot(grid[0, col])
                if panel == 'scaling':
                    if show_modes:
                        segments = np.stack([np.broadcast_to(x[None], ratio.T.shape), ratio.T], axis=-1)
                        axis.add_collection(LineCollection(segments, colors=[palette[0]], linewidths=.45, alpha=.025, rasterized=True))
                    axis.plot(x, np.median(ratio, axis=1), 'o-', color='#20242a', lw=1.7, ms=3.3, label='Median')
                    axis.plot(x, (x / d['reference']) ** d['target'], '--', color=palette[3], lw=1.5, label=f"Target k={d['target']:g}")
                    axis.set(xlabel=r'$\alpha$' if col == 0 else 'Extent', ylabel=r'$f/f_{\mathrm{ref}}$',
                             title='(a) Amplitude scaling' if col == 0 else '(b) Extent scaling')
                    axis.autoscale_view()
                    axis.set_xlim(x[0], x[-1])
                    axis.set_ylim(0, 7 if col == 0 else 2.35)
                    axis.legend(loc='upper left', frameon=False)
                else:
                    axis.scatter(d['slopes'], d['r2'], s=6, color=palette[0], alpha=.20, edgecolors='none', rasterized=True)
                    axis.axvline(d['target'], color=palette[3], ls='--', lw=1.3, label=f"Target k={d['target']:g}")
                    axis.axvline(d['stats']['median_exponent'], color='#20242a', lw=1.2,
                                 label=f"Median k={d['stats']['median_exponent']:.4f}")
                    axis.set(title='(a) Amplitude power-law fits' if col == 0 else '(b) Extent power-law fits',
                             xlabel='Fitted exponent k', ylabel=r'$R^2$ in log-log fit')
                    axis.legend(loc='lower right', frameon=False)
                    axis.margins(x=.07, y=.08)
                axis.grid(alpha=.18)
            return fig

    def export_figure(result, width=14., font_scale=1.9, show_modes=True):
        OUTPUT.mkdir(parents=True, exist_ok=True)
        for panel, stem in [('scaling', 'imagenet_scaling_curves'), ('fits', 'imagenet_power_law_fits')]:
            fig = make_figure(result, width, font_scale, show_modes, panel=panel)
            fig.savefig(OUTPUT / f'{stem}.pdf', bbox_inches='tight', pad_inches=.3,
                        metadata={'Title': f'ImageNet {panel}: historical 20-point reproduction'})
            fig.savefig(OUTPUT / f'{stem}.png', bbox_inches='tight', pad_inches=.3)
            plt.close(fig)
        pdf, png = OUTPUT / 'imagenet_scaling_curves.pdf', OUTPUT / 'imagenet_scaling_curves.png'
        summary = {name: result[name]['stats'] for name in ['amplitude', 'extent']}
        summary['cache_key'] = result['outcome']['key']
        summary['settings'] = result['outcome']['config']
        summary['reference_amplitude'] = result['amplitude']['reference']
        summary['reference_extent'] = result['extent']['reference']
        summary['fit'] = 'log(f/f_ref) = intercept + k log(x/x_ref); median over all class-mode fits'
        (OUTPUT / 'imagenet_scaling_reproduction.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        return pdf, png

    return export_figure, load_result, make_figure, statistic_rows


@app.cell(hide_code=True)
def _(mo):
    refresh = mo.ui.run_button(label='완료된 캐시 새로고침')
    mo.vstack([mo.md(r'''
    # ImageNet scaling

    **200 classes · images/class = max · 224 px · K <= 100 · bbox area 45–55%**

    Use single-bbox images and center the bbox. Hold each selected class's RGB mean/std fixed.
    Use 20 amplitude values from 0.2 to 2.0 and 20 extent values from 0.2 to 1.0.

    Fit $\log(f/f_{\rm ref})=a+k\log(x/x_{\rm ref})$ for each class and mode,
    then report the median coefficient across all valid fits, rather than averaging class coefficients.
    This notebook reads cached spectra; redrawing does not repeat image processing or GPU computation.
    '''), refresh])
    return (refresh,)


@app.cell
def _(json, load_result, mo, refresh, root):
    refresh.value
    _manifest = root / 'cache/historical_20point_result.json'
    _status_file = root / 'historical_20point_status.json'
    _status = json.loads(_status_file.read_text(encoding='utf-8')) if _status_file.exists() else {}
    mo.stop(not _manifest.exists(), mo.md(f"Preparing the original 20-point spectral cache. **{_status.get('completed_groups', 0)}/{_status.get('total_groups', 188)} classes** · {_status.get('status', 'waiting')}\n\nThe amplitude coefficient was reproduced as **1.586016** in a separate cache check. Refresh above when processing finishes."))
    result = load_result()
    return (result,)


@app.cell(hide_code=True)
def _(mo, result, statistic_rows):
    _a, _e = result['amplitude']['stats'], result['extent']['stats']
    mo.vstack([
        mo.md(f"## Amplitude **{_a['median_exponent']:.4f}** · Extent **{_e['median_exponent']:.4f}**\n\nOf the 200 classes, **188 classes, 9,073 images, and 9,063 mode fits** pass the filters. Reference coefficients are 1.5860 / 1.1235."),
        mo.ui.table([dict(zip(['Statistic', 'Amplitude', 'Extent'], row)) for row in statistic_rows(result)], selection=None),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    width = mo.ui.slider(start=12, stop=18, step=.5, value=14, label='그림 너비 (inch)')
    font_scale = mo.ui.slider(start=1.5, stop=2.3, step=.1, value=1.9, label='글자 배율 (기본 1.9)')
    show_modes = mo.ui.checkbox(value=True, label='전체 mode 곡선 표시')
    save_pdf = mo.ui.run_button(label='현재 설정으로 PDF / PNG 저장')
    mo.hstack([width, font_scale, show_modes, save_pdf], justify='start')
    return font_scale, save_pdf, show_modes, width


@app.cell(hide_code=True)
def _(font_scale, make_figure, mo, result, show_modes, width):
    scaling_figure = make_figure(result, width.value, font_scale.value, show_modes.value, panel='scaling')
    mo.vstack([mo.md('## Scaling curves'), mo.as_html(scaling_figure)])
    return


@app.cell(hide_code=True)
def _(font_scale, make_figure, mo, result, show_modes, width):
    fits_figure = make_figure(result, width.value, font_scale.value, show_modes.value, panel='fits')
    mo.vstack([mo.md('## Power-law fits'), mo.as_html(fits_figure)])
    return


@app.cell(hide_code=True)
def _(export_figure, font_scale, mo, result, save_pdf, show_modes, width):
    mo.stop(not save_pdf.value)
    pdf_path, png_path = export_figure(result, width.value, font_scale.value, show_modes.value)
    fits_pdf_path = pdf_path.with_name('imagenet_power_law_fits.pdf')
    fits_png_path = png_path.with_name('imagenet_power_law_fits.png')
    mo.vstack([
        mo.md(f'Saved: `{pdf_path}`'),
        mo.download(pdf_path.read_bytes(), filename=pdf_path.name, mimetype='application/pdf', label='Scaling PDF 다운로드'),
        mo.download(png_path.read_bytes(), filename=png_path.name, mimetype='image/png', label='Scaling PNG 다운로드'),
        mo.download(fits_pdf_path.read_bytes(), filename=fits_pdf_path.name, mimetype='application/pdf', label='Fits PDF 다운로드'),
        mo.download(fits_png_path.read_bytes(), filename=fits_png_path.name, mimetype='image/png', label='Fits PNG 다운로드'),
    ])
    return


if __name__ == "__main__":
    app.run()
