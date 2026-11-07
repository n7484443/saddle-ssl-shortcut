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
        lx, ly = np.log(np.asarray(x) / reference), np.log(ratios)
        slope, intercept = np.polyfit(lx, ly, 1)
        residual = np.sum((ly - lx[:, None] * slope - intercept) ** 2, axis=0)
        total = np.sum((ly - ly.mean(axis=0)) ** 2, axis=0)
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
            index = int(np.argmin(abs(x - 1)))
            reference = float(x[index])
            values = np.concatenate(curves[name], axis=1)
            ratios = values / values[index]
            slope, intercept, r2 = fit_modes(x, ratios, reference)
            stats = dict(median_exponent=float(np.median(slope)))
            result[name] = dict(x=x, reference=reference, ratios=ratios, slopes=slope, intercepts=intercept,
                                r2=r2, target=target, stats=stats)
        return result

    def make_figure(result, width=14., font_scale=1.9, show_modes=True, panel='scaling'):
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
        return pdf, png

    return export_figure, load_result, make_figure


@app.cell(hide_code=True)
def _(mo):
    refresh = mo.ui.run_button(label='완료된 캐시 새로고침')
    mo.vstack([mo.md("# ImageNet scaling"), refresh])
    return refresh,


@app.cell
def _(load_result, mo, refresh, root):
    refresh.value
    _manifest = root / 'cache/historical_20point_result.json'
    mo.stop(not _manifest.exists(), mo.md("Generate the spectral cache with reproduce_historical_20point.py."))
    result = load_result()
    return result,


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
