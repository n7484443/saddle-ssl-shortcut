import marimo

__generated_with = '0.23.16'
app = marimo.App(width='full')


@app.cell
def _():
    import marimo as mo
    import json
    import numpy as np
    from pathlib import Path
    import matplotlib.pyplot as plt
    root = Path(mo.notebook_dir())
    return json, mo, np, plt, root


@app.cell(hide_code=True)
def _(mo):
    mo.md('''
    # ImageNet: theory-aligned controls

    Cache-only viewer: no image decoding or GPU computation.
    Common-support amplitude=2 and coordinate-repetition extent=1 are algebraic
    controls, not independent natural-image findings. Original and shuffled
    backgrounds use a fixed paired subset. Extent uses a separate four-point grid.
    ''')
    refresh = mo.ui.run_button(label='Refresh saved results')
    refresh
    return (refresh,)


@app.cell
def _(json, mo, refresh, root):
    refresh.value
    _path = root / 'cache/theory_control_results.json'
    mo.stop(not _path.exists(), mo.md('The structural-control study has not finished yet.'))
    result = json.loads(_path.read_text(encoding='utf-8'))
    return (result,)


@app.cell(hide_code=True)
def _(mo, result):
    _rows = [dict(condition=name, quantity=s['quantity'], exponent=s['mean_class_curve_exponent'],
                  median_r2=s['median_r2'], conditional_class_bootstrap_95=str(s['conditional_class_bootstrap_95']),
                  classes=len(s['class_exponents'])) for name,s in result['summary'].items()]
    mo.ui.table(_rows, selection=None)
    return


@app.cell(hide_code=True)
def _(mo, np, plt, result):
    _fig, _axes = plt.subplots(1,2,figsize=(12,4.6),constrained_layout=True)
    for _name,_s in result['summary'].items():
        _ax = _axes[0] if _s['quantity']=='amplitude' else _axes[1]
        _curve = np.exp(np.log(np.asarray(_s['class_median_curves'])).mean(axis=1))
        _ax.loglog(_s['x'],_curve,'o-',label=f"{_name}: {_s['mean_class_curve_exponent']:.3f}",markersize=3)
    for _ax,_x,_theory,_label in zip(_axes,(np.linspace(.2,2,10),np.array([1,4,9,16])/16),(2,1),('Amplitude','Extent')):
        _ax.loglog(_x,_x**_theory,'k:',label=f'Theory {_theory}')
        _ax.set(xlabel=_label,ylabel='Normalized class-median strength')
        _ax.grid(True,which='both',alpha=.2); _ax.legend(fontsize=7)
    mo.vstack([mo.md('Curves: geometric means of per-class normalized median curves. Confidence intervals condition on the selected classes and do not correct for exploratory selection.'), _fig])
    return


@app.cell(hide_code=True)
def _(mo, result, root):
    _preview = root / 'cache/theory_controls' / result['key'] / 'example_components.png'
    mo.vstack([mo.md('Example preprocessing. RGB clipping is only for display.'), mo.image(str(_preview))])
    return


if __name__ == '__main__':
    app.run()
