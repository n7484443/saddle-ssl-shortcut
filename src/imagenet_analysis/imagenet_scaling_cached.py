import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import sys
    sys.path.insert(0, str(mo.notebook_dir()))
    from scaling_cache import run_cached, plot_result
    return mo, plot_result, run_cached


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # ImageNet validation: amplitude / extent scaling

    Images: `Imagenet/val` · XML: `Imagenet/bbox/val` · persistent results: `Imagenet/cache`.

    **pooled** mixes the selected classes into one group, matching the original
    notebook's flat-val-folder behavior. **classwise** groups images by the XML
    object class, and uses all available modes up to K. These are different protocols.

    With the 45–55% single-bbox filter, a validation class has at most 13 images;
    class-wise top-100 nonzero modes are therefore unavailable. Classwise mode
    explicitly keeps classes with at least two selected images, rather than the
    original train-oriented minimum of 20.

    Normalization is fixed RGB mean/std within each computation group. Amplitudes
    are 0.2–2.0 (10 points); extents are 0.2–1.0 (9 points). The original numerical
    kernel is reused. Sampling is deterministic and image IDs are saved.

    Submit the configuration to **load a matching disk cache or compute missing groups**.
    Changing graph presentation or aggregation below never recomputes image statistics.
    Data file metadata, selected conditions, and numerical code versions identify the cache.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    settings = mo.ui.dictionary({
        "area": mo.ui.range_slider(0.0, 1.0, value=[0.45, 0.55], step=0.01, label="Object area fraction after crop"),
        "grouping": mo.ui.dropdown(["pooled", "classwise"], value="pooled", label="Grouping"),
        "images_per_group": mo.ui.number(0, 4000, value=300, step=1, label="Maximum images/group (0 = all)"),
        "min_class_images": mo.ui.number(2, 50, value=2, step=1, label="Minimum images per actual class"),
        "k": mo.ui.number(1, 300, value=100, step=1, label="Maximum modes K"),
        "resolution": mo.ui.dropdown([28, 56, 112, 224], value=224, label="Spectral resolution"),
        "align": mo.ui.switch(value=True, label="Center bbox"),
        "seed": mo.ui.number(0, 10000, value=0, step=1, label="Subsampling seed"),
    }).form(submit_button_label="Load cache / compute")
    settings
    return (settings,)


@app.cell
def _(mo, run_cached, settings):
    mo.stop(settings.value is None, mo.md("Select conditions and press **Load cache / compute**."))
    with mo.status.spinner(title="Loading cache or computing missing spectra"):
        result = run_cached(settings.value)
    return (result,)


@app.cell(hide_code=True)
def _(mo, result):
    mo.md(f"""
    **Run `{result['key']}`** · {result['audit']['computed_images']:,} images ·
    {result['audit']['computed_groups']:,} groups ·
    **{result['group_cache_hits']:,} cache hits / {result['computed_now']:,} newly computed**

    Directory: `{result['directory']}`. Protocol, selected image IDs, arrays and summaries are saved there.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    aggregation = mo.ui.dropdown([
        "Mean exponent of group-median curves",
        "Mean of within-group median mode exponents",
        "Median of all group-mode exponents (original code)",
    ], value="Mean exponent of group-median curves", label="Coefficient aggregation (no recomputation)")
    aggregation
    return (aggregation,)


@app.cell(hide_code=True)
def _(aggregation, mo, result):
    _fields = {
        "Mean exponent of group-median curves": "mean_group_median_curve_exponent",
        "Mean of within-group median mode exponents": "mean_group_median_mode_exponent",
        "Median of all group-mode exponents (original code)": "pooled_mode_median_exponent",
    }
    _field = _fields[aggregation.value]
    _rows = [{"quantity": name, "exponent": stats[_field],
              "theory": 2 if name == "amplitude" else 1,
              "groups": stats["groups"], "total_modes": stats["total_modes"]}
             for name, stats in result["summary"].items()]
    mo.ui.table(_rows, selection=None)
    return


@app.cell(hide_code=True)
def _(mo, plot_result, result):
    from matplotlib.ticker import NullFormatter, ScalarFormatter
    _figure = plot_result(result)
    for _axis, _ticks in zip(_figure.axes, ([0.2, 0.5, 1.0, 2.0], [0.2, 0.5, 1.0])):
        _axis.set_xticks(_ticks)
        _axis.xaxis.set_major_formatter(ScalarFormatter())
        _axis.xaxis.set_minor_formatter(NullFormatter())
    _figure.tight_layout()
    mo.vstack([
        mo.md("The plot always shows the geometric mean of group-median curves; the coefficient table above also exposes the other aggregation definitions."),
        _figure,
    ])
    return


if __name__ == "__main__":
    app.run()
