import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import json
    import sys
    import numpy as np
    from pathlib import Path
    sys.path.insert(0, str(mo.notebook_dir()))
    from scaling_cache import summarize, plot_result
    root = Path(mo.notebook_dir())
    return json, mo, np, plot_result, root, summarize


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # ImageNet train: cached scaling results

    This notebook reads completed per-class spectra from disk. Plotting, changing
    K, and comparing coefficient aggregations do not decode images or use the GPU.
    The background pipeline saves every predeclared condition, including results
    that differ from the rebuttal. Reference exponents: amplitude 1.59, extent 1.12.
    """)
    refresh = mo.ui.run_button(label="Refresh completed results")
    refresh
    return (refresh,)


@app.cell
def _(json, mo, refresh, root):
    refresh.value
    _path = root / "cache/train_study_results.json"
    _status_path = root / "train_scaling_status.json"
    _state = json.loads(_status_path.read_text(encoding="utf-8")) if _status_path.exists() else {}
    mo.stop(not _path.exists(), mo.md(f"No completed run yet. Pipeline state: **{_state.get('status', 'not started')}**. Refresh after the first run finishes."))
    outcomes = json.loads(_path.read_text(encoding="utf-8"))
    mo.stop(not outcomes, mo.md("No completed runs yet."))
    return (outcomes,)


@app.cell(hide_code=True)
def _(mo, outcomes):
    run_choice = mo.ui.dropdown([r["name"] for r in outcomes], value=outcomes[0]["name"], label="Saved condition")
    modes = mo.ui.dropdown([100, 50, 20, 10], value=100, label="Maximum K (capped by class sample count)")
    aggregation = mo.ui.dropdown([
        "Mean exponent of class-median curves",
        "Mean of class median mode exponents",
        "Median of all class-mode exponents (original code)",
    ], value="Mean exponent of class-median curves", label="Coefficient aggregation")
    mo.vstack([run_choice, modes, aggregation])
    return aggregation, modes, run_choice


@app.cell
def _(modes, np, outcomes, root, run_choice, summarize):
    chosen = next(r for r in outcomes if r["name"] == run_choice.value)
    directory = root / "cache/train_runs" / chosen["key"]
    _arrays = {}
    for _cls in chosen["mode_counts"]:
        with np.load(directory / f"{_cls}.npz", allow_pickle=False) as _saved:
            _arrays[_cls] = {k: _saved[k][:, :modes.value] for k in _saved.files}
    _summary, _curves = summarize(_arrays, chosen["config"])
    result = dict(chosen, directory=directory, summary=_summary, curves=_curves)
    return chosen, directory, result


@app.cell(hide_code=True)
def _(aggregation, chosen, directory, mo, result):
    _fields = {
        "Mean exponent of class-median curves": "mean_group_median_curve_exponent",
        "Mean of class median mode exponents": "mean_group_median_mode_exponent",
        "Median of all class-mode exponents (original code)": "pooled_mode_median_exponent",
    }
    _rows = [{"quantity": label, "exponent": stats[_fields[aggregation.value]],
              "theory": 2 if label == "amplitude" else 1,
              "class-median curve R² (median)": stats["median_group_curve_r2"],
              "classes": stats["groups"], "modes": stats["total_modes"]}
             for label, stats in result["summary"].items()]
    mo.vstack([
        mo.md(f"**{chosen['audit']['computed_images']:,} images / {chosen['audit']['computed_groups']} classes** · normalization `{chosen['config']['normalization']}` · area `{chosen['config']['area']}`\n\nCache: `{directory}`"),
        mo.ui.table(_rows, selection=None),
    ])
    return


@app.cell(hide_code=True)
def _(mo, modes, plot_result, result):
    from matplotlib.ticker import NullFormatter, ScalarFormatter
    _figure = plot_result(result)
    for _axis, _ticks in zip(_figure.axes, ([.2, .5, 1, 2], [.2, .5, 1])):
        _axis.set_xticks(_ticks)
        _axis.xaxis.set_major_formatter(ScalarFormatter())
        _axis.xaxis.set_minor_formatter(NullFormatter())
    _figure.suptitle(f"TRAIN · {result['name']} · K ≤ {modes.value}")
    _figure.tight_layout()
    mo.vstack([
        mo.md("The plot shows the geometric mean of normalized class-median curves. Its fitted slope corresponds to the mean class-curve exponent; changing the table aggregation does not change the plotted curve."),
        _figure,
    ])
    return


if __name__ == "__main__":
    app.run()
