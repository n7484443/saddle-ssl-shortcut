import marimo

__generated_with = "0.13.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import seaborn as sns

    import marimo as mo

    sns.set_theme(style="whitegrid", context="notebook")

    # Point this at the directory holding result/<spur>/<seed>_seed/grid_*/
    ROOT = Path(os.environ.get("GRID_RESULT_DIR", "result"))
    return ROOT, mo, np, pl, plt, sns


@app.cell
def _(ROOT, mo, pl):
    files = sorted(ROOT.rglob("grid_long.parquet"))
    mo.stop(
        not files,
        mo.md(f"No `grid_long.parquet` under `{ROOT.resolve()}`. "
              "Set `GRID_RESULT_DIR` to the right directory."),
    )

    df = pl.concat(
        [pl.read_parquet(f).with_columns(src=pl.lit(str(f.parent))) for f in files],
        how="vertical_relaxed",
    )

    seeds = sorted(df["seed"].unique().to_list())
    mo.md(
        f"**{len(files)}** files &nbsp;|&nbsp; **{df.height}** rows &nbsp;|&nbsp; "
        f"seeds **{seeds}** &nbsp;|&nbsp; "
        f"sizes **{sorted(df['size_mult'].unique().to_list())}** &nbsp;|&nbsp; "
        f"intensities **{sorted(df['intensity'].unique().to_list())}**"
    )
    return df,


@app.cell
def _(df, pl):
    # Aggregate over seeds
    agg = (
        df.group_by(["spuriosity", "size_mult", "trigger_len", "intensity", "step"])
        .agg(
            acc=pl.col("test_acc").mean(),
            std=pl.col("test_acc").std().fill_null(0.0),
            n_seeds=pl.col("seed").n_unique(),
        )
        .with_columns(
            size_label=pl.format("{}x ({}px)", pl.col("size_mult"), pl.col("trigger_len"))
        )
        .sort(["spuriosity", "size_mult", "intensity", "step"])
    )
    return agg,


@app.cell
def _(df, mo):
    spur_opts = sorted(df["spuriosity"].unique().to_list())
    step_opts = sorted(df["step"].unique().to_list())

    sel_spur = mo.ui.dropdown(
        {s: s for s in spur_opts}, value=spur_opts[0], label="spuriosity"
    )
    sel_step = mo.ui.dropdown(
        {str(s): s for s in step_opts}, value=str(step_opts[-1]), label="step"
    )

    shown = ([sel_spur] if len(spur_opts) > 1 else []) + (
        [sel_step] if len(step_opts) > 1 else []
    )
    mo.hstack(shown, justify="start") if shown else mo.md("")
    return sel_spur, sel_step, step_opts


@app.cell
def _(agg, pl, sel_spur, sel_step):
    # Axis order and the selected-step slice, shared by the figures below
    view = agg.filter(pl.col("spuriosity") == sel_spur.value)
    sizes = view.sort("size_mult").select("size_mult", "size_label").unique(
        subset="size_mult", maintain_order=True
    )
    size_mults = sizes["size_mult"].to_list()
    size_labels = sizes["size_label"].to_list()
    intensities = sorted(view["intensity"].unique().to_list())
    cut = view.filter(pl.col("step") == sel_step.value)
    return cut, intensities, size_labels, size_mults, view


@app.cell
def _(cut, intensities, np, plt, sel_spur, sel_step, size_labels, size_mults, sns):
    # rows = trigger size, cols = intensity
    acc = np.full((len(size_mults), len(intensities)), np.nan)
    ann = np.full((len(size_mults), len(intensities)), "", dtype=object)
    for _r in cut.iter_rows(named=True):
        _i = size_mults.index(_r["size_mult"])
        _j = intensities.index(_r["intensity"])
        acc[_i, _j] = _r["acc"]
        ann[_i, _j] = (
            f"{_r['acc']:.1f}\n±{_r['std']:.1f}" if _r["n_seeds"] > 1 else f"{_r['acc']:.1f}"
        )

    fig_hm, ax_hm = plt.subplots(
        figsize=(1.05 * len(intensities) + 2.0, 0.62 * len(size_mults) + 1.4)
    )
    ax_hm.set_facecolor("#e9e9e9")
    sns.heatmap(
        acc, mask=np.isnan(acc), annot=ann, fmt="", cmap="viridis",
        linewidths=0.5, linecolor="white",
        xticklabels=[f"{v:g}" for v in intensities], yticklabels=size_labels,
        cbar_kws={"label": "test acc (%)"}, annot_kws={"fontsize": 9}, ax=ax_hm,
    )
    for _i, _j in zip(*np.where(np.isnan(acc))):
        ax_hm.text(_j + 0.5, _i + 0.5, "—", ha="center", va="center", color="#888888")
    ax_hm.set_xlabel("intensity")
    ax_hm.set_ylabel("trigger size")
    ax_hm.set_title(f"spuriosity={sel_spur.value}  step={sel_step.value}", fontsize=11, pad=8)
    fig_hm.tight_layout()
    fig_hm
    return


@app.cell
def _(cut, intensities, pl, plt, sel_spur, sel_step, size_labels, size_mults, sns):
    # Same numbers as the heatmap, read as two marginal profiles
    fig_p, (ax_i, ax_s) = plt.subplots(1, 2, figsize=(9.2, 3.4), sharey=True)
    pal_size = sns.color_palette("viridis", len(size_mults))
    pal_int = sns.color_palette("rocket", len(intensities))

    for _c, _sm, _lab in zip(pal_size, size_mults, size_labels):
        _s = cut.filter(pl.col("size_mult") == _sm).sort("intensity")
        if _s.is_empty():
            continue
        _x = [intensities.index(v) for v in _s["intensity"].to_list()]
        ax_i.errorbar(_x, _s["acc"].to_numpy(), yerr=_s["std"].to_numpy(),
                      marker="o", ms=5, capsize=3, color=_c, label=_lab)
    ax_i.set_xticks(range(len(intensities)))
    ax_i.set_xticklabels([f"{v:g}" for v in intensities])
    ax_i.set_xlabel("intensity")
    ax_i.set_ylabel("test acc (%)")
    ax_i.legend(title="size", fontsize=8, title_fontsize=8)

    for _c, _it in zip(pal_int, intensities):
        _s = cut.filter(pl.col("intensity") == _it).sort("size_mult")
        if _s.is_empty():
            continue
        _x = [size_mults.index(v) for v in _s["size_mult"].to_list()]
        ax_s.errorbar(_x, _s["acc"].to_numpy(), yerr=_s["std"].to_numpy(),
                      marker="o", ms=5, capsize=3, color=_c, label=f"{_it:g}")
    ax_s.set_xticks(range(len(size_mults)))
    ax_s.set_xticklabels(size_labels)
    ax_s.set_xlabel("trigger size")
    ax_s.legend(title="intensity", fontsize=8, title_fontsize=8)

    fig_p.suptitle(f"spuriosity={sel_spur.value}  step={sel_step.value}", fontsize=11)
    fig_p.tight_layout()
    fig_p
    return


@app.cell
def _(intensities, mo, pl, plt, sel_spur, size_labels, size_mults, sns, step_opts, view):
    mo.stop(len(step_opts) < 2)  # nothing to plot over time

    ncols = min(2, len(intensities))
    nrows = -(-len(intensities) // ncols)
    fig_c, axes = plt.subplots(
        nrows, ncols, figsize=(5.2 * ncols, 3.4 * nrows),
        sharex=True, sharey=True, squeeze=False,
    )
    palette = sns.color_palette("viridis", len(size_mults))

    for _ax, _inten in zip(axes.flat, intensities):
        for _c, _sm, _lab in zip(palette, size_mults, size_labels):
            _line = view.filter(
                (pl.col("intensity") == _inten) & (pl.col("size_mult") == _sm)
            ).sort("step")
            if _line.is_empty():
                continue
            _step = _line["step"].to_list()
            _mean = _line["acc"].to_numpy()
            _sd = _line["std"].to_numpy()
            _ax.plot(_step, _mean, marker="o", ms=4, color=_c, label=_lab)
            _ax.fill_between(_step, _mean - _sd, _mean + _sd, color=_c, alpha=0.15, lw=0)
        _ax.set_title(f"intensity = {_inten:g}", fontsize=11)
        _ax.set_xlabel("step")
        _ax.set_ylabel("test acc (%)")

    for _ax in axes.flat[len(intensities):]:
        _ax.set_visible(False)

    axes.flat[0].legend(title="size", fontsize=9, title_fontsize=9)
    fig_c.suptitle(f"spuriosity={sel_spur.value}", y=1.0)
    fig_c.tight_layout()
    fig_c
    return


if __name__ == "__main__":
    app.run()