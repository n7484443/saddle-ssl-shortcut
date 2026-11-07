import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path
    import os
    HERE = Path(__file__).resolve().parent
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(HERE / "mpl_config"))
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    import seaborn as sns
    import marimo as mo
    device = torch.device("cuda")
    sns.set_theme(style="whitegrid", font="cmr10", font_scale=1.9)
    plt.rcParams["axes.formatter.use_mathtext"] = True
    plt.rcParams.update({"mathtext.fontset": "cm", "pdf.fonttype": 42, "ps.fonttype": 42, "pdf.use14corefonts": False})
    colors = sns.color_palette()
    return HERE, colors, mo, np, os, plt


@app.cell
def _(np, HERE):
    run_dirs = {
        "DLN": HERE / "dln",
        "MLP": HERE / "mlp",
    }
    losses = {}
    for _name, _folder in run_dirs.items():
        _cache = _folder / "traces.npz"
        with np.load(_cache, allow_pickle=False) as _z:
            losses[_name] = _z["loss"].copy()
    return losses,


@app.cell
def _(mo):
    mo.md("""# Figure 8(c): linear alternative

    Cached four-layer DLN + SimCLR loss; no training is performed here.
    The comparison uses existing runs with different initialization scales
    (DLN 4e-9; ReLU MLP 4e-8), so transition-time differences are not an
    activation-only controlled comparison. These old runs constructed Linear
    parameters on CUDA; this redraw is not a new CPU-initialized experiment.
    """)
    return


@app.cell
def _(HERE, colors, losses, np, os, plt):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    dln_line, = ax.plot(np.arange(len(losses["DLN"])), losses["DLN"], color=colors[2], linewidth=2, alpha=0.7)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    fig.tight_layout()
    output_dir = HERE
    if os.environ.get("FIG8C_EXPORT") == "1":
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / "dim_loss.pdf", bbox_inches="tight", pad_inches=0.12)
        fig.savefig(output_dir / "dim_loss.png", bbox_inches="tight", pad_inches=0.12)
    fig
    return output_dir,


@app.cell
def _(colors, losses, np, os, output_dir, plt):
    comparison, axs = plt.subplots(1, 2, figsize=(12.8, 4.4))
    for _ax, _name, _title in zip(axs, ["DLN", "MLP"], ["DLN", "MLP (ReLU)"]):
        _line, = _ax.plot(np.arange(len(losses[_name])), losses[_name], color=colors[2], linewidth=2, alpha=0.7)
        _ax.set_xlabel("Step")
        _ax.set_ylabel("Loss")
        _ax.set_title(_title)
    comparison.tight_layout()
    if os.environ.get("FIG8C_EXPORT") == "1":
        comparison.savefig(output_dir / "comparison.pdf", bbox_inches="tight", pad_inches=0.12)
        comparison.savefig(output_dir / "comparison.png", bbox_inches="tight", pad_inches=0.12)
    comparison
    return


if __name__ == "__main__":
    app.run()
