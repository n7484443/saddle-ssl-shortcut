import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import torch
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, NullFormatter
    import seaborn as sns
    device = torch.device("cuda")
    sns.set_theme(style="whitegrid", font="cmr10", font_scale=1.9)
    plt.rcParams["axes.formatter.use_mathtext"] = True
    plt.rcParams.update({"mathtext.fontset": "cm", "pdf.fonttype": 42, "ps.fonttype": 42})
    colors = sns.color_palette()
    root = Path(__file__).resolve().parent
    return LogLocator, NullFormatter, colors, np, pd, plt, root


@app.cell
def _(np, root):
    with np.load(root / "heatmap_plot_data.npz", allow_pickle=False) as data:
        values = data["values"].copy()
        x_edges = data["x_edges"].copy()
        y_edges = data["y_edges"].copy()
    return values, x_edges, y_edges


@app.cell
def _(plt):
    def draw_original_heatmap(values, x_edges, y_edges):
        fig, ax = plt.subplots(figsize=(14, 10))
        im = ax.pcolormesh(x_edges, y_edges, values, cmap="RdBu_r", vmin=-60, vmax=60)
        ax.set_xlabel("Step")
        ax.set_ylabel("Object Ratio")
        colorbar = fig.colorbar(im, ax=ax, label="Top-5 Spuriosity")
        # Source's `cbar.ax.tick_params(12)` targeted an unrelated older figure:
        fig.tight_layout()
        return fig, ax, colorbar
    return draw_original_heatmap,


@app.cell
def _(draw_original_heatmap, root, values, x_edges, y_edges):
    figure, axes, colorbar = draw_original_heatmap(values, x_edges, y_edges)
    figure.savefig(root / "spuriosity_orig.pdf")
    figure
    return axes, colorbar, figure


if __name__ == "__main__":
    app.run()
