import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path
    import numpy as np
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
    return LogLocator, NullFormatter, colors, np, plt, root


@app.cell
def _(np, root):
    with np.load(root / "initial_eigenvalues.npz", allow_pickle=False) as data:
        slopes = data["slopes"].copy()
        depths = data["depths"].copy()
        values = data["top_eigenvalues"].copy()
    mean = values.mean(axis=0)
    std = values.std(axis=0, ddof=1)
    lower, upper = mean - std, mean + std
    eligible = np.all(lower > 0, axis=1)
    y_limits = [min(mean.min() * .65, lower[eligible].min() * .65),
                max(mean.max() * 1.3, upper[eligible].max() * 1.3)]
    return depths, eligible, lower, mean, slopes, upper, y_limits


@app.cell
def _(LogLocator, NullFormatter, colors, np, plt):
    def plot_five_seed_depths(
        slopes, depths, mean, lower, upper, eligible, y_limits, show_sd=False,
    ):
        """Plot all means, optionally adding full SD bands at eligible depths."""
        fig, ax = plt.subplots(figsize=(12, 9), layout="constrained")
        if show_sd:
            for index, depth in enumerate(depths):
                if eligible[index]:
                    ax.fill_between(
                        slopes, lower[index], upper[index],
                        facecolor=colors[index], alpha=0.15, linewidth=0,
                        zorder=1, rasterized=False,
                    )
        for index, depth in enumerate(depths):
            label = f"Depth {int(depth)}"
            if show_sd and eligible[index]:
                label += r" $\pm$ 1 SD"
            ax.plot(
                slopes, mean[index], color=colors[index], linewidth=2.3,
                label=label, zorder=3,
            )
        ax.set_yscale("log")
        ax.set_ylim(*y_limits)
        ax.set_xlim(-0.02, 1.02)
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_xlabel("LeakyReLU slope", labelpad=12)
        ax.set_ylabel(r"Initial top eigenvalue $\lambda_1(C)$", labelpad=12)
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, which="major", linewidth=0.8, alpha=0.65)
        ax.grid(False, which="minor")
        ax.tick_params(axis="both", pad=8)
        ax.legend(
            loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0,
            frameon=True, borderpad=0.55, labelspacing=0.4,
        )
        return fig, ax
    return plot_five_seed_depths,


@app.cell
def _(depths, eligible, lower, mean, plot_five_seed_depths, root, slopes, upper, y_limits):
    figures = []
    for show_sd, name in [(False, "leaky_relu_initial_eigenvalue"),
                          (True, "leaky_relu_initial_eigenvalue_std")]:
        fig, ax = plot_five_seed_depths(slopes, depths, mean, lower, upper, eligible, y_limits, show_sd)
        fig.savefig(root / f"{name}.pdf", bbox_inches="tight", pad_inches=.22)
        figures.append(fig)
    figures
    return


if __name__ == "__main__":
    app.run()
