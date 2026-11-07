import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path
    import hashlib
    import json
    import os
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import marimo as mo
    device = torch.device("cuda")
    sns.set_theme(style="whitegrid", font="cmr10", font_scale=1.9)
    plt.rcParams["axes.formatter.use_mathtext"] = True
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["pdf.use14corefonts"] = False
    colors = sns.color_palette()
    original_colors = sns.color_palette("dark")
    here = Path(__file__).resolve().parent
    export_target = os.environ.get("CMNIST_REPLOT_TARGET", "all")
    assert export_target in ("all", "cmnist", "alpha")
    matplotlib_version = matplotlib.__version__
    return export_target, hashlib, here, json, matplotlib_version, mo, np, original_colors, plt, torch


@app.cell
def _(here, np):
    with np.load(here / "plot_cache.npz", allow_pickle=False) as z:
        ratio_values = z["ratio_values"].copy()
        ratio_accuracy = z["ratio_accuracy"].copy()
        ratio_steps = z["ratio_steps"].copy()
        alpha_values = z["alpha_values"].copy()
        alpha_accuracy = z["alpha_accuracy"].copy()
        alpha_steps = z["alpha_steps"].copy()
    assert ratio_accuracy.shape == (3, 28)
    assert alpha_accuracy.shape == (4, 3, 118)
    return alpha_accuracy, alpha_steps, alpha_values, ratio_accuracy, ratio_steps, ratio_values


@app.cell
def _(np, original_colors, plt):
    def plot_cmnist(ratios, acc, step_list):
        background_predictability = 70
        fig, axs = plt.subplots(1, 1, figsize=(13, 5.3))
        positions = []
        ran = 90
        for r in range(len(ratios)):
            pos = np.where(acc[r] >= background_predictability * 1.005)
            if len(pos[0]) > 0:
                positions.append(pos[0][0])
                axs.axvline(step_list[r][positions[-1]], linestyle="--", color=original_colors[r], linewidth=4, alpha=0.7)
            axs.plot(step_list[r][:ran], acc[r][:ran], alpha=0.6, linewidth=5, color=original_colors[r], label=f"Object ratio : {ratios[r]:.3f}")
        axs.text(600, 68.5, "background-label correlation")
        axs.set_xlim(0-1, 5300)
        axs.set_ylim(65, 77.5)
        axs.axhline(background_predictability, linestyle="--", color="black", linewidth=2, alpha=0.6)
        axs.set_xlabel("Step")
        axs.set_ylabel("Linear Probing Accuracy")
        axs.legend()
        plt.tight_layout()
        return fig

    def plot_variable_alpha(alpha_values, accuracy, steps):
        background_predictability = 70
        fig, ax = plt.subplots(1, 1, figsize=(13, 5.3))
        ran, stride = 1800, 2
        statistics = {}
        for original_index in (1, 2, 3):
            alpha = float(alpha_values[original_index])
            acc_stack = accuracy[original_index]
            acc_mean = acc_stack[:, :ran].mean(axis=0)
            acc_std = acc_stack[:, :ran].std(axis=0, ddof=0)
            steps_plot = steps[:ran]
            color = original_colors[original_index]
            ax.plot(steps_plot, acc_mean, alpha=0.7, linewidth=3, color=color, label=f"$\\alpha$ = {alpha}")
            ax.errorbar(steps_plot[::stride], acc_mean[::stride], yerr=acc_std[::stride], fmt="none", ecolor=color, elinewidth=2, capsize=3, capthick=1.5, alpha=0.35)
            crossing = np.where(acc_mean >= background_predictability * 1.005)[0]
            escape_step = None
            if len(crossing) > 0:
                escape_step = float(steps_plot[crossing[0]])
                ax.axvline(escape_step, linestyle="--", color=color, linewidth=4, alpha=0.7)
            statistics[str(alpha)] = {"mean": acc_mean.tolist(), "population_sd_ddof0": acc_std.tolist(), "escape_step": escape_step, "original_palette_index": original_index}
        ax.text(600, 68.5, "background-label correlation")
        ax.set_ylim(65, 77.5)
        ax.axhline(70, linestyle="--", color="black", linewidth=2, alpha=0.6)
        ax.set_xlabel("Step")
        ax.set_ylabel("Linear Probing Accuracy")
        ax.legend(loc="upper left")
        ax.set_xlim(-1755.0, 36855.0)
        fig.tight_layout()
        return fig, statistics
    return plot_cmnist, plot_variable_alpha


@app.cell
def _(alpha_accuracy, alpha_steps, alpha_values, export_target, hashlib, here, json, matplotlib_version, mo, plot_cmnist, plot_variable_alpha, ratio_accuracy, ratio_steps, ratio_values, torch):
    target = here / "stage" / "dim" / "practical" / "cmnist"
    target.mkdir(parents=True, exist_ok=True)
    figure_ratio = plot_cmnist(ratio_values, ratio_accuracy, ratio_steps)
    if export_target in ("all", "cmnist"):
        figure_ratio.savefig(target / "cmnist.pdf", format="pdf")
        figure_ratio.savefig(here / "cmnist.png")
    figure_alpha, alpha_statistics = plot_variable_alpha(alpha_values, alpha_accuracy, alpha_steps)
    if export_target in ("all", "alpha"):
        figure_alpha.savefig(target / "cmnist_variable_alpha.pdf", bbox_inches="tight", pad_inches=0.1)
        figure_alpha.savefig(here / "cmnist_variable_alpha.png", bbox_inches="tight", pad_inches=0.1)
    record = {"font_settings": {"pdf.fonttype": 42, "ps.fonttype": 42, "pdf.use14corefonts": False}, "no_training": True, "cuda_initialized": torch.cuda.is_initialized(), "ratio_graph": {"ratios": ratio_values.tolist(), "accuracy": ratio_accuracy.tolist(), "steps": ratio_steps.tolist(), "x_limits": [-1, 5300], "y_limits": [65,77.5], "source": "feature_preference/practical/colored_mnist/cmnist_plot.ipynb cell4", "note": "NeurIPS cell has the xlim matching the original PDF; the newer ICLR cell comments it out. Both contain the same plotting operations and same 28-point raw data."}, "alpha_graph": {"statistics": alpha_statistics, "retained_alphas": [.05,.1,.2], "x_limits": [-1755,36855], "y_limits": [65,77.5]}, "cache_sha256": hashlib.sha256((here / "plot_cache.npz").read_bytes()).hexdigest()}
    assert not record["cuda_initialized"]
    record["export_target"] = export_target
    record["matplotlib_version"] = matplotlib_version
    (here / "export_record.json").write_text(json.dumps(record, indent=2)+"\n", encoding="utf8")
    (here / f"export_record_{export_target}.json").write_text(json.dumps(record, indent=2)+"\n", encoding="utf8")
    mo.mpl.interactive(figure_alpha)
    return


if __name__ == "__main__":
    app.run()
