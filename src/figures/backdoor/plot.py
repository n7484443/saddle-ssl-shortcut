import marimo

__generated_with = "0.17.7"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    import torch

    device = torch.device("cuda")
    sns.set_theme(style="whitegrid", font="cmr10", font_scale=1.9)
    plt.rcParams["axes.formatter.use_mathtext"] = True
    plt.rcParams["text.usetex"] = False
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["pdf.use14corefonts"] = False
    colors = sns.color_palette()
    return Path, pd, plt


@app.cell
def _(Path, pd):
    notebook_dir = Path(__file__).resolve().parent
    seeds = [0, 1, 2, 3, 4, 5]
    spuriosities = ["0.7", "None"]
    csv_filename = "0.0028_5step_0.2subset.csv"
    data = {}
    for _spur in spuriosities:
        _frames = []
        for _seed in seeds:
            _relative = f"data/{_spur}/{_seed}_seed/{csv_filename}"
            _path = notebook_dir / _relative
            _frame = pd.read_csv(_path)
            _frame["seed"] = _seed
            _frames.append(_frame)
        data[_spur] = pd.concat(_frames, ignore_index=True)
    return data, notebook_dir, spuriosities


@app.cell
def _(data, notebook_dir, plt, spuriosities):
    def plot_backdoor_mean_std(data_by_spur, destination):
        # axes, labels, explicit line colors, legend and error-bar styling.
        labels = {
            "0.7": "70% spurious correlation",
            "None": "10% spurious correlation (Random)",
        }
        curve_colors = {"0.7": "blue", "None": "red"}
        fig, ax = plt.subplots(figsize=(13, 5.3))
        for spur in spuriosities:
            stats = data_by_spur[spur].groupby("intensity")["test_acc"].agg(["mean", "std"]).reset_index()
            stats["std"] = stats["std"].fillna(0)
            ax.errorbar(stats["intensity"].values, stats["mean"].values,
                        yerr=stats["std"].values, label=labels[spur],
                        linewidth=3, alpha=0.6, color=curve_colors[spur],
                        capsize=0, capthick=0, elinewidth=5)
        ax.axhline(y=70, color="blue", linestyle="--", alpha=0.6, linewidth=2)
        ax.axhline(y=10, color="red", linestyle="--", alpha=0.6, linewidth=2)
        current_ticks = list(ax.get_yticks())
        ax.set_yticks(sorted(set(current_ticks + [10, 70])))
        ax.set_xlabel("Intensity of trigger pixels")
        ax.set_ylabel("Linear Probing Accuracy")
        ax.legend(loc="lower left")
        ax.grid(True)
        ax.set_xscale("log")
        fig.tight_layout()
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, format="pdf", bbox_inches="tight")
        fig.savefig(destination.with_suffix(".png"), bbox_inches="tight")
        return fig

    output_pdf = notebook_dir / "staging/freq/backdoor/backdoor_attack_acc_mean_std.pdf"
    figure = plot_backdoor_mean_std(data, output_pdf)
    figure
    return



if __name__ == "__main__":
    app.run()
