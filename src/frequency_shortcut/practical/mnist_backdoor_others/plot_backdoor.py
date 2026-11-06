import marimo

__generated_with = "0.23.1"
app = marimo.App()


@app.cell
def _():
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path

    return Path, pd, plt, sns


@app.cell
def _(plt, sns):
    sns.set_theme(style="whitegrid", font="cmr10", font_scale=1.9)
    plt.rcParams["axes.formatter.use_mathtext"] = True
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    colors = sns.color_palette("dark")
    colors
    return (colors,)


@app.cell
def _(Path):
    # seeds = [0, 1, 2, 3, 4, 5]
    seeds = [6]
    base_path = Path("result")
    csv_filename = "0.0028_5step_0.2subset.csv"
    spuriosities = ["0.7", "None"]
    return base_path, csv_filename, seeds, spuriosities


@app.cell
def _(base_path, csv_filename, pd, seeds, spuriosities):
    def load_all_seeds(base_path, spuriosity, seeds, csv_filename):
        dfs = []
        for seed in seeds:
            csv_path = base_path / spuriosity / f"{seed}_seed" / csv_filename
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                df["seed"] = seed
                dfs.append(df)
                print(f"  ✓ {csv_path}")
            else:
                print(f"  ✗ {csv_path}")
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


    data = {}
    for _spur in spuriosities:
        print(f"[{_spur}]")
        data[_spur] = load_all_seeds(base_path, _spur, seeds, csv_filename)
        print()
    return (data,)


@app.cell
def _(colors, plt, sns):
    _ = colors  # Enforce style-cell execution order in marimo.

    sns.set_theme(style="whitegrid", font="cmr10", font_scale=1.9)
    plt.rcParams["axes.formatter.use_mathtext"] = True
    plt.rcParams.update({
        'text.usetex': False,
        'mathtext.fontset': 'cm',
    })
    return


@app.cell
def _(data, plt, spuriosities):
    labels = {
        "0.7": "70% spurious correlation",
        "None": "10% spurious correlation (Random)",
    }
    plot_colors = {
        "0.7": "blue",
        "None": "red",
    }

    fig, ax = plt.subplots(figsize=(13, 5.3))

    for spur in spuriosities:
        df = data[spur]
        if df.empty:
            print(f"Warning: No data for spuriosity={spur}")
            continue

        # step, intensity, test_acc → intensity, mean/std
        stats = df.groupby("intensity")["test_acc"].agg(["mean", "std"]).reset_index()
        stats["std"] = stats["std"].fillna(0)

        x = stats["intensity"].values
        y_mean = stats["mean"].values
        y_std = stats["std"].values

        ax.errorbar(x, y_mean, yerr=y_std,
                    label=labels[spur],
                    linewidth=3, alpha=0.6,
                    color=plot_colors[spur],
                    capsize=0, capthick=0, elinewidth=5)

    ax.axhline(y=70, color='blue', linestyle='--', alpha=0.6, linewidth=2)
    ax.axhline(y=10, color='red', linestyle='--', alpha=0.6, linewidth=2)

    current_ticks = list(ax.get_yticks())
    ax.set_yticks(sorted(set(current_ticks + [10, 70])))

    ax.set_xlabel('Intensity of trigger pixels')
    ax.set_ylabel('Linear Probing Accuracy')
    ax.legend(loc='lower left')
    ax.grid(True)
    ax.set_xscale('log')

    # ax.text(0.98, 0.02, f"seeds={seeds}",
    #         transform=ax.transAxes, ha='right', va='bottom',
    #         fontsize=11, color='gray', style='italic')

    plt.tight_layout()
    plt.savefig("backdoor_attack_acc_mean_std.pdf", bbox_inches='tight')
    plt.gca()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
