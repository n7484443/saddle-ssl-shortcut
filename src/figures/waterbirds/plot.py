import marimo
__generated_with = "0.23.16"
app = marimo.App(width="full")

@app.cell
def _():
    from pathlib import Path
    import json
    import hashlib
    import os
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    device = torch.device("cuda")
    sns.set_theme(style="whitegrid", font="cmr10", font_scale=1.9)
    plt.rcParams["axes.formatter.use_mathtext"] = True
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["pdf.use14corefonts"] = False
    colors_darker = sns.color_palette("dark", 10)
    colors = sns.color_palette("deep", 10)
    out = Path(__file__).resolve().parent
    root = next(p for p in Path(__file__).resolve().parents if (p / "common.py").is_file())
    base = root / "feature_preference/practical/waterbirds"
    stage = out / "stage"
    loaded_files = {}
    figure_receipts = []
    return Path, base, colors, colors_darker, figure_receipts, hashlib, json, loaded_files, np, os, out, plt, root, stage

@app.cell
def _(Path, base, colors, colors_darker, figure_receipts, hashlib, json, loaded_files, np, os, out, plt, root, stage):
    def load_from_csv(path):
        p = Path(path)
        assert p.is_file(), str(p)
        loaded_files[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        a = np.loadtxt(p, delimiter=",")
        assert np.all(np.isfinite(a)), str(p)
        return a

    def load_from_csv_split(path):
        return load_from_csv(path).T

    def tqdm(items):
        return items

    def save_figure(fig, relative):
        dest = stage / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest, format="pdf")
        fig.savefig(dest.with_suffix(".png"))
        arrays = {}
        axes = []
        for aidx, ax in enumerate(fig.axes):
            axes.append(dict(xlim=list(ax.get_xlim()), ylim=list(ax.get_ylim()), xlabel=ax.get_xlabel(), ylabel=ax.get_ylabel()))
            for lidx, line in enumerate(ax.lines):
                arrays[f"a{aidx}_line{lidx}_x"] = np.asarray(line.get_xdata())
                arrays[f"a{aidx}_line{lidx}_y"] = np.asarray(line.get_ydata())
            for cidx, collection in enumerate(ax.collections):
                arrays[f"a{aidx}_collection{cidx}_offsets"] = np.asarray(collection.get_offsets())
        np.savez(out / (dest.stem + "_plotted_arrays.npz"), **arrays)
        figure_receipts.append(dict(relative=relative, figure_size_inches=list(fig.get_size_inches()), axes=axes, output_sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
        plt.close(fig)

    def plot_waterbirds_b():
        ratios = [0.05, 0.075, 0.1, 0.2]
        loss = []
        acc_object = []
        acc_background = []
        steps = []
        eig_list = []
        method = "waterbirds_b"
        directory = f"{base}/datas/BarlowTwins_single_background_0.05"
        print(directory)
        for r in ratios:
            loss_array = load_from_csv(f"{directory}/{r}/loss.csv")
            [acc_object_array, acc_background_array, step_array] = load_from_csv_split(f"{directory}/{r}/accuracy.csv")
            eigvals_lists_array = load_from_csv_split(f"{directory}/{r}/eig.csv")
            loss.append(loss_array)
            acc_object.append(acc_object_array)
            acc_background.append(acc_background_array)
            steps.append(step_array)
            eig_list.append(eigvals_lists_array)
        background_predictability = 70

        fig, axs = plt.subplots(1, 1, figsize=(13, 5.3))

        positions = []
        plateau_range = 3
        for r in range(len(ratios)):
            pos = np.where(acc_object[r] > background_predictability + plateau_range)
            if len(pos[0]) > 0:
                positions.append(pos[0][0])
                axs.axvline(steps[r][positions[-1]], linestyle="--", color=colors[r], linewidth=4, alpha=0.7)
                axs.axvline(steps[r][positions[-1]], linestyle="--", color=colors[r], linewidth=4, alpha=0.7)
            axs.plot(steps[r][:100], acc_object[r][:100], alpha=0.6, linewidth=5, color=colors_darker[r],
                     label=f"Object ratio : {ratios[r]:.3f}")

        upper_y = np.zeros_like(steps[0]) + background_predictability + plateau_range
        lower_y = np.zeros_like(steps[0]) + background_predictability - plateau_range
        axs.fill_between(steps[0], upper_y, lower_y, color='skyblue', alpha=0.4)
        axs.axhline(background_predictability, linestyle="--", color="black", linewidth=2, alpha=0.4)
        axs.set_xlabel("Step")
        axs.set_ylabel("Linear Probing Accuracy")
        margin_x = 500 * 0.02
        margin_y = 20 * 0.1
        axs.set_xlim(0, 500)
        axs.set_ylim(65 - margin_y, 80 + margin_y)
        axs.text(320, 68.5, "background-label correlation")

        axs.legend(ncols=2, loc='lower right')

        plt.tight_layout()
        save_figure(fig, 'dim/practical/modified_waterbird/waterbirds_b.pdf')

    def plot_waterbirds_c():
        ratios = [0.05, 0.1, 0.2, 0.5]
        lambdas = 0.005
        loss = []
        acc_object = []
        acc_background = []
        steps = []
        eig_list = []
        method = "waterbirds_c"
        directory = f"{base}/datas/lambda_0.005_simple_color"
        print(directory)
        for r in ratios:
            loss_array = load_from_csv(f"{directory}/{r}/loss.csv")
            [acc_object_array, acc_background_array, step_array] = load_from_csv_split(f"{directory}/{r}/accuracy.csv")
            eigvals_lists_array = load_from_csv_split(f"{directory}/{r}/eig.csv")
            loss.append(loss_array)
            acc_object.append(acc_object_array)
            acc_background.append(acc_background_array)
            steps.append(step_array)
            eig_list.append(eigvals_lists_array)
        background_predictability = 70

        fig, axs = plt.subplots(1, 1, figsize=(13, 5.3))

        positions = []
        plateau_range = 3
        for r in range(len(ratios)):
            pos = np.where(acc_object[r] > background_predictability + plateau_range)
            if len(pos[0]) > 0:
                positions.append(pos[0][0])
                if r != 0:
                    axs.axvline(steps[r][positions[-1]], linestyle="--", color=colors[r], linewidth=4, alpha=0.7)
                    axs.axvline(steps[r][positions[-1]], linestyle="--", color=colors[r], linewidth=4, alpha=0.7)
            axs.plot(steps[r][:140], acc_object[r][:140], alpha=0.6, linewidth=5, color=colors_darker[r],
                     label=f"Object ratio : {ratios[r]:.3f}")

        upper_y = np.zeros_like(steps[0]) + background_predictability + plateau_range
        lower_y = np.zeros_like(steps[0]) + background_predictability - plateau_range
        axs.fill_between(steps[0], upper_y, lower_y, color='skyblue', alpha=0.4)
        axs.axhline(background_predictability, linestyle="--", color="black", linewidth=2, alpha=0.4)
        axs.set_xlabel("Step")
        axs.set_ylabel("Linear Probing Accuracy")
        margin_x = 1400 * 0.02
        margin_y = 20 * 0.1
        axs.set_xlim(0 - margin_x, 1400 + margin_x)
        axs.set_ylim(65 - margin_y, 80 + margin_y)
        axs.text(520, 68.5, "background-label correlation")

        axs.legend(ncols=2, loc='lower right')

        plt.tight_layout()
        save_figure(fig, 'dim/practical/modified_waterbird/waterbirds_c.pdf')

    def plot_scatter():
        seeds = range(6)
        ratios_scatter = [0.03 + 0.01*i for i in range(7)]
        ratios = [0.03, 0.06, 0.09]
        lambdas = 0.01
        loss = []
        acc_object = []
        acc_background = []
        steps = []
        eig_list = []
        method = "BarlowTwins_multiple_filtered"
        for seed in range(6):
            loss.append([])
            acc_object.append([])
            acc_background.append([])
            steps.append([])
            eig_list.append([])
            directory_models = f"/models/jhkim/waterbirds/{method}_{lambdas}/{seed}"
            directory = f"{base}/datas/{method}_{lambdas}/{seed}"
            for r in ratios:
                loss_array = load_from_csv(f"{directory}/{r}/loss.csv")
                [acc_object_array, acc_background_array, step_array] = load_from_csv_split(f"{directory}/{r}/accuracy.csv")
                eigvals_lists_array = load_from_csv_split(f"{directory}/{r}/eig.csv")
                loss[-1].append(loss_array)
                acc_object[-1].append(acc_object_array)
                acc_background[-1].append(acc_background_array)
                steps[-1].append(step_array)
                eig_list[-1].append(eigvals_lists_array)

        fig, axs = plt.subplots(1, 2, figsize=(28, 10))

        for r_idx, r_val in enumerate(ratios):


            current_steps = steps[0][r_idx]
            obj_acc_np = acc_object[0][r_idx]

            axs[0].plot(current_steps, obj_acc_np, color=colors_darker[r_idx], linewidth=5, alpha=0.9, label=f"Object ratio : {r_val:.3f}")

        axs[0].set_xlabel("Step")
        axs[0].set_ylabel("Test Accuracy")
        axs[0].set_ylim(30, 100)
        axs[0].legend(loc='lower right')

        scatter_data = {ratio: {'steps': [], 'seeds': []} for ratio in ratios_scatter}

        for seed in tqdm(seeds):
            directory = f"{base}/datas/{method}_{lambdas}/{seed}"

            for ratio in ratios_scatter:
                try:
                    [acc_object_array, acc_background_array, step_array] = load_from_csv_split(f"{directory}/{ratio}/accuracy.csv")

                    if len(step_array) > 0:
                        step_at_target = step_array[-1]
                        scatter_data[ratio]['steps'].append(step_at_target)
                        scatter_data[ratio]['seeds'].append(seed)
                except Exception as e:
                    print(f"Error loading seed {seed}, ratio {ratio}: {e}")
                    continue

        means_x = []
        means_y = []
        stds_y = []

        for i, ratio in enumerate(ratios_scatter):
            if len(scatter_data[ratio]['steps']) > 0:
                seeds_list = scatter_data[ratio]['seeds']
                steps_list = scatter_data[ratio]['steps']

                max_seed = max(seeds_list)

                completed_x = []
                completed_y = []
                running_x = []
                running_y = []

                for seed, step in zip(seeds_list, steps_list):
                    completed_x.append(ratio)
                    completed_y.append(step)
                if len(completed_x) > 0:
                    axs[1].scatter(completed_x, completed_y, s=150, alpha=0.25, color=colors[i],)

                if len(completed_y) > 0:
                    steps_arr = np.array(completed_y)
                    mean_step = steps_arr.mean()
                    std_step = steps_arr.std()
                    means_x.append(ratio)
                    means_y.append(mean_step)
                    stds_y.append(std_step)

        if len(means_x) > 0:
            means_x_arr = np.array(means_x)
            means_y_arr = np.array(means_y)
            stds_y_arr = np.array(stds_y)
            axs[1].plot(means_x_arr, means_y_arr, color='red', linewidth=3, alpha=0.8, marker='o')
            axs[1].fill_between(means_x_arr,
                            means_y_arr - stds_y_arr,
                            means_y_arr + stds_y_arr,
                            alpha=0.12, color='red')

        axs[1].set_xlabel('Object Ratio')
        axs[1].set_ylabel('Training step at 80% accuracy')
        axs[1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_figure(fig, 'dim/practical/modified_waterbird/scatter_ratio_vs_steps.pdf')

    plot_waterbirds_b()
    plot_waterbirds_c()
    plot_scatter()
    for path, digest in loaded_files.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    return

if __name__ == "__main__":
    app.run()
