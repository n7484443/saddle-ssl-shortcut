import marimo

__generated_with = '0.19.11'
app = marimo.App(width='full')


@app.cell
def _():
    from pathlib import Path
    plot_dir = Path(__file__).resolve().parent
    import json
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    import torch
    import marimo as mo

    device = torch.device('cuda')
    sns.set_theme(style='whitegrid', font='cmr10', font_scale=1.9)
    plt.rcParams['axes.formatter.use_mathtext'] = True
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'pdf.use14corefonts': False})
    colors = sns.color_palette()
    return colors, json, mo, np, plot_dir, plt


@app.cell
def _(json, np, plot_dir):
    source = plot_dir / 'comparison.json'
    source_bytes = source.read_bytes()
    comparison = json.loads(source_bytes)
    steps = np.array([row['step'] for row in comparison])
    group_keys = ['CJ0_sat0_hue0', 'CJ0_sat005_hue005']
    means = np.empty((2, 61, 2))
    stds = np.empty((2, 61, 2))
    for group_index, group_key in enumerate(group_keys):
        for step_index, row in enumerate(comparison):
            for task_index, task in enumerate(['faint', 'vivid']):
                entry = row[group_key][task]
                means[group_index, step_index, task_index] = float(entry['mean'])
                stds[group_index, step_index, task_index] = float(entry['sample_SD'])
    return means, stds, steps


@app.cell
def _(colors, means, np, plot_dir, plt, stds, steps):
    def plot_cj0_augmentation_comparison():
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), sharex=True, sharey=True, layout='constrained')
        fig.get_layout_engine().set(w_pad=0.13, h_pad=0.08, wspace=0.07)
        lower_limit = min(0, 5 * np.floor(float((means - stds).min()) / 5))
        upper_limit = max(108, 5 * np.ceil(float((means + stds).max()) / 5))
        labels = ['BC', 'BC+SH']
        for task_index, ax in enumerate(axes):
            for group_index, label in enumerate(labels):
                mean = means[group_index, :, task_index]
                sd = stds[group_index, :, task_index]
                ax.fill_between(steps, mean - sd, mean + sd, color=colors[group_index], alpha=0.20, linewidth=0)
                ax.plot(steps, mean, color=colors[group_index], linewidth=2.8, label=label)
            ax.set_title(['Faint', 'Vivid'][task_index], pad=14, fontsize=24)
            ax.set(xlabel='Step', xlim=(0, 30000), ylim=(lower_limit, upper_limit))
            ax.spines[['top', 'right']].set_visible(False)
            ax.legend(loc='lower right', fontsize=19, frameon=True)
        axes[0].set_ylabel('Linear Probing Accuracy')
        for extension in ['png', 'pdf', 'svg']:
            fig.savefig(plot_dir / f'faint_vivid_colorjitter.{extension}', bbox_inches='tight', pad_inches=0.25)
        return fig

    figure = plot_cj0_augmentation_comparison()
    figure
    return


@app.cell
def _(mo):
    mo.md('Lines: mean over all ten seeds, 23–32. Shading: +/- one sample SD (ddof=1). S=saturation; H=hue. Values are saved every 500 steps from step 0, without smoothing, resampling, or seed exclusion.')
    return


if __name__ == '__main__':
    app.run()
