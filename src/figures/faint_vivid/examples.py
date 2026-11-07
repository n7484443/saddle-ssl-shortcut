import marimo

__generated_with = '0.23.1'
app = marimo.App(width='full')


@app.cell
def _():
    from pathlib import Path
    plot_dir = Path(__file__).resolve().parent
    workspace = next(p for p in Path(__file__).resolve().parents if (p / "common.py").is_file())
    import csv
    import random
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    import torch
    from PIL import Image
    device = torch.device('cuda')
    sns.set_theme(style='whitegrid', font='cmr10', font_scale=1.9)
    plt.rcParams['axes.formatter.use_mathtext'] = True
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'pdf.use14corefonts': False})
    colors = sns.color_palette()
    return Image, csv, plot_dir, plt, random


@app.cell
def _(Image, csv, plot_dir, plt, random):
    dataset = plot_dir / 'examples'
    manifest = dataset / 'manifest.csv'
    with manifest.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    selected = random.Random(0).sample(range(len(rows)), 8)

    def plot_dataset_examples():
        fig, axes = plt.subplots(4, 4, figsize=(10.8, 11.2))
        fig.subplots_adjust(left=0.025, right=0.975, bottom=0.025, top=0.93, wspace=0.055, hspace=0.055)
        for row_index in range(4):
            for column_index in range(4):
                condition = 0 if column_index < 2 else 1
                selected_index = selected[2 * row_index + column_index % 2]
                record = rows[selected_index]
                path = dataset / record[f'cond{condition}_image']
                with Image.open(path) as original:
                    tile = original.convert('RGB')
                    axes[row_index, column_index].imshow(tile, interpolation='nearest')
                axes[row_index, column_index].set_axis_off()
        fig.text(0.26, 0.963, 'Faint', ha='center', va='center', fontsize=40)
        fig.text(0.74, 0.963, 'Vivid', ha='center', va='center', fontsize=40)
        for extension in ['png', 'pdf', 'svg']:
            fig.savefig(plot_dir / f'faint_vivid_examples_4x4.{extension}', bbox_inches='tight', pad_inches=0.12)
        return fig

    figure = plot_dataset_examples()
    return


if __name__ == '__main__':
    app.run()
