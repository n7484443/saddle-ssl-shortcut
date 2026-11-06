import marimo

__generated_with = '0.23.1'
app = marimo.App(width='full')


@app.cell
def _():
    import sys
    from pathlib import Path
    plot_dir = Path(__file__).resolve().parent
    workspace = next(p for p in Path(__file__).resolve().parents if (p / "common.py").is_file())
    import csv
    import json
    import hashlib
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
    return Image, csv, hashlib, json, plot_dir, plt, random, workspace


@app.cell
def _(Image, csv, hashlib, json, plot_dir, plt, random, workspace):
    dataset = plot_dir / 'examples'
    manifest = dataset / 'manifest.csv'
    with manifest.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    selected = random.Random(0).sample(range(len(rows)), 8)

    def plot_dataset_examples():
        fig, axes = plt.subplots(4, 4, figsize=(10.8, 11.2))
        fig.subplots_adjust(left=0.025, right=0.975, bottom=0.025, top=0.93, wspace=0.055, hspace=0.055)
        samples = []
        for row_index in range(4):
            for column_index in range(4):
                condition = 0 if column_index < 2 else 1
                selected_index = selected[2 * row_index + column_index % 2]
                record = rows[selected_index]
                assert int(record[f'cond{condition}_p']) == [10, 45][condition]
                path = dataset / record[f'cond{condition}_image']
                with Image.open(path) as original:
                    tile = original.convert('RGB')
                    size = tile.size
                    axes[row_index, column_index].imshow(tile, interpolation='nearest')
                axes[row_index, column_index].set_axis_off()
                samples.append(dict(row=row_index, column=column_index, condition=['Faint', 'Vivid'][condition],
                    sample_id=record['sample_id'], label=int(record[f'cond{condition}_label']),
                    path=str(path), size=list(size), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        fig.text(0.26, 0.963, 'Faint', ha='center', va='center', fontsize=40)
        fig.text(0.74, 0.963, 'Vivid', ha='center', va='center', fontsize=40)
        for extension in ['png', 'pdf', 'svg']:
            fig.savefig(plot_dir / f'faint_vivid_examples_4x4.{extension}', bbox_inches='tight', pad_inches=0.12)
        return fig, samples

    figure, samples = plot_dataset_examples()
    provenance = dict(dataset=str(dataset), split='train', sampling_seed=0, selected_manifest_rows=selected,
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(), grid=[4, 4],
        faint_count=8, vivid_count=8, augmentation=False, normalization=False,
        image_interpolation='nearest', samples=samples, GPU_operations=0)
    (plot_dir / 'example_selection.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding='utf-8')
    return (figure,)


if __name__ == '__main__':
    app.run()
