from pathlib import Path
import sys
import os

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "common.py").is_file())
os.environ['MPLBACKEND'] = 'Agg'
os.environ['MPLCONFIGDIR'] = str(HERE/'mpl_config')
import marimo

__generated_with = '0.23.16'
app = marimo.App(width='full')


@app.cell
def _():
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    from matplotlib.transforms import Bbox
    import seaborn as sns
    device = torch.device('cuda')
    sns.set_theme(style='whitegrid', font='cmr10', font_scale=1.9)
    plt.rcParams['axes.formatter.use_mathtext'] = True
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['pdf.use14corefonts'] = False
    colors = sns.color_palette()
    colors_darker = sns.color_palette('dark', 10)

    def plot_scatter():
        base = ROOT/'feature_preference/practical/waterbirds/datas/BarlowTwins_multiple_filtered_0.01'
        def load_accuracy(seed, ratio):
            path = base/str(seed)/str(ratio)/'accuracy.csv'
            values = np.loadtxt(path, delimiter=',').T
            return values

        ratios = [.03, .06, .09]
        scatter_ratios = [.03+.01*i for i in range(7)]
        fig, axes = plt.subplots(1, 2, figsize=(22.4, 8))
        for index, ratio in enumerate(ratios):
            accuracy, _, steps = load_accuracy(0, ratio)
            axes[0].plot(steps, accuracy, color=colors_darker[index], linewidth=5, alpha=.9,
                         label=f'Object ratio : {ratio:.3f}')
        axes[0].set_xlabel('Step')
        axes[0].set_ylabel('Linear Probing Test Accuracy')
        axes[0].set_ylim(30, 100)
        axes[0].legend(loc='lower right')
        means, stds = [], []
        for index, ratio in enumerate(scatter_ratios):
            endpoints = [load_accuracy(seed, ratio)[2][-1] for seed in range(6)]
            axes[1].scatter([ratio]*6, endpoints, s=150, alpha=.25, color=colors[index])
            means.append(np.mean(endpoints))
            stds.append(np.std(endpoints))
        means, stds = np.asarray(means), np.asarray(stds)
        axes[1].plot(scatter_ratios, means, color='red', linewidth=3, alpha=.8, marker='o')
        axes[1].fill_between(scatter_ratios, means-stds, means+stds, alpha=.12, color='red')
        axes[1].set_xlabel('Object Ratio')
        axes[1].set_ylabel('Training step at 80% accuracy')
        axes[1].grid(True, alpha=.3)
        fig.tight_layout()

        bounds = Bbox.from_bounds(0, 0, 22.4, 8)
        fig.savefig(HERE/'scatter_ratio_vs_steps.pdf', bbox_inches=bounds, pad_inches=.1)
        fig.savefig(HERE/'scatter_ratio_vs_steps.png', bbox_inches=bounds, pad_inches=.1)
        fig.savefig(HERE/'scatter_ratio_vs_steps.svg', bbox_inches=bounds, pad_inches=.1)
        return fig

    figure = plot_scatter()
    return


if __name__ == '__main__':
    if '--export' in sys.argv:
        import ast
        tree = ast.parse(Path(__file__).read_text(encoding='utf-8'))
        scope = dict(globals())
        for cell in [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.decorator_list]:
            body = cell.body[:-1] if isinstance(cell.body[-1], ast.Return) else cell.body
            exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), __file__, 'exec'), scope)
    else:
        app.run()
